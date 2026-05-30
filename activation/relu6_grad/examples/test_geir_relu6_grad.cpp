/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// End-to-end GE-IR example for Relu6Grad on Ascend 950.
//
// Each test case builds a one-op graph, runs it on device, and checks the
// output against a pre-computed expected value. The example exits with
// status 0 only if every case passes.
//
// Semantics: dx = (0 < x < 6) ? dy : 0 (element-wise, broadcast).
// Strict open interval - x == 0 and x == 6 both yield 0.
// Cases below specifically exercise:
//   * 0 < x < 6                ->  dx = dy                  (interval pass-through)
//   * x = 0 or x = 6           ->  dx = 0                   (open-interval boundary)
//   * x < 0 or x > 6           ->  dx = 0                   (out-of-band)
//   * x = NaN                  ->  dx = 0                   (NaN comparisons are false)
//   * x = +inf / -inf          ->  dx = 0                   (out-of-band)
//   * x in (0, 6), dy = NaN    ->  dx = NaN                 (dy passes through verbatim)
//   * x in (0, 6), dy = +inf   ->  dx = +inf                (dy passes through verbatim)

#include <iostream>
#include <fstream>
#include <string.h>
#include <stdint.h>
#include <vector>
#include <string>
#include <map>
#include <cmath>
#include <limits>
#include "assert.h"

#include "graph.h"
#include "types.h"
#include "tensor.h"
#include "ge_error_codes.h"
#include "ge_api_types.h"
#include "ge_api.h"
#include "array_ops.h"
#include "ge_ir_build.h"

#include "nn_other.h"
#include "../op_graph/relu6_grad_proto.h"

#define FAILED -1
#define SUCCESS 0

using namespace ge;
using std::map;
using std::string;
using std::vector;

static string GetTime()
{
    time_t t;
    time(&t);
    char buf[64];
    strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", localtime(&t));
    return buf;
}

static uint32_t GetDataTypeSize(DataType dt)
{
    if (dt == ge::DT_FLOAT)   return 4;
    if (dt == ge::DT_FLOAT16) return 2;
    if (dt == ge::DT_BF16)    return 2;
    return 1;
}

// IEEE-754 fp32 -> half (round-to-nearest-even).
static uint16_t FloatToHalfBits(float v)
{
    uint32_t x;
    memcpy(&x, &v, sizeof(x));
    uint32_t sign  = (x >> 31) & 0x1;
    uint32_t exp32 = (x >> 23) & 0xff;
    uint32_t mant  = x & 0x7fffff;
    uint16_t out;
    if (exp32 == 0xff) {
        out = (uint16_t)((sign << 15) | 0x7c00 | (mant ? 0x200 : 0));
    } else if (exp32 == 0) {
        out = (uint16_t)(sign << 15);
    } else {
        int32_t newExp = (int32_t)exp32 - 127 + 15;
        if (newExp >= 31) {
            out = (uint16_t)((sign << 15) | 0x7c00);
        } else if (newExp <= 0) {
            out = (uint16_t)(sign << 15);
        } else {
            uint32_t roundBit = (mant >> 12) & 0x1;
            uint16_t m = (uint16_t)(mant >> 13);
            out = (uint16_t)((sign << 15) | (newExp << 10) | m);
            if (roundBit) out++;
        }
    }
    return out;
}

static float HalfBitsToFloat(uint16_t h)
{
    uint32_t sign  = (h >> 15) & 0x1;
    uint32_t exp16 = (h >> 10) & 0x1f;
    uint32_t mant  = h & 0x3ff;
    uint32_t out;
    if (exp16 == 0) {
        out = sign << 31;
    } else if (exp16 == 31) {
        out = (sign << 31) | 0x7f800000 | (mant << 13);
    } else {
        uint32_t newExp = exp16 - 15 + 127;
        out = (sign << 31) | (newExp << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &out, sizeof(f));
    return f;
}

// IEEE-754 fp32 -> bfloat16 (round-to-nearest-even truncating low 16 bits).
static uint16_t FloatToBf16Bits(float v)
{
    uint32_t x;
    memcpy(&x, &v, sizeof(x));
    // Preserve NaN signaling bit pattern: ensure result stays NaN.
    if (((x >> 23) & 0xff) == 0xff && (x & 0x7fffff) != 0) {
        return (uint16_t)((x >> 16) | 0x40);
    }
    uint32_t rounded = x + 0x7fff + ((x >> 16) & 1);
    return (uint16_t)(rounded >> 16);
}

static float Bf16BitsToFloat(uint16_t b)
{
    uint32_t x = ((uint32_t)b) << 16;
    float f;
    memcpy(&f, &x, sizeof(f));
    return f;
}

// Allocate a tensor and fill with a single value of the given dtype.
// `value` is a double so that callers can pass inf / nan / -0.0 directly.
static int32_t GenConstTensor(
    const vector<int64_t>& shape, DataType dtype, double value, Tensor& tensor, TensorDesc& desc)
{
    desc.SetRealDimCnt(shape.size());
    size_t numel = 1;
    for (auto d : shape) numel *= (size_t)d;
    size_t bytes = numel * GetDataTypeSize(dtype);

    if (dtype == ge::DT_FLOAT) {
        float* p = new (std::nothrow) float[numel];
        float v = (float)value;
        for (size_t i = 0; i < numel; ++i) p[i] = v;
        tensor = Tensor(desc, (uint8_t*)p, bytes);
    } else if (dtype == ge::DT_FLOAT16) {
        uint16_t* p = new (std::nothrow) uint16_t[numel];
        uint16_t v = FloatToHalfBits((float)value);
        for (size_t i = 0; i < numel; ++i) p[i] = v;
        tensor = Tensor(desc, (uint8_t*)p, bytes);
    } else if (dtype == ge::DT_BF16) {
        uint16_t* p = new (std::nothrow) uint16_t[numel];
        uint16_t v = FloatToBf16Bits((float)value);
        for (size_t i = 0; i < numel; ++i) p[i] = v;
        tensor = Tensor(desc, (uint8_t*)p, bytes);
    } else {
        return FAILED;
    }
    return SUCCESS;
}

// Build a graph that computes dx = Relu6Grad(dy, x) with two constant inputs.
static int BuildRelu6GradGraph(
    Graph& graph, std::vector<ge::Tensor>& inputData, std::vector<Operator>& inputs,
    std::vector<Operator>& outputs, DataType dtype, double dyVal, double xVal,
    const vector<int64_t>& sDy, const vector<int64_t>& sX,
    const vector<int64_t>& outShape, const string& caseTag)
{
    auto op = ge::op::Relu6Grad("r6g_" + caseTag);

    // Input gradients (dy)
    {
        auto ph = ge::op::Data("phDy_" + caseTag).set_attr_index(0);
        TensorDesc d(ge::Shape(sDy), FORMAT_ND, dtype);
        d.SetPlacement(ge::kPlacementHost);
        d.SetFormat(FORMAT_ND);
        Tensor t;
        if (GenConstTensor(sDy, dtype, dyVal, t, d) != SUCCESS) return FAILED;
        ph.update_input_desc_x(d);
        inputData.push_back(t);
        graph.AddOp(ph);
        op.set_input_gradients(ph);
        inputs.push_back(ph);
    }
    // Input features (x)
    {
        auto ph = ge::op::Data("phX_" + caseTag).set_attr_index(0);
        TensorDesc d(ge::Shape(sX), FORMAT_ND, dtype);
        d.SetPlacement(ge::kPlacementHost);
        d.SetFormat(FORMAT_ND);
        Tensor t;
        if (GenConstTensor(sX, dtype, xVal, t, d) != SUCCESS) return FAILED;
        ph.update_input_desc_x(d);
        inputData.push_back(t);
        graph.AddOp(ph);
        op.set_input_features(ph);
        inputs.push_back(ph);
    }
    // Output backprops (dx)
    {
        TensorDesc d(ge::Shape(outShape), FORMAT_ND, dtype);
        op.update_output_desc_backprops(d);
    }
    outputs.push_back(op);
    return SUCCESS;
}

// Expected comparison policy for a single element.
//   kExact: |got - expected| <= absTol (numeric)
//   kIsNaN: got must be NaN regardless of expected
struct ExpectPolicy {
    enum class Mode { kExact, kIsNaN };
    Mode   mode;
    double expected;
    double absTol;
};

struct CaseSpec {
    string                 tag;
    DataType               dtype;
    double                 dyVal, xVal;
    vector<int64_t>        sDy, sX, outShape;
    ExpectPolicy           expect;
};

static bool MatchOne(double got, const ExpectPolicy& expect)
{
    if (expect.mode == ExpectPolicy::Mode::kIsNaN) {
        return std::isnan(got);
    }
    return std::abs(got - expect.expected) <= expect.absTol;
}

static int RunOneCase(const CaseSpec& spec)
{
    printf("\n%s - INFO - [XIR]: ===== case %s start =====\n", GetTime().c_str(), spec.tag.c_str());
    Graph graph(("g_" + spec.tag).c_str());
    std::vector<ge::Tensor> inputData;
    std::vector<Operator> inputs;
    std::vector<Operator> outputs;
    if (BuildRelu6GradGraph(graph, inputData, inputs, outputs, spec.dtype,
                            spec.dyVal, spec.xVal,
                            spec.sDy, spec.sX, spec.outShape, spec.tag) != SUCCESS) {
        printf("BuildRelu6GradGraph failed\n");
        return FAILED;
    }
    graph.SetInputs(inputs).SetOutputs(outputs);

    std::map<AscendString, AscendString> buildOpts;
    ge::Session* session = new Session(buildOpts);
    if (!session) return FAILED;

    std::map<AscendString, AscendString> graphOpts;
    uint32_t graphId = 1;
    if (session->AddGraph(graphId, graph, graphOpts) != SUCCESS) {
        printf("AddGraph failed\n");
        delete session;
        return FAILED;
    }

    std::vector<ge::Tensor> output;
    Status ret = session->RunGraph(graphId, inputData, output);
    if (ret != SUCCESS) {
        printf("RunGraph failed\n");
        delete session;
        return FAILED;
    }

    int failCount = 0;
    for (size_t i = 0; i < output.size(); ++i) {
        int64_t numel = output[i].GetTensorDesc().GetShape().GetShapeSize();
        uint8_t* raw = output[i].GetData();
        for (int64_t j = 0; j < numel; ++j) {
            double got;
            if (spec.dtype == ge::DT_FLOAT) {
                got = (double)reinterpret_cast<const float*>(raw)[j];
            } else if (spec.dtype == ge::DT_FLOAT16) {
                got = (double)HalfBitsToFloat(reinterpret_cast<const uint16_t*>(raw)[j]);
            } else if (spec.dtype == ge::DT_BF16) {
                got = (double)Bf16BitsToFloat(reinterpret_cast<const uint16_t*>(raw)[j]);
            } else {
                got = 0.0;
            }
            if (!MatchOne(got, spec.expect)) {
                if (spec.expect.mode == ExpectPolicy::Mode::kIsNaN) {
                    printf("  MISMATCH @%ld: got=%.6f expected=NaN\n", (long)j, got);
                } else {
                    printf("  MISMATCH @%ld: got=%.6f expected=%.6f absTol=%.6f\n",
                           (long)j, got, spec.expect.expected, spec.expect.absTol);
                }
                failCount++;
            }
        }
    }
    delete session;

    if (failCount != 0) {
        printf("%s - FAIL - case %s: %d mismatches (numel=%ld)\n",
               GetTime().c_str(), spec.tag.c_str(), failCount,
               (long)output[0].GetTensorDesc().GetShape().GetShapeSize());
        return FAILED;
    }
    if (spec.expect.mode == ExpectPolicy::Mode::kIsNaN) {
        printf("%s - PASS - case %s (expected=NaN)\n", GetTime().c_str(), spec.tag.c_str());
    } else {
        printf("%s - PASS - case %s (expected=%.4f)\n",
               GetTime().c_str(), spec.tag.c_str(), spec.expect.expected);
    }
    return SUCCESS;
}

int main(int argc, char* argv[])
{
    (void)argc; (void)argv;

    std::map<AscendString, AscendString> globalOpts = {
        {"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    if (ge::GEInitialize(globalOpts) != SUCCESS) {
        printf("GEInitialize failed\n");
        return FAILED;
    }

    const double kInf  = std::numeric_limits<double>::infinity();
    const double kNeg0 = -0.0;
    const double kNaN  = std::numeric_limits<double>::quiet_NaN();

    auto exact = [](double v, double tol) {
        return ExpectPolicy{ExpectPolicy::Mode::kExact, v, tol};
    };
    auto nan = []() {
        return ExpectPolicy{ExpectPolicy::Mode::kIsNaN, 0.0, 0.0};
    };

    std::vector<CaseSpec> cases = {
        // tag                          dtype           dy     x      sDy     sX      out     expect
        // --- fp32 interval / boundary / out-of-band ---
        {"fp32_inside_band",          ge::DT_FLOAT,   2.5,   3.0,  {2,2},  {2,2},  {2,2},  exact(2.5,    1e-5)},
        {"fp32_boundary_x_eq_0",      ge::DT_FLOAT,   2.5,   0.0,  {2,2},  {2,2},  {2,2},  exact(0.0,    1e-5)},
        {"fp32_boundary_x_eq_6",      ge::DT_FLOAT,   2.5,   6.0,  {2,2},  {2,2},  {2,2},  exact(0.0,    1e-5)},
        {"fp32_x_neg",                ge::DT_FLOAT,   2.5,  -1.5,  {2,2},  {2,2},  {2,2},  exact(0.0,    1e-5)},
        {"fp32_x_gt_6",               ge::DT_FLOAT,   2.5,   6.5,  {2,2},  {2,2},  {2,2},  exact(0.0,    1e-5)},
        // --- fp32 IEEE-754 special values on x ---
        {"fp32_x_nan",                ge::DT_FLOAT,   2.5,   kNaN, {2,2},  {2,2},  {2,2},  exact(0.0,    1e-5)},
        {"fp32_x_pos_inf",            ge::DT_FLOAT,   2.5,   kInf, {2,2},  {2,2},  {2,2},  exact(0.0,    1e-5)},
        {"fp32_x_neg_inf",            ge::DT_FLOAT,   2.5,  -kInf, {2,2},  {2,2},  {2,2},  exact(0.0,    1e-5)},
        {"fp32_x_neg_zero",           ge::DT_FLOAT,   2.5,   kNeg0,{2,2},  {2,2},  {2,2},  exact(0.0,    1e-5)},
        // --- fp32 dy passes through verbatim when mask is true ---
        {"fp32_dy_nan_inband",        ge::DT_FLOAT,   kNaN,  3.0,  {2,2},  {2,2},  {2,2},  nan()              },
        {"fp32_dy_inf_inband",        ge::DT_FLOAT,   kInf,  3.0,  {2,2},  {2,2},  {2,2},  exact(kInf,   0.0 )},
        // --- fp16 ---
        {"fp16_inside_band",          ge::DT_FLOAT16, 1.5,   2.0,  {4},    {4},    {4},    exact(1.5,    5e-3)},
        {"fp16_boundary_x_eq_0",      ge::DT_FLOAT16, 1.5,   0.0,  {4},    {4},    {4},    exact(0.0,    5e-3)},
        {"fp16_boundary_x_eq_6",      ge::DT_FLOAT16, 1.5,   6.0,  {4},    {4},    {4},    exact(0.0,    5e-3)},
        // --- bf16 ---
        {"bf16_inside_band",          ge::DT_BF16,    1.0,   2.0,  {4},    {4},    {4},    exact(1.0,    2e-2)},
        {"bf16_boundary_x_eq_0",      ge::DT_BF16,    1.0,   0.0,  {4},    {4},    {4},    exact(0.0,    2e-2)},
        {"bf16_boundary_x_eq_6",      ge::DT_BF16,    1.0,   6.0,  {4},    {4},    {4},    exact(0.0,    2e-2)},
        {"bf16_x_pos_inf",            ge::DT_BF16,    1.0,   kInf, {4},    {4},    {4},    exact(0.0,    2e-2)},
        // --- broadcast ---
        {"fp32_broadcast_inside",     ge::DT_FLOAT,   2.0,   3.0,  {4,1},  {1,4},  {4,4},  exact(2.0,    1e-5)},
        {"fp32_broadcast_boundary",   ge::DT_FLOAT,   2.0,   6.0,  {4,1},  {1,4},  {4,4},  exact(0.0,    1e-5)},
        {"fp32_broadcast_cross_rank", ge::DT_FLOAT,   2.0,   3.0,  {4},    {3,4},  {3,4},  exact(2.0,    1e-5)},
    };

    int allFail = 0;
    for (const auto& c : cases) {
        if (RunOneCase(c) != SUCCESS) {
            allFail++;
        }
    }

    ge::AscendString errMsg = ge::GEGetErrorMsgV2();
    string errStr(errMsg.GetString());
    if (!errStr.empty()) std::cout << "GE error: " << errStr << std::endl;

    ge::GEFinalize();

    if (allFail != 0) {
        printf("\n%s - FAIL - [XIR]: %d / %zu cases failed\n",
               GetTime().c_str(), allFail, cases.size());
        return FAILED;
    }
    printf("\n%s - PASS - [XIR]: all %zu relu6_grad cases passed\n",
           GetTime().c_str(), cases.size());
    return SUCCESS;
}
