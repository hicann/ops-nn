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
 * \file test_geir_weight_quant_batch_matmul_v2.cpp
 * \brief
 */

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <cmath>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "graph.h"
#include "types.h"
#include "tensor.h"
#include "ge_error_codes.h"
#include "ge_api_types.h"
#include "ge_api.h"
#include "array_ops.h"
#include "ge_ir_build.h"

#include "../../op_graph/weight_quant_batch_matmul_v2_proto.h"

#define FAILED -1
#define SUCCESS 0

using namespace ge;
using std::map;
using std::string;
using std::vector;

// perchannel量化场景的shape常量，golden计算复用
static const int64_t M_DIM = 16;
static const int64_t K_DIM = 32;
static const int64_t N_DIM = 64;

static std::mt19937 g_rng(2026); // 固定种子，保证用例可复现

#define ADD_INPUT(inputIndex, inputName, inputDtype, inputShape)                                                  \
    vector<int64_t> placeholder##inputIndex##_shape = inputShape;                                                 \
    auto placeholder##inputIndex = op::Data("placeholder" + inputIndex).set_attr_index(0);                        \
    TensorDesc placeholder##inputIndex##_desc = TensorDesc(ge::Shape(placeholder##inputIndex##_shape), FORMAT_ND, \
                                                           inputDtype);                                           \
    placeholder##inputIndex##_desc.SetPlacement(ge::kPlacementHost);                                              \
    placeholder##inputIndex##_desc.SetFormat(FORMAT_ND);                                                          \
    Tensor tensor_placeholder##inputIndex;                                                                        \
    ret = GenRandomData(placeholder##inputIndex##_shape, tensor_placeholder##inputIndex,                          \
                        placeholder##inputIndex##_desc, inputDtype);                                              \
    if (ret != SUCCESS) {                                                                                         \
        printf("[XIR]: Generate input data failed\n");                                                            \
        return FAILED;                                                                                            \
    }                                                                                                             \
    placeholder##inputIndex.update_input_desc_x(placeholder##inputIndex##_desc);                                  \
    placeholder##inputIndex.update_output_desc_y(placeholder##inputIndex##_desc);                                 \
    input.push_back(tensor_placeholder##inputIndex);                                                              \
    graph.AddOp(placeholder##inputIndex);                                                                         \
    weight_quant_matmul.set_input_##inputName(placeholder##inputIndex);                                           \
    inputs.push_back(placeholder##inputIndex);

#define ADD_INPUT_ATTR(attrName, attrValue) weight_quant_matmul.set_attr_##attrName(attrValue);

#define ADD_OUTPUT(outputIndex, outputName, outputDtype, outputShape)                                       \
    TensorDesc outputName##outputIndex##_desc = TensorDesc(ge::Shape(outputShape), FORMAT_ND, outputDtype); \
    weight_quant_matmul.update_output_desc_##outputName(outputName##outputIndex##_desc);

// [lo, hi]均匀分布随机浮点
float RandFloat(float lo, float hi)
{
    std::uniform_real_distribution<float> dist(lo, hi);
    return dist(g_rng);
}

// float转FLOAT16比特（截断舍入；golden解码使用相同比特，无需四舍五入）
uint16_t FloatToFp16Bits(float f)
{
    uint32_t u;
    memcpy(&u, &f, sizeof(u));
    uint16_t sign = static_cast<uint16_t>((u >> 16) & 0x8000);
    int exp = static_cast<int>((u >> 23) & 0xFF) - 127 + 15;
    uint32_t man = u & 0x7FFFFF;
    if (exp <= 0) {
        return sign; // 小于FP16最小正规格化数的值按0处理
    }
    if (exp >= 31) {
        return sign | 0x7BFF; // 不会触发，钳位到FP16最大值
    }
    return sign | static_cast<uint16_t>((exp << 10) | (man >> 13));
}

// FLOAT16解码：1位符号、5位指数（偏置15）、10位尾数
float Fp16ToFloat(uint16_t h)
{
    int sign = (h >> 15) & 1;
    int exp = (h >> 10) & 0x1F;
    int man = h & 0x3FF;
    float v = (exp == 0) ? ldexpf(man / 1024.0f, -14) : ldexpf(1.0f + man / 1024.0f, exp - 15);
    return sign ? -v : v;
}

int32_t GenRandomData(vector<int64_t> shapes, Tensor& input_tensor, TensorDesc& input_tensor_desc, DataType data_type)
{
    input_tensor_desc.SetRealDimCnt(shapes.size());
    uint64_t elem_num = 1;
    for (auto dim : shapes) {
        elem_num *= dim;
    }
    uint64_t data_len = elem_num * ((data_type == ge::DT_FLOAT16) ? 2 : 1); // 本用例输入仅FLOAT16/INT8
    uint8_t* pData = new (std::nothrow) uint8_t[data_len];
    if (data_type == ge::DT_FLOAT16) {
        uint16_t* pFp16 = reinterpret_cast<uint16_t*>(pData);
        for (uint64_t i = 0; i < elem_num; ++i) {
            pFp16[i] = FloatToFp16Bits(RandFloat(-1.0f, 1.0f));
        }
    } else if (data_type == ge::DT_INT8) {
        for (uint64_t i = 0; i < elem_num; ++i) {
            pData[i] = static_cast<uint8_t>(static_cast<int8_t>(g_rng() % 16 - 8)); // [-8, 7]
        }
    } else {
        for (uint64_t i = 0; i < data_len; ++i) {
            pData[i] = static_cast<uint8_t>(g_rng() & 0xFF);
        }
    }
    input_tensor = Tensor(input_tensor_desc, pData, data_len);
    return SUCCESS;
}

// golden计算：perchannel量化公式 y[m,n] = sum_k(x[m,k] * weight[n,k] * antiquantScale[n])
// antiquant_offset未传入，按0处理
void ComputeGolden(const uint8_t* xData, const uint8_t* weightData, const uint8_t* scaleData, vector<float>& golden)
{
    golden.resize(M_DIM * N_DIM);
    for (int64_t m = 0; m < M_DIM; ++m) {
        for (int64_t n = 0; n < N_DIM; ++n) {
            float acc = 0.0f;
            for (int64_t k = 0; k < K_DIM; ++k) {
                float xv = Fp16ToFloat(reinterpret_cast<const uint16_t*>(xData)[m * K_DIM + k]);
                float wv = static_cast<float>(reinterpret_cast<const int8_t*>(weightData)[n * K_DIM + k]);
                acc += xv * wv;
            }
            golden[m * N_DIM + n] = acc * Fp16ToFloat(reinterpret_cast<const uint16_t*>(scaleData)[n]);
        }
    }
}

int32_t CompareWithGolden(const uint8_t* outData, const vector<float>& golden)
{
    float relTol = 0.02f; // FLOAT16输出相对精度2^-11，叠加累加顺序差异，取2%
    float absTol = 1e-2f;
    double maxDiff = 0.0;
    double maxRel = 0.0;
    int64_t badCount = 0;
    for (uint64_t i = 0; i < golden.size(); ++i) {
        float actual = Fp16ToFloat(reinterpret_cast<const uint16_t*>(outData)[i]);
        double diff = fabs(actual - golden[i]);
        double rel = diff / (fabs(golden[i]) + 1e-6);
        maxDiff = std::max(maxDiff, diff);
        maxRel = std::max(maxRel, rel);
        if (diff > absTol + relTol * fabs(golden[i])) {
            ++badCount;
        }
    }
    printf("[XIR]: golden compare: max abs diff = %g, max rel diff = %g, mismatch = %ld / %lu -> %s\n", maxDiff, maxRel,
           badCount, golden.size(), badCount == 0 ? "PASS" : "FAIL");
    return badCount == 0 ? SUCCESS : FAILED;
}

int CreateOppInGraph(std::vector<ge::Tensor>& input, std::vector<Operator>& inputs, std::vector<Operator>& outputs,
                     Graph& graph)
{
    Status ret = SUCCESS;
    // 自定义代码：添加单算子定义到图中，以perchannel量化场景为例
    auto weight_quant_matmul = op::WeightQuantBatchMatmulV2("weight_quant_batch_matmul_v2");
    std::vector<int64_t> xShape = {M_DIM, K_DIM};       // (M, K)
    std::vector<int64_t> weightShape = {N_DIM, K_DIM};  // (N, K)，transpose_weight=true
    std::vector<int64_t> antiquantScaleShape = {N_DIM}; // (N,)
    std::vector<int64_t> yShape = {M_DIM, N_DIM};       // (M, N)

    ADD_INPUT(1, x, DT_FLOAT16, xShape);
    ADD_INPUT(2, weight, DT_INT8, weightShape);
    ADD_INPUT(3, antiquant_scale, DT_FLOAT16, antiquantScaleShape);

    ADD_OUTPUT(1, y, DT_FLOAT16, yShape);

    ADD_INPUT_ATTR(transpose_x, false);
    ADD_INPUT_ATTR(transpose_weight, true);
    ADD_INPUT_ATTR(antiquant_group_size, 0);
    ADD_INPUT_ATTR(dtype, -1);
    ADD_INPUT_ATTR(inner_precise, 0);

    outputs.push_back(weight_quant_matmul);
    // 添加完毕
    return SUCCESS;
}

int main()
{
    std::map<AscendString, AscendString> globalOptions = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    if (ge::GEInitialize(globalOptions) != SUCCESS) {
        printf("[XIR]: GEInitialize failed\n");
        return FAILED;
    }

    Graph graph("tc_ge_irrun_test");
    std::vector<ge::Tensor> input;
    std::vector<Operator> inputs{};
    std::vector<Operator> outputs{};
    if (CreateOppInGraph(input, inputs, outputs, graph) != SUCCESS) {
        printf("[XIR]: Create graph failed\n");
        GEFinalize();
        return FAILED;
    }
    graph.SetInputs(inputs).SetOutputs(outputs);

    std::map<AscendString, AscendString> options = {};
    ge::Session* session = new Session(options);
    uint32_t graphId = 0;
    if (session == nullptr || session->AddGraph(graphId, graph, options) != SUCCESS) {
        printf("[XIR]: AddGraph failed\n");
        GEFinalize();
        return FAILED;
    }

    std::vector<ge::Tensor> output;
    if (session->RunGraph(graphId, input, output) != SUCCESS) {
        printf("[XIR]: RunGraph failed\n");
        delete session;
        GEFinalize();
        return FAILED;
    }

    // golden校验：host侧按perchannel量化公式计算期望值并与NPU输出比较
    vector<float> golden;
    ComputeGolden(input[0].GetData(), input[1].GetData(), input[2].GetData(), golden);
    int32_t ret = CompareWithGolden(output[0].GetData(), golden);

    delete session;
    GEFinalize();
    return ret;
}
