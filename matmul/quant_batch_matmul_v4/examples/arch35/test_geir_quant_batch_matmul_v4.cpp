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
 * \file test_geir_quant_batch_matmul_v4.cpp
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

#include "../../op_graph/quant_batch_matmul_v4_proto.h"

#define FAILED -1
#define SUCCESS 0

using namespace ge;
using std::map;
using std::string;
using std::vector;

// MxA8W4量化场景的shape常量，golden计算复用
static const int64_t M_DIM = 16;
static const int64_t K_DIM = 64;
static const int64_t N_DIM = 32;
static const int64_t GROUP_SIZE = 32;                  // MX量化模式的groupSizeK
static const int64_t K_GROUP_NUM = K_DIM / GROUP_SIZE; // K轴分组数

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
    quant_matmul.set_input_##inputName(placeholder##inputIndex);                                                  \
    inputs.push_back(placeholder##inputIndex);

#define ADD_INPUT_ATTR(attrName, attrValue) quant_matmul.set_attr_##attrName(attrValue);

#define ADD_OUTPUT(outputIndex, outputName, outputDtype, outputShape)                                       \
    TensorDesc outputName##outputIndex##_desc = TensorDesc(ge::Shape(outputShape), FORMAT_ND, outputDtype); \
    quant_matmul.update_output_desc_##outputName(outputName##outputIndex##_desc);

// 随机生成FLOAT8_E4M3FN比特：符号随机，指数域[5, 9]（对应2^-2 ~ 2^2），尾数随机，避开NaN编码
uint8_t GenFp8E4m3Bits()
{
    return static_cast<uint8_t>(((g_rng() & 1) << 7) | ((5 + g_rng() % 5) << 3) | (g_rng() & 0x7));
}

// 随机生成一个存放2个FLOAT4_E2M1的字节，E2M1的全部16个编码均为有效值
uint8_t GenFp4E2m1Pair() { return static_cast<uint8_t>(((g_rng() & 0xF) << 4) | (g_rng() & 0xF)); }

// 随机生成FLOAT8_E8M0比特：指数域[124, 130]，对应scale取值2^-3 ~ 2^3
uint8_t GenE8m0Bits() { return static_cast<uint8_t>(124 + g_rng() % 7); }

int32_t GenRandomData(vector<int64_t> shapes, Tensor& input_tensor, TensorDesc& input_tensor_desc, DataType data_type)
{
    input_tensor_desc.SetRealDimCnt(shapes.size());
    uint64_t elem_num = 1;
    for (auto dim : shapes) {
        elem_num *= dim;
    }
    // 本用例输入均为1字节/元素类型，FLOAT4_E2M1为每字节2个元素
    uint64_t data_len = (data_type == ge::DT_FLOAT4_E2M1) ? elem_num / 2 : elem_num;
    uint8_t* pData = new (std::nothrow) uint8_t[data_len];
    for (uint64_t i = 0; i < data_len; ++i) {
        if (data_type == ge::DT_FLOAT8_E4M3FN) {
            pData[i] = GenFp8E4m3Bits();
        } else if (data_type == ge::DT_FLOAT4_E2M1) {
            pData[i] = GenFp4E2m1Pair();
        } else if (data_type == ge::DT_FLOAT8_E8M0) {
            pData[i] = GenE8m0Bits();
        } else {
            pData[i] = static_cast<uint8_t>(g_rng() & 0xFF);
        }
    }
    input_tensor = Tensor(input_tensor_desc, pData, data_len);
    return SUCCESS;
}

// FLOAT8_E4M3FN解码：1位符号、4位指数（偏置7）、3位尾数
float Fp8E4m3ToFloat(uint8_t b)
{
    int sign = (b >> 7) & 1;
    int exp = (b >> 3) & 0xF;
    int man = b & 0x7;
    float v = (exp == 0) ? ldexpf(man / 8.0f, -6) : ldexpf(1.0f + man / 8.0f, exp - 7);
    return sign ? -v : v;
}

// FLOAT4_E2M1解码：1位符号、2位指数（偏置1）、1位尾数
float Fp4E2m1ToFloat(uint8_t nibble)
{
    int sign = (nibble >> 3) & 1;
    int exp = (nibble >> 1) & 0x3;
    int man = nibble & 1;
    float v = (exp == 0) ? 0.5f * man : ldexpf(1.0f + 0.5f * man, exp - 1);
    return sign ? -v : v;
}

// FLOAT8_E8M0解码：8位指数（偏置127），无符号尾数，值为2^(e-127)
float E8m0ToFloat(uint8_t e) { return ldexpf(1.0f, static_cast<int>(e) - 127); }

float Bf16ToFloat(uint16_t b)
{
    uint32_t u = static_cast<uint32_t>(b) << 16;
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

// 取x2第(n, k)个FLOAT4_E2M1元素。
// 本场景（transpose_x2=true、x2为FLOAT4_E2M1）下，x2传入数据按32个元素一组沿K轴分块排布，
// 即(k1, n, k0)顺序，k0=32：第k组的块内偏移为k % 32，块索引为k / 32，低半字节为偶数偏移
float GetFp4Element(const uint8_t* packed, int64_t n, int64_t k)
{
    int64_t blockIdx = k / GROUP_SIZE;
    int64_t inner = k % GROUP_SIZE;
    uint8_t byte = packed[blockIdx * (N_DIM * GROUP_SIZE / 2) + n * (GROUP_SIZE / 2) + inner / 2];
    return Fp4E2m1ToFloat((inner % 2 == 0) ? (byte & 0xF) : (byte >> 4));
}

// golden计算：MX量化模式公式 out[m,n] = sum_j((sum_k(x1[m,k] * x2[n,k])) * x1Scale[m,j] * x2Scale[n,j])
// scale的shape为(dim, ceil(K_GROUP_NUM / 2), 2)，j组的scale存放在扁平索引j处
void ComputeGolden(const uint8_t* x1Data, const uint8_t* x2Data, const uint8_t* x1ScaleData, const uint8_t* x2ScaleData,
                   vector<float>& golden)
{
    golden.resize(M_DIM * N_DIM);
    for (int64_t m = 0; m < M_DIM; ++m) {
        for (int64_t n = 0; n < N_DIM; ++n) {
            float acc = 0.0f;
            for (int64_t j = 0; j < K_GROUP_NUM; ++j) {
                float groupSum = 0.0f;
                for (int64_t k = j * GROUP_SIZE; k < (j + 1) * GROUP_SIZE; ++k) {
                    groupSum += Fp8E4m3ToFloat(x1Data[m * K_DIM + k]) * GetFp4Element(x2Data, n, k);
                }
                acc += groupSum * E8m0ToFloat(x1ScaleData[m * K_GROUP_NUM + j]) *
                       E8m0ToFloat(x2ScaleData[n * K_GROUP_NUM + j]);
            }
            golden[m * N_DIM + n] = acc;
        }
    }
}

int32_t CompareWithGolden(const uint8_t* outData, const vector<float>& golden)
{
    float relTol = 0.02f; // BF16输出相对精度2^-8，叠加累加顺序差异，取2%
    float absTol = 1e-3f;
    double maxDiff = 0.0;
    double maxRel = 0.0;
    int64_t badCount = 0;
    for (uint64_t i = 0; i < golden.size(); ++i) {
        float actual = Bf16ToFloat(reinterpret_cast<const uint16_t*>(outData)[i]);
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
    // 自定义代码：添加单算子定义到图中，以MxA8W4量化场景为例
    auto quant_matmul = op::QuantBatchMatmulV4("quant_batch_matmul_v4");
    std::vector<int64_t> x1Shape = {M_DIM, K_DIM};                   // (M, K)
    std::vector<int64_t> x2Shape = {N_DIM, K_DIM};                   // (N, K)，transpose_x2=true
    std::vector<int64_t> x1ScaleShape = {M_DIM, K_GROUP_NUM / 2, 2}; // (M, ceil(K / 64), 2)
    std::vector<int64_t> x2ScaleShape = {N_DIM, K_GROUP_NUM / 2, 2}; // (N, ceil(K / 64), 2)
    std::vector<int64_t> yShape = {M_DIM, N_DIM};                    // (M, N)

    ADD_INPUT(1, x1, DT_FLOAT8_E4M3FN, x1Shape);
    ADD_INPUT(2, x2, DT_FLOAT4_E2M1, x2Shape);
    ADD_INPUT(3, x1_scale, DT_FLOAT8_E8M0, x1ScaleShape);
    ADD_INPUT(4, x2_scale, DT_FLOAT8_E8M0, x2ScaleShape);

    ADD_OUTPUT(1, y, DT_BF16, yShape);

    ADD_INPUT_ATTR(dtype, 27); // 输出数据类型为BFLOAT16
    ADD_INPUT_ATTR(transpose_x1, false);
    ADD_INPUT_ATTR(transpose_x2, true);
    ADD_INPUT_ATTR(group_size, GROUP_SIZE); // MX量化模式的groupSizeK为32

    outputs.push_back(quant_matmul);
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

    // golden校验：host侧按MX量化公式计算期望值并与NPU输出比较
    vector<float> golden;
    ComputeGolden(input[0].GetData(), input[1].GetData(), input[2].GetData(), input[3].GetData(), golden);
    int32_t ret = CompareWithGolden(output[0].GetData(), golden);

    delete session;
    GEFinalize();
    return ret;
}
