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
 * \file test_geir_bn3d_training_reduce.cpp
 * \brief BN3DTrainingReduce 图模式（GE IR）调用样例。
 *
 * 用法：test_geir_bn3d_training_reduce [fp32|fp16|bf16] [rank5|rank4|rank3|rank2|ndhwc|NxCx...]
 * 默认 fp32 + rank5。第二参数亦可直接给自定义 shape（如 4x64x2x2x8192）以驱动特定 tiling 分支。
 * rank2~rank5 与自定义 shape 使用 NCDHW origin；ndhwc 使用 [N,D,H,W,C] origin。
 * storage format 均交由 GE 选择，本样例只覆盖 Ascend 950 当前公开的 NCDHW / NDHWC origin。
 * 样例内置 CPU 参考值比对，能直接暴露两类典型错误：
 *   - sum 与 square_sum 接反；
 *   - square_sum 被实现成 (Σx)² 而不是 Σ(x²)。
 */

#include <iostream>
#include <fstream>
#include <string.h>
#include <stdint.h>
#include <cmath>
#include <vector>
#include <string>
#include <map>
#include "assert.h"

#include "graph.h"
#include "types.h"
#include "tensor.h"
#include "ge_error_codes.h"
#include "ge_api_types.h"
#include "ge_api.h"
#include "array_ops.h"
#include "ge_ir_build.h"
#include "../../op_graph/bn3d_training_reduce_proto.h"

#define FAILED -1
#define SUCCESS 0

using namespace ge;
using std::map;
using std::string;
using std::vector;

string GetTime()
{
    time_t timep;
    time(&timep);
    char tmp[64];
    strftime(tmp, sizeof(tmp), "%Y-%m-%d %H:%M:%S,000", localtime(&timep));
    return string(tmp);
}

uint32_t GetDataTypeSize(DataType dt)
{
    if (dt == ge::DT_FLOAT)
        return 4;
    if (dt == ge::DT_FLOAT16 || dt == ge::DT_BF16)
        return 2;
    return 4;
}

// 样例输入取值：落在 [0, 1) 的确定性序列，保证跨次运行可复现。
static float RefValue(size_t i) { return static_cast<float>((i * 10 + 5) % 1000) / 1000.0f; }

static uint16_t Fp32ToFp16Bits(float val)
{
    uint32_t bits;
    memcpy(&bits, &val, sizeof(uint32_t));
    uint16_t sign = (bits >> 16) & 0x8000;
    uint16_t exponent = (bits >> 23) & 0xff;
    uint32_t mantissa = bits & 0x7fffff;
    if (exponent == 0) {
        return sign;
    } else if (exponent >= 113 && exponent <= 142) {
        return sign | ((exponent - 112) << 10) | (mantissa >> 13);
    } else if (exponent < 113) {
        if (exponent >= 103) {
            uint32_t shift = 113 - exponent;
            uint32_t shifted = (mantissa | 0x800000) >> shift;
            uint16_t fp16Bits = sign | static_cast<uint16_t>(shifted);
            if ((shifted >> 4) & 1)
                fp16Bits++;
            return fp16Bits;
        }
        return sign;
    }
    return sign | 0x7c00;
}

// bf16 就是 fp32 的高 16 位（round-to-nearest-even）。
static uint16_t Fp32ToBf16Bits(float val)
{
    uint32_t bits;
    memcpy(&bits, &val, sizeof(uint32_t));
    uint32_t lsb = (bits >> 16) & 1U;
    uint32_t rounded = bits + 0x7fffU + lsb;
    return static_cast<uint16_t>(rounded >> 16);
}

static float Fp16BitsToFp32(uint16_t h)
{
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t man = h & 0x3ff;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;
        } else {
            exp = 127 - 15 + 1;
            while ((man & 0x400) == 0) {
                man <<= 1;
                exp--;
            }
            man &= 0x3ff;
            bits = sign | (exp << 23) | (man << 13);
        }
    } else if (exp == 0x1f) {
        bits = sign | 0x7f800000 | (man << 13);
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (man << 13);
    }
    float out;
    memcpy(&out, &bits, sizeof(float));
    return out;
}

static float Bf16BitsToFp32(uint16_t b)
{
    uint32_t bits = static_cast<uint32_t>(b) << 16;
    float out;
    memcpy(&out, &bits, sizeof(float));
    return out;
}

// 生成输入数据，同时回填"实际参与计算的 fp32 值"，供 CPU 参考值使用。
// 低精度 dtype 必须用量化后的值算参考，否则比对会被量化误差淹没。
int32_t GenInputData(const vector<int64_t>& shapes, Tensor& inputTensor, TensorDesc& desc, DataType dtype,
                     vector<float>& actualFp32)
{
    desc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (size_t i = 0; i < shapes.size(); i++)
        size *= shapes[i];
    uint32_t dataLen = size * GetDataTypeSize(dtype);
    uint8_t* pData = new (std::nothrow) uint8_t[dataLen];
    if (pData == nullptr)
        return FAILED;

    actualFp32.resize(size);
    if (dtype == ge::DT_FLOAT) {
        float* p = reinterpret_cast<float*>(pData);
        for (size_t i = 0; i < size; ++i) {
            p[i] = RefValue(i);
            actualFp32[i] = p[i];
        }
    } else if (dtype == ge::DT_FLOAT16) {
        uint16_t* p = reinterpret_cast<uint16_t*>(pData);
        for (size_t i = 0; i < size; ++i) {
            p[i] = Fp32ToFp16Bits(RefValue(i));
            actualFp32[i] = Fp16BitsToFp32(p[i]);
        }
    } else {
        uint16_t* p = reinterpret_cast<uint16_t*>(pData);
        for (size_t i = 0; i < size; ++i) {
            p[i] = Fp32ToBf16Bits(RefValue(i));
            actualFp32[i] = Bf16BitsToFp32(p[i]);
        }
    }
    inputTensor = Tensor(desc, pData, dataLen);
    delete[] pData;
    return SUCCESS;
}

int main(int argc, char* argv[])
{
    const char* graphName = "tc_ge_irrun_bn3d_training_reduce";
    Graph graph(graphName);
    std::vector<ge::Tensor> input;

    printf("%s - INFO - [XIR]: Start to initialize ge\n", GetTime().c_str());
    std::map<AscendString, AscendString> globalOptions = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    Status ret = ge::GEInitialize(globalOptions);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: GEInitialize failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: GEInitialize success\n", GetTime().c_str());

    DataType xDtype = DT_FLOAT;
    Format xFormat = FORMAT_NCDHW;
    vector<int64_t> xShape = {2, 3, 4, 4, 8};

    if (argc > 1) {
        string arg = argv[1];
        if (arg == "fp16") {
            xDtype = DT_FLOAT16;
        } else if (arg == "bf16") {
            xDtype = DT_BF16;
        } else {
            xDtype = DT_FLOAT;
        }
    }
    // 默认 origin format 为 NCDHW：C 轴语义只由 origin format 决定，NCDHW 覆盖 rank 2~5。
    // origin NCHW 不构造用例——canndev 的两版 InferShape（legacy reduce_ops.cc 与 runtime
    // bn_3d_training.cc）均拒绝 origin NCHW。
    // op_info 中的 NCHW 行是留给 GE 的 storage format 选择的；实测 rank 4 时 GE 仍选 NCDHW
    // （见 PreRunAfterBuild dump 中该节点 layout="NCDHW"），故本样例不覆盖 NCHW kernel 行。
    if (argc > 2) {
        string rank = argv[2];
        if (rank == "rank4") {
            xShape = {2, 3, 8, 8};
        } else if (rank == "rank3") {
            xShape = {2, 3, 16};
        } else if (rank == "rank2") {
            xShape = {2, 3};
        } else if (rank == "ndhwc" || rank == "NDHWC") {
            // 公开 channel-last 格式 [N, D, H, W, C]：归约 N/D/H/W，保留 C。
            xFormat = FORMAT_NDHWC;
            xShape = {2, 4, 4, 8, 3};
        } else if (rank.find('x') != string::npos) {
            // 自定义 shape，形如 4x64x2x2x8192：用于驱动 tiling 的分支（多核分通道、
            // R0 全载 nTile>1 的跨 N 跳搬、以及单行 R0 放不下时的 sub-R 分块）。
            xShape.clear();
            size_t pos = 0;
            string s = rank;
            while (pos <= s.size()) {
                size_t next = s.find('x', pos);
                string tok = (next == string::npos) ? s.substr(pos) : s.substr(pos, next - pos);
                if (!tok.empty()) {
                    xShape.push_back(std::stoll(tok));
                }
                if (next == string::npos) {
                    break;
                }
                pos = next + 1;
            }
        } else {
            xShape = {2, 3, 4, 4, 8};
        }
    }
    printf("%s - INFO - [XIR]: x dtype=%d format=%d shape=[", GetTime().c_str(), xDtype, xFormat);
    for (size_t i = 0; i < xShape.size(); i++)
        printf("%ld,", xShape[i]);
    printf("]\n");

    auto xData = op::Data("placeholder_x").set_attr_index(0);
    TensorDesc xDesc = TensorDesc(ge::Shape(xShape), xFormat, xDtype);
    xDesc.SetPlacement(ge::kPlacementHost);
    xDesc.SetFormat(xFormat);
    xDesc.SetOriginFormat(xFormat);
    Tensor tensorX;
    vector<float> actualFp32;
    ret = GenInputData(xShape, tensorX, xDesc, xDtype, actualFp32);
    if (ret != SUCCESS) {
        printf("GenInputData x failed\n");
        return FAILED;
    }
    xData.update_input_desc_x(xDesc);
    xData.update_output_desc_y(xDesc);
    input.push_back(tensorX);
    graph.AddOp(xData);

    auto reduceOp = op::BN3DTrainingReduce("bn3d_training_reduce");
    reduceOp.set_input_x(xData);

    std::vector<Operator> inputs = {xData};
    std::vector<Operator> outputs = {reduceOp};
    graph.SetInputs(inputs).SetOutputs(outputs);

    std::map<AscendString, AscendString> buildOptions = {};
    printf("%s - INFO - [XIR]: Create session\n", GetTime().c_str());
    ge::Session* session = new Session(buildOptions);
    if (session == nullptr) {
        printf("Create session failed\n");
        return FAILED;
    }

    std::map<AscendString, AscendString> graphOptions = {};
    uint32_t graphId = 0;
    ret = session->AddGraph(graphId, graph, graphOptions);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: AddGraph failed ret=%u\n", GetTime().c_str(), ret);
        delete session;
        GEFinalize();
        return FAILED;
    }
    printf("%s - INFO - [XIR]: AddGraph success\n", GetTime().c_str());

    printf("%s - INFO - [XIR]: RunGraph start\n", GetTime().c_str());
    std::vector<ge::Tensor> output;
    ret = session->RunGraph(graphId, input, output);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: RunGraph failed ret=%u\n", GetTime().c_str(), ret);
        ge::AscendString errorMsg = ge::GEGetErrorMsgV2();
        printf("%s - ERROR - [XIR]: error: %s\n", GetTime().c_str(), errorMsg.GetString());
        delete session;
        GEFinalize();
        return FAILED;
    }
    printf("%s - INFO - [XIR]: RunGraph success\n", GetTime().c_str());

    // CPU 参考值。公开的两种 origin 布局都归一化为 R1-A-R0：
    //   idx(r1, a, r0) = r1 * (A * R0) + a * R0 + r0
    //   * channel-first（NCDHW）：R1 = N，A = C，R0 = product(dim2:)；
    //   * channel-last（NDHWC）：R1 = N*D*H*W，A = C，R0 = 1。
    int64_t numN = 0;  // R1
    int64_t numC = 0;  // A
    int64_t numR0 = 1; // R0
    if (xFormat == FORMAT_NDHWC) {
        numN = xShape[0] * xShape[1] * xShape[2] * xShape[3];
        numC = xShape[4];
    } else {
        numN = xShape[0];
        numC = xShape[1];
        for (size_t i = 2; i < xShape.size(); i++)
            numR0 *= xShape[i];
    }
    const int64_t outLen = numC;
    vector<double> refSum(outLen, 0.0);
    vector<double> refSquareSum(outLen, 0.0);
    for (int64_t n = 0; n < numN; ++n) {
        for (int64_t c = 0; c < numC; ++c) {
            for (int64_t r = 0; r < numR0; ++r) {
                const double v = static_cast<double>(actualFp32[n * numC * numR0 + c * numR0 + r]);
                refSum[c] += v;
                refSquareSum[c] += v * v;
            }
        }
    }

    int outputNum = output.size();
    printf("output_num=%d\n", outputNum);
    if (outputNum < 2) {
        printf("%s - ERROR - [XIR]: expect 2 outputs (sum, square_sum), got %d\n", GetTime().c_str(), outputNum);
        delete session;
        GEFinalize();
        return FAILED;
    }

    // fp32 混合容差：rtol = 2^-10，atol = 2^-16。
    const double rtol = 9.765625e-4;
    const double atol = 1.52587890625e-5;
    int failCnt = 0;
    const char* outName[2] = {"sum", "square_sum"};
    for (int i = 0; i < 2; i++) {
        DataType outDtype = output[i].GetTensorDesc().GetDataType();
        int64_t outSize = output[i].GetTensorDesc().GetShape().GetShapeSize();
        printf("output[%d] (%s) dtype=%d shapeSize=%ld\n", i, outName[i], outDtype, outSize);
        if (outDtype != ge::DT_FLOAT) {
            printf("  ERROR: output dtype must be float32\n");
            failCnt++;
            continue;
        }
        if (outSize != outLen) {
            printf("  ERROR: output shapeSize %ld != expected %ld\n", outSize, outLen);
            failCnt++;
            continue;
        }
        const float* p = reinterpret_cast<const float*>(output[i].GetData());
        const vector<double>& ref = (i == 0) ? refSum : refSquareSum;
        for (int64_t k = 0; k < outLen; ++k) {
            const double expect = ref[k];
            const double got = static_cast<double>(p[k]);
            const double tol = atol + rtol * std::fabs(expect);
            const bool ok = std::fabs(got - expect) <= tol;
            // 元素多时只打印前若干个，避免刷屏；不匹配的一律打印。
            if (k < 8 || !ok) {
                printf("  %s[%ld] = %f (expect %f)%s\n", outName[i], k, got, expect, ok ? "" : "   <== MISMATCH");
            }
            if (!ok)
                failCnt++;
        }
    }

    delete session;
    session = nullptr;
    ret = ge::GEFinalize();
    if (ret != SUCCESS) {
        printf("GEFinalize failed\n");
        return FAILED;
    }
    printf("%s - INFO - [XIR]: GEFinalize success\n", GetTime().c_str());

    if (failCnt != 0) {
        printf("%s - ERROR - [XIR]: %d value(s) mismatched\n", GetTime().c_str(), failCnt);
        return FAILED;
    }
    printf("%s - INFO - [XIR]: all outputs matched CPU reference\n", GetTime().c_str());
    return SUCCESS;
}
