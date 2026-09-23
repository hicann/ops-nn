/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the License).
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file test_geir_fused_bias_leaky_relu.cpp
 * @brief FusedBiasLeakyRelu GE IR 静态 shape 调用示例
 *
 * 构造静态图：Data(x) + Data(bias) -> FusedBiasLeakyRelu -> y，RunGraph 后用
 * CPU Golden 校验输出的 Shape、dtype 和逐元素数值。
 *
 * 算子接口（REG_OP FusedBiasLeakyRelu）：
 *   2 输入: x (ND), bias (ND, shape 与 x 完全相同)
 *   2 属性: negative_slope (Float, 默认 0.2), scale (Float, 默认 1.414213562373)
 *   1 输出: y (ND, 同 x)
 *
 * Formula:
 *   t = x + bias
 *   y = (t >= 0) ? t * scale : t * negative_slope * scale
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
#include "array_ops.h"
#include "ge_ir_build.h"

#include "../../op_graph/fused_bias_leaky_relu_proto.h"

#define FAILED (-1)
#define SUCCESS 0

using namespace ge;
using std::map;
using std::string;
using std::vector;

namespace {
string GetTime()
{
    time_t timep;
    time(&timep);
    char tmp[64];
    strftime(tmp, sizeof(tmp), "%Y-%m-%d %H:%M:%S,000", localtime(&timep));
    return tmp;
}

uint32_t GetDataTypeSize(DataType dt)
{
    if (dt == ge::DT_FLOAT || dt == ge::DT_INT32 || dt == ge::DT_UINT32) {
        return 4;
    }
    if (dt == ge::DT_FLOAT16 || dt == ge::DT_BF16 || dt == ge::DT_INT16) {
        return 2;
    }
    if (dt == ge::DT_INT8 || dt == ge::DT_UINT8) {
        return 1;
    }
    if (dt == ge::DT_INT64 || dt == ge::DT_UINT64) {
        return 8;
    }
    return 4;
}

string ShapeToStr(const vector<int64_t>& shape)
{
    string s = "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        s += std::to_string(shape[i]);
        if (i + 1 < shape.size()) {
            s += ",";
        }
    }
    s += "]";
    return s;
}

int64_t ShapeSize(const vector<int64_t>& shape)
{
    int64_t n = 1;
    for (auto d : shape) {
        n *= d;
    }
    return n;
}

// 简单截断式 FP32->FP16 转换，仅用于构造恰好可被 FP16 精确表示的输入值
void Fp32ToFp16Bytes(float v, uint8_t* out)
{
    uint32_t fp32;
    std::memcpy(&fp32, &v, sizeof(fp32));
    uint16_t sign = static_cast<uint16_t>((fp32 >> 31) & 0x1) << 15;
    uint32_t exp = (fp32 >> 23) & 0xFF;
    uint32_t mant = (fp32 >> 13) & 0x3FF;
    uint16_t fp16;
    if (exp >= 113 && exp <= 142) {
        fp16 = static_cast<uint16_t>(sign | ((exp - 112) << 10) | mant);
    } else if (exp < 113) {
        fp16 = static_cast<uint16_t>(sign);
    } else {
        fp16 = static_cast<uint16_t>(sign | 0x7C00);
    }
    std::memcpy(out, &fp16, sizeof(fp16));
}

float Fp16BytesToFp32(const uint8_t* in)
{
    uint16_t fp16;
    std::memcpy(&fp16, in, sizeof(fp16));
    uint32_t sign = static_cast<uint32_t>((fp16 >> 15) & 0x1) << 31;
    uint32_t exp = (fp16 >> 10) & 0x1F;
    uint32_t mant = fp16 & 0x3FF;
    uint32_t fp32;
    if (exp == 0) {
        if (mant == 0) {
            fp32 = sign;
        } else {
            // subnormal
            uint32_t e = 0;
            uint32_t m = mant;
            while ((m & 0x400) == 0) {
                m <<= 1;
                e++;
            }
            m &= 0x3FF;
            fp32 = sign | ((127 - 15 - e + 1) << 23) | (m << 13);
        }
    } else if (exp == 0x1F) {
        fp32 = sign | 0x7F800000 | (mant << 13);
    } else {
        fp32 = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float v;
    std::memcpy(&v, &fp32, sizeof(v));
    return v;
}

// 确定性输入：取值均为 0.25 的整数倍，可被 FP16 精确表示
float XValueAt(int64_t idx) { return static_cast<float>((idx % 17) - 8) * 0.25f; }

float BiasValueAt(int64_t idx) { return static_cast<float>((idx % 5) - 2) * 0.5f; }

vector<uint8_t> MakeInputBuffer(int64_t n, DataType dtype, bool isBias)
{
    uint32_t elemSize = GetDataTypeSize(dtype);
    vector<uint8_t> buf(static_cast<size_t>(n) * elemSize, 0);
    for (int64_t i = 0; i < n; ++i) {
        float v = isBias ? BiasValueAt(i) : XValueAt(i);
        if (dtype == DT_FLOAT) {
            std::memcpy(buf.data() + static_cast<size_t>(i) * elemSize, &v, elemSize);
        } else {
            Fp32ToFp16Bytes(v, buf.data() + static_cast<size_t>(i) * elemSize);
        }
    }
    return buf;
}

// CPU Golden：FP16 输入按 kernel 语义先升 FP32 计算
vector<float> ComputeGolden(const vector<float>& x, const vector<float>& bias, float negativeSlope, float scale)
{
    vector<float> y(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        float t = x[i] + bias[i];
        y[i] = (t >= 0.0f) ? (t * scale) : (t * negativeSlope * scale);
    }
    return y;
}

bool VerifyOutput(const vector<int64_t>& shape, DataType dtype, float negativeSlope, float scale,
                  const vector<ge::Tensor>& output)
{
    if (output.size() != 1) {
        printf("%s - ERROR - [XIR]: expect 1 output, got %zu\n", GetTime().c_str(), output.size());
        return false;
    }
    const ge::Tensor& out = output[0];
    if (out.GetTensorDesc().GetDataType() != dtype) {
        printf("%s - ERROR - [XIR]: output dtype mismatch, expect %d, got %d\n", GetTime().c_str(),
               static_cast<int>(dtype), static_cast<int>(out.GetTensorDesc().GetDataType()));
        return false;
    }
    auto outShape = out.GetTensorDesc().GetShape();
    if (static_cast<size_t>(outShape.GetDimNum()) != shape.size()) {
        printf("%s - ERROR - [XIR]: output dim num mismatch, expect %zu, got %zu\n", GetTime().c_str(), shape.size(),
               static_cast<size_t>(outShape.GetDimNum()));
        return false;
    }
    for (size_t i = 0; i < shape.size(); ++i) {
        if (outShape.GetDim(i) != shape[i]) {
            printf("%s - ERROR - [XIR]: output dim %zu mismatch, expect %ld, got %ld\n", GetTime().c_str(), i, shape[i],
                   outShape.GetDim(i));
            return false;
        }
    }

    int64_t n = ShapeSize(shape);
    vector<float> x(n);
    vector<float> bias(n);
    for (int64_t i = 0; i < n; ++i) {
        if (dtype == DT_FLOAT) {
            x[i] = XValueAt(i);
            bias[i] = BiasValueAt(i);
        } else {
            uint8_t xb[2];
            uint8_t bb[2];
            Fp32ToFp16Bytes(XValueAt(i), xb);
            Fp32ToFp16Bytes(BiasValueAt(i), bb);
            x[i] = Fp16BytesToFp32(xb);
            bias[i] = Fp16BytesToFp32(bb);
        }
    }
    vector<float> golden = ComputeGolden(x, bias, negativeSlope, scale);

    const uint8_t* data = out.GetData();
    int64_t dataSize = out.GetSize();
    if (data == nullptr || dataSize != n * static_cast<int64_t>(GetDataTypeSize(dtype))) {
        printf("%s - ERROR - [XIR]: output data size mismatch, expect %ld, got %ld\n", GetTime().c_str(),
               n * static_cast<int64_t>(GetDataTypeSize(dtype)), dataSize);
        return false;
    }
    float tol = (dtype == DT_FLOAT) ? 1.0e-5f : 1.0e-3f;
    for (int64_t i = 0; i < n; ++i) {
        float actual;
        if (dtype == DT_FLOAT) {
            std::memcpy(&actual, data + static_cast<size_t>(i) * 4, 4);
        } else {
            actual = Fp16BytesToFp32(data + static_cast<size_t>(i) * 2);
        }
        float expect = golden[static_cast<size_t>(i)];
        float diff = std::fabs(actual - expect);
        float rel = diff / std::fmax(std::fabs(expect), 1.0e-6f);
        if (diff > tol && rel > tol) {
            printf("%s - ERROR - [XIR]: output value mismatch at %ld, expect %.7f, got %.7f\n", GetTime().c_str(), i,
                   expect, actual);
            return false;
        }
    }
    return true;
}

int RunStaticCase(const vector<int64_t>& shape, DataType dtype, float negativeSlope, float scale)
{
    const char* graphName = "fused_bias_leaky_relu_geir_static";
    Graph graph(graphName);
    vector<Operator> inputs{};
    vector<Operator> outputs{};

    auto op1 = op::FusedBiasLeakyRelu("fused_bias_leaky_relu_static");
    op1.set_attr_negative_slope(negativeSlope);
    op1.set_attr_scale(scale);

    vector<ge::Tensor> feeds;
    const char* inputNames[2] = {"x", "bias"};
    for (int idx = 0; idx < 2; ++idx) {
        string dataName = string(inputNames[idx]) + "_data";
        auto data = op::Data(dataName.c_str()).set_attr_index(idx);
        TensorDesc desc = TensorDesc(ge::Shape(shape), FORMAT_ND, dtype);
        desc.SetPlacement(ge::kPlacementHost);
        desc.SetFormat(FORMAT_ND);
        desc.SetRealDimCnt(static_cast<int32_t>(shape.size()));
        data.update_input_desc_x(desc);
        data.update_output_desc_y(desc);
        graph.AddOp(data);
        if (idx == 0) {
            op1.set_input_x(data);
        } else {
            op1.set_input_bias(data);
        }
        vector<uint8_t> buf = MakeInputBuffer(ShapeSize(shape), dtype, idx == 1);
        feeds.push_back(Tensor(desc, buf.data(), buf.size()));
        inputs.push_back(data);
    }

    TensorDesc outDesc = TensorDesc(ge::Shape(shape), FORMAT_ND, dtype);
    outDesc.SetFormat(FORMAT_ND);
    outDesc.SetRealDimCnt(static_cast<int32_t>(shape.size()));
    op1.update_output_desc_y(outDesc);
    outputs.push_back(op1);

    graph.SetInputs(inputs).SetOutputs(outputs);

    std::map<AscendString, AscendString> buildOptions = {};
    ge::Session* session = new (std::nothrow) Session(buildOptions);
    if (session == nullptr) {
        printf("%s - ERROR - [XIR]: create session failed\n", GetTime().c_str());
        return FAILED;
    }
    std::map<AscendString, AscendString> graphOptions = {};
    uint32_t graphId = 0;
    Status ret = session->AddGraph(graphId, graph, graphOptions);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: AddGraph failed\n", GetTime().c_str());
        ge::AscendString errorMsg = ge::GEGetErrorMsgV2();
        std::cout << "Error message: " << errorMsg.GetString() << std::endl;
        delete session;
        return FAILED;
    }

    vector<ge::Tensor> output;
    ret = session->RunGraph(graphId, feeds, output);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: RunGraph failed\n", GetTime().c_str());
        ge::AscendString errorMsg = ge::GEGetErrorMsgV2();
        std::cout << "Error message: " << errorMsg.GetString() << std::endl;
        delete session;
        return FAILED;
    }

    bool ok = VerifyOutput(shape, dtype, negativeSlope, scale, output);
    delete session;
    if (!ok) {
        printf("%s - ERROR - [XIR]: Shape, dtype and values FAILED for %s\n", GetTime().c_str(),
               ShapeToStr(shape).c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Shape, dtype and values PASSED for %s\n", GetTime().c_str(), ShapeToStr(shape).c_str());
    return SUCCESS;
}
} // namespace

int main(int argc, char* argv[])
{
    (void)argc;
    (void)argv;

    printf("%s - INFO - [XIR]: FusedBiasLeakyRelu static GEIR example start\n", GetTime().c_str());
    std::map<AscendString, AscendString> globalOptions = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    Status ret = ge::GEInitialize(globalOptions);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: GEInitialize failed\n", GetTime().c_str());
        return FAILED;
    }

    int failed = 0;
    // case 1: FP32, shape [2,4], default attrs
    if (RunStaticCase({2, 4}, DT_FLOAT, 0.2f, 1.414213562373f) != SUCCESS) {
        failed++;
    }
    // case 2: FP16, shape [3,5], default attrs
    if (RunStaticCase({3, 5}, DT_FLOAT16, 0.2f, 1.414213562373f) != SUCCESS) {
        failed++;
    }
    // case 3: FP32, shape [8], non-default attrs
    if (RunStaticCase({8}, DT_FLOAT, 0.5f, 2.0f) != SUCCESS) {
        failed++;
    }

    ret = ge::GEFinalize();
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: GEFinalize failed\n", GetTime().c_str());
        return FAILED;
    }

    if (failed > 0) {
        printf("%s - ERROR - [XIR]: FusedBiasLeakyRelu static GEIR verification FAILED (%d case(s) failed)\n",
               GetTime().c_str(), failed);
        return FAILED;
    }
    printf("%s - INFO - [XIR]: FusedBiasLeakyRelu static GEIR verification PASSED\n", GetTime().c_str());
    return SUCCESS;
}
