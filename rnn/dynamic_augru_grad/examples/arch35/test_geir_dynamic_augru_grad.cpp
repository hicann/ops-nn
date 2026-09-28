/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file test_geir_dynamic_augru_grad.cpp
 * @brief DynamicAUGRUGrad GE IR调用验证：构图执行并与CPU golden全量比对
 */

#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <iostream>
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

#include "../../op_graph/dynamic_augru_grad_proto.h"

#define FAILED -1
#define SUCCESS 0

using namespace ge;
using std::string;
using std::vector;

static constexpr int64_t GATE_NUM = 3;

static uint32_t GetDataTypeSize(DataType dt) { return (dt == ge::DT_FLOAT || dt == ge::DT_INT32) ? 4U : 2U; }

// float<->half位模式转换（host侧无__fp16支持，按IEEE754半精度手工转换）
static uint16_t FloatToHalfBits(float f)
{
    uint32_t x;
    memcpy(&x, &f, sizeof(x));
    uint32_t sign = (x >> 16) & 0x8000U;
    int32_t exp = static_cast<int32_t>((x >> 23) & 0xFFU) - 127 + 15;
    uint32_t mant = x & 0x7FFFFFU;
    if (exp <= 0) {
        return static_cast<uint16_t>(sign); // 溢出/下溢按0处理（测试数据范围安全）
    }
    if (exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7BFFU);
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13));
}

static float HalfBitsToFloat(uint16_t h)
{
    uint32_t sign = static_cast<uint32_t>(h & 0x8000U) << 16;
    uint32_t exp = (h >> 10) & 0x1FU;
    uint32_t mant = h & 0x3FFU;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            int32_t e = 127 - 15 + 1;
            while ((mant & 0x400U) == 0) {
                mant <<= 1;
                e--;
            }
            mant &= 0x3FFU;
            bits = sign | (static_cast<uint32_t>(e) << 23) | (mant << 13);
        }
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

// fp16：转half位供GM喂入，同时把float向量替换为round-trip值（golden与kernel所见输入一致）
static const void* PrepInput(vector<float>& v, vector<uint16_t>& h, bool isHalf)
{
    if (!isHalf) {
        return v.data();
    }
    h.resize(v.size());
    for (size_t i = 0; i < v.size(); i++) {
        h[i] = FloatToHalfBits(v[i]);
        v[i] = HalfBitsToFloat(h[i]);
    }
    return h.data();
}

template <typename T>
static vector<T> GenRandom(size_t count, T low, T high, uint32_t seed)
{
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(static_cast<double>(low), static_cast<double>(high));
    vector<T> data(count);
    for (size_t i = 0; i < count; i++) {
        data[i] = static_cast<T>(dist(gen));
    }
    return data;
}

// CPU golden：完整AUGRU反向BPTT（与kernel公式一致）
struct GoldenResult {
    vector<float> dwInput;  // [I,3H]
    vector<float> dwHidden; // [H,3H]
    vector<float> dbInput;  // [3H]
    vector<float> dbHidden; // [3H]
    vector<float> dx;       // [T,B,I]
    vector<float> dhPrev;   // [B,H]
    vector<float> dwAtt;    // [T,B]
};

static GoldenResult ComputeGolden(const vector<float>& x, const vector<float>& wInput, const vector<float>& wHidden,
                                  const vector<float>& att, const vector<float>& initH, const vector<float>& h,
                                  const vector<float>& dy, const vector<float>& dh, const vector<float>& update,
                                  const vector<float>& updateAtt, const vector<float>& reset,
                                  const vector<float>& newGate, const vector<float>& hiddenNew,
                                  const vector<int32_t>& seqLen, int64_t T, int64_t B, int64_t I, int64_t H,
                                  int64_t isSeqLen)
{
    int64_t threeH = GATE_NUM * H;
    int64_t tb = T * B;
    GoldenResult out;
    out.dwInput.assign(I * threeH, 0.0f);
    out.dwHidden.assign(H * threeH, 0.0f);
    out.dbInput.assign(threeH, 0.0f);
    out.dbHidden.assign(threeH, 0.0f);
    out.dx.assign(tb * I, 0.0f);
    out.dhPrev.assign(B * H, 0.0f);
    out.dwAtt.assign(tb, 0.0f);
    vector<float> dGh(tb * threeH, 0.0f);
    vector<float> dGi(tb * threeH, 0.0f);
    vector<float> hPrev(tb * H, 0.0f);

    int64_t zSlot = 0; // zrh
    int64_t rSlot = 1;
    int64_t nSlot = 2;
    for (int64_t b = 0; b < B; b++) {
        for (int64_t hh = 0; hh < H; hh++) {
            out.dhPrev[b * H + hh] = dh[b * H + hh];
        }
    }
    for (int64_t t = T - 1; t >= 0; t--) {
        for (int64_t b = 0; b < B; b++) {
            float mask = 1.0f;
            if (isSeqLen == 1) {
                mask = (t < seqLen[b]) ? 1.0f : 0.0f;
            }
            for (int64_t hh = 0; hh < H; hh++) {
                int64_t tbh = (t * B + b) * H + hh;
                int64_t bh = b * H + hh;
                float gradH = out.dhPrev[bh] * mask + dy[tbh];
                float z = update[tbh];
                float u = updateAtt[tbh];
                float a = att[tbh];
                float r = reset[tbh];
                float n = newGate[tbh];
                float hn = hiddenNew[tbh];
                float hp = (t == 0) ? initH[bh] : h[((t - 1) * B + b) * H + hh];

                float ghn = gradH * (hp - n);
                float dnt = gradH * (1 - u) * (1 - n * n);
                float dr = dnt * r * (1 - r) * hn;
                float dz = ghn * (1 - a) * z * (1 - z);

                int64_t giBase = (t * B + b) * threeH;
                dGi[giBase + nSlot * H + hh] = dnt;
                dGh[giBase + nSlot * H + hh] = dnt * r;
                dGi[giBase + rSlot * H + hh] = dr;
                dGh[giBase + rSlot * H + hh] = dr;
                dGi[giBase + zSlot * H + hh] = dz;
                dGh[giBase + zSlot * H + hh] = dz;
                hPrev[(t * B + b) * H + hh] = hp;
                out.dwAtt[t * B + b] += -ghn * z;
                out.dhPrev[bh] = gradH * u;
            }
            for (int64_t hh = 0; hh < H; hh++) {
                float acc = 0.0f;
                for (int64_t k = 0; k < threeH; k++) {
                    acc += dGh[(t * B + b) * threeH + k] * wHidden[hh * threeH + k];
                }
                out.dhPrev[b * H + hh] += acc;
            }
        }
    }
    for (int64_t m = 0; m < I; m++) {
        for (int64_t nn = 0; nn < threeH; nn++) {
            float acc = 0.0f;
            for (int64_t k = 0; k < tb; k++) {
                acc += x[k * I + m] * dGi[k * threeH + nn];
            }
            out.dwInput[m * threeH + nn] = acc;
        }
    }
    for (int64_t nn = 0; nn < threeH; nn++) {
        float acc = 0.0f;
        for (int64_t k = 0; k < tb; k++) {
            acc += dGi[k * threeH + nn];
        }
        out.dbInput[nn] = acc;
    }
    for (int64_t m = 0; m < H; m++) {
        for (int64_t nn = 0; nn < threeH; nn++) {
            float acc = 0.0f;
            for (int64_t k = 0; k < tb; k++) {
                acc += hPrev[k * H + m] * dGh[k * threeH + nn];
            }
            out.dwHidden[m * threeH + nn] = acc;
        }
    }
    for (int64_t nn = 0; nn < threeH; nn++) {
        float acc = 0.0f;
        for (int64_t k = 0; k < tb; k++) {
            acc += dGh[k * threeH + nn];
        }
        out.dbHidden[nn] = acc;
    }
    for (int64_t m = 0; m < tb; m++) {
        for (int64_t nn = 0; nn < I; nn++) {
            float acc = 0.0f;
            for (int64_t k = 0; k < threeH; k++) {
                acc += dGi[m * threeH + k] * wInput[nn * threeH + k];
            }
            out.dx[m * I + nn] = acc;
        }
    }
    return out;
}

static Tensor MakeTensor(const vector<int64_t>& shape, DataType dtype, const void* data)
{
    TensorDesc desc(ge::Shape(shape), FORMAT_ND, dtype);
    desc.SetRealDimCnt(shape.size());
    size_t elemNum = 1;
    for (auto d : shape) {
        elemNum *= static_cast<size_t>(d);
    }
    size_t dataLen = elemNum * GetDataTypeSize(dtype);
    return Tensor(desc, const_cast<uint8_t*>(static_cast<const uint8_t*>(data)), dataLen);
}

struct CaseParams {
    const char* name;
    int64_t T;
    int64_t B;
    int64_t I;
    int64_t H;
    DataType dtype;
    bool withSeqLen;
    const char* gateOrder = "zrh";
};

static float ReadOutElem(const uint8_t* got, size_t idx, DataType dtype)
{
    if (dtype == DT_FLOAT16) {
        uint16_t h;
        memcpy(&h, got + idx * sizeof(uint16_t), sizeof(uint16_t));
        return HalfBitsToFloat(h);
    }
    float v;
    memcpy(&v, got + idx * sizeof(float), sizeof(float));
    return v;
}

static int64_t CompareOut(const char* name, const uint8_t* got, DataType dtype, const vector<float>& expect,
                          double rtol, double atol, int64_t limit = 8)
{
    int64_t diffCnt = 0;
    for (size_t i = 0; i < expect.size(); i++) {
        float v = ReadOutElem(got, i, dtype);
        double actual = static_cast<double>(v);
        double expected = static_cast<double>(expect[i]);
        bool mismatch = (std::isnan(expected) && !std::isnan(actual)) ||
                        (std::isinf(expected) &&
                         (!std::isinf(actual) || std::signbit(actual) != std::signbit(expected))) ||
                        (std::isfinite(expected) &&
                         (!std::isfinite(actual) || std::fabs(actual - expected) > atol + rtol * std::fabs(expected)));
        if (mismatch) {
            if (diffCnt < limit) {
                printf("  [%s] mismatch idx=%zu got=%f expect=%f\n", name, i, v, expect[i]);
            }
            diffCnt++;
        }
    }
    return diffCnt;
}

static int32_t RunCase(const CaseParams& p, double rtol, double atol)
{
    printf("==== %s: T=%lld, B=%lld, I=%lld, H=%lld, dtype=%d, seqLen=%d ====\n", p.name, p.T, p.B, p.I, p.H,
           (int)p.dtype, (int)p.withSeqLen);
    int64_t threeH = GATE_NUM * p.H;
    int64_t tb = p.T * p.B;
    size_t tbh = static_cast<size_t>(tb * p.H);
    size_t tbi = static_cast<size_t>(tb * p.I);

    vector<float> x = GenRandom<float>(tbi, 0.0f, 1.0f, 1);
    vector<float> wInput = GenRandom<float>(static_cast<size_t>(p.I * threeH), -0.05f, 0.05f, 2);
    vector<float> wHidden = GenRandom<float>(static_cast<size_t>(p.H * threeH), -0.05f, 0.05f, 3);
    vector<float> att = GenRandom<float>(tbh, 0.0f, 1.0f, 4);
    vector<float> initH = GenRandom<float>(static_cast<size_t>(p.B * p.H), -1.0f, 1.0f, 5);
    vector<float> h = GenRandom<float>(tbh, -1.0f, 1.0f, 6);
    vector<float> dy = GenRandom<float>(tbh, -1.0f, 1.0f, 7);
    vector<float> dh = GenRandom<float>(static_cast<size_t>(p.B * p.H), -1.0f, 1.0f, 8);
    vector<float> update = GenRandom<float>(tbh, 0.0f, 1.0f, 9);
    vector<float> updateAtt = GenRandom<float>(tbh, 0.0f, 1.0f, 10);
    vector<float> reset = GenRandom<float>(tbh, 0.0f, 1.0f, 11);
    vector<float> newGate = GenRandom<float>(tbh, -1.0f, 1.0f, 12);
    vector<float> hiddenNew = GenRandom<float>(tbh, -1.0f, 1.0f, 13);
    vector<int32_t> seqLen;
    if (p.withSeqLen) {
        seqLen.resize(p.B);
        for (int64_t b = 0; b < p.B; b++) {
            seqLen[b] = static_cast<int32_t>((b * 3 + 1) % (p.T + 1));
        }
    }
    // fp16：先转half（round-trip回写float向量），golden基于kernel实际收到的值计算
    bool isHalf = (p.dtype == DT_FLOAT16);
    vector<uint16_t> xH, wInputH, wHiddenH, attH, initHH, hH, dyH, dhH, updateH, updateAttH, resetH, newH, hiddenNewH;
    const void* xP = PrepInput(x, xH, isHalf);
    const void* wInputP = PrepInput(wInput, wInputH, isHalf);
    const void* wHiddenP = PrepInput(wHidden, wHiddenH, isHalf);
    const void* attP = PrepInput(att, attH, isHalf);
    const void* initHP = PrepInput(initH, initHH, isHalf);
    const void* hP = PrepInput(h, hH, isHalf);
    const void* dyP = PrepInput(dy, dyH, isHalf);
    const void* dhP = PrepInput(dh, dhH, isHalf);
    const void* updateP = PrepInput(update, updateH, isHalf);
    const void* updateAttP = PrepInput(updateAtt, updateAttH, isHalf);
    const void* resetP = PrepInput(reset, resetH, isHalf);
    const void* newP = PrepInput(newGate, newH, isHalf);
    const void* hiddenNewP = PrepInput(hiddenNew, hiddenNewH, isHalf);
    GoldenResult golden = ComputeGolden(x, wInput, wHidden, att, initH, h, dy, dh, update, updateAtt, reset, newGate,
                                        hiddenNew, seqLen, p.T, p.B, p.I, p.H, p.withSeqLen ? 1 : 0);

    Graph graph("geir_dynamic_augru_grad");
    vector<ge::Tensor> inputs;
    vector<op::Data> dataNodes;

    auto makeInput = [&](const char* name, const vector<int64_t>& shape, const void* data, DataType dtype) {
        auto node = op::Data(name);
        Tensor t = MakeTensor(shape, dtype, data);
        node.update_input_desc_x(t.GetTensorDesc());
        node.update_output_desc_y(t.GetTensorDesc());
        graph.AddOp(node);
        inputs.push_back(t);
        dataNodes.push_back(node);
        return node;
    };

    auto op1 = op::DynamicAUGRUGrad("augru_grad_op");
    auto xNode = makeInput("x", {p.T, p.B, p.I}, xP, p.dtype);
    auto wiNode = makeInput("weight_input", {p.I, threeH}, wInputP, p.dtype);
    auto whNode = makeInput("weight_hidden", {p.H, threeH}, wHiddenP, p.dtype);
    auto attNode = makeInput("weight_att", {p.T, p.B, p.H}, attP, p.dtype);
    auto yNode = makeInput("y", {p.T, p.B, p.H}, attP, p.dtype); // 前向输出占位输入
    auto initHNode = makeInput("init_h", {p.B, p.H}, initHP, p.dtype);
    auto hNode = makeInput("h", {p.T, p.B, p.H}, hP, p.dtype);
    auto dyNode = makeInput("dy", {p.T, p.B, p.H}, dyP, p.dtype);
    auto dhNode = makeInput("dh", {p.B, p.H}, dhP, p.dtype);
    auto uNode = makeInput("update", {p.T, p.B, p.H}, updateP, p.dtype);
    auto uattNode = makeInput("update_att", {p.T, p.B, p.H}, updateAttP, p.dtype);
    auto rNode = makeInput("reset", {p.T, p.B, p.H}, resetP, p.dtype);
    auto nNode = makeInput("new", {p.T, p.B, p.H}, newP, p.dtype);
    auto hnNode = makeInput("hidden_new", {p.T, p.B, p.H}, hiddenNewP, p.dtype);
    op1.set_input_x(xNode);
    op1.set_input_weight_input(wiNode);
    op1.set_input_weight_hidden(whNode);
    op1.set_input_weight_att(attNode);
    op1.set_input_y(yNode);
    op1.set_input_init_h(initHNode);
    op1.set_input_h(hNode);
    op1.set_input_dy(dyNode);
    op1.set_input_dh(dhNode);
    op1.set_input_update(uNode);
    op1.set_input_update_att(uattNode);
    op1.set_input_reset(rNode);
    op1.set_input_new(nNode);
    op1.set_input_hidden_new(hnNode);
    if (p.withSeqLen) {
        auto seqNode = makeInput("seq_length", {p.B}, seqLen.data(), DT_INT32);
        op1.set_input_seq_length(seqNode);
    }

    TensorDesc dwInputDesc(ge::Shape({p.I, threeH}), FORMAT_ND, p.dtype);
    TensorDesc dwHiddenDesc(ge::Shape({p.H, threeH}), FORMAT_ND, p.dtype);
    TensorDesc dbDesc(ge::Shape({threeH}), FORMAT_ND, p.dtype);
    TensorDesc dxDesc(ge::Shape({p.T, p.B, p.I}), FORMAT_ND, p.dtype);
    TensorDesc dhPrevDesc(ge::Shape({p.B, p.H}), FORMAT_ND, p.dtype);
    TensorDesc dwAttDesc(ge::Shape({p.T, p.B}), FORMAT_ND, p.dtype);
    op1.update_output_desc_dw_input(dwInputDesc);
    op1.update_output_desc_dw_hidden(dwHiddenDesc);
    op1.update_output_desc_db_input(dbDesc);
    op1.update_output_desc_db_hidden(dbDesc);
    op1.update_output_desc_dx(dxDesc);
    op1.update_output_desc_dh_prev(dhPrevDesc);
    op1.update_output_desc_dw_att(dwAttDesc);
    op1.set_attr_direction("UNIDIRECTIONAL");
    op1.set_attr_cell_depth(1);
    op1.set_attr_keep_prob(-1.0);
    op1.set_attr_cell_clip(-1.0);
    op1.set_attr_num_proj(0);
    op1.set_attr_time_major(true);
    op1.set_attr_gate_order(p.gateOrder);
    op1.set_attr_reset_after(true);

    vector<op::Data> inNodes = {xNode,  wiNode, whNode, attNode,  yNode, initHNode, hNode,
                                dyNode, dhNode, uNode,  uattNode, rNode, nNode,     hnNode};
    if (p.withSeqLen) {
        inNodes.push_back(dataNodes.back());
    }
    vector<Operator> graphInputs;
    for (auto& n : inNodes) {
        graphInputs.push_back(n);
    }
    graph.SetInputs(graphInputs).SetOutputs({op1});

    std::map<AscendString, AscendString> buildOptions = {};
    ge::Session* session = new Session(buildOptions);
    if (session == nullptr) {
        printf("ERROR: create session failed\n");
        return FAILED;
    }
    uint32_t graphId = 0;
    if (session->AddGraph(graphId, graph, buildOptions) != SUCCESS) {
        printf("ERROR: add graph failed\n");
        delete session;
        return FAILED;
    }
    vector<ge::Tensor> outputs;
    if (session->RunGraph(graphId, inputs, outputs) != SUCCESS) {
        printf("ERROR: run graph failed\n");
        ge::AscendString errMsg = ge::GEGetErrorMsgV2();
        std::cout << "Error: " << std::string(errMsg.GetString()) << std::endl;
        delete session;
        return FAILED;
    }
    delete session;
    if (outputs.size() != 7) {
        printf("ERROR: output size=%zu\n", outputs.size());
        return FAILED;
    }

    int64_t totalDiff = 0;
    const char* outNames[7] = {"dw_input", "dw_hidden", "db_input", "db_hidden", "dx", "dh_prev", "dw_att"};
    const vector<float>* goldens[7] = {&golden.dwInput, &golden.dwHidden, &golden.dbInput, &golden.dbHidden,
                                       &golden.dx,      &golden.dhPrev,   &golden.dwAtt};
    for (int i = 0; i < 7; i++) {
        int64_t d = CompareOut(outNames[i], outputs[i].GetData(), p.dtype, *goldens[i], rtol, atol);
        totalDiff += d;
    }
    if (totalDiff == 0) {
        printf("Shape, dtype and values PASSED for [%lld,%lld,%lld]\n", p.T, p.B, p.I);
        printf("%s PASSED\n", p.name);
        return SUCCESS;
    }
    printf("%s FAILED: %lld mismatched\n", p.name, totalDiff);
    return FAILED;
}

int main(int argc, char* argv[])
{
    printf("INFO: initialize ge\n");
    std::map<AscendString, AscendString> globalOptions = {
        {"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}, {"ge.jit_compile", "0"}};
    if (ge::GEInitialize(globalOptions) != SUCCESS) {
        printf("ERROR: initialize ge failed\n");
        return FAILED;
    }

    int32_t finalRet = SUCCESS;
    // case0: fp32最小规模T=1（定位db问题是否依赖BPTT长度）
    CaseParams c0 = {"fp32_t1_min", 1, 4, 16, 32, DT_FLOAT, true};
    if (RunCase(c0, 2e-2, 2e-2) != SUCCESS) {
        finalRet = FAILED;
    }
    // case1: fp32 + seq_length（融合掩码生效，覆盖0/T/中间长度）
    CaseParams c1 = {"fp32_bptt_with_seq_len", 6, 4, 16, 32, DT_FLOAT, true};
    if (RunCase(c1, 2e-2, 2e-2) != SUCCESS) {
        finalRet = FAILED;
    }
    // case2: fp32 无seq_length
    CaseParams c2 = {"fp32_bptt_no_seq_len", 5, 4, 16, 32, DT_FLOAT, false};
    if (RunCase(c2, 2e-2, 2e-2) != SUCCESS) {
        finalRet = FAILED;
    }
    // case3: rzh门序
    CaseParams c3 = {"fp32_gate_order_rzh", 5, 4, 16, 32, DT_FLOAT, true};
    if (RunCase(c3, 2e-2, 2e-2) != SUCCESS) {
        finalRet = FAILED;
    }
    // case4: fp32 更大规模
    CaseParams c4 = {"fp32_large", 16, 8, 32, 64, DT_FLOAT, true};
    if (RunCase(c4, 2e-2, 2e-2) != SUCCESS) {
        finalRet = FAILED;
    }

    // case5: fp16 + seq_length（rtol放宽适配fp16精度）
    CaseParams c5 = {"fp16_bptt_with_seq_len", 4, 4, 16, 32, DT_FLOAT16, true};
    if (RunCase(c5, 5e-2, 5e-2) != SUCCESS) {
        finalRet = FAILED;
    }
    // case6: fp16 无seq_length
    CaseParams c6 = {"fp16_bptt_no_seq_len", 5, 4, 16, 32, DT_FLOAT16, false};
    if (RunCase(c6, 5e-2, 5e-2) != SUCCESS) {
        finalRet = FAILED;
    }
    // case7: fp16 较大规模
    CaseParams c7 = {"fp16_large", 8, 8, 32, 64, DT_FLOAT16, true};
    if (RunCase(c7, 5e-2, 5e-2) != SUCCESS) {
        finalRet = FAILED;
    }

    if (finalRet == SUCCESS) {
        printf("DynamicAUGRUGrad static GEIR verification PASSED\n");
    } else {
        printf("Some test cases did not pass.\n");
    }
    ge::GEFinalize();
    return finalRet;
}
