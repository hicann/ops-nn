/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the License).
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file test_geir_dynamic_augru_grad_dynamic.cpp
 * @brief DynamicAUGRUGrad动态GEIR验证：-1未知维与-2未知Rank两场景，
 *        同一Session单张图AddGraph一次，连续RunGraph多组具体Shape，
 *        与CPU golden全量比对全部输出的Shape、dtype和数值
 */

#include <cmath>
#include <cstring>
#include <cstdint>
#include <iostream>
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

#include "../../op_graph/dynamic_augru_grad_proto.h"

#define FAILED -1
#define SUCCESS 0

using namespace ge;
using std::string;
using std::vector;

static constexpr int64_t GATE_NUM = 3;
// 声明维中H保持具体值：InferShape含3*H算术校验（weight_hidden=[H,3H]），
// H未知(-1)时3*hSize=-3无法与-1通过build期校验；T/B/I全部声明-1
static constexpr int64_t DECL_H = 32;
static constexpr int64_t DECL_3H = GATE_NUM * DECL_H;

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

// CPU golden：完整AUGRU反向BPTT（与kernel公式一致，同静态example）
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

// 具体Shape列表序列化为无空格JSON（与证据校验器canonical格式一致）
static string ShapeListJson(const vector<vector<int64_t>>& shapes)
{
    string s = "[";
    for (size_t i = 0; i < shapes.size(); i++) {
        if (i > 0) {
            s += ",";
        }
        s += "[";
        for (size_t j = 0; j < shapes[i].size(); j++) {
            if (j > 0) {
                s += ",";
            }
            s += std::to_string(shapes[i][j]);
        }
        s += "]";
    }
    s += "]";
    return s;
}

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

// 一组具体Shape的全部输入数据与golden（确定随机种子，失败可复现）
struct GroupData {
    int64_t T = 0;
    int64_t B = 0;
    int64_t I = 0;
    int64_t H = 0;
    bool withMask = false;
    vector<vector<int64_t>> shapes; // 具体输入Shape（构图输入顺序）
    vector<const void*> datas;
    vector<DataType> dtypes;
    GoldenResult golden;
    vector<float> x, wInput, wHidden, att, initH, h, dy, dh, update, updateAtt, reset, newGate, hiddenNew;
    vector<int32_t> seqLen;
    vector<uint8_t> mask;
    vector<uint16_t> xH, wInputH, wHiddenH, attH, initHH, hH, dyH, dhH, updateH, updateAttH, resetH, newH, hiddenNewH;
};

static GroupData MakeGroupData(int64_t t, int64_t b, int64_t iDim, int64_t hDim, DataType dtype, bool withMask,
                               const vector<int64_t>& maskShape, uint32_t seedBase)
{
    GroupData g;
    g.T = t;
    g.B = b;
    g.I = iDim;
    g.H = hDim;
    g.withMask = withMask;
    int64_t threeH = GATE_NUM * hDim;
    size_t tbh = static_cast<size_t>(t * b * hDim);
    size_t tbi = static_cast<size_t>(t * b * iDim);

    g.x = GenRandom<float>(tbi, 0.0f, 1.0f, seedBase + 1);
    g.wInput = GenRandom<float>(static_cast<size_t>(iDim * threeH), -0.05f, 0.05f, seedBase + 2);
    g.wHidden = GenRandom<float>(static_cast<size_t>(hDim * threeH), -0.05f, 0.05f, seedBase + 3);
    g.att = GenRandom<float>(tbh, 0.0f, 1.0f, seedBase + 4);
    g.initH = GenRandom<float>(static_cast<size_t>(b * hDim), -1.0f, 1.0f, seedBase + 5);
    g.h = GenRandom<float>(tbh, -1.0f, 1.0f, seedBase + 6);
    g.dy = GenRandom<float>(tbh, -1.0f, 1.0f, seedBase + 7);
    g.dh = GenRandom<float>(static_cast<size_t>(b * hDim), -1.0f, 1.0f, seedBase + 8);
    g.update = GenRandom<float>(tbh, 0.0f, 1.0f, seedBase + 9);
    g.updateAtt = GenRandom<float>(tbh, 0.0f, 1.0f, seedBase + 10);
    g.reset = GenRandom<float>(tbh, 0.0f, 1.0f, seedBase + 11);
    g.newGate = GenRandom<float>(tbh, -1.0f, 1.0f, seedBase + 12);
    g.hiddenNew = GenRandom<float>(tbh, -1.0f, 1.0f, seedBase + 13);
    g.seqLen.resize(b);
    for (int64_t bb = 0; bb < b; bb++) {
        g.seqLen[bb] = static_cast<int32_t>((bb * 3 + 1) % (t + 1));
    }
    if (withMask) {
        size_t maskNum = 1;
        for (auto d : maskShape) {
            maskNum *= static_cast<size_t>(d);
        }
        g.mask.assign(maskNum, 1U); // mask为dropout占位（keep_prob=-1不生效），数值不参与计算
    }

    bool isHalf = (dtype == DT_FLOAT16);
    const void* xP = PrepInput(g.x, g.xH, isHalf);
    const void* wInputP = PrepInput(g.wInput, g.wInputH, isHalf);
    const void* wHiddenP = PrepInput(g.wHidden, g.wHiddenH, isHalf);
    const void* attP = PrepInput(g.att, g.attH, isHalf);
    const void* initHP = PrepInput(g.initH, g.initHH, isHalf);
    const void* hP = PrepInput(g.h, g.hH, isHalf);
    const void* dyP = PrepInput(g.dy, g.dyH, isHalf);
    const void* dhP = PrepInput(g.dh, g.dhH, isHalf);
    const void* updateP = PrepInput(g.update, g.updateH, isHalf);
    const void* updateAttP = PrepInput(g.updateAtt, g.updateAttH, isHalf);
    const void* resetP = PrepInput(g.reset, g.resetH, isHalf);
    const void* newP = PrepInput(g.newGate, g.newH, isHalf);
    const void* hiddenNewP = PrepInput(g.hiddenNew, g.hiddenNewH, isHalf);

    g.shapes = {{t, b, iDim}, {iDim, threeH}, {hDim, threeH}, {t, b, hDim}, {t, b, hDim},
                {b, hDim},    {t, b, hDim},   {t, b, hDim},   {b, hDim},    {t, b, hDim},
                {t, b, hDim}, {t, b, hDim},   {t, b, hDim},   {t, b, hDim}, {b}};
    g.datas = {xP,      wInputP,    wHiddenP, attP, attP,       initHP,         hP, dyP, dhP,
               updateP, updateAttP, resetP,   newP, hiddenNewP, g.seqLen.data()};
    g.dtypes.assign(15, dtype);
    g.dtypes[14] = DT_INT32;
    if (withMask) {
        g.shapes.push_back(maskShape);
        g.datas.push_back(g.mask.data());
        g.dtypes.push_back(DT_UINT8);
    }

    g.golden = ComputeGolden(g.x, g.wInput, g.wHidden, g.att, g.initH, g.h, g.dy, g.dh, g.update, g.updateAtt, g.reset,
                             g.newGate, g.hiddenNew, g.seqLen, t, b, iDim, hDim, 1);
    return g;
}

// 校验全部7个输出的Shape、dtype与逐元素数值
static bool CheckOutputs(const vector<ge::Tensor>& outputs, const GroupData& g, DataType dtype, double rtol,
                         double atol)
{
    if (outputs.size() != 7) {
        printf("  output size=%zu mismatch\n", outputs.size());
        return false;
    }
    int64_t threeH = GATE_NUM * g.H;
    vector<vector<int64_t>> expectShapes = {{g.I, threeH},   {g.H, threeH}, {threeH},  {threeH},
                                            {g.T, g.B, g.I}, {g.B, g.H},    {g.T, g.B}};
    const char* names[7] = {"dw_input", "dw_hidden", "db_input", "db_hidden", "dx", "dh_prev", "dw_att"};
    const vector<float>* goldens[7] = {&g.golden.dwInput, &g.golden.dwHidden, &g.golden.dbInput, &g.golden.dbHidden,
                                       &g.golden.dx,      &g.golden.dhPrev,   &g.golden.dwAtt};
    for (int i = 0; i < 7; i++) {
        vector<int64_t> gotDims = outputs[i].GetTensorDesc().GetShape().GetDims();
        if (gotDims != expectShapes[i]) {
            printf("  [%s] shape mismatch: got %s expect %s\n", names[i], ShapeListJson({gotDims}).c_str(),
                   ShapeListJson({expectShapes[i]}).c_str());
            return false;
        }
        if (outputs[i].GetTensorDesc().GetDataType() != dtype) {
            printf("  [%s] dtype mismatch: got %d expect %d\n", names[i],
                   static_cast<int>(outputs[i].GetTensorDesc().GetDataType()), static_cast<int>(dtype));
            return false;
        }
        if (CompareOut(names[i], outputs[i].GetData(), dtype, *goldens[i], rtol, atol) != 0) {
            printf("  [%s] values mismatch\n", names[i]);
            return false;
        }
    }
    return true;
}

// 构建一张动态图：declaredShapes为声明Shape（含-1/-2），seq_length与mask是否连接由入参决定
static bool BuildDynamicGraph(Graph& graph, const string& scenario, const vector<vector<int64_t>>& declaredShapes,
                              bool withMask, DataType dtype, vector<op::Data>& dataNodes)
{
    string suffix = "_" + scenario;
    auto makeData = [&](const char* name, size_t idx, DataType nodeDtype) {
        string nodeName = string(name) + suffix;
        auto node = op::Data(nodeName.c_str());
        TensorDesc desc(ge::Shape(declaredShapes[idx]), FORMAT_ND, nodeDtype);
        desc.SetRealDimCnt(declaredShapes[idx].size());
        node.update_input_desc_x(desc);
        node.update_output_desc_y(desc);
        graph.AddOp(node);
        dataNodes.push_back(node);
        return node;
    };

    auto op1 = op::DynamicAUGRUGrad(("dynamic_augru_grad" + suffix).c_str());
    auto xNode = makeData("x", 0, dtype);
    auto wiNode = makeData("weight_input", 1, dtype);
    auto whNode = makeData("weight_hidden", 2, dtype);
    auto attNode = makeData("weight_att", 3, dtype);
    auto yNode = makeData("y", 4, dtype);
    auto initHNode = makeData("init_h", 5, dtype);
    auto hNode = makeData("h", 6, dtype);
    auto dyNode = makeData("dy", 7, dtype);
    auto dhNode = makeData("dh", 8, dtype);
    auto uNode = makeData("update", 9, dtype);
    auto uattNode = makeData("update_att", 10, dtype);
    auto rNode = makeData("reset", 11, dtype);
    auto nNode = makeData("new", 12, dtype);
    auto hnNode = makeData("hidden_new", 13, dtype);
    auto seqNode = makeData("seq_length", 14, DT_INT32);
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
    op1.set_input_seq_length(seqNode);
    if (withMask) {
        auto maskNode = makeData("mask", 15, DT_UINT8);
        op1.set_input_mask(maskNode);
    }

    TensorDesc dwInputDesc(ge::Shape({-1, DECL_3H}), FORMAT_ND, dtype);
    TensorDesc dwHiddenDesc(ge::Shape({DECL_H, DECL_3H}), FORMAT_ND, dtype);
    TensorDesc dbDesc(ge::Shape({DECL_3H}), FORMAT_ND, dtype);
    TensorDesc dxDesc(ge::Shape({-1, -1, -1}), FORMAT_ND, dtype);
    TensorDesc dhPrevDesc(ge::Shape({-1, DECL_H}), FORMAT_ND, dtype);
    TensorDesc dwAttDesc(ge::Shape({-1, -1}), FORMAT_ND, dtype);
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
    op1.set_attr_gate_order("zrh");
    op1.set_attr_reset_after(true);

    vector<Operator> graphInputs;
    for (auto& n : dataNodes) {
        graphInputs.push_back(n);
    }
    graph.SetInputs(graphInputs).SetOutputs({op1});
    return true;
}

// 单场景：AddGraph一次后同Session连续RunGraph全部具体Shape组
static int32_t RunScenario(ge::Session* session, uint32_t graphId, const string& scenario,
                           const vector<vector<int64_t>>& declaredShapes, bool withMask,
                           const vector<GroupData>& groups, DataType dtype, double rtol, double atol)
{
    printf("Scenario %s, declared shape %s\n", scenario.c_str(), ShapeListJson(declaredShapes).c_str());
    Graph graph(scenario.c_str());
    vector<op::Data> dataNodes;
    BuildDynamicGraph(graph, scenario, declaredShapes, withMask, dtype, dataNodes);

    std::map<AscendString, AscendString> buildOptions = {};
    if (session->AddGraph(graphId, graph, buildOptions) != SUCCESS) {
        printf("add graph %s not success\n", scenario.c_str());
        return FAILED;
    }

    int32_t passed = 0;
    for (const auto& g : groups) {
        string concreteJson = ShapeListJson(g.shapes);
        printf("Run concrete shape %s\n", concreteJson.c_str());
        vector<ge::Tensor> inputs;
        for (size_t idx = 0; idx < g.shapes.size(); idx++) {
            inputs.push_back(MakeTensor(g.shapes[idx], g.dtypes[idx], g.datas[idx]));
        }
        vector<ge::Tensor> outputs;
        if (session->RunGraph(graphId, inputs, outputs) != SUCCESS) {
            ge::AscendString errMsg = ge::GEGetErrorMsgV2();
            std::cout << "run graph not success: " << std::string(errMsg.GetString()) << std::endl;
            continue;
        }
        if (CheckOutputs(outputs, g, dtype, rtol, atol)) {
            printf("Shape, dtype and values PASSED for %s\n", concreteJson.c_str());
            passed++;
        }
    }
    printf("Scenario %s summary: %d/%zu passed\n", scenario.c_str(), passed, groups.size());
    return (passed == static_cast<int32_t>(groups.size())) ? SUCCESS : FAILED;
}

int main(int argc, char* argv[])
{
    printf("INFO: initialize ge\n");
    std::map<AscendString, AscendString> globalOptions = {
        {"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}, {"ge.jit_compile", "0"}};
    if (ge::GEInitialize(globalOptions) != SUCCESS) {
        printf("initialize ge not success\n");
        return FAILED;
    }

    std::map<AscendString, AscendString> buildOptions = {};
    ge::Session* session = new Session(buildOptions);
    if (session == nullptr) {
        printf("create session not success\n");
        ge::GEFinalize();
        return FAILED;
    }

    int32_t finalRet = SUCCESS;

    // 场景一：-1未知维（T/B/I声明-1，H因InferShape的3*H算术校验保持具体值）
    const vector<vector<int64_t>> declaredMinus1 = {
        {-1, -1, -1},     {-1, DECL_3H},    {DECL_H, DECL_3H}, {-1, -1, DECL_H}, {-1, -1, DECL_H},
        {-1, DECL_H},     {-1, -1, DECL_H}, {-1, -1, DECL_H},  {-1, DECL_H},     {-1, -1, DECL_H},
        {-1, -1, DECL_H}, {-1, -1, DECL_H}, {-1, -1, DECL_H},  {-1, -1, DECL_H}, {-1}};
    vector<GroupData> groupsMinus1;
    groupsMinus1.push_back(MakeGroupData(4, 2, 16, DECL_H, DT_FLOAT, false, {}, 100));
    groupsMinus1.push_back(MakeGroupData(6, 3, 24, DECL_H, DT_FLOAT, false, {}, 200));
    groupsMinus1.push_back(MakeGroupData(2, 1, 48, DECL_H, DT_FLOAT, false, {}, 300));
    if (RunScenario(session, 1, "unknown_dim_minus_1", declaredMinus1, false, groupsMinus1, DT_FLOAT, 2e-2, 2e-2) !=
        SUCCESS) {
        finalRet = FAILED;
    }

    // 场景二：-2未知Rank（mask为不参与数值计算的占位输入，协议允许任意Rank，实跑覆盖rank 1/2/3）
    const vector<vector<int64_t>> declaredMinus2 = {{-1, -1, -1},
                                                    {-1, DECL_3H},
                                                    {DECL_H, DECL_3H},
                                                    {-1, -1, DECL_H},
                                                    {-1, -1, DECL_H},
                                                    {-1, DECL_H},
                                                    {-1, -1, DECL_H},
                                                    {-1, -1, DECL_H},
                                                    {-1, DECL_H},
                                                    {-1, -1, DECL_H},
                                                    {-1, -1, DECL_H},
                                                    {-1, -1, DECL_H},
                                                    {-1, -1, DECL_H},
                                                    {-1, -1, DECL_H},
                                                    {-1},
                                                    {-2}};
    vector<GroupData> groupsMinus2;
    groupsMinus2.push_back(MakeGroupData(3, 2, 16, DECL_H, DT_FLOAT16, true, {3 * 2}, 400));
    groupsMinus2.push_back(MakeGroupData(4, 2, 32, DECL_H, DT_FLOAT16, true, {4, 2}, 500));
    groupsMinus2.push_back(MakeGroupData(5, 1, 24, DECL_H, DT_FLOAT16, true, {5, 1, 1}, 600));
    if (RunScenario(session, 2, "unknown_rank_minus_2", declaredMinus2, true, groupsMinus2, DT_FLOAT16, 5e-2, 5e-2) !=
        SUCCESS) {
        finalRet = FAILED;
    }

    delete session;
    if (finalRet == SUCCESS) {
        printf("DynamicAUGRUGrad dynamic GEIR verification PASSED (-1 and -2)\n");
    }
    ge::GEFinalize();
    return finalRet;
}
