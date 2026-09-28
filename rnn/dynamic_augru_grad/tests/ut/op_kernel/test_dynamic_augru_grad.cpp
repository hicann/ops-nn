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
 * \file test_dynamic_augru_grad.cpp
 * \brief DynamicAUGRUGrad kernel UT（CPU仿真）：与CPU golden全量比对
 *
 * kernel按AIC模式运行（matmul+向量混合），tiling由本文件手工填入，
 * 单核（blockDim=1）执行完整BPTT，验证含融合掩码在内的全部输出。
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <vector>
#include "gtest/gtest.h"

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#endif

#include "data_utils.h"
#include "../../../op_kernel/arch35/dynamic_augru_grad.h"

using namespace std;

constexpr int64_t GATE_NUM = 3;

// 各输入用独立seed：同seed同分布的多个张量数值完全相同，会掩盖同类缓冲
// 交换（如update/reset、h/dy）产生的接线错误
template <typename T>
static vector<T> GenRandomVector(size_t count, T low, T high, uint32_t seed)
{
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(static_cast<double>(low), static_cast<double>(high));
    vector<T> data(count);
    for (size_t i = 0; i < count; i++) {
        data[i] = static_cast<T>(dist(gen));
    }
    return data;
}

// 按dtype实例化的kernel入口（与真实入口dynamic_augru_grad一致，供fp32/fp16用例分别实例化）
template <typename T>
__global__ __aicore__ void augru_grad_ut_entry(GM_ADDR x, GM_ADDR weightInput, GM_ADDR weightHidden, GM_ADDR weightAtt,
                                               GM_ADDR y, GM_ADDR initH, GM_ADDR h, GM_ADDR dy, GM_ADDR dh,
                                               GM_ADDR update, GM_ADDR updateAtt, GM_ADDR reset, GM_ADDR newGate,
                                               GM_ADDR hiddenNew, GM_ADDR seqLength, GM_ADDR mask, GM_ADDR dwInput,
                                               GM_ADDR dwHidden, GM_ADDR dbInput, GM_ADDR dbHidden, GM_ADDR dx,
                                               GM_ADDR dhPrev, GM_ADDR dwAtt, GM_ADDR workspace, GM_ADDR tiling)
{
#ifdef __CCE_KT_TEST__
    auto tilingData = *reinterpret_cast<DynamicAUGRUGradTilingData*>(tiling);
#else
    REGISTER_TILING_DEFAULT(DynamicAUGRUGradTilingData);
    GET_TILING_DATA_WITH_STRUCT(DynamicAUGRUGradTilingData, tilingData, tiling);
#endif
    NsDynamicAUGRUGrad::DynamicAUGRUGradKernel<T> op;
    REGIST_MATMUL_OBJ(&op.pipe, GetSysWorkSpacePtr(), op.dgateMM, &tilingData.dgateMMParam, op.dwInputMM,
                      &tilingData.dwInputMMParam, op.dwHiddenMM, &tilingData.dwHiddenMMParam, op.dxMM,
                      &tilingData.dxMMParam);
    op.Init(x, weightInput, weightHidden, weightAtt, y, initH, h, dy, dh, update, updateAtt, reset, newGate, hiddenNew,
            seqLength, mask, dwInput, dwHidden, dbInput, dbHidden, dx, dhPrev, dwAtt, workspace, &tilingData);
    op.Process();
}

class DynamicAUGRUGradKernelTest : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "DynamicAUGRUGradKernelTest SetUp" << endl; }
    static void TearDownTestCase() { cout << "DynamicAUGRUGradKernelTest TearDown" << endl; }
};

// 单核跑完整(M,N,K)的matmul tiling（参照gru_grad UT）；K/N超L0时按baseK/baseN分块
static void InitTCubeTiling(TCubeTiling* t, int64_t m, int64_t n, int64_t k)
{
    t->usedCoreNum = 1;
    t->M = m;
    t->N = n;
    t->Ka = k;
    t->Kb = k;
    t->singleCoreM = m;
    t->singleCoreN = n;
    t->singleCoreK = k;
    // L0B为64KB且depthB1=2双缓冲：baseN*baseK*4*2<=64KB -> baseN*baseK<=8192
    constexpr int64_t L0_LIMIT = 8192;
    int64_t baseK = k;
    int64_t baseN = n;
    while (baseN * baseK > L0_LIMIT && baseK > 128) {
        baseK /= 2;
    }
    while (baseN * baseK > L0_LIMIT && baseN > 128) {
        baseN /= 2;
    }
    t->baseM = m;
    t->baseN = baseN;
    t->baseK = baseK;
    t->depthA1 = 1;
    t->depthB1 = 1;
    t->depthAL1CacheUB = 0;
    t->depthBL1CacheUB = 0;
    t->stepM = 1;
    t->stepN = 1;
    t->isBias = 0;
    t->transLength = 0;
    t->iterateOrder = 0;
    t->shareMode = 0;
    t->shareL1Size = 6144;
    t->shareL0CSize = 2048;
    t->shareUbSize = 0;
    t->batchM = 1;
    t->batchN = 1;
    t->singleBatchM = 1;
    t->singleBatchN = 1;
    t->stepKa = 1;
    t->stepKb = 1;
    t->dbL0A = 2;
    t->dbL0B = 2;
    t->dbL0C = 1;
    t->ALayoutInfoB = 0;
    t->ALayoutInfoS = 0;
    t->ALayoutInfoN = 0;
    t->ALayoutInfoG = 0;
    t->ALayoutInfoD = 0;
    t->BLayoutInfoB = 0;
    t->BLayoutInfoS = 0;
    t->BLayoutInfoN = 0;
    t->BLayoutInfoG = 0;
    t->BLayoutInfoD = 0;
    t->CLayoutInfoB = 0;
    t->CLayoutInfoS1 = 0;
    t->CLayoutInfoN = 0;
    t->CLayoutInfoG = 0;
    t->CLayoutInfoS2 = 0;
    t->BatchNum = 0;
}

static void InitTiling(uint8_t* tilingBuf, int64_t t, int64_t b, int64_t iSize, int64_t hSize, int64_t isSeqLen,
                       int64_t gateOrder, int64_t enableDbInline = 0, int64_t enablePipeline = 0)
{
    auto* tiling = reinterpret_cast<DynamicAUGRUGradTilingData*>(tilingBuf);
    memset(tilingBuf, 0, sizeof(DynamicAUGRUGradTilingData));
    int64_t tb = t * b;
    int64_t hPad = (hSize + 15) / 16 * 16;
    int64_t threeHPad = GATE_NUM * hPad;
    // MM维度与host侧GetMatmulTilings一致（padded维）
    InitTCubeTiling(&tiling->dgateMMParam, b, hPad, std::min(hPad / 2, MM_RECURRENT_CHUNK));
    tiling->dgateMMParam.Ka = threeHPad;
    tiling->dgateMMParam.Kb = threeHPad;
    InitTCubeTiling(&tiling->dwInputMMParam, iSize, threeHPad, tb);
    InitTCubeTiling(&tiling->dwHiddenMMParam, hPad, threeHPad, tb);
    InitTCubeTiling(&tiling->dxMMParam, tb, iSize, std::min(hPad, MM_RECURRENT_CHUNK));
    tiling->dxMMParam.Ka = threeHPad;
    tiling->dxMMParam.Kb = threeHPad;
    tiling->timeStep = t;
    tiling->batchSize = b;
    tiling->hiddenSize = hSize;
    tiling->inputSize = iSize;
    tiling->hPad = hPad;
    tiling->isSeqLength = isSeqLen;
    tiling->gateOrder = gateOrder;
    // H整块进UB（用例保证H<=1024），bTile取min(B, 32)
    tiling->hTile = ((hSize + 7) / 8) * 8;
    tiling->bTile = b < 32 ? b : 32;
    // 15个fp32向量缓冲（VF操作数9+结果6）+辅助缓冲需在256KB UB内：15*2048*4+20KB
    tiling->ubLength = 2048;
    // bias归约：按FP32_ALIGN列切分
    tiling->singleCoreReduceN = 8;
    tiling->enableDbInline = enableDbInline;
    tiling->enablePipeline = enablePipeline && threeHPad == MM_RECURRENT_CHUNK;
}

// CPU golden：完整AUGRU反向BPTT
struct GoldenInputs {
    vector<float> x;         // [T,B,I]
    vector<float> wInput;    // [I,3H]
    vector<float> wHidden;   // [H,3H]
    vector<float> att;       // [T,B,H]
    vector<float> initH;     // [B,H]
    vector<float> h;         // [T,B,H]
    vector<float> dy;        // [T,B,H]
    vector<float> dh;        // [B,H]
    vector<float> update;    // [T,B,H]
    vector<float> updateAtt; // [T,B,H]
    vector<float> reset;     // [T,B,H]
    vector<float> newGate;   // [T,B,H]
    vector<float> hiddenNew; // [T,B,H]
    vector<int32_t> seqLen;  // [B]
};

struct GoldenOutputs {
    vector<float> dwInput;  // [I,3H]
    vector<float> dwHidden; // [H,3H]
    vector<float> dbInput;  // [3H]
    vector<float> dbHidden; // [3H]
    vector<float> dx;       // [T,B,I]
    vector<float> dhPrev;   // [B,H]
    vector<float> dwAtt;    // [T,B]
};

static GoldenOutputs ComputeGolden(const GoldenInputs& in, int64_t T, int64_t B, int64_t I, int64_t H, int64_t isSeqLen,
                                   int64_t gateOrder)
{
    int64_t threeH = GATE_NUM * H;
    GoldenOutputs out;
    out.dwInput.assign(I * threeH, 0.0f);
    out.dwHidden.assign(H * threeH, 0.0f);
    out.dbInput.assign(threeH, 0.0f);
    out.dbHidden.assign(threeH, 0.0f);
    out.dx.assign(T * B * I, 0.0f);
    out.dhPrev.assign(B * H, 0.0f);
    out.dwAtt.assign(T * B, 0.0f);

    int64_t zSlot = gateOrder == 0 ? 0 : 1;
    int64_t rSlot = 1 - zSlot;
    int64_t nSlot = 2;

    // dh回传：dhPrev[b,h]，从dh起
    for (int64_t b = 0; b < B; b++) {
        for (int64_t hh = 0; hh < H; hh++) {
            out.dhPrev[b * H + hh] = in.dh[b * H + hh];
        }
    }
    // dGi/dGh按[t,b,g,h]组织
    vector<float> dGh(T * B * threeH, 0.0f);
    vector<float> dGi(T * B * threeH, 0.0f);
    vector<float> hPrev(T * B * H, 0.0f);

    for (int64_t t = T - 1; t >= 0; t--) {
        for (int64_t b = 0; b < B; b++) {
            float mask = 1.0f;
            if (isSeqLen == 1) {
                mask = (t < in.seqLen[b]) ? 1.0f : 0.0f;
            }
            for (int64_t hh = 0; hh < H; hh++) {
                int64_t tbh = (t * B + b) * H + hh;
                int64_t bh = b * H + hh;
                float gradH = out.dhPrev[bh] * mask + in.dy[tbh];
                float z = in.update[tbh];
                float u = in.updateAtt[tbh];
                float a = in.att[tbh];
                float r = in.reset[tbh];
                float n = in.newGate[tbh];
                float hn = in.hiddenNew[tbh];
                float hp = (t == 0) ? in.initH[bh] : in.h[((t - 1) * B + b) * H + hh];

                float ghn = gradH * (hp - n); // dz公用项
                float dnt = gradH * (1 - u) * (1 - n * n);
                float dr = dnt * r * (1 - r) * hn;
                float dz = ghn * (1 - a) * z * (1 - z);
                float dwAttElem = -ghn * z;

                int64_t giBase = (t * B + b) * threeH;
                dGi[giBase + nSlot * H + hh] = dnt;
                dGh[giBase + nSlot * H + hh] = dnt * r;
                dGi[giBase + rSlot * H + hh] = dr;
                dGh[giBase + rSlot * H + hh] = dr;
                dGi[giBase + zSlot * H + hh] = dz;
                dGh[giBase + zSlot * H + hh] = dz;
                hPrev[(t * B + b) * H + hh] = hp;
                out.dwAtt[t * B + b] += dwAttElem;

                // dh回传：dh_prev_from_h = gradH * u; 下一轮 dh = dGh @ wHidden^T + from_h
                out.dhPrev[bh] = gradH * u;
            }
            // dh += dGh[t,b,:] @ wHidden^T
            for (int64_t hh = 0; hh < H; hh++) {
                float acc = 0.0f;
                for (int64_t k = 0; k < threeH; k++) {
                    acc += dGh[(t * B + b) * threeH + k] * in.wHidden[hh * threeH + k];
                }
                out.dhPrev[b * H + hh] += acc;
            }
        }
    }

    // dw_input = x^T @ dGi; db_input = sum(dGi)
    for (int64_t m = 0; m < I; m++) {
        for (int64_t nn = 0; nn < threeH; nn++) {
            float acc = 0.0f;
            for (int64_t k = 0; k < T * B; k++) {
                acc += in.x[k * I + m] * dGi[k * threeH + nn];
            }
            out.dwInput[m * threeH + nn] = acc;
        }
    }
    for (int64_t nn = 0; nn < threeH; nn++) {
        float acc = 0.0f;
        for (int64_t k = 0; k < T * B; k++) {
            acc += dGi[k * threeH + nn];
        }
        out.dbInput[nn] = acc;
    }
    // dw_hidden = hPrev^T @ dGh; db_hidden = sum(dGh)
    for (int64_t m = 0; m < H; m++) {
        for (int64_t nn = 0; nn < threeH; nn++) {
            float acc = 0.0f;
            for (int64_t k = 0; k < T * B; k++) {
                acc += hPrev[k * H + m] * dGh[k * threeH + nn];
            }
            out.dwHidden[m * threeH + nn] = acc;
        }
    }
    for (int64_t nn = 0; nn < threeH; nn++) {
        float acc = 0.0f;
        for (int64_t k = 0; k < T * B; k++) {
            acc += dGh[k * threeH + nn];
        }
        out.dbHidden[nn] = acc;
    }
    // dx = dGi @ wInput^T
    for (int64_t m = 0; m < T * B; m++) {
        for (int64_t nn = 0; nn < I; nn++) {
            float acc = 0.0f;
            for (int64_t k = 0; k < threeH; k++) {
                acc += dGi[m * threeH + k] * in.wInput[nn * threeH + k];
            }
            out.dx[m * I + nn] = acc;
        }
    }
    return out;
}

static int64_t CompareBuffer(const char* name, const float* got, const vector<float>& expect, double rtol, double atol,
                             int64_t limit = 8)
{
    int64_t diffCnt = 0;
    for (size_t i = 0; i < expect.size(); i++) {
        double actual = static_cast<double>(got[i]);
        double expected = static_cast<double>(expect[i]);
        bool mismatch = (std::isnan(expected) && !std::isnan(actual)) ||
                        (std::isinf(expected) &&
                         (!std::isinf(actual) || std::signbit(actual) != std::signbit(expected))) ||
                        (std::isfinite(expected) &&
                         (!std::isfinite(actual) || std::fabs(actual - expected) > atol + rtol * std::fabs(expected)));
        if (mismatch) {
            if (diffCnt < limit) {
                printf("  [%s] mismatch idx=%zu got=%f expect=%f\n", name, i, got[i], expect[i]);
            }
            diffCnt++;
        }
    }
    return diffCnt;
}

template <typename DT>
static float ToFloat(DT v)
{
    return static_cast<float>(v);
}

static float ToFloat(half v) { return static_cast<float>(v); }

template <typename DT>
static std::vector<DT> ToHalfVec(const std::vector<float>& in)
{
    std::vector<DT> out(in.size());
    for (size_t i = 0; i < in.size(); i++) {
        out[i] = static_cast<DT>(in[i]);
    }
    return out;
}

template <typename DT>
static void RunKernelCase(int64_t T, int64_t B, int64_t I, int64_t H, bool withSeqLen, int64_t gateOrder,
                          uint32_t blockDim, double rtol, double atol, bool dbInline = false, bool pipeline = false)
{
    int64_t threeH = GATE_NUM * H;
    int64_t tb = T * B;
    size_t tbh = static_cast<size_t>(tb * H);
    size_t tbi = static_cast<size_t>(tb * I);

    GoldenInputs in;
    in.x = GenRandomVector<float>(tbi, 0.0f, 1.0f, 1);
    in.wInput = GenRandomVector<float>(static_cast<size_t>(I * threeH), -0.05f, 0.05f, 2);
    in.wHidden = GenRandomVector<float>(static_cast<size_t>(H * threeH), -0.05f, 0.05f, 3);
    in.att = GenRandomVector<float>(tbh, 0.0f, 1.0f, 4);
    in.initH = GenRandomVector<float>(static_cast<size_t>(B * H), -1.0f, 1.0f, 5);
    in.h = GenRandomVector<float>(tbh, -1.0f, 1.0f, 6);
    in.dy = GenRandomVector<float>(tbh, -1.0f, 1.0f, 7);
    in.dh = GenRandomVector<float>(static_cast<size_t>(B * H), -1.0f, 1.0f, 8);
    in.update = GenRandomVector<float>(tbh, 0.0f, 1.0f, 9);
    in.updateAtt = GenRandomVector<float>(tbh, 0.0f, 1.0f, 10);
    in.reset = GenRandomVector<float>(tbh, 0.0f, 1.0f, 11);
    in.newGate = GenRandomVector<float>(tbh, -1.0f, 1.0f, 12);
    in.hiddenNew = GenRandomVector<float>(tbh, -1.0f, 1.0f, 13);
    if (withSeqLen) {
        in.seqLen.resize(B);
        for (int64_t b = 0; b < B; b++) {
            in.seqLen[b] = static_cast<int32_t>((b * 3 + 1) % (T + 1)); // 覆盖0与T
        }
    }
    // fp16：输入先量化round-trip（golden与kernel所见输入一致，仅比对累加序差异）
    if constexpr (sizeof(DT) == 2) {
        auto roundTrip = [](vector<float>& v) {
            for (size_t i = 0; i < v.size(); i++) {
                v[i] = static_cast<float>(static_cast<half>(v[i]));
            }
        };
        roundTrip(in.x);
        roundTrip(in.wInput);
        roundTrip(in.wHidden);
        roundTrip(in.att);
        roundTrip(in.initH);
        roundTrip(in.h);
        roundTrip(in.dy);
        roundTrip(in.dh);
        roundTrip(in.update);
        roundTrip(in.updateAtt);
        roundTrip(in.reset);
        roundTrip(in.newGate);
        roundTrip(in.hiddenNew);
    }
    GoldenOutputs golden = ComputeGolden(in, T, B, I, H, withSeqLen ? 1 : 0, gateOrder);

    // 输入按dtype转换后灌入（golden按fp32计算，比对时kernel输出转回fp32）
    std::vector<DT> xT = ToHalfVec<DT>(in.x);
    std::vector<DT> wInputT = ToHalfVec<DT>(in.wInput);
    std::vector<DT> wHiddenT = ToHalfVec<DT>(in.wHidden);
    std::vector<DT> attT = ToHalfVec<DT>(in.att);
    std::vector<DT> initHT = ToHalfVec<DT>(in.initH);
    std::vector<DT> hT = ToHalfVec<DT>(in.h);
    std::vector<DT> dyT = ToHalfVec<DT>(in.dy);
    std::vector<DT> dhT = ToHalfVec<DT>(in.dh);
    std::vector<DT> updateT = ToHalfVec<DT>(in.update);
    std::vector<DT> uAttT = ToHalfVec<DT>(in.updateAtt);
    std::vector<DT> resetT = ToHalfVec<DT>(in.reset);
    std::vector<DT> newT = ToHalfVec<DT>(in.newGate);
    std::vector<DT> hnT = ToHalfVec<DT>(in.hiddenNew);
    uint8_t* xBuf = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(tbi * sizeof(DT)));
    uint8_t* wInputBuf = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(I * threeH * sizeof(DT)));
    uint8_t* wHiddenBuf = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(H * threeH * sizeof(DT)));
    uint8_t* attBuf = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(tbh * sizeof(DT)));
    uint8_t* yBuf = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(tbh * sizeof(DT))); // 占位输入
    uint8_t* maskBuf = nullptr;                                                     // 占位输入
    uint8_t* initHBuf = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(B * H * sizeof(DT)));
    uint8_t* hBuf = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(tbh * sizeof(DT)));
    uint8_t* dyBuf = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(tbh * sizeof(DT)));
    uint8_t* dhBuf = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(B * H * sizeof(DT)));
    uint8_t* updateBuf = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(tbh * sizeof(DT)));
    uint8_t* uAttBuf = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(tbh * sizeof(DT)));
    uint8_t* resetBuf = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(tbh * sizeof(DT)));
    uint8_t* newBuf = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(tbh * sizeof(DT)));
    uint8_t* hnBuf = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(tbh * sizeof(DT)));
    uint8_t* seqLenBuf = nullptr;
    if (withSeqLen) {
        seqLenBuf = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(B * sizeof(int32_t)));
    }
    uint8_t* dwInputBuf = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(I * threeH * sizeof(DT)));
    uint8_t* dwHiddenBuf = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(H * threeH * sizeof(DT)));
    uint8_t* dbInputBuf = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(threeH * sizeof(DT)));
    uint8_t* dbHiddenBuf = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(threeH * sizeof(DT)));
    uint8_t* dxBuf = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(tbi * sizeof(DT)));
    uint8_t* dhPrevBuf = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(B * H * sizeof(DT)));
    uint8_t* dwAttBuf = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(tb * sizeof(DT)));
    int64_t hPadT = (H + 15) / 16 * 16;
    int64_t threeHPadT = GATE_NUM * hPadT;
    int64_t tbPadT = (tb + 15) / 16 * 16;
    // kernel布局: dGh/dGi[tbPad,3HPad] hPrev[tbPad,HPad] dhPrevWs[B,HPad]
    //   dhFromH[B,H] + (needCopy)权重padded副本[HPad/I,3HPad] + 输出中转
    int64_t wsFloats = 2 * tbPadT * threeHPadT + tbPadT * hPadT + B * hPadT + B * H;
    if (sizeof(DT) == 2 || hPadT != H) {
        wsFloats += hPadT * threeHPadT + I * threeHPadT; // 权重padded副本(wHidden含尾行)
    }
    if (sizeof(DT) == 2) {
        wsFloats += tbPadT * I + I * threeHPadT + hPadT * threeHPadT + tb * I; // x/dwIn/dwHid/dx中转
    } else if (hPadT != H) {
        wsFloats += I * threeHPadT + hPadT * threeHPadT; // dw系列padded中转
    }
    if (dbInline) {
        wsFloats += 2 * static_cast<int64_t>(blockDim) * threeH; // dbPartialGi/Gh[blockDim,3H]
    }
    wsFloats += 2 * std::max(B * hPadT, tb * I);
    size_t wsElems = wsFloats;
    uint8_t* wsBuf = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(wsElems * sizeof(float)));
    uint8_t* tilingBuf = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(sizeof(DynamicAUGRUGradTilingData)));
    std::vector<uint8_t*> allocedBufs = {xBuf,   wInputBuf, wHiddenBuf, attBuf,      yBuf,       initHBuf,
                                         hBuf,   dyBuf,     dhBuf,      updateBuf,   uAttBuf,    resetBuf,
                                         newBuf, hnBuf,     dwInputBuf, dwHiddenBuf, dbInputBuf, dbHiddenBuf,
                                         dxBuf,  dhPrevBuf, dwAttBuf,   wsBuf,       tilingBuf};
    for (uint8_t* buf : allocedBufs) {
        ASSERT_NE(buf, nullptr);
    }

    memcpy(xBuf, xT.data(), tbi * sizeof(DT));
    memcpy(wInputBuf, wInputT.data(), I * threeH * sizeof(DT));
    memcpy(wHiddenBuf, wHiddenT.data(), H * threeH * sizeof(DT));
    memcpy(attBuf, attT.data(), tbh * sizeof(DT));
    memcpy(initHBuf, initHT.data(), B * H * sizeof(DT));
    memcpy(hBuf, hT.data(), tbh * sizeof(DT));
    memcpy(dyBuf, dyT.data(), tbh * sizeof(DT));
    memcpy(dhBuf, dhT.data(), B * H * sizeof(DT));
    memcpy(updateBuf, updateT.data(), tbh * sizeof(DT));
    memcpy(uAttBuf, uAttT.data(), tbh * sizeof(DT));
    memcpy(resetBuf, resetT.data(), tbh * sizeof(DT));
    memcpy(newBuf, newT.data(), tbh * sizeof(DT));
    memcpy(hnBuf, hnT.data(), tbh * sizeof(DT));
    if (withSeqLen) {
        memcpy(seqLenBuf, in.seqLen.data(), B * sizeof(int32_t));
    }

    InitTiling(tilingBuf, T, B, I, H, withSeqLen ? 1 : 0, gateOrder, dbInline ? 1 : 0, pipeline ? 1 : 0);

    ICPU_RUN_KF(augru_grad_ut_entry<DT>, blockDim, xBuf, wInputBuf, wHiddenBuf, attBuf, yBuf, initHBuf, hBuf, dyBuf,
                dhBuf, updateBuf, uAttBuf, resetBuf, newBuf, hnBuf, seqLenBuf, maskBuf, dwInputBuf, dwHiddenBuf,
                dbInputBuf, dbHiddenBuf, dxBuf, dhPrevBuf, dwAttBuf, wsBuf, tilingBuf);

    const DT* dwInput = reinterpret_cast<const DT*>(dwInputBuf);
    const DT* dwHidden = reinterpret_cast<const DT*>(dwHiddenBuf);
    const DT* dbInput = reinterpret_cast<const DT*>(dbInputBuf);
    const DT* dbHidden = reinterpret_cast<const DT*>(dbHiddenBuf);
    const DT* dx = reinterpret_cast<const DT*>(dxBuf);
    const DT* dhPrev = reinterpret_cast<const DT*>(dhPrevBuf);
    const DT* dwAtt = reinterpret_cast<const DT*>(dwAttBuf);
    auto toFloatVec = [&](const DT* p, size_t n) {
        std::vector<float> v(n);
        for (size_t i = 0; i < n; i++) {
            v[i] = ToFloat(p[i]);
        }
        return v;
    };

    int64_t totalDiff = 0;
    totalDiff += CompareBuffer("dw_input", toFloatVec(dwInput, I * threeH).data(), golden.dwInput, rtol, atol);
    totalDiff += CompareBuffer("dw_hidden", toFloatVec(dwHidden, H * threeH).data(), golden.dwHidden, rtol, atol);
    totalDiff += CompareBuffer("db_input", toFloatVec(dbInput, threeH).data(), golden.dbInput, rtol, atol);
    totalDiff += CompareBuffer("db_hidden", toFloatVec(dbHidden, threeH).data(), golden.dbHidden, rtol, atol);
    totalDiff += CompareBuffer("dx", toFloatVec(dx, tbi).data(), golden.dx, rtol, atol);
    totalDiff += CompareBuffer("dh_prev", toFloatVec(dhPrev, B * H).data(), golden.dhPrev, rtol, atol);
    totalDiff += CompareBuffer("dw_att", toFloatVec(dwAtt, tb).data(), golden.dwAtt, rtol, atol);
    EXPECT_EQ(totalDiff, 0);

    AscendC::GmFree(xBuf);
    AscendC::GmFree(wInputBuf);
    AscendC::GmFree(wHiddenBuf);
    AscendC::GmFree(attBuf);
    AscendC::GmFree(yBuf);
    AscendC::GmFree(initHBuf);
    AscendC::GmFree(hBuf);
    AscendC::GmFree(dyBuf);
    AscendC::GmFree(dhBuf);
    AscendC::GmFree(updateBuf);
    AscendC::GmFree(uAttBuf);
    AscendC::GmFree(resetBuf);
    AscendC::GmFree(newBuf);
    AscendC::GmFree(hnBuf);
    if (seqLenBuf != nullptr) {
        AscendC::GmFree(seqLenBuf);
    }
    AscendC::GmFree(dwInputBuf);
    AscendC::GmFree(dwHiddenBuf);
    AscendC::GmFree(dbInputBuf);
    AscendC::GmFree(dbHiddenBuf);
    AscendC::GmFree(dxBuf);
    AscendC::GmFree(dhPrevBuf);
    AscendC::GmFree(dwAttBuf);
    AscendC::GmFree(wsBuf);
    AscendC::GmFree(tilingBuf);
}

// 单步T=1（无matmul链回传）、带seq_length
TEST_F(DynamicAUGRUGradKernelTest, test_single_step_with_seq_len)
{
    RunKernelCase<float>(1, 4, 16, 16, true, 0, 1, 2e-3, 2e-3);
}

TEST_F(DynamicAUGRUGradKernelTest, compare_buffer_rejects_nonfinite_mismatch)
{
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    EXPECT_EQ(CompareBuffer("finite_vs_nan", &nan, {1.0f}, 1e-5, 1e-6), 1);
    EXPECT_EQ(CompareBuffer("nan_vs_nan", &nan, {nan}, 1e-5, 1e-6), 0);
    EXPECT_EQ(CompareBuffer("inf_sign", &inf, {-inf}, 1e-5, 1e-6), 1);
    EXPECT_EQ(CompareBuffer("inf_match", &inf, {inf}, 1e-5, 1e-6), 0);
}

// 多步BPTT + 融合掩码（seq_length含0/T/中间值）
TEST_F(DynamicAUGRUGradKernelTest, test_bptt_with_seq_len)
{
    RunKernelCase<float>(6, 4, 16, 32, true, 0, 1, 2e-2, 2e-2);
}

// 多步BPTT 无seq_length（全1掩码语义）
TEST_F(DynamicAUGRUGradKernelTest, test_bptt_without_seq_len)
{
    RunKernelCase<float>(5, 4, 16, 32, false, 0, 1, 2e-2, 2e-2);
}

// rzh门序
TEST_F(DynamicAUGRUGradKernelTest, test_bptt_gate_order_rzh)
{
    RunKernelCase<float>(4, 4, 16, 16, true, 1, 1, 2e-2, 2e-2);
}

// 较大H（H=48，16对齐）
TEST_F(DynamicAUGRUGradKernelTest, test_large_hidden) { RunKernelCase<float>(4, 4, 16, 48, true, 0, 1, 2e-2, 2e-2); }

// 多核blockDim（向量阶段按核切batch）
TEST_F(DynamicAUGRUGradKernelTest, test_multi_block) { RunKernelCase<float>(4, 8, 16, 16, true, 0, 4, 2e-2, 2e-2); }

// 复现NPU条件：blockDim=32全核（db疑似丢写）
TEST_F(DynamicAUGRUGradKernelTest, test_npu_repro_block32)
{
    RunKernelCase<float>(6, 4, 16, 32, true, 0, 32, 2e-2, 2e-2);
}

// fp16：staging队列转换路径（BPTT+seq）
TEST_F(DynamicAUGRUGradKernelTest, test_fp16_bptt_with_seq_len)
{
    RunKernelCase<half>(5, 4, 16, 32, true, 0, 1, 2e-2, 2e-2);
}

// fp16：无seq_length
TEST_F(DynamicAUGRUGradKernelTest, test_fp16_bptt_without_seq_len)
{
    RunKernelCase<half>(4, 4, 16, 32, false, 0, 1, 2e-2, 2e-2);
}

// fp16：较大H与多核blockDim
TEST_F(DynamicAUGRUGradKernelTest, test_fp16_large_hidden_multiblock)
{
    RunKernelCase<half>(4, 8, 16, 48, true, 0, 4, 2e-2, 2e-2);
}

// db内联路径：BPTT循环内累加进UB累加器，收尾仅对partial小表做列归约
TEST_F(DynamicAUGRUGradKernelTest, test_db_inline_bptt)
{
    RunKernelCase<float>(6, 4, 16, 32, true, 0, 1, 2e-2, 2e-2, true);
}

// db内联 + 多核（各核partial表跨核归约）
TEST_F(DynamicAUGRUGradKernelTest, test_db_inline_multiblock)
{
    RunKernelCase<float>(5, 8, 16, 32, true, 0, 4, 2e-2, 2e-2, true);
}

// db内联 + fp16（staging队列与db累加器共存）
TEST_F(DynamicAUGRUGradKernelTest, test_fp16_db_inline)
{
    RunKernelCase<half>(5, 4, 16, 32, true, 0, 1, 2e-2, 2e-2, true);
}

// 单tile流水路径：dgateMM异步IterateAll窗口内预取下一时间步操作数
TEST_F(DynamicAUGRUGradKernelTest, test_pipeline_bptt)
{
    RunKernelCase<float>(6, 4, 16, 32, true, 0, 1, 2e-2, 2e-2, false, true);
}

// fp16误开流水（手工tiling）：kernel侧dtype守卫兜底回退同步路径，数值仍正确
TEST_F(DynamicAUGRUGradKernelTest, test_fp16_pipeline_guard_fallback)
{
    RunKernelCase<half>(6, 4, 16, 32, true, 0, 1, 2e-2, 2e-2, false, true);
}

// H非16对齐：padded布局路径
// 说明：手工matmul tiling难以满足lib内部L0A/L0B布局约束（baseN/baseK需
// 与M/N/K精确匹配），非对齐组合的完整验证走NPU标杆（真实host tiling）：
// fp32_h20_unaligned / fp32_h52_seq / fp16_h20_unaligned 三个用例。
TEST_F(DynamicAUGRUGradKernelTest, DISABLED_test_h20_unaligned)
{
    RunKernelCase<float>(5, 4, 16, 20, true, 0, 4, 2e-2, 2e-2);
}

TEST_F(DynamicAUGRUGradKernelTest, DISABLED_test_h52_unaligned_rzh)
{
    RunKernelCase<float>(4, 4, 20, 52, true, 1, 4, 2e-2, 2e-2);
}
