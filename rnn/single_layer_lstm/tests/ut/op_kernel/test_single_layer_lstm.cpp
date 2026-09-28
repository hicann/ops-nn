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
 * \file test_single_layer_lstm.cpp
 * \brief SingleLayerLstm kernel unit test on the CPU simulator.
 *
 * The reference is computed IN THIS FILE in double precision rather than read from a generated
 * fixture, for two reasons: the gate order and the fused weight layout are exactly what a fixture
 * would have to encode a second time, and a wrong gate permutation stays smooth and plausible -- so
 * the comparison has to be against an independently written formula, not a recorded output.
 *
 * The tiling is filled by calling THE SAME host pickers the operator's TilingFunc calls, so the test
 * cannot pass with a tiling the real operator would never produce.
 *
 * The kernel entry is DECLARED here and defined by the separately compiled copy of
 * op_kernel/arch35/single_layer_lstm.cpp that the UT build stages; it is deliberately not #included, because the
 * entry is `extern "C"` and including it would define the symbol twice.
 *
 * WHAT THIS FILE CAN AND CANNOT ATTRIBUTE, AND WHY THE SPLIT IS WHERE IT IS
 * ------------------------------------------------------------------------
 * The registered case scores PHASE A -- the input projection -- against the reference, and the UT
 * CMake therefore builds the kernel with -DSINGLE_LAYER_LSTM_ONLY_PHASE_A=1. Phase A is where the hand-written
 * GEMM, the bias table, the NZ addressing, the K blocking, the L0C -> UB drain and the AIV scatter
 * all live, and comparing the intermediate rather than the final outputs is what separates "the
 * projection is wrong" from "the recurrence consumed it wrongly".
 *
 * The end-to-end case is DISABLED_, and the reason is a tool fault rather than a shortcut: with
 * phases B and C enabled, the CPU simulator segfaults inside its OWN predicate-register model
 * (AscendC::PregCompute -> PemRvecTop::SimdPvStep) while executing the Compares/Select pair that
 * implements the range-split tanh. Neither Compares nor Select reports a contract violation, and the
 * pair is load bearing -- the devkit's default tanh has ~1.67e7 ULP of cancellation near zero -- so
 * the kernel is not going to be reshaped to suit the simulator. ops-nn's own MIX cube kernel UT
 * (conv/conv2d_v2) likewise runs its kernel without asserting any output value.
 *
 * TO RUN THE END-TO-END CASE ON HARDWARE: drop -DSINGLE_LAYER_LSTM_ONLY_PHASE_A=1 from
 * tests/ut/op_kernel/CMakeLists.txt and the DISABLED_ prefix below.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <vector>
#include <gtest/gtest.h>
#include "tikicpulib.h"

#include "../../../op_host/arch35/single_layer_lstm_budget.h"
#include "../../../op_kernel/arch35/single_layer_lstm_tiling_data.h"

extern "C" __global__ __aicore__ void single_layer_lstm(GM_ADDR x, GM_ADDR w, GM_ADDR b, GM_ADDR init_h, GM_ADDR init_c,
                                                        GM_ADDR seq_length, GM_ADDR bias_hh, GM_ADDR y,
                                                        GM_ADDR output_h, GM_ADDR output_c, GM_ADDR out_i,
                                                        GM_ADDR out_j, GM_ADDR out_f, GM_ADDR out_o, GM_ADDR out_tanhc,
                                                        GM_ADDR workspace, GM_ADDR tiling);

namespace {

constexpr uint32_t kGates = 4;
constexpr float kTol = 1e-4F;

/* One shape under test. Two are run, and the pair is the point:
 *   H = 64 is a whole vector repeat, so NO padding path is exercised -- a mismatch there is
 *          arithmetic, not an allocation-rounding artefact.
 *   H = 8  is the smallest legal hidden size and drives every padding path at once: the bias slice
 *          shorter than the bias-table burst, the byte-wide mask plane, and the compare count that
 *          has to be rounded up to a repeat. */
struct Shape {
    uint32_t t, b, i, h;
};

struct GmDeleter {
    void operator()(uint8_t* ptr) const
    {
        if (ptr != nullptr) {
            AscendC::GmFree(ptr);
        }
    }
};
using GmBuffer = std::unique_ptr<uint8_t, GmDeleter>;

GmBuffer AllocGm(size_t bytes)
{
    auto buf = GmBuffer(static_cast<uint8_t*>(AscendC::GmAlloc(bytes)));
    if (buf != nullptr) {
        std::memset(buf.get(), 0, bytes);
    }
    return buf;
}

float* AsFloat(const GmBuffer& buf) { return reinterpret_cast<float*>(buf.get()); }

/* Deterministic, small, and spread across zero: the gate activations then all sit in their
 * well-conditioned range, so a mismatch means the arithmetic is wrong rather than that the inputs
 * sat on a saturation boundary. */
float FillValue(uint32_t seed) { return 0.05F * static_cast<float>((seed * 37U) % 41U) - 1.0F; }

double Sigmoid(double v) { return 1.0 / (1.0 + std::exp(-v)); }

/* Gate columns of `w` and `b` are ordered i, f, j, o -- the "ifjo" order the operator declares and
 * refuses to deviate from. Rows [0, I) of `w` pair with x; rows [I, I+H) are W_hh ALREADY TRANSPOSED
 * and pair with h_{t-1}. */
struct Reference {
    std::vector<double> y, h, c, gi, gj, gf, go, tanhc; // each [T][B][H]
};

Reference ComputeReference(const Shape& s, const float* x, const float* w, const float* bias, const float* h0,
                           const float* c0, bool withHgates = true)
{
    const uint32_t plane = s.t * s.b * s.h;
    Reference ref;
    for (auto* v : {&ref.y, &ref.h, &ref.c, &ref.gi, &ref.gj, &ref.gf, &ref.go, &ref.tanhc}) {
        v->assign(plane, 0.0);
    }

    std::vector<double> hPrev(s.b * s.h);
    std::vector<double> cPrev(s.b * s.h);
    for (uint32_t k = 0; k < s.b * s.h; ++k) {
        hPrev[k] = h0[k];
        cPrev[k] = c0[k];
    }

    const uint32_t wCols = kGates * s.h;
    for (uint32_t t = 0; t < s.t; ++t) {
        for (uint32_t b = 0; b < s.b; ++b) {
            for (uint32_t hh = 0; hh < s.h; ++hh) {
                double pre[kGates];
                for (uint32_t g = 0; g < kGates; ++g) {
                    double acc = bias[g * s.h + hh];
                    for (uint32_t k = 0; k < s.i; ++k) {
                        acc += static_cast<double>(x[(t * s.b + b) * s.i + k]) * w[k * wCols + g * s.h + hh];
                    }
                    if (withHgates) {
                        for (uint32_t k = 0; k < s.h; ++k) {
                            acc += hPrev[b * s.h + k] * w[(s.i + k) * wCols + g * s.h + hh];
                        }
                    }
                    pre[g] = acc;
                }
                const double gateI = Sigmoid(pre[0]);
                const double gateF = Sigmoid(pre[1]);
                const double gateJ = std::tanh(pre[2]);
                const double gateO = Sigmoid(pre[3]);
                const double cNew = gateF * cPrev[b * s.h + hh] + gateI * gateJ;
                const double tanhCNew = std::tanh(cNew);

                const uint32_t off = (t * s.b + b) * s.h + hh;
                ref.gi[off] = gateI;
                ref.gf[off] = gateF;
                ref.gj[off] = gateJ;
                ref.go[off] = gateO;
                ref.c[off] = cNew;
                ref.tanhc[off] = tanhCNew;
                ref.h[off] = gateO * tanhCNew;
                ref.y[off] = ref.h[off];
            }
        }
        /* The row is advanced only after the whole step: h_{t-1} feeds every gate of step t, so
         * updating it inside the hidden loop would leak step t into itself. */
        for (uint32_t b = 0; b < s.b; ++b) {
            for (uint32_t hh = 0; hh < s.h; ++hh) {
                const uint32_t off = (t * s.b + b) * s.h + hh;
                hPrev[b * s.h + hh] = ref.h[off];
                cPrev[b * s.h + hh] = ref.c[off];
            }
        }
    }
    return ref;
}

void ExpectPlaneNear(const char* name, const float* actual, const std::vector<double>& expect)
{
    for (size_t k = 0; k < expect.size(); ++k) {
        EXPECT_NEAR(actual[k], static_cast<float>(expect[k]), kTol) << name << " element " << k;
    }
}

void RunShape(const Shape& s, bool fullOutputs)
{
    const uint32_t mBlk = SingleLayerLstmFwd::PickRowsPerBlock(s.b, SingleLayerLstmBudget::MAX_CLUSTERS);
    ASSERT_GT(mBlk, 0U);
    ASSERT_EQ(s.b % mBlk, 0U);
    /* sizeof(DTYPE_W), which this test fixes at float through its CMakeLists. Written out rather
     * than hard-coded as 4 so that the budget this test asks for is the budget the translation unit
     * it links against was compiled for. */
    constexpr uint32_t kInBytes = static_cast<uint32_t>(sizeof(DTYPE_W));
    ASSERT_TRUE(SingleLayerLstmBudget::RecurrenceFits(s.b, s.h, s.t, mBlk, kInBytes));
    uint32_t tChunk = 0;
    uint32_t kChunk = 0;
    uint32_t nc = 0;
    ASSERT_TRUE(SingleLayerLstmBudget::PickProjTiling(mBlk, s.i, s.h, s.t, kInBytes, &tChunk, &kChunk, &nc));

    const size_t seqBytes = static_cast<size_t>(s.t) * s.b * s.h * sizeof(float);
    GmBuffer x = AllocGm(static_cast<size_t>(s.t) * s.b * s.i * sizeof(float));
    GmBuffer w = AllocGm(static_cast<size_t>(s.i + s.h) * kGates * s.h * sizeof(float));
    GmBuffer bias = AllocGm(static_cast<size_t>(kGates) * s.h * sizeof(float));
    GmBuffer initH = AllocGm(static_cast<size_t>(s.b) * s.h * sizeof(float));
    GmBuffer initC = AllocGm(static_cast<size_t>(s.b) * s.h * sizeof(float));
    GmBuffer y = AllocGm(seqBytes);
    GmBuffer outH = AllocGm(seqBytes);
    GmBuffer outC = AllocGm(seqBytes);
    GmBuffer outI = AllocGm(seqBytes);
    GmBuffer outJ = AllocGm(seqBytes);
    GmBuffer outF = AllocGm(seqBytes);
    GmBuffer outO = AllocGm(seqBytes);
    GmBuffer outTanhc = AllocGm(seqBytes);
    /* seq_length gets a real one-element buffer rather than nullptr: the CPU simulator stages every
     * GM argument into its virtual memory, and a null pointer faults inside that staging rather than
     * being treated as an absent optional input. The kernel never reads its value -- the host
     * resolved it into TilingData. */
    GmBuffer seqLength = AllocGm(sizeof(int64_t));
    GmBuffer tiling = AllocGm(sizeof(SingleLayerLstmTilingData));
    ASSERT_NE(x.get(), nullptr);
    ASSERT_NE(tiling.get(), nullptr);

    auto* td = reinterpret_cast<SingleLayerLstmTilingData*>(tiling.get());
    // AllocGm supplies raw storage, not a constructed TilingData. Initialize
    // every field explicitly before filling this FP32-only fixture.
    *td = {};
    td->batch = s.b;
    td->inputSize = s.i;
    td->hiddenSize = s.h;
    td->logicalInputSize = s.i;
    td->logicalHiddenSize = s.h;
    td->timeStep = s.t;
    td->seqLenMax = s.t;
    td->rowsPerBlock = mBlk;
    td->blockDim = s.b / mBlk;
    td->tChunk = tChunk;
    td->kChunk = kChunk;
    td->projNChunk = nc;
    uint32_t offset = 0;
    td->offIgates = offset;
    offset += s.t * s.b * kGates * s.h;
    td->offHAll = offset;
    offset += (s.t + 1) * s.b * s.h;
    td->offCAll = offset;
    offset += (s.t + 1) * s.b * s.h;
    td->offStore = offset;
    offset += s.t * s.b * kGates * s.h;
    td->offXF32 = offset;
    td->offWF32 = offset;
    GmBuffer workspace = AllocGm(static_cast<size_t>(offset) * sizeof(float));
    ASSERT_NE(workspace.get(), nullptr);

    for (uint32_t k = 0; k < s.t * s.b * s.i; ++k) {
        AsFloat(x)[k] = FillValue(k + 1U);
    }
    for (uint32_t k = 0; k < (s.i + s.h) * kGates * s.h; ++k) {
        AsFloat(w)[k] = 0.1F * FillValue(k + 7U);
    }
    for (uint32_t k = 0; k < kGates * s.h; ++k) {
        AsFloat(bias)[k] = 0.2F * FillValue(k + 3U);
    }
    for (uint32_t k = 0; k < s.b * s.h; ++k) {
        AsFloat(initH)[k] = 0.3F * FillValue(k + 11U);
        AsFloat(initC)[k] = 0.4F * FillValue(k + 13U);
    }

    const Reference ref = ComputeReference(s, AsFloat(x), AsFloat(w), AsFloat(bias), AsFloat(initH), AsFloat(initC));

    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    ICPU_RUN_KF(single_layer_lstm, td->blockDim, x.get(), w.get(), bias.get(), initH.get(), initC.get(),
                seqLength.get(), static_cast<GM_ADDR>(nullptr), y.get(), outH.get(), outC.get(), outI.get(), outJ.get(),
                outF.get(), outO.get(), outTanhc.get(), workspace.get(), tiling.get());

    /* PHASE A. `igates` in workspace is x @ w[0:I] + b, laid out [T, B, 4H] gate-minor. Comparing it
     * element by element against the double-precision reference scores the projection on its own. */
    const float* igates = AsFloat(workspace) + td->offIgates;
    const uint32_t wCols = kGates * s.h;
    for (uint32_t t = 0; t < s.t; ++t) {
        for (uint32_t b = 0; b < s.b; ++b) {
            for (uint32_t g = 0; g < kGates; ++g) {
                for (uint32_t hh = 0; hh < s.h; ++hh) {
                    double acc = AsFloat(bias)[g * s.h + hh];
                    for (uint32_t k = 0; k < s.i; ++k) {
                        acc += static_cast<double>(AsFloat(x)[(t * s.b + b) * s.i + k]) *
                               AsFloat(w)[k * wCols + g * s.h + hh];
                    }
                    const size_t idx = ((t * s.b + b) * kGates + g) * s.h + hh;
                    EXPECT_NEAR(igates[idx], static_cast<float>(acc), kTol)
                        << "igates t=" << t << " b=" << b << " gate=" << g << " h=" << hh;
                }
            }
        }
    }

    if (fullOutputs) {
        ExpectPlaneNear("y", AsFloat(y), ref.y);
        ExpectPlaneNear("output_h", AsFloat(outH), ref.h);
        ExpectPlaneNear("output_c", AsFloat(outC), ref.c);
        ExpectPlaneNear("i", AsFloat(outI), ref.gi);
        ExpectPlaneNear("j", AsFloat(outJ), ref.gj);
        ExpectPlaneNear("f", AsFloat(outF), ref.gf);
        ExpectPlaneNear("o", AsFloat(outO), ref.go);
        ExpectPlaneNear("tanhc", AsFloat(outTanhc), ref.tanhc);
    }
}

} // namespace

class SingleLayerLstmKernel : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "SingleLayerLstmKernel SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "SingleLayerLstmKernel TearDown" << std::endl; }
};

TEST_F(SingleLayerLstmKernel, input_projection_matches_reference_at_a_whole_vector_repeat)
{
    RunShape({2U, 2U, 16U, 64U}, false);
}

/* The smallest legal hidden size, which drives every padding path at once: the bias slice shorter
 * than the bias-table burst, the byte-wide mask plane, and the compare count rounded to a repeat.
 * Each of those was a defect found here. */
TEST_F(SingleLayerLstmKernel, input_projection_matches_reference_at_the_smallest_hidden_size)
{
    RunShape({2U, 2U, 8U, 8U}, false);
}

/* HARDWARE ONLY -- see the note at the top of this file for why the CPU simulator cannot run it.
 * Drop -DSINGLE_LAYER_LSTM_ONLY_PHASE_A=1 from the UT CMake and the DISABLED_ prefix to enable. */
TEST_F(SingleLayerLstmKernel, DISABLED_all_outputs_match_a_double_precision_reference)
{
    RunShape({2U, 2U, 16U, 64U}, true);
    RunShape({2U, 2U, 8U, 8U}, true);
}
