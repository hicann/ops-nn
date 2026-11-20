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
 * \file test_ascend_dequant.cpp
 * \brief CPU-side numeric UT of the AscendDequant regbase kernel.
 *
 *   The expected value is computed by an independent scalar reference in this
 *   file:
 *     s = bit-cast_fp32(low 32 bits of deq_scale)   (uint64 deq_scale)
 *     p = fp32(x) * s
 *     if relu_flag: p = max(p, 0)
 *     if sqrt_mode: p = p * s          (same scale again, i.e. s^2)
 *     y = fp16(p)                      (attr dtype=1, bit-exact)
 *     y = p                            (attr dtype=0, fp32, small rel tolerance)
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "tikicpulib.h"

#include "ascend_dequant_tiling_def.h"

#include "../../../op_kernel/arch35/ascend_dequant.cpp"

using namespace std;

namespace {

// Flat element index of x -> element index of deq_scale after numpy
// broadcasting (right-aligned; scale axes of size 1 add nothing).
int64_t ScaleIndexOfFlat(const std::vector<int64_t>& xShape, const std::vector<int64_t>& scaleShape, int64_t flat)
{
    const int64_t xRank = static_cast<int64_t>(xShape.size());
    const int64_t sRank = static_cast<int64_t>(scaleShape.size());
    int64_t idx = 0;
    int64_t xStride = 1;
    int64_t sStride = 1;
    for (int64_t d = xRank - 1; d >= 0; --d) {
        const int64_t coord = (flat / xStride) % xShape[static_cast<size_t>(d)];
        const int64_t sd = d - (xRank - sRank);
        if (sd >= 0) {
            if (scaleShape[static_cast<size_t>(sd)] != 1) {
                idx += coord * sStride;
            }
            sStride *= scaleShape[static_cast<size_t>(sd)];
        }
        xStride *= xShape[static_cast<size_t>(d)];
    }
    return idx;
}

std::vector<int64_t> ProductStrides(const std::vector<int64_t>& shape, const std::vector<int64_t>& maxBro)
{
    const int64_t rank = static_cast<int64_t>(shape.size());
    std::vector<int64_t> strides(static_cast<size_t>(rank), 0);
    int64_t stride = 1;
    for (int64_t d = rank - 1; d >= 0; --d) {
        const bool brc = (shape[static_cast<size_t>(d)] == 1 && maxBro[static_cast<size_t>(d)] != 1);
        strides[static_cast<size_t>(d)] = brc ? 0 : stride;
        if (!brc) {
            stride *= shape[static_cast<size_t>(d)];
        }
    }
    return strides;
}

template <int64_t R>
void FillTiling(AscendDequantTilingData<R>& td, const std::vector<int64_t>& maxBro,
                const std::vector<std::vector<int64_t>>& inShapes, const std::vector<std::vector<int64_t>>& outShapes,
                const SplitResult& split, const MultiCoreResult& multicore, int64_t sqrtMode, int64_t reluFlag,
                int64_t outDtype)
{
    td.split = split;
    td.multicore = multicore;
    td.rank = static_cast<int64_t>(maxBro.size());
    td.perBufBytes = 87360; // (262144 / 3) & ~31
    td.numInputs = kMaxInputSlots;
    td.numOutputs = kMaxOutputSlots;
    td.sqrtMode = sqrtMode;
    td.reluFlag = reluFlag;
    td.outDtype = outDtype; // 0 -> float32, 1 -> float16

    for (int64_t d = 0; d < R; ++d) {
        td.maxBroShape[d] = (d < td.rank) ? maxBro[static_cast<size_t>(d)] : 1;
    }
    for (int64_t i = 0; i < kMaxInputSlots; ++i) {
        for (int64_t d = 0; d < R; ++d) {
            td.inputShapes[i][d] = 1;
            td.inputStrides[i][d] = 0;
        }
    }
    for (int64_t i = 0; i < kMaxOutputSlots; ++i) {
        for (int64_t d = 0; d < R; ++d) {
            td.outputShapes[i][d] = 1;
            td.outputStrides[i][d] = 0;
        }
    }
    for (size_t i = 0; i < inShapes.size() && i < static_cast<size_t>(kMaxInputSlots); ++i) {
        const auto strides = ProductStrides(inShapes[i], maxBro);
        for (int64_t d = 0; d < td.rank; ++d) {
            td.inputShapes[i][static_cast<size_t>(d)] = inShapes[i][static_cast<size_t>(d)];
            td.inputStrides[i][static_cast<size_t>(d)] = strides[static_cast<size_t>(d)];
        }
        if (i == 1) { // uint64 deq_scale is read as a uint32 low-word view
            for (int64_t d = 0; d < R; ++d) {
                td.inputStrides[i][d] *= 2;
            }
        }
    }
    for (size_t i = 0; i < outShapes.size() && i < static_cast<size_t>(kMaxOutputSlots); ++i) {
        const auto strides = ProductStrides(outShapes[i], maxBro);
        for (int64_t d = 0; d < td.rank; ++d) {
            td.outputShapes[i][static_cast<size_t>(d)] = outShapes[i][static_cast<size_t>(d)];
            td.outputStrides[i][static_cast<size_t>(d)] = strides[static_cast<size_t>(d)];
        }
    }
}

uint64_t MakeScaleBits(float s)
{
    uint32_t bits = 0;
    memcpy(&bits, &s, sizeof(bits));
    return static_cast<uint64_t>(bits);
}

float DecodeScale(uint64_t bits)
{
    const uint32_t low = static_cast<uint32_t>(bits & 0xFFFFFFFFULL);
    float s = 0.0f;
    memcpy(&s, &low, sizeof(s));
    return s;
}

template <int64_t R>
void RunKernelCase(const std::vector<int64_t>& xShape, const std::vector<int64_t>& scaleShape,
                   const std::vector<int64_t>& maxBro, const std::vector<std::vector<int64_t>>& inShapes,
                   const std::vector<std::vector<int64_t>>& outShapes, const SplitResult& split,
                   const MultiCoreResult& multicore, int64_t sqrtMode, int64_t reluFlag,
                   const std::vector<int32_t>& xVals, const std::vector<float>& scaleVals, bool empty,
                   bool fp32Out = false, const std::vector<float>* expectPattern = nullptr)
{
    const int64_t elemCount = static_cast<int64_t>(xVals.size());
    const int64_t scaleCount = static_cast<int64_t>(scaleVals.size());
    const int64_t outDtype = fp32Out ? 0 : 1;
    const size_t xBytes = static_cast<size_t>(std::max<int64_t>(elemCount, 1)) * sizeof(int32_t);
    const size_t scaleBytes = static_cast<size_t>(std::max<int64_t>(scaleCount, 1)) * sizeof(uint64_t);
    const size_t yBytes = static_cast<size_t>(std::max<int64_t>(elemCount, 1)) *
                          (fp32Out ? sizeof(float) : sizeof(half));
    const size_t tilingBytes = sizeof(AscendDequantTilingData<R>);

    uint8_t* x = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(xBytes));
    uint8_t* scale = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(scaleBytes));
    uint8_t* y = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(yBytes));
    uint8_t* ws = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(16 * 1024 * 1024));
    uint8_t* tiling = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(tilingBytes));

    auto* td = reinterpret_cast<AscendDequantTilingData<R>*>(tiling);
    FillTiling<R>(*td, maxBro, inShapes, outShapes, split, multicore, sqrtMode, reluFlag, outDtype);

    for (int64_t i = 0; i < elemCount; ++i) {
        reinterpret_cast<int32_t*>(x)[i] = xVals[static_cast<size_t>(i)];
    }
    for (int64_t i = 0; i < scaleCount; ++i) {
        reinterpret_cast<uint64_t*>(scale)[i] = MakeScaleBits(scaleVals[static_cast<size_t>(i)]);
    }
    const half sentinel(1.0f);
    const float sentinelF = 1.0f;
    for (int64_t i = 0; i < std::max<int64_t>(elemCount, 1); ++i) {
        if (fp32Out) {
            reinterpret_cast<float*>(y)[i] = sentinelF;
        } else {
            reinterpret_cast<half*>(y)[i] = sentinel;
        }
    }

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(R == 4 ? 0U : 1U);
    auto kernelLambda = [](GM_ADDR xAddr, GM_ADDR scaleAddr, GM_ADDR yAddr, GM_ADDR wsAddr, GM_ADDR tilingAddr) {
        ::ascend_dequant<R>(xAddr, scaleAddr, yAddr, wsAddr, tilingAddr);
    };
    const uint32_t blockDim = static_cast<uint32_t>(multicore.usedCoreNum == 0 ? 1 : multicore.usedCoreNum);
    ICPU_RUN_KF(kernelLambda, blockDim, x, scale, y, ws, tiling);

    if (empty) {
        if (fp32Out) {
            EXPECT_EQ(reinterpret_cast<float*>(y)[0], sentinelF) << "empty tensor must early-return without writing y";
        } else {
            EXPECT_EQ(reinterpret_cast<half*>(y)[0].val, sentinel.val)
                << "empty tensor must early-return without writing y";
        }
    } else {
        for (int64_t i = 0; i < elemCount; ++i) {
            const int64_t ci = ScaleIndexOfFlat(xShape, scaleShape, i);
            const float s = scaleVals[static_cast<size_t>(ci)];
            float v = static_cast<float>(xVals[static_cast<size_t>(i)]) * s;
            if (reluFlag != 0) {
                v = std::max(v, 0.0f);
            }
            if (sqrtMode != 0) {
                v = v * s;
            }
            if (fp32Out) {
                const float got = reinterpret_cast<float*>(y)[i];
                const float tol = 1e-6f * std::max(1.0f, std::fabs(v));
                EXPECT_LE(std::fabs(got - v), tol) << "mismatch at " << i << " (x=" << xVals[static_cast<size_t>(i)]
                                                   << ", s=" << s << ", got=" << got << ", expect=" << v << ")";
            } else {
                const half expect(v);
                EXPECT_EQ(reinterpret_cast<half*>(y)[i].val, expect.val)
                    << "mismatch at " << i << " (x=" << xVals[static_cast<size_t>(i)] << ", s=" << s << ")";
            }
        }
        if (expectPattern != nullptr) { // hand-derived per-element pattern
            ASSERT_EQ(expectPattern->size(), static_cast<size_t>(elemCount));
            for (int64_t i = 0; i < elemCount; ++i) {
                const float pattern = (*expectPattern)[static_cast<size_t>(i)];
                if (fp32Out) {
                    EXPECT_LE(std::fabs(reinterpret_cast<float*>(y)[i] - pattern), 1e-6f)
                        << "pattern mismatch at " << i;
                } else {
                    EXPECT_EQ(reinterpret_cast<half*>(y)[i].val, half(pattern).val) << "pattern mismatch at " << i;
                }
            }
        }
    }

    AscendC::GmFree(x);
    AscendC::GmFree(scale);
    AscendC::GmFree(y);
    AscendC::GmFree(ws);
    AscendC::GmFree(tiling);
}

} // namespace

class ascend_dequant_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "ascend_dequant_test SetUp" << endl; }
    static void TearDownTestCase() { cout << "ascend_dequant_test TearDown" << endl; }
};

TEST_F(ascend_dequant_test, Rank1PerChannel)
{
    const std::vector<int64_t> xShape{16};
    const std::vector<int64_t> scaleShape{16};
    std::vector<int32_t> x(16);
    std::vector<float> s(16);
    const float patterns[4] = {0.5f, 1.0f, 2.0f, 4.0f};
    for (int64_t i = 0; i < 16; ++i) {
        x[static_cast<size_t>(i)] = static_cast<int32_t>(i) - 8;
        s[static_cast<size_t>(i)] = patterns[i % 4];
    }
    RunKernelCase<4>(xShape, scaleShape, {16}, {{16}, {16}}, {{16}}, {0, 16, 1, 16}, {1, 1, 1, 1}, 0, 0, x, s, false);
}

TEST_F(ascend_dequant_test, Rank2PerChannelRelu)
{
    const std::vector<int64_t> xShape{4, 8};
    const std::vector<int64_t> scaleShape{8};
    std::vector<int32_t> x(32);
    std::vector<float> s(8);
    for (int64_t i = 0; i < 32; ++i) {
        x[static_cast<size_t>(i)] = static_cast<int32_t>(i % 9) - 4;
    }
    for (int64_t i = 0; i < 8; ++i) {
        s[static_cast<size_t>(i)] = (i % 2 == 0) ? 1.0f : 0.5f;
    }
    RunKernelCase<4>(xShape, scaleShape, {4, 8}, {{4, 8}, {1, 8}}, {{4, 8}}, {0, 4, 1, 4}, {1, 1, 1, 1}, 0, 1, x, s,
                     false);
}

TEST_F(ascend_dequant_test, Rank4PerChannelSqrtRelu)
{
    const std::vector<int64_t> xShape{2, 3, 2, 2};
    const std::vector<int64_t> scaleShape{1, 3, 1, 1};
    std::vector<int32_t> x(24);
    std::vector<float> s(3);
    for (int64_t i = 0; i < 24; ++i) {
        x[static_cast<size_t>(i)] = static_cast<int32_t>(i % 7) - 3;
    }
    s = {0.5f, 1.0f, 2.0f};
    RunKernelCase<4>(xShape, scaleShape, {2, 3, 4}, {{2, 3, 4}, {1, 3, 1}}, {{2, 3, 4}}, {0, 2, 1, 2}, {1, 1, 1, 1}, 1,
                     1, x, s, false);
}

TEST_F(ascend_dequant_test, Rank5PerTensor)
{
    const std::vector<int64_t> xShape{2, 3, 2, 2, 4};
    const std::vector<int64_t> scaleShape{1};
    std::vector<int32_t> x(96);
    std::vector<float> s{2.0f};
    for (int64_t i = 0; i < 96; ++i) {
        x[static_cast<size_t>(i)] = static_cast<int32_t>(i % 11) - 5;
    }
    RunKernelCase<8>(xShape, scaleShape, {96}, {{96}, {1}}, {{96}}, {0, 96, 1, 96}, {1, 1, 1, 1}, 0, 0, x, s, false);
}

TEST_F(ascend_dequant_test, Rank5PerChannelKey1)
{
    const std::vector<int64_t> xShape{2, 3, 4, 2, 4}; // broadcast on dim1 (C = 3)
    const std::vector<int64_t> scaleShape{1, 3, 1, 1, 1};
    std::vector<int32_t> x(192);
    std::vector<float> s{0.5f, 1.0f, 2.0f};
    for (int64_t i = 0; i < 192; ++i) {
        x[static_cast<size_t>(i)] = static_cast<int32_t>(i % 13) - 6;
    }
    RunKernelCase<8>(xShape, scaleShape, {2, 3, 32}, {{2, 3, 32}, {1, 3, 1}}, {{2, 3, 32}}, {0, 2, 1, 2}, {1, 1, 1, 1},
                     0, 0, x, s, false);
}

// A same-rank deq_scale broadcasts on dim1: with x all ones, C = 2 and D = 3
// the (C, D) plane must read [2,2,2,3,3,3].
TEST_F(ascend_dequant_test, Rank5BroadcastOnDim1)
{
    const std::vector<int64_t> xShape{1, 2, 3, 1, 1};
    const std::vector<int64_t> scaleShape{1, 2, 1, 1, 1};
    const std::vector<int32_t> x(6, 1);
    const std::vector<float> s{2.0f, 3.0f};
    const std::vector<float> pattern{2.0f, 2.0f, 2.0f, 3.0f, 3.0f, 3.0f}; // one scale step per C, D within it
    RunKernelCase<4>(xShape, scaleShape, {2, 3}, {{2, 3}, {2, 1}}, {{2, 3}}, {0, 2, 1, 2}, {1, 1, 1, 1}, 0, 0, x, s,
                     false, false, &pattern);
}

// Arbitrary broadcast axes: x={2,3,4} with scale={2,1,4}. The middle axis is
// broadcast in the NDDMA descriptor (source stride 0), so every (i,j,k) element
// must read scale[i*4 + k].
TEST_F(ascend_dequant_test, Rank3BroadcastMiddleAxis)
{
    const std::vector<int64_t> xShape{2, 3, 4};
    const std::vector<int64_t> scaleShape{2, 1, 4};
    const std::vector<int32_t> x(24, 1);
    const std::vector<float> s{2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
    std::vector<float> pattern(24);
    for (int64_t i = 0; i < 2; ++i) {
        for (int64_t j = 0; j < 3; ++j) {
            for (int64_t k = 0; k < 4; ++k) {
                pattern[static_cast<size_t>((i * 3 + j) * 4 + k)] = s[static_cast<size_t>(i * 4 + k)];
            }
        }
    }
    RunKernelCase<4>(xShape, scaleShape, {2, 3, 4}, {{2, 3, 4}, {2, 1, 4}}, {{2, 3, 4}}, {0, 2, 1, 2}, {1, 1, 1, 1}, 0,
                     0, x, s, false, false, &pattern);
}

// uint64 deq_scale (low 32 bits = fp32 bit pattern) -> float32 output: no
// narrowing cast, the fp32 product is stored directly.
TEST_F(ascend_dequant_test, Fp32OutUint64ScalePerChannel)
{
    const std::vector<int64_t> xShape{4, 8};
    const std::vector<int64_t> scaleShape{8};
    std::vector<int32_t> x(32);
    std::vector<float> s(8);
    for (int64_t i = 0; i < 32; ++i) {
        x[static_cast<size_t>(i)] = static_cast<int32_t>(i) - 16;
    }
    for (int64_t i = 0; i < 8; ++i) {
        s[static_cast<size_t>(i)] = 0.125f * static_cast<float>(i + 1);
    }
    RunKernelCase<4>(xShape, scaleShape, {4, 8}, {{4, 8}, {1, 8}}, {{4, 8}}, {0, 4, 1, 4}, {1, 1, 1, 1}, 0, 0, x, s,
                     false, true);
}

// uint64 deq_scale -> float32 output with per-tensor broadcast and sqrt/relu.
// A per-tensor scale never blocks axis merging, so the host hands the kernel a
// one-dimensional coordinate system.
TEST_F(ascend_dequant_test, Fp32OutUint64ScalePerTensorSqrtRelu)
{
    const std::vector<int64_t> xShape{2, 3, 2, 2};
    const std::vector<int64_t> scaleShape{1};
    std::vector<int32_t> x(24);
    std::vector<float> s{0.5f};
    for (int64_t i = 0; i < 24; ++i) {
        x[static_cast<size_t>(i)] = static_cast<int32_t>(i % 9) - 4;
    }
    RunKernelCase<4>(xShape, scaleShape, {24}, {{24}, {1}}, {{24}}, {0, 24, 1, 24}, {1, 1, 1, 1}, 1, 1, x, s, false,
                     true);
}

// uint64 deq_scale with a rank-5 tile routed to the RANK_8 instantiation.
TEST_F(ascend_dequant_test, Fp32OutUint64ScaleRank5)
{
    const std::vector<int64_t> xShape{2, 3, 4, 2, 4}; // broadcast on dim1 (C = 3)
    const std::vector<int64_t> scaleShape{1, 3, 1, 1, 1};
    std::vector<int32_t> x(192);
    std::vector<float> s{0.5f, 1.0f, 2.0f};
    for (int64_t i = 0; i < 192; ++i) {
        x[static_cast<size_t>(i)] = static_cast<int32_t>(i % 13) - 6;
    }
    RunKernelCase<8>(xShape, scaleShape, {2, 3, 32}, {{2, 3, 32}, {1, 3, 1}}, {{2, 3, 32}}, {0, 2, 1, 2}, {1, 1, 1, 1},
                     0, 0, x, s, false, true);
}

TEST_F(ascend_dequant_test, EmptyTensorEarlyReturn)
{
    const std::vector<int64_t> xShape{0, 3};
    const std::vector<int64_t> scaleShape{3};
    std::vector<int32_t> x(0);
    std::vector<float> s{1.0f, 2.0f, 0.5f};
    // An empty tensor has no elements to compare; pass 0 x values and let the
    // helper allocate a one-element sentinel buffer that must stay untouched.
    RunKernelCase<4>(xShape, scaleShape, {0, 3}, {{0, 3}, {1, 3}}, {{0, 3}}, {}, {0, 1, 1, 1}, 0, 0, x, s, true);
}
