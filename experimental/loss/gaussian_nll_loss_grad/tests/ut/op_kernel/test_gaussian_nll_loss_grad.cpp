/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "../../../op_kernel/gaussian_nll_loss_grad.h"

using std::vector;

constexpr size_t GM_GUARD_BYTES = 64;
constexpr uint8_t GM_GUARD_VALUE = 0xa5;

extern "C" __global__ __aicore__ void gaussian_nll_loss_grad(GM_ADDR gradOutput, GM_ADDR input, GM_ADDR target,
                                                             GM_ADDR var, GM_ADDR gradInput, GM_ADDR gradVar,
                                                             GM_ADDR workspace, GM_ADDR tiling);

extern "C" __global__ __aicore__ void gaussian_nll_loss_grad_half_test(GM_ADDR gradOutput, GM_ADDR input,
                                                                       GM_ADDR target, GM_ADDR var, GM_ADDR gradInput,
                                                                       GM_ADDR gradVar, GM_ADDR workspace,
                                                                       GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(GaussianNllLossGradTilingData);
    GET_TILING_DATA_WITH_STRUCT(GaussianNllLossGradTilingData, data, tiling);
    (void)workspace;
    AscendC::TPipe pipe;
    NsGaussianNllLossGrad::KernelGaussianNllLossGrad<half> op;
    op.Init(gradOutput, input, target, var, gradInput, gradVar, &data, pipe);
    op.Process();
}

extern "C" __global__ __aicore__ void gaussian_nll_loss_grad_bf16_test(GM_ADDR gradOutput, GM_ADDR input,
                                                                       GM_ADDR target, GM_ADDR var, GM_ADDR gradInput,
                                                                       GM_ADDR gradVar, GM_ADDR workspace,
                                                                       GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(GaussianNllLossGradTilingData);
    GET_TILING_DATA_WITH_STRUCT(GaussianNllLossGradTilingData, data, tiling);
    (void)workspace;
    AscendC::TPipe pipe;
    NsGaussianNllLossGrad::KernelGaussianNllLossGrad<bfloat16_t> op;
    op.Init(gradOutput, input, target, var, gradInput, gradVar, &data, pipe);
    op.Process();
}

static uint16_t FloatToHalf(float value)
{
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000;
    int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xff) - 127 + 15;
    uint32_t mantissa = (bits >> 13) & 0x3ff;
    if (exponent <= 0) {
        return static_cast<uint16_t>(sign);
    }
    if (exponent >= 31) {
        return static_cast<uint16_t>(sign | 0x7c00);
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) | mantissa);
}

static float HalfToFloat(uint16_t value)
{
    uint32_t sign = static_cast<uint32_t>(value & 0x8000) << 16;
    uint32_t exponent = (value >> 10) & 0x1f;
    uint32_t mantissa = value & 0x3ff;
    uint32_t bits;
    if (exponent == 0) {
        bits = sign;
    } else if (exponent == 31) {
        bits = sign | 0x7f800000 | (mantissa << 13);
    } else {
        bits = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
    }
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

static uint16_t FloatToBFloat16(float value)
{
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t bias = 0x7fff + ((bits >> 16) & 1);
    return static_cast<uint16_t>((bits + bias) >> 16);
}

static float BFloat16ToFloat(uint16_t value)
{
    uint32_t bits = static_cast<uint32_t>(value) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

struct HalfTraits {
    static constexpr size_t SIZE = 2;
    static constexpr float TOLERANCE = 2e-2f;
    static uint16_t Encode(float value) { return FloatToHalf(value); }
    static float Decode(uint16_t value) { return HalfToFloat(value); }
    static void Run(uint32_t blocks, uint8_t* gradOutput, uint8_t* input, uint8_t* target, uint8_t* var,
                    uint8_t* gradInput, uint8_t* gradVar, uint8_t* workspace, uint8_t* tiling)
    {
        ICPU_RUN_KF(gaussian_nll_loss_grad_half_test, blocks, gradOutput, input, target, var, gradInput, gradVar,
                    workspace, tiling);
    }
};

struct BFloat16Traits {
    static constexpr size_t SIZE = 2;
    static constexpr float TOLERANCE = 8e-2f;
    static uint16_t Encode(float value) { return FloatToBFloat16(value); }
    static float Decode(uint16_t value) { return BFloat16ToFloat(value); }
    static void Run(uint32_t blocks, uint8_t* gradOutput, uint8_t* input, uint8_t* target, uint8_t* var,
                    uint8_t* gradInput, uint8_t* gradVar, uint8_t* workspace, uint8_t* tiling)
    {
        ICPU_RUN_KF(gaussian_nll_loss_grad_bf16_test, blocks, gradOutput, input, target, var, gradInput, gradVar,
                    workspace, tiling);
    }
};

struct FloatTraits {
    static constexpr size_t SIZE = 4;
    static constexpr float TOLERANCE = 1e-5f;
    static void Run(uint32_t blocks, uint8_t* gradOutput, uint8_t* input, uint8_t* target, uint8_t* var,
                    uint8_t* gradInput, uint8_t* gradVar, uint8_t* workspace, uint8_t* tiling)
    {
        ICPU_RUN_KF(gaussian_nll_loss_grad, blocks, gradOutput, input, target, var, gradInput, gradVar, workspace,
                    tiling);
    }
};

template <typename Traits>
static void Write(uint8_t* gm, const vector<float>& values)
{
    if constexpr (Traits::SIZE == sizeof(float)) {
        std::memcpy(gm, values.data(), values.size() * sizeof(float));
    } else {
        for (size_t i = 0; i < values.size(); ++i) {
            const uint16_t value = Traits::Encode(values[i]);
            std::memcpy(gm + i * sizeof(value), &value, sizeof(value));
        }
    }
}

template <typename Traits>
static vector<float> Read(const uint8_t* gm, size_t count)
{
    vector<float> values(count);
    if constexpr (Traits::SIZE == sizeof(float)) {
        std::memcpy(values.data(), gm, count * sizeof(float));
    } else {
        for (size_t i = 0; i < count; ++i) {
            uint16_t value;
            std::memcpy(&value, gm + i * sizeof(value), sizeof(value));
            values[i] = Traits::Decode(value);
        }
    }
    return values;
}

static void ExpectGuardBytes(const uint8_t* allocation, size_t payloadBytes)
{
    for (size_t i = 0; i < GM_GUARD_BYTES; ++i) {
        EXPECT_EQ(allocation[i], GM_GUARD_VALUE) << "prefix guard byte " << i;
        EXPECT_EQ(allocation[GM_GUARD_BYTES + payloadBytes + i], GM_GUARD_VALUE) << "suffix guard byte " << i;
    }
}

struct KernelCase {
    vector<float> gradOutput;
    vector<float> input;
    vector<float> target;
    vector<float> var;
    uint32_t targetMode = 0;
    uint32_t targetAxisSize = 1;
    uint32_t targetInnerStride = 1;
    uint32_t varMode = 0;
    uint32_t varReduceSize = 1;
    uint32_t reduction = 0;
    float eps = 1e-6f;
    uint32_t blocks = 1;
    uint32_t tile = 64;
};

static uint32_t TargetIndex(const KernelCase& c, uint32_t logical)
{
    if (c.targetMode == 0) {
        return logical;
    }
    return logical / (c.targetInnerStride * c.targetAxisSize) * c.targetInnerStride + logical % c.targetInnerStride;
}

static uint32_t VarIndex(const KernelCase& c, uint32_t logical)
{
    return c.varMode == 0 ? logical : (c.varReduceSize == 0 ? 0 : logical / c.varReduceSize);
}

static void Golden(const KernelCase& c, vector<float>& gradInput, vector<float>& gradVar)
{
    gradInput.resize(c.input.size());
    gradVar.assign(c.var.size(), 0.0f);
    const float meanScale = c.input.empty() ? 0.0f : 1.0f / static_cast<float>(c.input.size());
    for (uint32_t i = 0; i < c.input.size(); ++i) {
        const uint32_t varIndex = VarIndex(c, i);
        const float d = c.input[i] - c.target[TargetIndex(c, i)];
        const float v = std::max(c.var[varIndex], c.eps);
        float go = c.gradOutput[c.reduction == 0 ? i : 0];
        if (c.reduction == 2) {
            go *= meanScale;
        }
        gradInput[i] = go * d / v;
        gradVar[varIndex] += go * 0.5f * (1.0f / v - d * d / (v * v));
    }
}

template <typename Traits>
static void RunCase(const KernelCase& c)
{
    const size_t inputBytes = std::max<size_t>(1, c.input.size()) * Traits::SIZE;
    const size_t targetBytes = std::max<size_t>(1, c.target.size()) * Traits::SIZE;
    const size_t varBytes = std::max<size_t>(1, c.var.size()) * Traits::SIZE;
    const size_t gradOutputBytes = std::max<size_t>(1, c.gradOutput.size()) * Traits::SIZE;
    uint8_t* gradOutput = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(gradOutputBytes));
    uint8_t* input = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(inputBytes));
    uint8_t* target = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(targetBytes));
    uint8_t* var = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(varBytes));
    const size_t gradInputAllocationBytes = inputBytes + 2 * GM_GUARD_BYTES;
    const size_t gradVarAllocationBytes = varBytes + 2 * GM_GUARD_BYTES;
    uint8_t* gradInputAllocation = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(gradInputAllocationBytes));
    uint8_t* gradVarAllocation = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(gradVarAllocationBytes));
    uint8_t* gradInput = nullptr;
    uint8_t* gradVar = nullptr;
    uint8_t* workspace = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(32));
    uint8_t* tiling = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(sizeof(GaussianNllLossGradTilingData)));
    ASSERT_NE(gradOutput, nullptr);
    ASSERT_NE(input, nullptr);
    ASSERT_NE(target, nullptr);
    ASSERT_NE(var, nullptr);
    ASSERT_NE(gradInputAllocation, nullptr);
    ASSERT_NE(gradVarAllocation, nullptr);
    gradInput = gradInputAllocation + GM_GUARD_BYTES;
    gradVar = gradVarAllocation + GM_GUARD_BYTES;
    ASSERT_NE(workspace, nullptr);
    ASSERT_NE(tiling, nullptr);
    std::memset(gradInputAllocation, GM_GUARD_VALUE, gradInputAllocationBytes);
    std::memset(gradVarAllocation, GM_GUARD_VALUE, gradVarAllocationBytes);
    std::memset(gradInput, 0, inputBytes);
    std::memset(gradVar, 0, varBytes);
    Write<Traits>(gradOutput, c.gradOutput);
    Write<Traits>(input, c.input);
    Write<Traits>(target, c.target);
    Write<Traits>(var, c.var);

    auto* data = reinterpret_cast<GaussianNllLossGradTilingData*>(tiling);
    const uint32_t small = static_cast<uint32_t>(c.input.size()) / c.blocks;
    const uint32_t tailBlocks = static_cast<uint32_t>(c.input.size()) % c.blocks;
    const uint32_t big = small + (tailBlocks > 0 ? 1 : 0);
    data->smallCoreDataNum = small;
    data->bigCoreDataNum = big;
    data->finalSmallTileNum = small == 0 ? 0 : (small + c.tile - 1) / c.tile;
    data->finalBigTileNum = big == 0 ? 0 : (big + c.tile - 1) / c.tile;
    data->tileDataNum = c.tile;
    data->smallTailDataNum = small == 0 ? 0 : small - (data->finalSmallTileNum - 1) * c.tile;
    data->bigTailDataNum = big == 0 ? 0 : big - (data->finalBigTileNum - 1) * c.tile;
    data->tailBlockNum = tailBlocks;
    data->totalDataNum = static_cast<uint32_t>(c.input.size());
    data->targetDataNum = static_cast<uint32_t>(c.target.size());
    data->varDataNum = static_cast<uint32_t>(c.var.size());
    data->targetBroadcastAxisSize = c.targetAxisSize;
    data->targetInnerStride = c.targetInnerStride;
    data->targetBroadcastMode = c.targetMode;
    data->varBroadcastMode = c.varMode;
    data->varReduceSize = c.varReduceSize;
    data->reduction = c.reduction;
    data->eps = c.eps;
    data->meanScale = c.input.empty() ? 0.0f : 1.0f / static_cast<float>(c.input.size());

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    Traits::Run(c.blocks, gradOutput, input, target, var, gradInput, gradVar, workspace, tiling);
    vector<float> actualInput = Read<Traits>(gradInput, c.input.size());
    vector<float> actualVar = Read<Traits>(gradVar, c.var.size());
    ExpectGuardBytes(gradInputAllocation, inputBytes);
    ExpectGuardBytes(gradVarAllocation, varBytes);
    vector<float> expectedInput;
    vector<float> expectedVar;
    Golden(c, expectedInput, expectedVar);
    for (size_t i = 0; i < expectedInput.size(); ++i) {
        EXPECT_NEAR(actualInput[i], expectedInput[i], Traits::TOLERANCE) << "gradInput index " << i;
    }
    for (size_t i = 0; i < expectedVar.size(); ++i) {
        EXPECT_NEAR(actualVar[i], expectedVar[i], Traits::TOLERANCE) << "gradVar index " << i;
    }
    AscendC::GmFree(gradOutput);
    AscendC::GmFree(input);
    AscendC::GmFree(target);
    AscendC::GmFree(var);
    AscendC::GmFree(gradInputAllocation);
    AscendC::GmFree(gradVarAllocation);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST(GaussianNllLossGradKernel, FloatNoneSameShapeClampBoundary)
{
    KernelCase c;
    c.gradOutput = {1.0f, -0.5f, 2.0f, 0.25f, -1.0f};
    c.input = {0.0f, 1.0f, -2.0f, 3.0f, 0.5f};
    c.target = {0.5f, 0.0f, -1.0f, 2.0f, 1.5f};
    c.var = {0.0f, 0.1f, 0.2f, 1.0f, 2.0f};
    c.eps = 0.1f;
    RunCase<FloatTraits>(c);
}

TEST(GaussianNllLossGradKernel, FloatMeanTargetAndVarBroadcastMultiCoreMultiTile)
{
    KernelCase c;
    c.input.resize(513);
    c.gradOutput = {1.25f};
    c.target.resize(171);
    c.var.resize(171);
    for (size_t i = 0; i < c.input.size(); ++i) {
        c.input[i] = static_cast<float>(static_cast<int>(i % 19) - 9) * 0.1f;
    }
    for (size_t i = 0; i < c.target.size(); ++i) {
        c.target[i] = static_cast<float>(static_cast<int>(i % 11) - 5) * 0.05f;
        c.var[i] = 0.5f + static_cast<float>(i % 5) * 0.2f;
    }
    c.targetMode = 1;
    c.targetAxisSize = 3;
    c.targetInnerStride = 1;
    c.varMode = 1;
    c.varReduceSize = 3;
    c.reduction = 2;
    c.blocks = 3;
    c.tile = 64;
    RunCase<FloatTraits>(c);
}

TEST(GaussianNllLossGradKernel, HalfSumMissingLastDimensionTail)
{
    KernelCase c;
    c.input = {0.0f, 0.5f, 1.0f, 1.5f, -1.0f, -0.5f, 0.25f};
    c.gradOutput = {0.75f};
    c.target = {0.0f, 0.25f, 1.5f, 1.0f, -0.5f, -0.25f, 0.0f};
    c.var = {0.5f};
    c.varMode = 2;
    c.varReduceSize = 7;
    c.reduction = 1;
    c.blocks = 4;
    c.tile = 3;
    RunCase<HalfTraits>(c);
}

TEST(GaussianNllLossGradKernel, BFloat16NoneScalarVar)
{
    KernelCase c;
    c.input = {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 2.5f, 3.0f};
    c.target = {0.25f, 1.25f, 2.25f};
    c.gradOutput = {1.0f, 0.5f, -1.0f, 1.5f, -0.5f, 2.0f, 0.25f, 0.75f, -1.25f};
    c.var = {1.5f};
    c.targetMode = 1;
    c.targetAxisSize = 3;
    c.targetInnerStride = 1;
    c.varMode = 3;
    c.varReduceSize = 9;
    c.blocks = 5;
    c.tile = 2;
    RunCase<BFloat16Traits>(c);
}

TEST(GaussianNllLossGradKernel, EmptyTensorAllReductions)
{
    for (uint32_t reduction : {0U, 1U, 2U}) {
        KernelCase c;
        c.gradOutput = reduction == 0 ? vector<float>{} : vector<float>{1.0f};
        c.input = {};
        c.target = {};
        c.var = {1.0f};
        c.varMode = 3;
        c.varReduceSize = 0;
        c.reduction = reduction;
        RunCase<FloatTraits>(c);
    }
}

TEST(GaussianNllLossGradKernel, MatchesFiniteDifference)
{
    KernelCase c;
    c.gradOutput = {1.0f, -0.75f, 0.5f};
    c.input = {0.2f, -0.4f, 1.3f};
    c.target = {0.0f, -0.1f, 0.8f};
    c.var = {0.7f, 1.2f, 2.0f};
    vector<float> analyticInput;
    vector<float> analyticVar;
    Golden(c, analyticInput, analyticVar);
    constexpr float h = 1e-3f;
    auto objective = [&](const vector<float>& input, const vector<float>& var) {
        float value = 0.0f;
        for (size_t i = 0; i < input.size(); ++i) {
            const float d = input[i] - c.target[i];
            const float v = std::max(var[i], c.eps);
            value += c.gradOutput[i] * 0.5f * (std::log(v) + d * d / v);
        }
        return value;
    };
    for (size_t i = 0; i < c.input.size(); ++i) {
        vector<float> plus = c.input;
        vector<float> minus = c.input;
        plus[i] += h;
        minus[i] -= h;
        EXPECT_NEAR(analyticInput[i], (objective(plus, c.var) - objective(minus, c.var)) / (2.0f * h), 2e-4f);
        vector<float> varPlus = c.var;
        vector<float> varMinus = c.var;
        varPlus[i] += h;
        varMinus[i] -= h;
        EXPECT_NEAR(analyticVar[i], (objective(c.input, varPlus) - objective(c.input, varMinus)) / (2.0f * h), 2e-4f);
    }
    RunCase<FloatTraits>(c);
}

TEST(GaussianNllLossGradKernel, VectorRepeatBoundariesAndCanaries)
{
    for (uint32_t tile : {63U, 64U, 65U, 72U, 127U, 128U, 129U}) {
        KernelCase c;
        const size_t count = static_cast<size_t>(tile) * 2 + 1;
        c.gradOutput.assign(count, 1.0f);
        c.input.assign(count, 0.5f);
        c.target.assign(count, 0.0f);
        c.var.assign(count, 1.0f);
        c.tile = tile;
        RunCase<FloatTraits>(c);
    }
}

TEST(GaussianNllLossGradKernel, FloatMaximumTileBroadcastReduction)
{
    KernelCase c;
    constexpr size_t count = 4097;
    c.gradOutput.assign(count, 1.0f);
    c.input.assign(count, 0.5f);
    c.target.assign(count, 0.0f);
    c.var = {1.0f};
    c.varMode = 3;
    c.varReduceSize = static_cast<uint32_t>(count);
    c.tile = 4096;
    RunCase<FloatTraits>(c);
}

TEST(GaussianNllLossGradKernel, HalfAndBFloat16BroadcastVarMultiCore)
{
    KernelCase c;
    constexpr size_t varCount = 33;
    constexpr uint32_t reduceSize = 3;
    constexpr size_t count = varCount * reduceSize;
    c.gradOutput.assign(count, 1.0f);
    c.input.assign(count, 0.5f);
    c.target.assign(count, 0.0f);
    c.var.assign(varCount, 1.0f);
    c.varMode = 1;
    c.varReduceSize = reduceSize;
    c.blocks = 4;
    c.tile = 72;
    RunCase<HalfTraits>(c);
    RunCase<BFloat16Traits>(c);
}
