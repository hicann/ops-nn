/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "../../../op_kernel/gaussian_nll_loss.cpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>
#include "gtest/gtest.h"
#include "tikicpulib.h"

namespace {
struct HalfStorage {
    uint16_t value;
};
struct BFloat16Storage {
    uint16_t value;
};

uint16_t FloatToHalf(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000;
    const int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xff) - 112;
    const uint32_t mantissa = (bits >> 13) & 0x3ff;
    if (exponent <= 0) {
        return static_cast<uint16_t>(sign);
    }
    if (exponent >= 31) {
        return static_cast<uint16_t>(sign | 0x7c00);
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) | mantissa);
}

float HalfToFloat(uint16_t value)
{
    const uint32_t sign = static_cast<uint32_t>(value & 0x8000) << 16;
    const uint32_t exponent = (value >> 10) & 0x1f;
    const uint32_t mantissa = value & 0x3ff;
    const uint32_t bits = exponent == 0  ? sign :
                          exponent == 31 ? sign | 0x7f800000 | (mantissa << 13) :
                                           sign | ((exponent + 112) << 23) | (mantissa << 13);
    float result = 0;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

uint16_t FloatToBFloat16(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return static_cast<uint16_t>((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
}

float BFloat16ToFloat(uint16_t value)
{
    const uint32_t bits = static_cast<uint32_t>(value) << 16;
    float result = 0;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

template <typename T, uint32_t reductionMode>
__global__ __aicore__ void GaussianNllLossUt(GM_ADDR input, GM_ADDR target, GM_ADDR var, GM_ADDR loss,
                                             GM_ADDR workspace, GM_ADDR tiling)
{
    GaussianNllLossTilingData data;
    std::memcpy(&data, tiling, sizeof(data));
    AscendC::TPipe pipe;
    NsGaussianNllLoss::KernelGaussianNllLoss<T, reductionMode> op;
    op.Init(input, target, var, loss, workspace, &data, pipe);
    op.Process();
}

template <typename Storage>
struct Traits;

template <>
struct Traits<float> {
    using KernelType = float;
    static constexpr float kTolerance = 2e-5f;
    static void Write(uint8_t* address, const std::vector<float>& values)
    {
        std::memcpy(address, values.data(), values.size() * sizeof(float));
    }
    static std::vector<float> Read(const uint8_t* address, size_t size)
    {
        std::vector<float> values(size);
        std::memcpy(values.data(), address, size * sizeof(float));
        return values;
    }
};

template <>
struct Traits<HalfStorage> {
    using KernelType = half;
    static constexpr float kTolerance = 3e-2f;
    static void Write(uint8_t* address, const std::vector<float>& values)
    {
        for (size_t i = 0; i < values.size(); ++i) {
            const uint16_t value = FloatToHalf(values[i]);
            std::memcpy(address + i * sizeof(value), &value, sizeof(value));
        }
    }
    static std::vector<float> Read(const uint8_t* address, size_t size)
    {
        std::vector<float> values(size);
        for (size_t i = 0; i < size; ++i) {
            uint16_t value = 0;
            std::memcpy(&value, address + i * sizeof(value), sizeof(value));
            values[i] = HalfToFloat(value);
        }
        return values;
    }
};

template <>
struct Traits<BFloat16Storage> {
    using KernelType = bfloat16_t;
    static constexpr float kTolerance = 1e-1f;
    static void Write(uint8_t* address, const std::vector<float>& values)
    {
        for (size_t i = 0; i < values.size(); ++i) {
            const uint16_t value = FloatToBFloat16(values[i]);
            std::memcpy(address + i * sizeof(value), &value, sizeof(value));
        }
    }
    static std::vector<float> Read(const uint8_t* address, size_t size)
    {
        std::vector<float> values(size);
        for (size_t i = 0; i < size; ++i) {
            uint16_t value = 0;
            std::memcpy(&value, address + i * sizeof(value), sizeof(value));
            values[i] = BFloat16ToFloat(value);
        }
        return values;
    }
};

uint64_t TargetIndex(uint64_t logicalIndex, uint32_t mode, uint64_t axisSpan, uint64_t innerSize)
{
    if (mode == 0) {
        return logicalIndex;
    }
    return logicalIndex / axisSpan * innerSize + logicalIndex % innerSize;
}

uint64_t VarIndex(uint64_t logicalIndex, uint32_t mode, uint64_t innerSize)
{
    if (mode == 0) {
        return logicalIndex;
    }
    return mode == 2 ? 0 : logicalIndex / innerSize;
}

std::vector<float> Golden(const std::vector<float>& input, const std::vector<float>& target,
                          const std::vector<float>& var, uint32_t targetMode, uint64_t targetAxisSpan,
                          uint64_t targetInnerSize, uint32_t varMode, uint64_t varInnerSize, bool full, float eps,
                          uint32_t reduction)
{
    constexpr float halfLogTwoPi = 0.91893853320467274178f;
    std::vector<float> loss(input.size());
    float sum = 0.0f;
    for (size_t i = 0; i < input.size(); ++i) {
        const float difference = input[i] - target[TargetIndex(i, targetMode, targetAxisSpan, targetInnerSize)];
        const float variance = std::max(var[VarIndex(i, varMode, varInnerSize)], eps);
        loss[i] = 0.5f * (std::log(variance) + difference * difference / variance);
        if (full) {
            loss[i] += halfLogTwoPi;
        }
        sum += loss[i];
    }
    if (reduction == 0) {
        return loss;
    }
    if (reduction == 2) {
        sum = input.empty() ? NAN : sum / static_cast<float>(input.size());
    }
    return {sum};
}

GaussianNllLossTilingData BuildTiling(uint32_t total, uint32_t blocks, uint32_t tile, uint32_t targetMode,
                                      uint64_t targetAxisSpan, uint64_t targetInnerSize, uint64_t targetElementCount,
                                      uint32_t varMode, uint64_t varInnerSize, uint64_t varElementCount, bool full,
                                      float eps)
{
    GaussianNllLossTilingData data{};
    const uint32_t small = total / blocks;
    const uint32_t tailBlocks = total % blocks;
    const uint32_t big = small + (tailBlocks == 0 ? 0 : 1);
    data.smallCoreDataNum = small;
    data.bigCoreDataNum = big;
    data.finalBigTileNum = big == 0 ? 0 : (big + tile - 1) / tile;
    data.finalSmallTileNum = small == 0 ? 0 : (small + tile - 1) / tile;
    data.tileDataNum = tile;
    data.smallTailDataNum = small == 0 ? 0 : small - (data.finalSmallTileNum - 1) * tile;
    data.bigTailDataNum = big == 0 ? 0 : big - (data.finalBigTileNum - 1) * tile;
    data.tailBlockNum = tailBlocks;
    data.blockNum = blocks;
    data.workspaceFloatsPerCore = 8;
    data.targetBroadcastMode = targetMode;
    data.varBroadcastMode = varMode;
    data.targetAxisSpan = targetAxisSpan;
    data.targetInnerSize = targetInnerSize;
    data.targetElementCount = targetElementCount;
    data.varInnerSize = varInnerSize;
    data.varElementCount = varElementCount;
    data.eps = eps;
    data.fullConstant = full ? 0.91893853320467274178f : 0.0f;
    data.meanScale = total == 0 ? NAN : 1.0f / static_cast<float>(total);
    return data;
}

template <typename Storage, uint32_t reductionMode>
void RunCase(const std::vector<float>& inputHost, const std::vector<float>& targetHost,
             const std::vector<float>& varHost, uint32_t targetMode = 0, uint64_t targetAxisSpan = 1,
             uint64_t targetInnerSize = 1, uint32_t varMode = 0, uint64_t varInnerSize = 1, bool full = false,
             float eps = 1e-6f, uint32_t blocks = 1, uint32_t tile = 32, bool useProductionEntry = false)
{
    constexpr size_t kSystemWorkspaceBytes = 16 * 1024 * 1024;
    ASSERT_FALSE(targetHost.empty());
    ASSERT_FALSE(varHost.empty());
    const size_t outputSize = reductionMode == 0 ? inputHost.size() : 1;
    const size_t inputBytes = std::max<size_t>(1, inputHost.size()) * sizeof(Storage);
    const size_t targetBytes = targetHost.size() * sizeof(Storage);
    const size_t varBytes = varHost.size() * sizeof(Storage);
    uint8_t* input = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(inputBytes));
    uint8_t* target = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(targetBytes));
    uint8_t* var = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(varBytes));
    uint8_t* loss = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(outputSize * sizeof(Storage)));
    const size_t userWorkspaceBytes = blocks * 8 * sizeof(float);
    const size_t workspaceBytes = useProductionEntry && reductionMode != 0 && blocks > 1 ?
                                      kSystemWorkspaceBytes + userWorkspaceBytes :
                                      userWorkspaceBytes;
    uint8_t* workspace = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(workspaceBytes));
    uint8_t* tiling = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(sizeof(GaussianNllLossTilingData)));
    ASSERT_NE(input, nullptr);
    ASSERT_NE(target, nullptr);
    ASSERT_NE(var, nullptr);
    ASSERT_NE(loss, nullptr);
    ASSERT_NE(workspace, nullptr);
    ASSERT_NE(tiling, nullptr);
    Traits<Storage>::Write(input, inputHost);
    Traits<Storage>::Write(target, targetHost);
    Traits<Storage>::Write(var, varHost);
    std::memset(loss, 0, outputSize * sizeof(Storage));
    const auto data = BuildTiling(static_cast<uint32_t>(inputHost.size()), blocks, tile, targetMode, targetAxisSpan,
                                  targetInnerSize, targetHost.size(), varMode, varInnerSize, varHost.size(), full, eps);
    std::memcpy(tiling, &data, sizeof(data));
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    using KernelType = typename Traits<Storage>::KernelType;
    if (useProductionEntry) {
        ICPU_RUN_KF((gaussian_nll_loss<reductionMode>), blocks, input, target, var, loss, workspace, tiling);
    } else {
        ICPU_RUN_KF((GaussianNllLossUt<KernelType, reductionMode>), blocks, input, target, var, loss, workspace,
                    tiling);
    }
    const auto actual = Traits<Storage>::Read(loss, outputSize);
    const auto expected = Golden(inputHost, targetHost, varHost, targetMode, targetAxisSpan, targetInnerSize, varMode,
                                 varInnerSize, full, eps, reductionMode);
    for (size_t i = 0; i < outputSize; ++i) {
        if (std::isnan(expected[i])) {
            EXPECT_TRUE(std::isnan(actual[i])) << "index=" << i;
        } else {
            EXPECT_NEAR(actual[i], expected[i], Traits<Storage>::kTolerance) << "index=" << i;
        }
    }
    AscendC::GmFree(input);
    AscendC::GmFree(target);
    AscendC::GmFree(var);
    AscendC::GmFree(loss);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST(GaussianNllLossKernel, FloatNoneCoversClampAndFull)
{
    RunCase<float, 0>({0.0f, 1.0f, -1.0f, 2.0f}, {0.5f, 0.5f, -0.5f, 1.0f}, {0.0f, 1e-6f, 0.5f, 2.0f}, 0, 1, 1, 0, 1,
                      true);
}

TEST(GaussianNllLossKernel, HalfSumUsesFloatComputation)
{
    RunCase<HalfStorage, 1>({0.0f, 0.5f, 1.0f, 1.5f}, {0.25f, 0.25f, 0.25f, 0.25f}, {0.5f, 0.5f, 0.5f, 0.5f});
}

TEST(GaussianNllLossKernel, BFloat16MeanTargetAxisAndScalarVar)
{
    const std::vector<float> input = {0.0f, 0.5f, 1.0f, 1.5f, -1.0f, -0.5f, 0.0f, 0.5f};
    const std::vector<float> target = {0.25f, 0.25f, 0.25f, 0.25f};
    RunCase<BFloat16Storage, 2>(input, target, {0.75f}, 1, 8, 4, 2, 1, true);
}

TEST(GaussianNllLossKernel, TargetAxisAndTrailingVarBroadcast)
{
    std::vector<float> input(24);
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>(i) * 0.1f;
    }
    const std::vector<float> target = {0.0f, 0.1f, 0.2f, 0.3f, 1.0f, 1.1f, 1.2f, 1.3f};
    const std::vector<float> var = {0.5f, 1.0f, 2.0f, 0.75f, 1.25f, 1.5f};
    RunCase<float, 0>(input, target, var, 1, 12, 4, 1, 4);
}

TEST(GaussianNllLossKernel, FloatMeanUnevenMultiCoreMultipleTiles)
{
    std::vector<float> input(257);
    std::vector<float> target(257);
    std::vector<float> var(257);
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>(static_cast<int>(i % 17) - 8) * 0.125f;
        target[i] = static_cast<float>(i % 7) * 0.1f;
        var[i] = 0.25f + static_cast<float>(i % 5) * 0.25f;
    }
    RunCase<float, 2>(input, target, var, 0, 1, 1, 0, 1, false, 1e-6f, 4, 32);
}

TEST(GaussianNllLossKernel, EmptySumIsZero) { RunCase<float, 1>({}, {0.0f}, {1.0f}, 0, 1, 1, 2, 1); }

TEST(GaussianNllLossKernel, ProductionEntryEmptyMeanIsNan)
{
    RunCase<float, 2>({}, {0.0f}, {1.0f}, 0, 1, 1, 2, 1, false, 1e-6f, 1, 32, true);
}

TEST(GaussianNllLossKernel, ProductionEntryMeanUsesUserWorkspace)
{
    std::vector<float> input(257);
    std::vector<float> target(257);
    std::vector<float> var(257);
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>(static_cast<int>(i % 17) - 8) * 0.125f;
        target[i] = static_cast<float>(i % 7) * 0.1f;
        var[i] = 0.25f + static_cast<float>(i % 5) * 0.25f;
    }
    RunCase<float, 2>(input, target, var, 0, 1, 1, 0, 1, false, 1e-6f, 4, 32, true);
}

TEST(GaussianNllLossKernel, ProductionEntrySingleCoreSum)
{
    RunCase<float, 1>({0.0f, 0.5f, 1.0f}, {0.25f, 0.25f, 0.25f}, {0.5f, 0.5f, 0.5f}, 0, 1, 1, 0, 1, false, 1e-6f, 1, 32,
                      true);
}
} // namespace
