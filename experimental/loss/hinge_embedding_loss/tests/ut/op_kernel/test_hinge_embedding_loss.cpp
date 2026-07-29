/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "../../../op_kernel/hinge_embedding_loss.h"
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
    if (exponent <= 0)
        return static_cast<uint16_t>(sign);
    if (exponent >= 31)
        return static_cast<uint16_t>(sign | 0x7c00);
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
__global__ __aicore__ void HingeEmbeddingLossUt(GM_ADDR input, GM_ADDR target, GM_ADDR loss, GM_ADDR workspace,
                                                GM_ADDR tiling)
{
    HingeEmbeddingLossTilingData data;
    std::memcpy(&data, tiling, sizeof(data));
    NsHingeEmbeddingLoss::KernelHingeEmbeddingLoss<T, reductionMode> op;
    op.Init(input, target, loss, workspace, &data);
    op.Process();
}

template <typename Storage>
struct Traits;

template <>
struct Traits<float> {
    using KernelType = float;
    static constexpr float kTolerance = 1e-5f;
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
    static constexpr float kTolerance = 2e-2f;
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
    static constexpr float kTolerance = 8e-2f;
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

std::vector<float> Golden(const std::vector<float>& input, const std::vector<float>& target, float margin,
                          uint32_t reduction)
{
    std::vector<float> loss(input.size());
    float sum = 0;
    for (size_t i = 0; i < input.size(); ++i) {
        loss[i] = target[i] == 1.0f ? input[i] : std::max(0.0f, margin - input[i]);
        sum += loss[i];
    }
    if (reduction == 0)
        return loss;
    if (reduction == 2)
        sum = input.empty() ? NAN : sum / static_cast<float>(input.size());
    return {sum};
}

HingeEmbeddingLossTilingData BuildTiling(uint32_t total, uint32_t blocks, uint32_t tile, float margin)
{
    HingeEmbeddingLossTilingData data{};
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
    data.margin = margin;
    data.meanScale = total == 0 ? NAN : 1.0f / static_cast<float>(total);
    return data;
}

template <typename Storage, uint32_t reductionMode>
void RunCase(const std::vector<float>& inputHost, const std::vector<float>& targetHost, float margin,
             uint32_t blocks = 1, uint32_t tile = 32)
{
    ASSERT_EQ(inputHost.size(), targetHost.size());
    ASSERT_FALSE(inputHost.empty());
    const size_t outputSize = reductionMode == 0 ? inputHost.size() : 1;
    const size_t inputBytes = inputHost.size() * sizeof(Storage);
    uint8_t* input = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(inputBytes));
    uint8_t* target = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(inputBytes));
    uint8_t* loss = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(outputSize * sizeof(Storage)));
    uint8_t* workspace = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(blocks * 8 * sizeof(float)));
    uint8_t* tiling = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(sizeof(HingeEmbeddingLossTilingData)));
    ASSERT_NE(input, nullptr);
    ASSERT_NE(target, nullptr);
    ASSERT_NE(loss, nullptr);
    ASSERT_NE(workspace, nullptr);
    ASSERT_NE(tiling, nullptr);
    Traits<Storage>::Write(input, inputHost);
    Traits<Storage>::Write(target, targetHost);
    std::memset(loss, 0, outputSize * sizeof(Storage));
    const auto data = BuildTiling(static_cast<uint32_t>(inputHost.size()), blocks, tile, margin);
    std::memcpy(tiling, &data, sizeof(data));
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    using KernelType = typename Traits<Storage>::KernelType;
    ICPU_RUN_KF((HingeEmbeddingLossUt<KernelType, reductionMode>), blocks, input, target, loss, workspace, tiling);
    const auto actual = Traits<Storage>::Read(loss, outputSize);
    const auto expected = Golden(inputHost, targetHost, margin, reductionMode);
    for (size_t i = 0; i < outputSize; ++i) {
        EXPECT_NEAR(actual[i], expected[i], Traits<Storage>::kTolerance) << "index=" << i;
    }
    AscendC::GmFree(input);
    AscendC::GmFree(target);
    AscendC::GmFree(loss);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST(HingeEmbeddingLossKernel, FloatNoneCoversBranchesAndBoundary)
{
    RunCase<float, 0>({-1.0f, 0.0f, 0.5f, 1.0f, 2.0f}, {1.0f, -1.0f, -1.0f, -1.0f, 1.0f}, 1.0f);
}

TEST(HingeEmbeddingLossKernel, HalfSumCustomMargin)
{
    RunCase<HalfStorage, 1>({0.0f, 0.5f, 1.0f, 2.0f}, {-1.0f, -1.0f, 1.0f, 1.0f}, 0.5f);
}

TEST(HingeEmbeddingLossKernel, BFloat16Mean)
{
    RunCase<BFloat16Storage, 2>({-1.0f, 0.0f, 1.0f, 3.0f}, {1.0f, -1.0f, -1.0f, 1.0f}, 2.0f);
}

TEST(HingeEmbeddingLossKernel, FloatNoneUnevenMultiCoreMultipleTiles)
{
    std::vector<float> input(257);
    std::vector<float> target(257);
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>(static_cast<int>(i % 17) - 8) * 0.25f;
        target[i] = i % 2 == 0 ? 1.0f : -1.0f;
    }
    RunCase<float, 0>(input, target, 1.0f, 4, 32);
}

TEST(HingeEmbeddingLossKernel, FloatMeanUnevenMultiCoreMultipleTiles)
{
    std::vector<float> input(257);
    std::vector<float> target(257);
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>(static_cast<int>(i % 17) - 8) * 0.25f;
        target[i] = i % 2 == 0 ? 1.0f : -1.0f;
    }
    RunCase<float, 2>(input, target, 1.0f, 4, 32);
}
} // namespace
