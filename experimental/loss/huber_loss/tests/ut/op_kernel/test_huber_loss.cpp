/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "huber_loss_tiling.h"
#include "../../../op_kernel/huber_loss.h"

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
    const int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xff) - 127 + 15;
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
    uint32_t exponent = (value >> 10) & 0x1f;
    uint32_t mantissa = value & 0x03ff;
    uint32_t bits = 0;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            exponent = 1;
            while ((mantissa & 0x0400) == 0) {
                mantissa <<= 1;
                --exponent;
            }
            mantissa &= 0x03ff;
            bits = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
        }
    } else if (exponent == 31) {
        bits = sign | 0x7f800000 | (mantissa << 13);
    } else {
        bits = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
    }
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

uint16_t FloatToBFloat16(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t roundingBias = 0x7fff + ((bits >> 16) & 1);
    return static_cast<uint16_t>((bits + roundingBias) >> 16);
}

float BFloat16ToFloat(uint16_t value)
{
    const uint32_t bits = static_cast<uint32_t>(value) << 16;
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

template <typename T>
__global__ __aicore__ void HuberLossUtKernel(GM_ADDR predictions, GM_ADDR targets, GM_ADDR loss, GM_ADDR workspace,
                                             GM_ADDR tiling)
{
    (void)workspace;
    HuberLossTilingData tilingData;
    InitTilingData(tiling, &tilingData);
    NsHuberLoss::KernelHuberLoss<T> op;
    op.Init(predictions, targets, loss, &tilingData);
    op.Process();
}

template <typename Storage>
struct KernelTraits;

template <>
struct KernelTraits<float> {
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
struct KernelTraits<HalfStorage> {
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
struct KernelTraits<BFloat16Storage> {
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

std::vector<float> Golden(const std::vector<float>& predictions, const std::vector<float>& targets, float delta)
{
    std::vector<float> loss(predictions.size());
    for (size_t i = 0; i < predictions.size(); ++i) {
        const float diff = predictions[i] - targets[i];
        const float absDiff = std::abs(diff);
        loss[i] = absDiff <= delta ? 0.5f * diff * diff : delta * (absDiff - 0.5f * delta);
    }
    return loss;
}

HuberLossTilingData BuildTiling(uint32_t total, uint32_t blockNum, uint32_t tileDataNum, float delta)
{
    HuberLossTilingData data{};
    const uint32_t small = total / blockNum;
    const uint32_t tailBlocks = total % blockNum;
    const uint32_t big = small + (tailBlocks == 0 ? 0 : 1);
    data.smallCoreDataNum = small;
    data.bigCoreDataNum = big;
    data.finalBigTileNum = big == 0 ? 0 : (big + tileDataNum - 1) / tileDataNum;
    data.finalSmallTileNum = small == 0 ? 0 : (small + tileDataNum - 1) / tileDataNum;
    data.tileDataNum = tileDataNum;
    data.smallTailDataNum = small == 0 ? 0 : small - (data.finalSmallTileNum - 1) * tileDataNum;
    data.bigTailDataNum = big == 0 ? 0 : big - (data.finalBigTileNum - 1) * tileDataNum;
    data.tailBlockNum = tailBlocks;
    data.delta = delta;
    return data;
}

template <typename Storage>
void RunCase(const std::vector<float>& predictionsHost, const std::vector<float>& targetsHost, float delta,
             uint32_t blockNum = 1, uint32_t tileDataNum = 32)
{
    ASSERT_EQ(predictionsHost.size(), targetsHost.size());
    ASSERT_FALSE(predictionsHost.empty());
    ASSERT_LE(blockNum, predictionsHost.size());
    const size_t size = predictionsHost.size();
    const size_t bytes = size * sizeof(Storage);
    uint8_t* predictions = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(bytes));
    uint8_t* targets = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(bytes));
    uint8_t* loss = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(bytes));
    uint8_t* workspace = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(32));
    uint8_t* tiling = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(sizeof(HuberLossTilingData)));
    ASSERT_NE(predictions, nullptr);
    ASSERT_NE(targets, nullptr);
    ASSERT_NE(loss, nullptr);
    ASSERT_NE(workspace, nullptr);
    ASSERT_NE(tiling, nullptr);

    KernelTraits<Storage>::Write(predictions, predictionsHost);
    KernelTraits<Storage>::Write(targets, targetsHost);
    std::memset(loss, 0, bytes);
    const HuberLossTilingData tilingData = BuildTiling(static_cast<uint32_t>(size), blockNum, tileDataNum, delta);
    std::memcpy(tiling, &tilingData, sizeof(tilingData));

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    using KernelType = typename KernelTraits<Storage>::KernelType;
    ICPU_RUN_KF((HuberLossUtKernel<KernelType>), blockNum, predictions, targets, loss, workspace, tiling);

    const std::vector<float> actual = KernelTraits<Storage>::Read(loss, size);
    const std::vector<float> expected = Golden(predictionsHost, targetsHost, delta);
    for (size_t i = 0; i < size; ++i) {
        EXPECT_NEAR(actual[i], expected[i], KernelTraits<Storage>::kTolerance) << "index=" << i;
    }

    AscendC::GmFree(predictions);
    AscendC::GmFree(targets);
    AscendC::GmFree(loss);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST(HuberLossKernelTest, Float32CoversBoundaryAndBothBranches)
{
    RunCase<float>({-2.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 2.0f}, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}, 1.0f);
}

TEST(HuberLossKernelTest, Float16UsesCustomDelta)
{
    RunCase<HalfStorage>({-1.5f, -0.5f, 0.0f, 0.5f, 1.5f}, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f}, 0.5f);
}

TEST(HuberLossKernelTest, BFloat16UsesCustomDelta)
{
    RunCase<BFloat16Storage>({-3.0f, -2.0f, 0.0f, 2.0f, 3.0f}, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f}, 2.0f);
}

TEST(HuberLossKernelTest, Float32HandlesUnevenMultiCoreAndMultipleTiles)
{
    std::vector<float> predictions(257);
    std::vector<float> targets(257);
    for (size_t i = 0; i < predictions.size(); ++i) {
        predictions[i] = static_cast<float>(static_cast<int>(i % 23) - 11) * 0.125f;
        targets[i] = static_cast<float>(static_cast<int>(i % 17) - 8) * 0.0625f;
    }
    RunCase<float>(predictions, targets, 1.0f, 4, 64);
}

TEST(HuberLossKernelTest, Float32HandlesVectorAlignedFullTileAndTail)
{
    std::vector<float> predictions(97);
    std::vector<float> targets(97);
    for (size_t i = 0; i < predictions.size(); ++i) {
        predictions[i] = static_cast<float>(static_cast<int>(i % 19) - 9) * 0.25f;
        targets[i] = static_cast<float>(static_cast<int>(i % 13) - 6) * 0.125f;
    }
    RunCase<float>(predictions, targets, 1.0f, 1, 64);
}

TEST(HuberLossKernelTest, Float16HandlesOneElement) { RunCase<HalfStorage>({2.0f}, {0.5f}, 1.0f); }
} // namespace
