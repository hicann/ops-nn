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
 * \file test_mse_loss.cpp
 * \brief MseLoss 算子 kernel UT 测试
 */

#include "mse_loss_tiling.h"
#include "../../../op_kernel/mse_loss.cpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>
#include "gtest/gtest.h"
#include "tikicpulib.h"

using namespace std;

static uint16_t FloatToHalf(float f)
{
    uint32_t bits;
    memcpy(&bits, &f, sizeof(float));
    uint32_t sign = (bits >> 16) & 0x8000;
    int32_t exp = static_cast<int32_t>((bits >> 23) & 0xff) - 127 + 15;
    uint32_t mant = (bits >> 13) & 0x3ff;
    if (exp <= 0) {
        return static_cast<uint16_t>(sign);
    }
    if (exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7c00);
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | mant);
}

static uint16_t FloatToBFloat16(float f)
{
    uint32_t bits;
    memcpy(&bits, &f, sizeof(float));
    uint32_t roundingBias = 0x7fff + ((bits >> 16) & 1);
    return static_cast<uint16_t>((bits + roundingBias) >> 16);
}

static float HalfToFloat(uint16_t h)
{
    uint32_t sign = (static_cast<uint32_t>(h & 0x8000)) << 16;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x03ff;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400) == 0) {
                mant <<= 1;
                --exp;
            }
            mant &= 0x03ff;
            bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000 | (mant << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &bits, sizeof(float));
    return f;
}

static float BFloat16ToFloat(uint16_t h)
{
    uint32_t bits = static_cast<uint32_t>(h) << 16;
    float f;
    memcpy(&f, &bits, sizeof(float));
    return f;
}

static void FillHalf(uint8_t* gm, const vector<float>& values)
{
    for (size_t i = 0; i < values.size(); ++i) {
        uint16_t h = FloatToHalf(values[i]);
        memcpy(gm + i * sizeof(uint16_t), &h, sizeof(uint16_t));
    }
}

static void FillBFloat16(uint8_t* gm, const vector<float>& values)
{
    for (size_t i = 0; i < values.size(); ++i) {
        uint16_t h = FloatToBFloat16(values[i]);
        memcpy(gm + i * sizeof(uint16_t), &h, sizeof(uint16_t));
    }
}

static vector<float> ReadHalf(uint8_t* gm, size_t size)
{
    vector<float> out(size);
    for (size_t i = 0; i < size; ++i) {
        uint16_t h;
        memcpy(&h, gm + i * sizeof(uint16_t), sizeof(uint16_t));
        out[i] = HalfToFloat(h);
    }
    return out;
}

static vector<float> ReadBFloat16(uint8_t* gm, size_t size)
{
    vector<float> out(size);
    for (size_t i = 0; i < size; ++i) {
        uint16_t h;
        memcpy(&h, gm + i * sizeof(uint16_t), sizeof(uint16_t));
        out[i] = BFloat16ToFloat(h);
    }
    return out;
}

static void FillFloat(uint8_t* gm, const vector<float>& values)
{
    memcpy(gm, values.data(), values.size() * sizeof(float));
}

static vector<float> ReadFloat(uint8_t* gm, size_t size)
{
    vector<float> out(size);
    memcpy(out.data(), gm, size * sizeof(float));
    return out;
}

static vector<float> Golden(const vector<float>& predict, const vector<float>& label, int64_t reduction)
{
    vector<float> loss(predict.size());
    float sum = 0.0f;
    for (size_t i = 0; i < predict.size(); ++i) {
        float diff = predict[i] - label[i];
        loss[i] = diff * diff;
        sum += loss[i];
    }
    if (reduction == 0) {
        return loss;
    }
    if (reduction == 2) {
        if (predict.empty()) {
            sum = std::numeric_limits<float>::quiet_NaN();
        } else {
            sum /= static_cast<float>(predict.size());
        }
    }
    return {sum};
}

template <typename Element>
struct KernelCaseTraits;

struct BFloat16Element {
    uint16_t value;
};

template <>
struct KernelCaseTraits<uint16_t> {
    static constexpr uint32_t TILING_KEY = 0;
    static constexpr float EPS = 2e-2f;

    static void Fill(uint8_t* gm, const vector<float>& values) { FillHalf(gm, values); }

    static vector<float> Read(uint8_t* gm, size_t size) { return ReadHalf(gm, size); }

    static void Run(uint32_t numBlocks, uint8_t* predict, uint8_t* label, uint8_t* y, uint8_t* workspace,
                    uint8_t* tiling)
    {
        ICPU_SET_TILING_KEY(TILING_KEY);
        ICPU_RUN_KF((mse_loss<TILING_KEY>), numBlocks, predict, label, y, workspace, tiling);
    }
};

template <>
struct KernelCaseTraits<BFloat16Element> {
    static constexpr uint32_t TILING_KEY = 2;
    static constexpr float EPS = 8e-2f;

    static void Fill(uint8_t* gm, const vector<float>& values) { FillBFloat16(gm, values); }

    static vector<float> Read(uint8_t* gm, size_t size) { return ReadBFloat16(gm, size); }

    static void Run(uint32_t numBlocks, uint8_t* predict, uint8_t* label, uint8_t* y, uint8_t* workspace,
                    uint8_t* tiling)
    {
        ICPU_SET_TILING_KEY(TILING_KEY);
        ICPU_RUN_KF((mse_loss<TILING_KEY>), numBlocks, predict, label, y, workspace, tiling);
    }
};

template <>
struct KernelCaseTraits<float> {
    static constexpr uint32_t TILING_KEY = 1;
    static constexpr float EPS = 1e-5f;

    static void Fill(uint8_t* gm, const vector<float>& values) { FillFloat(gm, values); }

    static vector<float> Read(uint8_t* gm, size_t size) { return ReadFloat(gm, size); }

    static void Run(uint32_t numBlocks, uint8_t* predict, uint8_t* label, uint8_t* y, uint8_t* workspace,
                    uint8_t* tiling)
    {
        ICPU_SET_TILING_KEY(TILING_KEY);
        ICPU_RUN_KF((mse_loss<TILING_KEY>), numBlocks, predict, label, y, workspace, tiling);
    }
};

template <typename Element>
static void RunCase(const vector<float>& predictHost, const vector<float>& labelHost, int64_t reduction,
                    uint32_t numBlocks = 1, int64_t ubFactor = 64)
{
    const size_t size = predictHost.size();
    const size_t outputSize = (reduction == 0) ? size : 1;
    const size_t tilingDataSize = sizeof(MseLossTilingData);
    const size_t elemSize = sizeof(Element);
    const size_t workspaceSize = static_cast<size_t>(numBlocks) * 32;

    const size_t inputBytes = std::max(size * elemSize, elemSize);
    uint8_t* predict = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(inputBytes));
    uint8_t* label = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(inputBytes));
    uint8_t* y = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(outputSize * elemSize));
    uint8_t* workspace = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(workspaceSize));
    uint8_t* tiling = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(tilingDataSize));
    if (predict == nullptr || label == nullptr || y == nullptr || workspace == nullptr || tiling == nullptr) {
        if (predict != nullptr) {
            AscendC::GmFree(predict);
        }
        if (label != nullptr) {
            AscendC::GmFree(label);
        }
        if (y != nullptr) {
            AscendC::GmFree(y);
        }
        if (workspace != nullptr) {
            AscendC::GmFree(workspace);
        }
        if (tiling != nullptr) {
            AscendC::GmFree(tiling);
        }
        FAIL() << "GmAlloc failed";
    }
    memset(y, 0, outputSize * elemSize);
    memset(workspace, 0, workspaceSize);
    KernelCaseTraits<Element>::Fill(predict, predictHost);
    KernelCaseTraits<Element>::Fill(label, labelHost);

    auto* tilingData = reinterpret_cast<MseLossTilingData*>(tiling);
    tilingData->totalNum = static_cast<int64_t>(size);
    tilingData->blockFactor = static_cast<int64_t>((size + numBlocks - 1) / numBlocks);
    tilingData->ubFactor = ubFactor;
    tilingData->reduction = reduction;
    tilingData->blockNum = numBlocks;
    tilingData->workspaceFloatsPerCore = 8;
    tilingData->meanScale = size == 0 ? std::numeric_limits<float>::quiet_NaN() : 1.0f / static_cast<float>(size);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    KernelCaseTraits<Element>::Run(numBlocks, predict, label, y, workspace, tiling);

    vector<float> actual = KernelCaseTraits<Element>::Read(y, outputSize);
    vector<float> expect = Golden(predictHost, labelHost, reduction);
    ASSERT_EQ(actual.size(), expect.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        if (std::isnan(expect[i])) {
            EXPECT_TRUE(std::isnan(actual[i])) << "index=" << i;
        } else {
            EXPECT_NEAR(actual[i], expect[i], KernelCaseTraits<Element>::EPS) << "index=" << i;
        }
    }

    AscendC::GmFree(predict);
    AscendC::GmFree(label);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

static void RunHalfCase(const vector<float>& predictHost, const vector<float>& labelHost, int64_t reduction)
{
    RunCase<uint16_t>(predictHost, labelHost, reduction, 1, 64);
}

class MseLossKernelTest : public testing::Test {};

TEST_F(MseLossKernelTest, mean_fp16)
{
    vector<float> predict = {1.0f, 2.0f, -3.0f, 4.5f, -5.5f, 6.0f, 0.25f, -0.75f};
    vector<float> label = {0.0f, 1.5f, -1.0f, 1.5f, -4.5f, 2.0f, -0.75f, -1.25f};
    RunHalfCase(predict, label, 2);
}

TEST_F(MseLossKernelTest, mean_fp32_empty) { RunCase<float>({}, {}, 2); }

TEST_F(MseLossKernelTest, sum_fp16)
{
    vector<float> predict = {1.0f, -2.0f, 3.0f, -4.0f, 5.0f};
    vector<float> label = {0.5f, -1.0f, 1.0f, -1.0f, 0.0f};
    RunHalfCase(predict, label, 1);
}

TEST_F(MseLossKernelTest, none_fp16_large)
{
    vector<float> predict(4096);
    vector<float> label(4096);
    for (size_t i = 0; i < predict.size(); ++i) {
        predict[i] = static_cast<float>(static_cast<int>(i % 17) - 8) * 0.25f;
        label[i] = static_cast<float>(static_cast<int>(i % 11) - 5) * 0.125f;
    }
    RunHalfCase(predict, label, 0);
}

TEST_F(MseLossKernelTest, mean_fp16_multi_block_tail)
{
    vector<float> predict(513);
    vector<float> label(513);
    for (size_t i = 0; i < predict.size(); ++i) {
        predict[i] = static_cast<float>(static_cast<int>(i % 23) - 11) * 0.125f;
        label[i] = static_cast<float>(static_cast<int>(i % 19) - 9) * 0.0625f;
    }
    RunCase<uint16_t>(predict, label, 2, 4, 64);
}

TEST_F(MseLossKernelTest, sum_fp32_multi_tile)
{
    vector<float> predict(1025);
    vector<float> label(1025);
    for (size_t i = 0; i < predict.size(); ++i) {
        predict[i] = static_cast<float>(static_cast<int>(i % 31) - 15) * 0.03125f;
        label[i] = static_cast<float>(static_cast<int>(i % 29) - 14) * 0.015625f;
    }
    RunCase<float>(predict, label, 1, 1, 64);
}

TEST_F(MseLossKernelTest, mean_bf16_multi_block_tail)
{
    vector<float> predict(769);
    vector<float> label(769);
    for (size_t i = 0; i < predict.size(); ++i) {
        predict[i] = static_cast<float>(static_cast<int>(i % 37) - 18) * 0.0625f;
        label[i] = static_cast<float>(static_cast<int>(i % 29) - 14) * 0.03125f;
    }
    RunCase<BFloat16Element>(predict, label, 2, 5, 64);
}

TEST_F(MseLossKernelTest, mean_bf16_single_block_fast_path)
{
    vector<float> predict(257);
    vector<float> label(257);
    for (size_t i = 0; i < predict.size(); ++i) {
        predict[i] = static_cast<float>(static_cast<int>(i % 17) - 8) * 0.125f;
        label[i] = static_cast<float>(static_cast<int>(i % 13) - 6) * 0.0625f;
    }
    RunCase<BFloat16Element>(predict, label, 2, 1, 320);
}

TEST_F(MseLossKernelTest, none_fp32_unaligned_small)
{
    vector<float> predict = {1.0f, -2.0f, 3.5f, -4.25f, 5.125f, -6.0f, 7.25f};
    vector<float> label = {0.25f, -1.5f, 2.0f, -5.0f, 4.5f, -4.0f, 7.0f};
    RunCase<float>(predict, label, 0);
}
