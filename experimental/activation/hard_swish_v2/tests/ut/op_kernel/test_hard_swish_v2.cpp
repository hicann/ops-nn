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
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "../../../op_kernel/hard_swish_v2_tiling_data.h"

extern "C" __global__ __aicore__ void hard_swish_v2_fp32_ut(GM_ADDR x, GM_ADDR out, GM_ADDR tiling);
extern "C" __global__ __aicore__ void hard_swish_v2_fp16_ut(GM_ADDR x, GM_ADDR out, GM_ADDR tiling);
extern "C" __global__ __aicore__ void hard_swish_v2_bf16_ut(GM_ADDR x, GM_ADDR out, GM_ADDR tiling);

namespace {
enum class Dtype { FLOAT, HALF, BFLOAT16 };
struct GmDeleter {
    void operator()(uint8_t* ptr) const { AscendC::GmFree(ptr); }
};
using GmBuffer = std::unique_ptr<uint8_t, GmDeleter>;

uint16_t Encode(float value, Dtype dtype)
{
    if (dtype == Dtype::BFLOAT16) {
        uint32_t bits;
        std::memcpy(&bits, &value, sizeof(bits));
        bits += 0x7FFFU + ((bits >> 16) & 1U);
        return static_cast<uint16_t>(bits >> 16);
    }
    half value16 = static_cast<half>(value);
    uint16_t bits;
    std::memcpy(&bits, &value16, sizeof(bits));
    return bits;
}

float Decode(uint16_t bits, Dtype dtype)
{
    if (dtype == Dtype::BFLOAT16) {
        uint32_t bits32 = static_cast<uint32_t>(bits) << 16;
        float value;
        std::memcpy(&value, &bits32, sizeof(value));
        return value;
    }
    half value16;
    std::memcpy(&value16, &bits, sizeof(bits));
    return static_cast<float>(value16);
}

float Golden(float x)
{
    return static_cast<float>(static_cast<double>(x) * std::clamp(static_cast<double>(x) + 3.0, 0.0, 6.0) / 6.0);
}
} // namespace

class HardSwishV2RegressionTest : public testing::TestWithParam<Dtype> {
protected:
    void Run(int64_t count, int64_t tile, int64_t blockNum, bool nearBoundary = false, bool sweep = false)
    {
        Dtype dtype = GetParam();
        bool fp32 = dtype == Dtype::FLOAT;
        int64_t width = fp32 ? sizeof(float) : sizeof(uint16_t);
        int64_t align = 32 / width;
        int64_t storage = ((count + align - 1) / align) * align;
        int64_t storageBlocks = storage / align;
        ASSERT_GT(count, 0);
        ASSERT_EQ(tile % align, 0);
        ASSERT_LE(blockNum, storageBlocks);
        size_t bytes = (storage + align) * width;
        GmBuffer x(static_cast<uint8_t*>(AscendC::GmAlloc(bytes)));
        GmBuffer out(static_cast<uint8_t*>(AscendC::GmAlloc(bytes)));
        GmBuffer tiling(static_cast<uint8_t*>(AscendC::GmAlloc(sizeof(HardSwishV2TilingData))));
        ASSERT_NE(x, nullptr);
        ASSERT_NE(out, nullptr);
        ASSERT_NE(tiling, nullptr);
        std::memset(x.get(), 0, bytes);
        std::memset(out.get(), 0x5A, bytes);
        const std::array<float, 16> values = {-8.0f, -4.0f, -3.0f, -2.0f, -1.0f, -0.5f, -0.0f, 0.0f,
                                              0.5f,  1.0f,  2.0f,  3.0f,  4.0f,  6.0f,  8.0f,  16.0f};
        const std::array<float, 16> around = {
            -3.03125f,  -3.015625f, -3.001953125f, -3.0f,        -2.998046875f, -2.984375f,   -2.96875f, -0.0078125f,
            0.0078125f, 2.96875f,   2.984375f,     2.998046875f, 3.0f,          3.001953125f, 3.015625f, 3.03125f};
        auto* x32 = reinterpret_cast<float*>(x.get());
        auto* x16 = reinterpret_cast<uint16_t*>(x.get());
        for (int64_t i = 0; i < storage; ++i) {
            float value = sweep ? (static_cast<float>((i * 977) % 20001) - 10000.0f) / 256.0f :
                                  (nearBoundary ? around[i % around.size()] : values[i % values.size()]);
            if (fp32) {
                x32[i] = value;
            } else {
                x16[i] = Encode(value, dtype);
            }
        }
        HardSwishV2TilingData t{};
        t.smallCoreDataNum = (storageBlocks / blockNum) * align;
        t.bigCoreDataNum = t.smallCoreDataNum + align;
        t.tailBlockNum = storageBlocks % blockNum;
        t.tileDataNum = tile;
        t.finalSmallTileNum = (t.smallCoreDataNum + tile - 1) / tile;
        t.finalBigTileNum = (t.bigCoreDataNum + tile - 1) / tile;
        t.smallTailDataNum = t.smallCoreDataNum % tile ? t.smallCoreDataNum % tile : tile;
        t.bigTailDataNum = t.bigCoreDataNum % tile ? t.bigCoreDataNum % tile : tile;
        std::memcpy(tiling.get(), &t, sizeof(t));
        AscendC::SetKernelMode(KernelMode::AIV_MODE);
        if (fp32) {
            ICPU_RUN_KF(hard_swish_v2_fp32_ut, blockNum, x.get(), out.get(), tiling.get());
        } else if (dtype == Dtype::HALF) {
            ICPU_RUN_KF(hard_swish_v2_fp16_ut, blockNum, x.get(), out.get(), tiling.get());
        } else {
            ICPU_RUN_KF(hard_swish_v2_bf16_ut, blockNum, x.get(), out.get(), tiling.get());
        }
        auto* out32 = reinterpret_cast<float*>(out.get());
        auto* out16 = reinterpret_cast<uint16_t*>(out.get());
        for (int64_t i = 0; i < count; ++i) {
            SCOPED_TRACE(i);
            float input = fp32 ? x32[i] : Decode(x16[i], dtype);
            float expected = Golden(input);
            if (!fp32) {
                expected = Decode(Encode(expected, dtype), dtype);
            }
            float actual = fp32 ? out32[i] : Decode(out16[i], dtype);
            ASSERT_TRUE(std::isfinite(actual));
            float tolerance = fp32 ? 1e-5f : (dtype == Dtype::BFLOAT16 ? 0.008f : 0.0015f);
            EXPECT_NEAR(actual, expected, std::max(1e-6f, tolerance * std::fabs(expected)));
        }
        // The host tiler rounds storage to 32 bytes, so check after that rounded boundary.
        for (size_t i = storage * width; i < bytes; ++i) {
            EXPECT_EQ(out.get()[i], 0x5A) << "output guard byte " << i;
        }
    }
};

TEST_P(HardSwishV2RegressionTest, FiniteValuesAcrossActivationRegions) { Run(64, 64, 1); }
TEST_P(HardSwishV2RegressionTest, ValuesAroundMinusThreeAndThree) { Run(48, 64, 1, true); }
TEST_P(HardSwishV2RegressionTest, ShortAlignedTile) { Run(16, 16, 1); }
TEST_P(HardSwishV2RegressionTest, MultipleTiles) { Run(256, 64, 1); }
TEST_P(HardSwishV2RegressionTest, UnequalCoreSplitsAndFinalTile) { Run(272, 32, 3); }
TEST_P(HardSwishV2RegressionTest, UnalignedLogicalLengthWithPaddedStorage) { Run(257, 64, 3); }
TEST_P(HardSwishV2RegressionTest, LargeFiniteSweep) { Run(4097, 256, 4, false, true); }

INSTANTIATE_TEST_SUITE_P(AllDtypes, HardSwishV2RegressionTest,
                         testing::Values(Dtype::FLOAT, Dtype::HALF, Dtype::BFLOAT16));
