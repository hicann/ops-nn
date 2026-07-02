/*
 * Copyright (c) 2026 联通（广东）产业互联网有限公司.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <array>
#include <cstddef>
#include <cstdint>
#include "securec.h"
#include "gtest/gtest.h"
#include "matmul_add_tiling_def.h"

class MatmulAddKernelTest : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

TEST_F(MatmulAddKernelTest, tiling_struct_layout)
{
    optiling::MatmulAddTilingData td{};
    td.M = 128;
    td.N = 256;
    td.K = 64;
    td.has_bias = true;

    EXPECT_EQ(td.M, 128u);
    EXPECT_EQ(td.N, 256u);
    EXPECT_EQ(td.K, 64u);
    EXPECT_TRUE(td.has_bias);
}

TEST_F(MatmulAddKernelTest, tiling_no_bias)
{
    optiling::MatmulAddTilingData td{};
    td.M = 32;
    td.N = 32;
    td.K = 32;
    td.has_bias = false;

    EXPECT_EQ(td.M, 32u);
    EXPECT_FALSE(td.has_bias);
}

TEST_F(MatmulAddKernelTest, full_tiling_buffer_round_trip)
{
    constexpr std::uint32_t kM = 128;
    constexpr std::uint32_t kN = 256;
    constexpr std::uint32_t kK = 64;
    constexpr std::uint32_t kHasBias = 1U;
    constexpr std::size_t kCubeOffset =
        offsetof(optiling::MatmulAddTilingData, cubeTiling);
    constexpr std::size_t kCubeSize =
        sizeof(optiling::MatmulAddTilingData) - kCubeOffset;

    EXPECT_EQ(kCubeOffset, 4U * sizeof(std::uint32_t));
    EXPECT_EQ(alignof(optiling::MatmulAddTilingData), 8U);

    alignas(optiling::MatmulAddTilingData)
    std::array<std::uint8_t, sizeof(optiling::MatmulAddTilingData)> buffer{};

    auto writeU32 = [&](std::size_t offset, const std::uint32_t& value) {
        EXPECT_EQ(memcpy_s(buffer.data() + offset, buffer.size() - offset,
            &value, sizeof(value)), EOK);
    };
    writeU32(offsetof(optiling::MatmulAddTilingData, M), kM);
    writeU32(offsetof(optiling::MatmulAddTilingData, N), kN);
    writeU32(offsetof(optiling::MatmulAddTilingData, K), kK);
    writeU32(offsetof(optiling::MatmulAddTilingData, has_bias), kHasBias);

    for (std::size_t i = 0; i < kCubeSize; ++i) {
        buffer[kCubeOffset + i] =
            static_cast<std::uint8_t>((i * 17U + 3U) & 0xFFU);
    }

    auto readU32 = [&](std::size_t offset) {
        std::uint32_t value = 0;
        EXPECT_EQ(memcpy_s(&value, sizeof(value),
            buffer.data() + offset, sizeof(value)), EOK);
        return value;
    };

    EXPECT_EQ(readU32(offsetof(optiling::MatmulAddTilingData, M)), kM);
    EXPECT_EQ(readU32(offsetof(optiling::MatmulAddTilingData, N)), kN);
    EXPECT_EQ(readU32(offsetof(optiling::MatmulAddTilingData, K)), kK);
    EXPECT_EQ(readU32(offsetof(optiling::MatmulAddTilingData, has_bias)), kHasBias);

    for (std::size_t i = 0; i < kCubeSize; ++i) {
        EXPECT_EQ(buffer[kCubeOffset + i],
            static_cast<std::uint8_t>((i * 17U + 3U) & 0xFFU));
    }
}

