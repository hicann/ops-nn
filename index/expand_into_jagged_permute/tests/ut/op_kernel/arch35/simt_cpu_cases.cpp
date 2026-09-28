/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Execute the production SIMT functions serially to verify indexing and branch behavior.
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>
#include "expand_into_jagged_permute_simt.h"

namespace {
constexpr int32_t SENTINEL = -777;
constexpr int64_t TEST_CORE_COUNT = 4;
constexpr int64_t MIN_CORE_OUTPUT = 1024;
constexpr int64_t WARP_SIZE = 32;
constexpr size_t GUARD_ELEMENTS = 16;
constexpr uint32_t RANDOM_SEED = 8583;
constexpr int RANDOM_CASE_COUNT = 100;
constexpr int MAX_SEGMENTS = 100;
constexpr int MAX_SEGMENT_LENGTH = 500;
constexpr int64_t STRIDED_OUTPUT = 3;

void RunCase(std::vector<int32_t> permute, std::vector<int32_t> inputOffsets, std::vector<int32_t> outputOffsets,
             int64_t stride)
{
    const int64_t outputSize = outputOffsets.back();
    const int64_t segments = static_cast<int64_t>(permute.size());
    const int64_t ideal = (outputSize + TEST_CORE_COUNT - 1) / TEST_CORE_COUNT;
    const int64_t perCore = std::max(MIN_CORE_OUTPUT, (ideal + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE);
    const uint32_t cores = static_cast<uint32_t>((outputSize + perCore - 1) / perCore);
    std::vector<int32_t> actual(outputSize * stride + GUARD_ELEMENTS, SENTINEL);
    std::vector<int32_t> expected(actual.size(), SENTINEL);
    for (int64_t segment = 0; segment < segments; ++segment) {
        for (int64_t pos = outputOffsets[segment]; pos < outputOffsets[segment + 1]; ++pos) {
            expected[pos * stride] = inputOffsets[permute[segment]] + pos - outputOffsets[segment];
        }
    }
    ExpandIntoJaggedPermuteTilingData tiling{segments, outputSize, perCore, stride, cores, 0};
    for (uint32_t core = 0; core < cores; ++core) {
        AscendC::blockIdx.x = core;
        NsExpandIntoJaggedPermute::Process<int32_t>(
            reinterpret_cast<GM_ADDR>(permute.data()), reinterpret_cast<GM_ADDR>(inputOffsets.data()),
            reinterpret_cast<GM_ADDR>(outputOffsets.data()), reinterpret_cast<GM_ADDR>(actual.data()), &tiling);
    }
    // Also checks untouched stride padding and the trailing guard region.
    assert(actual == expected);
}
void CheckOverflowRejected(int64_t stride)
{
    // A three-element range starting at INT32_MAX - 1 cannot fit in INT32 output.
    std::vector<int32_t> permute{1, 0};
    std::vector<int32_t> inputOffsets{0, INT32_MAX - 1, INT32_MAX};
    std::vector<int32_t> outputOffsets{0, 3, 6};
    constexpr int64_t OUTPUT_SIZE = 6;
    constexpr int64_t OVERFLOW_SEGMENT_LENGTH = 3;
    std::vector<int32_t> output(OUTPUT_SIZE * stride + GUARD_ELEMENTS, SENTINEL);
    ExpandIntoJaggedPermuteTilingData tiling{2, OUTPUT_SIZE, MIN_CORE_OUTPUT, stride, 1, 0};
    AscendC::blockIdx.x = 0;
    NsExpandIntoJaggedPermute::Process<int32_t>(
        reinterpret_cast<GM_ADDR>(permute.data()), reinterpret_cast<GM_ADDR>(inputOffsets.data()),
        reinterpret_cast<GM_ADDR>(outputOffsets.data()), reinterpret_cast<GM_ADDR>(output.data()), &tiling);
    for (int64_t pos = 0; pos < OVERFLOW_SEGMENT_LENGTH * stride; ++pos) {
        assert(output[pos] == SENTINEL);
    }
    for (size_t pos = OUTPUT_SIZE * stride; pos < output.size(); ++pos) {
        assert(output[pos] == SENTINEL);
    }
}
} // namespace

int main()
{
    using NsExpandIntoJaggedPermute::ShouldUseOutputCentric;
    constexpr int32_t LARGE_OUTPUT = 8192;
    constexpr int32_t HALF_OUTPUT = LARGE_OUTPUT / 2;
    constexpr int32_t INT32_MAX_VALUE = INT32_MAX;
    assert(ShouldUseOutputCentric(2, 4, 1));
    assert(!ShouldUseOutputCentric(2, LARGE_OUTPUT, TEST_CORE_COUNT));
    for (int64_t stride : {int64_t{1}, STRIDED_OUTPUT}) {
        // Target segments may be longer than their corresponding source segments.
        RunCase({0, 1}, {0, 1, 4}, {0, 2, 4}, stride);
        // Force segment-centric execution, including boundaries spanning multiple cores.
        RunCase({0, 1}, {0, 1, LARGE_OUTPUT}, {0, HALF_OUTPUT, LARGE_OUTPUT}, stride);
        // The final generated index may equal INT32_MAX without overflowing.
        RunCase({1, 0}, {0, INT32_MAX_VALUE - 1, INT32_MAX_VALUE}, {0, 2, 4}, stride);
        CheckOverflowRejected(stride);
    }
    std::mt19937 random(RANDOM_SEED);
    for (int trial = 0; trial < RANDOM_CASE_COUNT; ++trial) {
        const int segments = 1 + random() % MAX_SEGMENTS;
        std::vector<int32_t> permute(segments), inputOffsets(segments + 1), outputOffsets(segments + 1);
        std::iota(permute.begin(), permute.end(), 0);
        std::shuffle(permute.begin(), permute.end(), random);
        for (int i = 0; i < segments; ++i) {
            inputOffsets[i + 1] = inputOffsets[i] + 1 + random() % MAX_SEGMENT_LENGTH;
        }
        for (int i = 0; i < segments; ++i) {
            outputOffsets[i + 1] = outputOffsets[i] + inputOffsets[permute[i] + 1] - inputOffsets[permute[i]];
        }
        RunCase(permute, inputOffsets, outputOffsets, 1);
        RunCase(permute, inputOffsets, outputOffsets, STRIDED_OUTPUT);
    }
    std::cout << "208 SIMT CPU regression cases passed (not device execution)\n";
}
