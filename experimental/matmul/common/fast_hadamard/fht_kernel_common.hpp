/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FAST_HADAMARD_COMMON_FHT_KERNEL_COMMON_HPP
#define FAST_HADAMARD_COMMON_FHT_KERNEL_COMMON_HPP

namespace ascend_ops {
namespace FastHadamardCommon {
using namespace pto;

struct TileWork {
    uint32_t gm_offset, sample_count, elements;
};

PTO_INTERNAL void InitVectorMask()
{
    set_mask_norm();
    set_vector_mask(-1, -1);
}

PTO_INTERNAL void InitPipeEvents()
{
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
}

PTO_INTERNAL void DrainPipeEvents()
{
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
}

PTO_INTERNAL bool ResolveCoreWork(uint32_t batch, uint32_t num_cores, uint32_t vid, uint32_t& sample_offset,
                                  uint32_t& samples_to_process)
{
    if (num_cores == 0) {
        return false;
    }
    const uint32_t samples_per_core = (batch + num_cores - 1) / num_cores;
    sample_offset = samples_per_core * vid;
    if (sample_offset >= batch) {
        return false;
    }
    samples_to_process = samples_per_core;
    if (sample_offset + samples_to_process > batch) {
        samples_to_process = batch - sample_offset;
    }
    return samples_to_process != 0;
}

PTO_INTERNAL bool NextTile(uint32_t& sample_done, uint32_t gm_offset_base, uint32_t samples_to_process,
                           uint32_t samples_per_load, uint32_t n, TileWork& tile)
{
    if (sample_done >= samples_to_process) {
        return false;
    }
    tile.sample_count = min(samples_per_load, samples_to_process - sample_done);
    tile.elements = tile.sample_count * n;
    tile.gm_offset = gm_offset_base + sample_done * n;
    sample_done += tile.sample_count;
    return true;
}

template <typename InputT, uint32_t kElementsPerTile, unsigned kEvenBase, unsigned kOddBase, uint32_t kN,
          uint32_t kLog2N>
AICORE void RunBatchedHadamardInPlace(unsigned x_base, uint32_t sample_count)
{
    constexpr uint32_t kNHalf = kN >> 1;
    constexpr uint32_t kSamplesPerLoad = kElementsPerTile / kN;
    using FullTile = Tile<TileType::Vec, InputT, kSamplesPerLoad, kN, BLayout::RowMajor, DYNAMIC, kN>;
    using HalfTile = Tile<TileType::Vec, InputT, kSamplesPerLoad, kNHalf, BLayout::RowMajor, DYNAMIC, kNHalf>;
    using RowHalfTile = Tile<TileType::Vec, InputT, 1, kNHalf, BLayout::RowMajor, 1, kNHalf>;
    FullTile xBulkTile(sample_count);
    HalfTile evenTile(sample_count);
    HalfTile oddTile(sample_count);
    TASSIGN(xBulkTile, x_base);
    TASSIGN(evenTile, kEvenBase);
    TASSIGN(oddTile, kOddBase);
    for (uint32_t iter_m = 0; iter_m < kLog2N; ++iter_m) {
        TGATHER<HalfTile, FullTile, MaskPattern::P0101>(evenTile, xBulkTile);
        TGATHER<HalfTile, FullTile, MaskPattern::P1010>(oddTile, xBulkTile);
        pipe_barrier(PIPE_V);
        for (uint32_t s = 0; s < sample_count; ++s) {
            const unsigned row_base = x_base + s * kN * sizeof(InputT);
            RowHalfTile evenRow;
            RowHalfTile oddRow;
            RowHalfTile xFirstHalf;
            RowHalfTile xSecondHalf;
            TASSIGN(evenRow, kEvenBase + s * kNHalf * sizeof(InputT));
            TASSIGN(oddRow, kOddBase + s * kNHalf * sizeof(InputT));
            TASSIGN(xFirstHalf, row_base);
            TASSIGN(xSecondHalf, row_base + kNHalf * sizeof(InputT));
            TADD(xFirstHalf, evenRow, oddRow);
            TSUB(xSecondHalf, evenRow, oddRow);
        }
        pipe_barrier(PIPE_V);
    }
}

template <typename InputT, uint32_t kElementsPerTile>
AICORE void IssueTLoad(__gm__ InputT* x, const TileWork& tile, unsigned x_base, event_t ev)
{
    using ShapeDim5 = pto::Shape<1, 1, 1, 1, kElementsPerTile>;
    using StridDim5 = pto::Stride<1, 1, 1, 1, 1>;
    using InGlobal = pto::GlobalTensor<InputT, ShapeDim5, StridDim5>;
    using FullTile = Tile<TileType::Vec, InputT, 1, kElementsPerTile, BLayout::RowMajor, -1, -1>;
    FullTile xBulkTile(1, tile.elements);
    TASSIGN(xBulkTile, x_base);
    InGlobal xGlobal(x + tile.gm_offset);
    TASSIGN(xGlobal, (x + tile.gm_offset));
    wait_flag(PIPE_V, PIPE_MTE2, ev);
    TLOAD(xBulkTile, xGlobal);
    set_flag(PIPE_MTE2, PIPE_V, ev);
}

template <typename InputT, uint32_t kElementsPerTile, unsigned kEvenBase, unsigned kOddBase>
AICORE bool TryRunBatchedHadamard(unsigned x_base, uint32_t sample_count, uint32_t n, uint32_t log2_n)
{
    switch (n) {
#define FAST_HADAMARD_COMMON_DISPATCH_CASE(N, LOG2)                                                                  \
    case N:                                                                                                          \
        if (log2_n == LOG2) {                                                                                        \
            RunBatchedHadamardInPlace<InputT, kElementsPerTile, kEvenBase, kOddBase, N, LOG2>(x_base, sample_count); \
            return true;                                                                                             \
        }                                                                                                            \
        break;
        FAST_HADAMARD_COMMON_DISPATCH_CASE(64, 6)
        FAST_HADAMARD_COMMON_DISPATCH_CASE(128, 7)
        FAST_HADAMARD_COMMON_DISPATCH_CASE(256, 8)
        FAST_HADAMARD_COMMON_DISPATCH_CASE(512, 9)
        FAST_HADAMARD_COMMON_DISPATCH_CASE(1024, 10)
        FAST_HADAMARD_COMMON_DISPATCH_CASE(2048, 11)
        FAST_HADAMARD_COMMON_DISPATCH_CASE(4096, 12)
        FAST_HADAMARD_COMMON_DISPATCH_CASE(8192, 13)
        FAST_HADAMARD_COMMON_DISPATCH_CASE(16384, 14)
#undef FAST_HADAMARD_COMMON_DISPATCH_CASE
        default:
            break;
    }
    return false;
}

template <typename InputT, uint32_t kElementsPerTile, unsigned kEvenBase, unsigned kOddBase>
AICORE void RunSingleHadamardRow(unsigned row_base, uint32_t n, uint32_t log2_n)
{
    const uint32_t n_half = n >> 1;
    using FullTile = Tile<TileType::Vec, InputT, 1, kElementsPerTile, BLayout::RowMajor, -1, -1>;
    using HalfTile = Tile<TileType::Vec, InputT, 1, kElementsPerTile / 2, BLayout::RowMajor, -1, -1>;
    FullTile xRowTile(1, n);
    HalfTile xFirstHalf(1, n_half);
    HalfTile xSecondHalf(1, n_half);
    HalfTile evenTile(1, n_half);
    HalfTile oddTile(1, n_half);
    TASSIGN(xRowTile, row_base);
    TASSIGN(xFirstHalf, row_base);
    TASSIGN(xSecondHalf, row_base + n_half * sizeof(InputT));
    TASSIGN(evenTile, kEvenBase);
    TASSIGN(oddTile, kOddBase);
    for (uint32_t iter_m = 0; iter_m < log2_n; ++iter_m) {
        TGATHER<HalfTile, FullTile, MaskPattern::P0101>(evenTile, xRowTile);
        TGATHER<HalfTile, FullTile, MaskPattern::P1010>(oddTile, xRowTile);
        pipe_barrier(PIPE_V);
        TADD(xFirstHalf, evenTile, oddTile);
        TSUB(xSecondHalf, evenTile, oddTile);
        pipe_barrier(PIPE_V);
    }
}

template <typename InputT, uint32_t kElementsPerTile, unsigned kEvenBase, unsigned kOddBase>
AICORE void RunTileHadamardInPlace(unsigned tile_base, uint32_t sample_count, uint32_t full_n, uint32_t hadamard_n,
                                   uint32_t log2_hadamard_n)
{
    if (full_n == hadamard_n && TryRunBatchedHadamard<InputT, kElementsPerTile, kEvenBase, kOddBase>(
                                    tile_base, sample_count, full_n, log2_hadamard_n)) {
        return;
    }
    const uint32_t num_blocks = full_n / hadamard_n;
    for (uint32_t s = 0; s < sample_count; ++s) {
        const unsigned row_base = tile_base + s * full_n * sizeof(InputT);
        for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
            const unsigned block_base = row_base + block_idx * hadamard_n * sizeof(InputT);
            if (!TryRunBatchedHadamard<InputT, kElementsPerTile, kEvenBase, kOddBase>(block_base, 1, hadamard_n,
                                                                                      log2_hadamard_n)) {
                RunSingleHadamardRow<InputT, kElementsPerTile, kEvenBase, kOddBase>(block_base, hadamard_n,
                                                                                    log2_hadamard_n);
            }
        }
    }
}

} // namespace FastHadamardCommon
} // namespace ascend_ops

#endif // FAST_HADAMARD_COMMON_FHT_KERNEL_COMMON_HPP
