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
 * \file fast_hadamard_dynamic_quant.cpp
 * \brief FHT + per-row dynamic int4 quant, exposed as
 *        torch.ops.ascend_ops_nn.fast_hadamard_dynamic_quant.
 *
 * Ecosystem "最简算子" delivery: one TU with the pto-ISA device kernel (verbatim, templated on
 * the compile-time row width kFullN; int4 packing via int4_cvt.hpp), host launch, and torch
 * registration. The entry dispatches on fullN and is launched with <<<>>> on the current stream.
 */

#include "experimental/matmul/common/fast_hadamard/op_includes.hpp"
#include "experimental/matmul/common/fast_hadamard/fht_kernel_common.hpp"

namespace ascend_ops {
namespace FastHadamardDynamicQuant {
using namespace pto;
namespace Common = ascend_ops::FastHadamardCommon;
// int4_cvt stays nested so its `namespace fast_hadamard_int4` is per-op isolated.
#include "experimental/matmul/common/fast_hadamard/int4_cvt.hpp"

// Keep some slack below PTO's TMP_UB_OFFSET (184KB) while increasing the
// batched x tile a bit to improve samples_per_load for smaller block widths.
constexpr uint32_t X_BUFFER_BYTES = 32 * 1024;
constexpr uint32_t UB_HALF_BYTES = X_BUFFER_BYTES / 2;
constexpr uint32_t ELEMENTS_PER_TILE = X_BUFFER_BYTES / sizeof(half);
constexpr uint32_t Y_BUFFER_BYTES = ELEMENTS_PER_TILE / 2;
constexpr uint32_t UB_USABLE_BYTES = 184 * 1024;
// The public dynamic fused wrapper routes NPU execution only for n >= 64, so
// the largest reachable batched row count per load is 16384 / 64 = 256.
constexpr uint32_t MAX_DYNAMIC_SAMPLES_PER_LOAD = ELEMENTS_PER_TILE / 64;
constexpr unsigned X_PING = 0x00000;
constexpr unsigned Y_PING = X_PING + X_BUFFER_BYTES + 0x100;
constexpr unsigned X_PONG = Y_PING + Y_BUFFER_BYTES + 0x100;
constexpr unsigned Y_PONG = X_PONG + X_BUFFER_BYTES + 0x100;
constexpr unsigned EVEN_BASE = Y_PONG + Y_BUFFER_BYTES + 0x100;
constexpr unsigned ODD_BASE = EVEN_BASE + UB_HALF_BYTES + 0x100;
constexpr unsigned SCALE_BASE = ODD_BASE + UB_HALF_BYTES + 0x100;
constexpr unsigned REDUCE_TMP_BASE = SCALE_BASE + MAX_DYNAMIC_SAMPLES_PER_LOAD * sizeof(float) + 0x100;
constexpr unsigned ROWMAX_BASE = REDUCE_TMP_BASE + X_BUFFER_BYTES + 0x100;
constexpr unsigned ROWMIN_BASE = ROWMAX_BASE + MAX_DYNAMIC_SAMPLES_PER_LOAD * sizeof(half) + 0x100;
static_assert(ODD_BASE + UB_HALF_BYTES <= UB_USABLE_BYTES, "Fused Hadamard+quantize UB layout exceeds usable UB.");
static_assert(SCALE_BASE + MAX_DYNAMIC_SAMPLES_PER_LOAD * sizeof(float) <= UB_USABLE_BYTES,
              "Dynamic quant scale UB layout exceeds usable UB.");
static_assert(REDUCE_TMP_BASE + X_BUFFER_BYTES <= UB_USABLE_BYTES,
              "Dynamic quant reduce-temp UB layout exceeds usable UB.");
static_assert(ROWMAX_BASE + MAX_DYNAMIC_SAMPLES_PER_LOAD * sizeof(half) <= UB_USABLE_BYTES,
              "Dynamic quant row-max UB layout exceeds usable UB.");
static_assert(ROWMIN_BASE + MAX_DYNAMIC_SAMPLES_PER_LOAD * sizeof(half) <= UB_USABLE_BYTES,
              "Dynamic quant row-min UB layout exceeds usable UB.");

#define FAST_HADAMARD_BATCHED_CASES(X) \
    X(64, 6)                           \
    X(128, 7)                          \
    X(256, 8)                          \
    X(512, 9)                          \
    X(1024, 10)                        \
    X(2048, 11)                        \
    X(4096, 12)                        \
    X(8192, 13)                        \
    X(16384, 14)

namespace {

template <typename InputT, typename BulkTile, typename TmpTile, typename ColTile, typename RowTile>
AICORE void ReduceAbsMax(BulkTile& x_tile, TmpTile& tmp, ColTile& row_max, ColTile& row_min, RowTile& row_max_rm,
                         RowTile& row_min_rm)
{
    TROWMAX(row_max, x_tile, tmp);
    TROWMIN(row_min, x_tile, tmp);
    pipe_barrier(PIPE_V);
    TRESHAPE(row_max_rm, row_max);
    TRESHAPE(row_min_rm, row_min);
    pipe_barrier(PIPE_V);
    TMULS(row_min_rm, row_min_rm, (InputT)-1.0f);
    pipe_barrier(PIPE_V);
    TMAX(row_max_rm, row_max_rm, row_min_rm);
    pipe_barrier(PIPE_V);
}

template <typename InputT, typename BulkTile, typename ScaleTile, typename ColTile, typename RowTile>
AICORE void NormalizeForInt4(BulkTile& x_tile, ScaleTile& scale_tile, ScaleTile& scale_floor, ColTile& row_max,
                             RowTile& row_max_rm, RowTile& scratch, float inv_sqrt_hadamard_n)
{
    constexpr float scale_divisor = 7.0f;
    TCVT(scale_tile, row_max_rm, RoundMode::CAST_RINT);
    pipe_barrier(PIPE_V);
    TMULS(scale_tile, scale_tile, inv_sqrt_hadamard_n / scale_divisor);
    pipe_barrier(PIPE_V);
    TMULS(scratch, row_max_rm, (InputT)0.0f);
    pipe_barrier(PIPE_V);
    TADDS(scratch, scratch, (InputT)1e-6f);
    pipe_barrier(PIPE_V);
    TMAX(row_max_rm, row_max_rm, scratch);
    pipe_barrier(PIPE_V);
    TRESHAPE(row_max, row_max_rm);
    pipe_barrier(PIPE_V);
    TROWEXPANDDIV(x_tile, x_tile, row_max);
    pipe_barrier(PIPE_V);
    TMULS(x_tile, x_tile, (InputT)scale_divisor);
    pipe_barrier(PIPE_V);
    TMULS(scale_floor, scale_tile, 0.0f);
    pipe_barrier(PIPE_V);
    TADDS(scale_floor, scale_floor, 1e-6f);
    pipe_barrier(PIPE_V);
    TMAX(scale_tile, scale_tile, scale_floor);
    pipe_barrier(PIPE_V);
}

template <typename OutputT, typename QuantTile, typename ScaleTile, typename OutGlobal, typename ScaleGlobal>
AICORE void StoreDynamicTile(QuantTile& y_tile, ScaleTile& scale_tile, OutGlobal& y_global, ScaleGlobal& scale_global,
                             event_t ev)
{
    set_flag(PIPE_V, PIPE_MTE3, ev);
    wait_flag(PIPE_V, PIPE_MTE3, ev);
    TSTORE(y_global, y_tile);
    TSTORE(scale_global, scale_tile);
    set_flag(PIPE_MTE3, PIPE_V, ev);
    set_flag(PIPE_V, PIPE_MTE2, ev);
}

template <typename TmpTile, typename ColTile, typename RowTile, typename ScaleTile>
AICORE void AssignReduceScratch(TmpTile& tmp, ColTile& row_max, ColTile& row_min, RowTile& row_max_rm,
                                RowTile& row_min_rm, ScaleTile& scale_floor)
{
    TASSIGN(tmp, REDUCE_TMP_BASE);
    TASSIGN(row_max, ROWMAX_BASE);
    TASSIGN(row_min, ROWMIN_BASE);
    TASSIGN(row_max_rm, EVEN_BASE);
    TASSIGN(row_min_rm, ODD_BASE);
    TASSIGN(scale_floor, REDUCE_TMP_BASE);
}

template <typename InputT, typename OutputT, uint32_t kFullN>
AICORE void ProcessDynamicTile(__gm__ OutputT* y, __gm__ float* row_scales, const Common::TileWork& tile,
                               unsigned x_base, unsigned y_base, event_t ev, uint32_t hadamard_n,
                               uint32_t log2_hadamard_n, float inv_sqrt_hadamard_n)
{
    constexpr uint32_t samples_per_load = ELEMENTS_PER_TILE / kFullN;
    using StridDim5 = pto::Stride<1, 1, 1, 1, 1>;
    using OutGlobal = pto::GlobalTensor<OutputT, pto::Shape<1, 1, 1, 1, ELEMENTS_PER_TILE / 2>, StridDim5>;
    using ScaleGlobal = pto::GlobalTensor<float, pto::Shape<1, 1, 1, 1, MAX_DYNAMIC_SAMPLES_PER_LOAD>, StridDim5>;
    using QuantTile = Tile<TileType::Vec, OutputT, 1, ELEMENTS_PER_TILE / 2, BLayout::RowMajor, -1, -1>;
    using BulkTile = Tile<TileType::Vec, InputT, samples_per_load, kFullN, BLayout::RowMajor, -1, -1>;
    using BulkQuantTile = Tile<TileType::Vec, OutputT, samples_per_load, kFullN / 2, BLayout::RowMajor, -1, -1>;
    using TmpTile = Tile<TileType::Vec, InputT, samples_per_load, kFullN, BLayout::RowMajor, -1, -1>;
    using ColTile = Tile<TileType::Vec, InputT, MAX_DYNAMIC_SAMPLES_PER_LOAD, 1, BLayout::ColMajor, -1, -1>;
    using RowTile = Tile<TileType::Vec, InputT, 1, MAX_DYNAMIC_SAMPLES_PER_LOAD, BLayout::RowMajor, -1, -1>;
    using ScaleTile = Tile<TileType::Vec, float, 1, MAX_DYNAMIC_SAMPLES_PER_LOAD, BLayout::RowMajor, -1, -1>;
    QuantTile yTile(1, tile.elements >> 1);
    BulkTile xTile(tile.sample_count, kFullN);
    BulkQuantTile yTile2D(tile.sample_count, kFullN >> 1);
    ScaleTile scaleTile(1, tile.sample_count);
    TASSIGN(yTile, y_base);
    TASSIGN(xTile, x_base);
    TASSIGN(yTile2D, y_base);
    TASSIGN(scaleTile, SCALE_BASE);
    OutGlobal yGlobal(y + (tile.gm_offset >> 1));
    ScaleGlobal scaleGlobal(row_scales + tile.gm_offset / kFullN);
    TASSIGN(yGlobal, (y + (tile.gm_offset >> 1)));
    TASSIGN(scaleGlobal, (row_scales + tile.gm_offset / kFullN));
    wait_flag(PIPE_MTE3, PIPE_V, ev);
    Common::RunTileHadamardInPlace<InputT, ELEMENTS_PER_TILE, EVEN_BASE, ODD_BASE>(x_base, tile.sample_count, kFullN,
                                                                                   hadamard_n, log2_hadamard_n);
    pipe_barrier(PIPE_V);
    TmpTile tmp(tile.sample_count, kFullN);
    ColTile rowMax(tile.sample_count, 1);
    ColTile rowMin(tile.sample_count, 1);
    RowTile rowMaxRm(1, tile.sample_count);
    RowTile rowMinRm(1, tile.sample_count);
    ScaleTile scaleFloor(1, tile.sample_count);
    AssignReduceScratch(tmp, rowMax, rowMin, rowMaxRm, rowMinRm, scaleFloor);
    ReduceAbsMax<InputT>(xTile, tmp, rowMax, rowMin, rowMaxRm, rowMinRm);
    NormalizeForInt4<InputT>(xTile, scaleTile, scaleFloor, rowMax, rowMaxRm, rowMinRm, inv_sqrt_hadamard_n);
    fast_hadamard_int4::TCVT_FP16_TO_INT4_PACKED(yTile2D, xTile, RoundMode::CAST_RINT);
    pipe_barrier(PIPE_V);
    StoreDynamicTile<OutputT>(yTile, scaleTile, yGlobal, scaleGlobal, ev);
}

template <typename InputT, typename OutputT, uint32_t kFullN>
AICORE void RunDynamicTiles(__gm__ InputT* x, __gm__ OutputT* y, __gm__ float* row_scales, uint32_t sample_offset,
                            uint32_t samples_to_process, uint32_t hadamard_n, uint32_t log2_hadamard_n,
                            float inv_sqrt_hadamard_n)
{
    constexpr uint32_t samples_per_load = ELEMENTS_PER_TILE / kFullN;
    uint32_t sample_done = 0;
    Common::TileWork tile;
    if (!Common::NextTile(sample_done, sample_offset * kFullN, samples_to_process, samples_per_load, kFullN, tile)) {
        return;
    }
    bool ping = true;
    Common::IssueTLoad<InputT, ELEMENTS_PER_TILE>(x, tile, X_PING, EVENT_ID0);
    while (true) {
        const event_t ev = ping ? (event_t)EVENT_ID0 : (event_t)EVENT_ID1;
        wait_flag(PIPE_MTE2, PIPE_V, ev);
        Common::TileWork next_tile;
        const bool has_next = Common::NextTile(sample_done, sample_offset * kFullN, samples_to_process,
                                               samples_per_load, kFullN, next_tile);
        if (has_next) {
            Common::IssueTLoad<InputT, ELEMENTS_PER_TILE>(x, next_tile, ping ? X_PONG : X_PING,
                                                          ping ? (event_t)EVENT_ID1 : (event_t)EVENT_ID0);
        }
        ProcessDynamicTile<InputT, OutputT, kFullN>(y, row_scales, tile, ping ? X_PING : X_PONG, ping ? Y_PING : Y_PONG,
                                                    ev, hadamard_n, log2_hadamard_n, inv_sqrt_hadamard_n);
        if (!has_next) {
            break;
        }
        tile = next_tile;
        ping = !ping;
    }
}

// kFullN keeps PTO row-reduce/expand tile strides equal to the true row width.
template <typename InputT, typename OutputT, uint32_t kFullN>
AICORE void runTFastHadamardDynamicQuant(__gm__ InputT* x, __gm__ OutputT* y, __gm__ float* row_scales, uint32_t batch,
                                         uint32_t hadamard_n, uint32_t log2_hadamard_n, float inv_sqrt_hadamard_n,
                                         uint32_t num_cores, uint32_t vid)
{
    Common::InitVectorMask();
    if (hadamard_n == 0 || kFullN % hadamard_n != 0) {
        return;
    }
    uint32_t sample_offset = 0;
    uint32_t samples_to_process = 0;
    if (!Common::ResolveCoreWork(batch, num_cores, vid, sample_offset, samples_to_process)) {
        return;
    }
    Common::InitPipeEvents();
    RunDynamicTiles<InputT, OutputT, kFullN>(x, y, row_scales, sample_offset, samples_to_process, hadamard_n,
                                             log2_hadamard_n, inv_sqrt_hadamard_n);
    Common::DrainPipeEvents();
}

} // namespace

// ---- Kernel launch entry: dispatch on fullN to template the reduce/expand tiles on kFullN ----
__global__ __aicore__ void FastHadamardDynamicQuantKernel(__gm__ uint8_t* x, __gm__ uint8_t* y,
                                                          __gm__ uint8_t* row_scales, uint32_t batch, uint32_t fullN,
                                                          uint32_t hadamardN, uint32_t logHadamardN,
                                                          float invSqrtHadamardN)
{
#if defined(__DAV_VEC__)
    const uint32_t num_cores = get_block_num() * get_subblockdim();
    const uint32_t vid = get_block_idx() * get_subblockdim() + get_subblockid();
    switch (fullN) {
#define FAST_HADAMARD_DYNAMIC_DISPATCH_CASE(N, LOG2)                                                                 \
    case N:                                                                                                          \
        runTFastHadamardDynamicQuant<half, int8_t, N>((__gm__ half*)x, (__gm__ int8_t*)y, (__gm__ float*)row_scales, \
                                                      batch, hadamardN, logHadamardN, invSqrtHadamardN, num_cores,   \
                                                      vid);                                                          \
        break;
        FAST_HADAMARD_BATCHED_CASES(FAST_HADAMARD_DYNAMIC_DISPATCH_CASE)
#undef FAST_HADAMARD_DYNAMIC_DISPATCH_CASE
        default:
            break;
    }
#endif
}

void FastHadamardDynamicQuantKernelLaunch(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* row_scales,
                                          uint32_t batch, uint32_t fullN, uint32_t hadamardN, uint32_t logHadamardN,
                                          float invSqrtHadamardN)
{
    FastHadamardDynamicQuantKernel<<<blockDim, nullptr, stream>>>(x, y, row_scales, batch, fullN, hadamardN,
                                                                  logHadamardN, invSqrtHadamardN);
}

constexpr int64_t MIN_FULL_N = 64;
constexpr int64_t MAX_HADAMARD_N = 16384;
constexpr uint32_t AIV_NUM = 40; // vector cores on Atlas A2 (Ascend910B)

int64_t FastHadamardDynamicQuantNpu(torch::Tensor& x, int64_t hadamard_n, torch::Tensor& y, torch::Tensor& row_scales)
{
    TORCH_CHECK(torch_npu::utils::is_npu(x), "x must be on the NPU device");
    TORCH_CHECK(torch_npu::utils::is_npu(y), "out must be on the NPU device");
    TORCH_CHECK(torch_npu::utils::is_npu(row_scales), "row_scales must be on the NPU device");
    TORCH_CHECK(x.scalar_type() == at::kHalf, "x dtype must be float16");
    TORCH_CHECK(y.scalar_type() == at::kChar, "out dtype must be int8 (packed int4)");
    TORCH_CHECK(row_scales.scalar_type() == at::kFloat, "row_scales dtype must be float32");
    TORCH_CHECK(x.is_contiguous() && y.is_contiguous() && row_scales.is_contiguous(),
                "x, out, row_scales must be contiguous");
    TORCH_CHECK(x.dim() >= 1, "x rank must be >= 1");

    const int64_t fullN = x.size(x.dim() - 1);
    const int64_t totalElems = x.numel();
    TORCH_CHECK(fullN > 0, "last dim fullN must be positive");
    TORCH_CHECK(fullN >= MIN_FULL_N && fullN <= MAX_HADAMARD_N && (fullN & (fullN - 1)) == 0, "last dim fullN=", fullN,
                " must be a power of two in [", MIN_FULL_N, ", ", MAX_HADAMARD_N, "]");
    TORCH_CHECK(totalElems % fullN == 0, "total elements not divisible by fullN");

    int64_t hadamardN = (hadamard_n > 0) ? hadamard_n : fullN;
    TORCH_CHECK(hadamardN > 1 && hadamardN <= MAX_HADAMARD_N && (hadamardN & (hadamardN - 1)) == 0,
                "hadamard_n=", hadamardN, " must be a power of two in (2, ", MAX_HADAMARD_N, "]");
    TORCH_CHECK(fullN % hadamardN == 0, "fullN must be a multiple of hadamard_n");

    uint32_t logHadamardN = 0;
    for (int64_t v = hadamardN; v > 1; v >>= 1) {
        ++logHadamardN;
    }
    const uint32_t batch = static_cast<uint32_t>(totalElems / fullN);
    const float invSqrtHadamardN = 1.0f / std::sqrt(static_cast<float>(hadamardN));

    uint32_t blockDim = (batch < AIV_NUM) ? batch : AIV_NUM;
    if (blockDim == 0) {
        blockDim = 1;
    }

    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    at_npu::native::OpCommand::RunOpApi("fast_hadamard_dynamic_quant", [=]() -> int {
        FastHadamardDynamicQuantKernelLaunch(blockDim, stream, (uint8_t*)x.data_ptr(), (uint8_t*)y.data_ptr(),
                                             (uint8_t*)row_scales.data_ptr(), batch, static_cast<uint32_t>(fullN),
                                             static_cast<uint32_t>(hadamardN), logHadamardN, invSqrtHadamardN);
        return 0;
    });
    return 0;
}

TORCH_LIBRARY_FRAGMENT(EXTENSION_MODULE_NAME, m)
{
    m.def("fast_hadamard_dynamic_quant(Tensor x, int hadamard_n, Tensor out, Tensor row_scales) -> "
          "int");
}

TORCH_LIBRARY_IMPL(EXTENSION_MODULE_NAME, PrivateUse1, m)
{
    m.impl("fast_hadamard_dynamic_quant", FastHadamardDynamicQuantNpu);
}

} // namespace FastHadamardDynamicQuant
} // namespace ascend_ops
