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
 * \file fast_hadamard_quant.cpp
 * \brief FHT + grouped static int4 quant, exposed as torch.ops.ascend_ops_nn.fast_hadamard_quant.
 *
 * Ecosystem "最简算子" delivery: one TU with the pto-ISA device kernel (verbatim from the
 * standard-aclnn op_kernel, int4 packing via int4_cvt.hpp), the host launch, and the torch
 * registration. The kernel is launched with <<<>>> on torch_npu's current stream.
 */

#include "experimental/matmul/common/fast_hadamard/op_includes.hpp"
#include "experimental/matmul/common/fast_hadamard/fht_kernel_common.hpp"

namespace ascend_ops {
namespace FastHadamardQuant {
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
constexpr unsigned X_PING = 0x00000;
constexpr unsigned Y_PING = X_PING + X_BUFFER_BYTES + 0x100;
constexpr unsigned X_PONG = Y_PING + Y_BUFFER_BYTES + 0x100;
constexpr unsigned Y_PONG = X_PONG + X_BUFFER_BYTES + 0x100;
constexpr unsigned EVEN_BASE = Y_PONG + Y_BUFFER_BYTES + 0x100;
constexpr unsigned ODD_BASE = EVEN_BASE + UB_HALF_BYTES + 0x100;
static_assert(ODD_BASE + UB_HALF_BYTES <= UB_USABLE_BYTES, "Fused Hadamard+quantize UB layout exceeds usable UB.");

namespace {

template <typename InputT, typename FullTile, typename QuantTile>
AICORE void ApplyUniformQuant(FullTile& x_tile, QuantTile& y_tile, event_t ev, float scale, float q_offset)
{
    TMULS(x_tile, x_tile, (InputT)scale);
    pipe_barrier(PIPE_V);
    if (q_offset != 0.0f) {
        TADDS(x_tile, x_tile, (InputT)q_offset);
        pipe_barrier(PIPE_V);
    }
    wait_flag(PIPE_MTE3, PIPE_V, ev);
    fast_hadamard_int4::TCVT_FP16_TO_INT4_PACKED(y_tile, x_tile, RoundMode::CAST_NONE);
    pipe_barrier(PIPE_V);
}

template <typename InputT, typename OutputT>
AICORE void ApplyGroupedQuant(unsigned x_base, unsigned y_base, const Common::TileWork& tile, __gm__ InputT* scales,
                              __gm__ InputT* offsets, uint32_t scale_stride, uint32_t offset_stride, uint32_t n,
                              uint32_t group_size, float scale, float q_offset)
{
    using FullTile = Tile<TileType::Vec, InputT, 1, ELEMENTS_PER_TILE, BLayout::RowMajor, -1, -1>;
    using QuantTile = Tile<TileType::Vec, OutputT, 1, ELEMENTS_PER_TILE / 2, BLayout::RowMajor, -1, -1>;
    const uint32_t groups_per_row = n / group_size;
    const uint32_t packed_n = n >> 1;
    const uint32_t packed_group_size = group_size >> 1;
    const bool has_scales = scales != nullptr;
    const bool has_offsets = offsets != nullptr;
    for (uint32_t s = 0; s < tile.sample_count; ++s) {
        const uint32_t row_index = tile.gm_offset / n + s;
        const unsigned row_x_base = x_base + s * n * sizeof(InputT);
        const unsigned row_y_base = y_base + s * packed_n * sizeof(OutputT);
        for (uint32_t g = 0; g < groups_per_row; ++g) {
            FullTile xGroupTile(1, group_size);
            QuantTile yGroupTile(1, packed_group_size);
            TASSIGN(xGroupTile, row_x_base + g * group_size * sizeof(InputT));
            TASSIGN(yGroupTile, row_y_base + g * packed_group_size * sizeof(OutputT));
            InputT group_scale = has_scales ? scales[(scale_stride == 0) ? g : row_index * scale_stride + g] :
                                              (InputT)scale;
            TMULS(xGroupTile, xGroupTile, group_scale);
            pipe_barrier(PIPE_V);
            if (has_offsets || q_offset != 0.0f) {
                InputT group_offset = has_offsets ? offsets[(offset_stride == 0) ? g : row_index * offset_stride + g] :
                                                    (InputT)q_offset;
                TADDS(xGroupTile, xGroupTile, group_offset);
                pipe_barrier(PIPE_V);
            }
            fast_hadamard_int4::TCVT_FP16_TO_INT4_PACKED(yGroupTile, xGroupTile, RoundMode::CAST_NONE);
            pipe_barrier(PIPE_V);
        }
    }
}

template <typename InputT, typename OutputT>
AICORE void ProcessQuantTile(__gm__ OutputT* y, const Common::TileWork& tile, unsigned x_base, unsigned y_base,
                             event_t ev, __gm__ InputT* scales, __gm__ InputT* offsets, uint32_t scale_stride,
                             uint32_t offset_stride, uint32_t n, uint32_t log2_n, uint32_t group_size, float scale,
                             float q_offset)
{
    using OutShapeDim5 = pto::Shape<1, 1, 1, 1, ELEMENTS_PER_TILE / 2>;
    using StridDim5 = pto::Stride<1, 1, 1, 1, 1>;
    using OutGlobal = pto::GlobalTensor<OutputT, OutShapeDim5, StridDim5>;
    using FullTile = Tile<TileType::Vec, InputT, 1, ELEMENTS_PER_TILE, BLayout::RowMajor, -1, -1>;
    using QuantTile = Tile<TileType::Vec, OutputT, 1, ELEMENTS_PER_TILE / 2, BLayout::RowMajor, -1, -1>;
    FullTile xBulkTile(1, tile.elements);
    QuantTile yBulkTile(1, tile.elements >> 1);
    TASSIGN(xBulkTile, x_base);
    TASSIGN(yBulkTile, y_base);
    OutGlobal yGlobal(y + (tile.gm_offset >> 1));
    TASSIGN(yGlobal, (y + (tile.gm_offset >> 1)));
    Common::RunTileHadamardInPlace<InputT, ELEMENTS_PER_TILE, EVEN_BASE, ODD_BASE>(x_base, tile.sample_count, n, n,
                                                                                   log2_n);
    if (scales == nullptr && offsets == nullptr) {
        ApplyUniformQuant<InputT>(xBulkTile, yBulkTile, ev, scale, q_offset);
    } else {
        wait_flag(PIPE_MTE3, PIPE_V, ev);
        ApplyGroupedQuant<InputT, OutputT>(x_base, y_base, tile, scales, offsets, scale_stride, offset_stride, n,
                                           group_size, scale, q_offset);
    }
    set_flag(PIPE_V, PIPE_MTE3, ev);
    wait_flag(PIPE_V, PIPE_MTE3, ev);
    TSTORE(yGlobal, yBulkTile);
    set_flag(PIPE_MTE3, PIPE_V, ev);
    set_flag(PIPE_V, PIPE_MTE2, ev);
}

template <typename InputT, typename OutputT>
AICORE void RunQuantTiles(__gm__ InputT* x, __gm__ OutputT* y, __gm__ InputT* scales, __gm__ InputT* offsets,
                          uint32_t scale_stride, uint32_t offset_stride, uint32_t sample_offset,
                          uint32_t samples_to_process, uint32_t n, uint32_t log2_n, uint32_t group_size, float scale,
                          float q_offset)
{
    const uint32_t samples_per_load = (n < ELEMENTS_PER_TILE) ? ELEMENTS_PER_TILE / n : 1;
    uint32_t sample_done = 0;
    Common::TileWork tile;
    if (!Common::NextTile(sample_done, sample_offset * n, samples_to_process, samples_per_load, n, tile)) {
        return;
    }
    bool ping = true;
    Common::IssueTLoad<InputT, ELEMENTS_PER_TILE>(x, tile, X_PING, EVENT_ID0);
    while (true) {
        const event_t ev = ping ? (event_t)EVENT_ID0 : (event_t)EVENT_ID1;
        const unsigned x_base = ping ? X_PING : X_PONG;
        const unsigned y_base = ping ? Y_PING : Y_PONG;
        wait_flag(PIPE_MTE2, PIPE_V, ev);
        Common::TileWork next_tile;
        const bool has_next = Common::NextTile(sample_done, sample_offset * n, samples_to_process, samples_per_load, n,
                                               next_tile);
        if (has_next) {
            Common::IssueTLoad<InputT, ELEMENTS_PER_TILE>(x, next_tile, ping ? X_PONG : X_PING,
                                                          ping ? (event_t)EVENT_ID1 : (event_t)EVENT_ID0);
        }
        ProcessQuantTile(y, tile, x_base, y_base, ev, scales, offsets, scale_stride, offset_stride, n, log2_n,
                         group_size, scale, q_offset);
        if (!has_next) {
            break;
        }
        tile = next_tile;
        ping = !ping;
    }
}

template <typename InputT, typename OutputT>
AICORE void runTFastHadamardQuant(__gm__ InputT* x, __gm__ OutputT* y, __gm__ InputT* group_scales,
                                  __gm__ InputT* group_offsets, uint32_t scale_group_stride,
                                  uint32_t offset_group_stride, uint32_t batch, uint32_t n, uint32_t log2_n,
                                  uint32_t num_cores, uint32_t vid, float scale, uint32_t group_size, float q_offset)
{
    Common::InitVectorMask();
    if (n == 0 || (n & 1U) != 0 || n > ELEMENTS_PER_TILE) {
        return;
    }
    uint32_t sample_offset = 0;
    uint32_t samples_to_process = 0;
    if (!Common::ResolveCoreWork(batch, num_cores, vid, sample_offset, samples_to_process)) {
        return;
    }
    Common::InitPipeEvents();
    RunQuantTiles(x, y, group_scales, group_offsets, scale_group_stride, offset_group_stride, sample_offset,
                  samples_to_process, n, log2_n, group_size, scale, q_offset);
    Common::DrainPipeEvents();
}

} // namespace

// ---- Kernel launch entry (scalar args; num_cores/vid computed device-side) ----
__global__ __aicore__ void FastHadamardQuantKernel(__gm__ uint8_t* x, __gm__ uint8_t* group_scales,
                                                   __gm__ uint8_t* group_offsets, __gm__ uint8_t* y,
                                                   uint32_t scaleGroupStride, uint32_t offsetGroupStride,
                                                   uint32_t batch, uint32_t n, uint32_t logN, float scale,
                                                   uint32_t groupSize, float qOffset)
{
#if defined(__DAV_VEC__)
    const uint32_t num_cores = get_block_num() * get_subblockdim();
    const uint32_t vid = get_block_idx() * get_subblockdim() + get_subblockid();
    runTFastHadamardQuant<half, int8_t>((__gm__ half*)x, (__gm__ int8_t*)y, (__gm__ half*)group_scales,
                                        (__gm__ half*)group_offsets, scaleGroupStride, offsetGroupStride, batch, n,
                                        logN, num_cores, vid, scale, groupSize, qOffset);
#endif
}

void FastHadamardQuantKernelLaunch(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* group_scales,
                                   uint8_t* group_offsets, uint8_t* y, uint32_t scaleGroupStride,
                                   uint32_t offsetGroupStride, uint32_t batch, uint32_t n, uint32_t logN, float scale,
                                   uint32_t groupSize, float qOffset)
{
    FastHadamardQuantKernel<<<blockDim, nullptr, stream>>>(x, group_scales, group_offsets, y, scaleGroupStride,
                                                           offsetGroupStride, batch, n, logN, scale, groupSize,
                                                           qOffset);
}

constexpr int64_t MAX_HADAMARD_N = 16384;
constexpr uint32_t AIV_NUM = 40; // vector cores on Atlas A2 (Ascend910B)

// Per-row stride into a grouped scale/offset tensor: groupsPerRow for a 2D [batch, groups]
// tensor, 0 (shared) for a 1D [groups] / broadcast tensor or when absent.
static uint32_t GroupStride(const c10::optional<at::Tensor>& t, uint32_t batch, uint32_t groupsPerRow)
{
    if (!t.has_value() || !t->defined()) {
        return 0;
    }
    const at::Tensor& s = t.value();
    if (s.dim() >= 2 && static_cast<uint32_t>(s.size(0)) == batch) {
        return groupsPerRow;
    }
    return 0;
}

int64_t FastHadamardQuantNpu(torch::Tensor& x, const c10::optional<torch::Tensor>& group_scales,
                             const c10::optional<torch::Tensor>& group_offsets, double scale, int64_t group_size,
                             double q_offset, torch::Tensor& y)
{
    TORCH_CHECK(torch_npu::utils::is_npu(x), "x must be on the NPU device");
    TORCH_CHECK(torch_npu::utils::is_npu(y), "out must be on the NPU device");
    TORCH_CHECK(x.scalar_type() == at::kHalf, "x dtype must be float16");
    TORCH_CHECK(y.scalar_type() == at::kChar, "out dtype must be int8 (packed int4)");
    TORCH_CHECK(x.is_contiguous() && y.is_contiguous(), "x and out must be contiguous");
    TORCH_CHECK(x.dim() >= 1, "x rank must be >= 1");

    const int64_t n = x.size(x.dim() - 1);
    const int64_t totalElems = x.numel();
    TORCH_CHECK(n > 1 && n <= MAX_HADAMARD_N && (n & (n - 1)) == 0, "last dim n=", n, " must be a power of two in (2, ",
                MAX_HADAMARD_N, "]");
    TORCH_CHECK(totalElems % n == 0, "total elements not divisible by n");

    uint32_t logN = 0;
    for (int64_t v = n; v > 1; v >>= 1) {
        ++logN;
    }
    const uint32_t batch = static_cast<uint32_t>(totalElems / n);

    // Resolve group size: explicit attr wins; else infer from group_scales last dim; else whole row.
    uint32_t groupSize = static_cast<uint32_t>(n);
    if (group_size > 0) {
        groupSize = static_cast<uint32_t>(group_size);
    } else if (group_scales.has_value() && group_scales->defined()) {
        const int64_t gpr = group_scales->size(group_scales->dim() - 1);
        TORCH_CHECK(gpr > 0 && n % gpr == 0, "group_scales last dim must divide n");
        groupSize = static_cast<uint32_t>(n / gpr);
    }
    TORCH_CHECK(groupSize > 0 && (groupSize & 1) == 0 && n % groupSize == 0, "group_size=", groupSize,
                " must be an even divisor of n");
    const uint32_t groupsPerRow = static_cast<uint32_t>(n) / groupSize;

    uint32_t blockDim = (batch < AIV_NUM) ? batch : AIV_NUM;
    if (blockDim == 0) {
        blockDim = 1;
    }
    const uint32_t scaleGroupStride = GroupStride(group_scales, batch, groupsPerRow);
    const uint32_t offsetGroupStride = GroupStride(group_offsets, batch, groupsPerRow);
    uint8_t* gsPtr = (group_scales.has_value() && group_scales->defined()) ? (uint8_t*)group_scales->data_ptr() :
                                                                             nullptr;
    uint8_t* goPtr = (group_offsets.has_value() && group_offsets->defined()) ? (uint8_t*)group_offsets->data_ptr() :
                                                                               nullptr;
    const float scaleF = static_cast<float>(scale);
    const float qOffsetF = static_cast<float>(q_offset);

    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    at_npu::native::OpCommand::RunOpApi("fast_hadamard_quant", [=]() -> int {
        FastHadamardQuantKernelLaunch(blockDim, stream, (uint8_t*)x.data_ptr(), gsPtr, goPtr, (uint8_t*)y.data_ptr(),
                                      scaleGroupStride, offsetGroupStride, batch, static_cast<uint32_t>(n), logN,
                                      scaleF, groupSize, qOffsetF);
        return 0;
    });
    return 0;
}

TORCH_LIBRARY_FRAGMENT(EXTENSION_MODULE_NAME, m)
{
    m.def("fast_hadamard_quant(Tensor x, Tensor? group_scales, Tensor? group_offsets, float scale, "
          "int group_size, float q_offset, Tensor out) -> int");
}

TORCH_LIBRARY_IMPL(EXTENSION_MODULE_NAME, PrivateUse1, m) { m.impl("fast_hadamard_quant", FastHadamardQuantNpu); }

} // namespace FastHadamardQuant
} // namespace ascend_ops
