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
 * \file fast_hadamard.cpp
 * \brief Fused fast Hadamard transform (fp16), out-of-place, exposed as a torch op.
 *
 * Ecosystem "最简算子" delivery: a single translation unit that holds the AscendC/pto
 * device kernel, the host launch, and the torch registration. After the ascend_ops_nn
 * PyTorch extension is imported, the op is reachable as
 *     torch.ops.ascend_ops_nn.fast_hadamard(x, out)
 *
 * The transform body is the proven pto-ISA kernel from pto-kernels
 * (examples/jit_cpp/fast_hadamard/standard/fast_hadamard.cpp); the pto ISA headers ship
 * inside the CANN toolkit (${ASCEND_TOOLKIT_HOME}/include/pto, already on the extension
 * build's include path). Only the entry/host/registration layer differs from the
 * standard-aclnn delivery: the kernel is launched directly with <<<>>> on torch_npu's
 * current stream instead of via the generated aclnn API.
 */

#include "experimental/matmul/common/fast_hadamard/op_includes.hpp"
#include "experimental/matmul/common/fast_hadamard/fht_kernel_common.hpp"

namespace ascend_ops {
namespace FastHadamard {
using namespace pto;
namespace Common = ascend_ops::FastHadamardCommon;

constexpr uint32_t X_BUFFER_BYTES = 32 * 1024;
constexpr uint32_t UB_HALF_BYTES = X_BUFFER_BYTES / 2;
constexpr uint32_t ELEMENTS_PER_TILE = X_BUFFER_BYTES / sizeof(half);
constexpr uint32_t UB_USABLE_BYTES = 184 * 1024;

constexpr unsigned X_PING = 0x00000;
constexpr unsigned X_PONG = X_PING + X_BUFFER_BYTES + 0x100;
constexpr unsigned EVEN_BASE = X_PONG + X_BUFFER_BYTES + 0x100;
constexpr unsigned ODD_BASE = EVEN_BASE + UB_HALF_BYTES + 0x100;
static_assert(ODD_BASE + UB_HALF_BYTES <= UB_USABLE_BYTES, "Fast Hadamard UB layout exceeds usable UB.");

namespace {

template <typename T>
AICORE void StoreHadamardTile(__gm__ T* y, const Common::TileWork& tile, unsigned x_base, event_t ev, uint32_t n,
                              uint32_t log2_n)
{
    using ShapeDim5 = pto::Shape<1, 1, 1, 1, ELEMENTS_PER_TILE>;
    using StridDim5 = pto::Stride<1, 1, 1, 1, 1>;
    using GlobalData = pto::GlobalTensor<T, ShapeDim5, StridDim5>;
    using FullTile = Tile<TileType::Vec, T, 1, ELEMENTS_PER_TILE, BLayout::RowMajor, -1, -1>;
    FullTile xBulkTile(1, tile.elements);
    TASSIGN(xBulkTile, x_base);
    GlobalData yGlobal(y + tile.gm_offset);
    TASSIGN(yGlobal, (y + tile.gm_offset));
    Common::RunTileHadamardInPlace<T, ELEMENTS_PER_TILE, EVEN_BASE, ODD_BASE>(x_base, tile.sample_count, n, n, log2_n);
    wait_flag(PIPE_MTE3, PIPE_V, ev);
    set_flag(PIPE_V, PIPE_MTE3, ev);
    wait_flag(PIPE_V, PIPE_MTE3, ev);
    TSTORE(yGlobal, xBulkTile);
    set_flag(PIPE_MTE3, PIPE_V, ev);
    set_flag(PIPE_V, PIPE_MTE2, ev);
}

template <typename T>
AICORE void RunHadamardTiles(__gm__ T* x, __gm__ T* y, uint32_t sample_offset, uint32_t samples_to_process, uint32_t n,
                             uint32_t log2_n)
{
    const uint32_t samples_per_load = (n < ELEMENTS_PER_TILE) ? ELEMENTS_PER_TILE / n : 1;
    uint32_t sample_done = 0;
    Common::TileWork current_tile;
    const uint32_t gm_offset_base = sample_offset * n;
    if (!Common::NextTile(sample_done, gm_offset_base, samples_to_process, samples_per_load, n, current_tile)) {
        return;
    }

    bool ping = true;
    Common::IssueTLoad<T, ELEMENTS_PER_TILE>(x, current_tile, X_PING, EVENT_ID0);
    while (true) {
        const event_t current_ev = ping ? (event_t)EVENT_ID0 : (event_t)EVENT_ID1;
        const unsigned x_base = ping ? X_PING : X_PONG;
        wait_flag(PIPE_MTE2, PIPE_V, current_ev);
        Common::TileWork next_tile;
        const bool has_next = Common::NextTile(sample_done, gm_offset_base, samples_to_process, samples_per_load, n,
                                               next_tile);
        if (has_next) {
            const event_t next_ev = ping ? (event_t)EVENT_ID1 : (event_t)EVENT_ID0;
            const unsigned next_x_base = ping ? X_PONG : X_PING;
            Common::IssueTLoad<T, ELEMENTS_PER_TILE>(x, next_tile, next_x_base, next_ev);
        }
        StoreHadamardTile(y, current_tile, x_base, current_ev, n, log2_n);
        if (!has_next) {
            break;
        }
        current_tile = next_tile;
        ping = !ping;
    }
}

template <typename T>
AICORE void runTFastHadamard(__gm__ T* x, __gm__ T* y, uint32_t batch, uint32_t n, uint32_t log2_n)
{
#if defined(__DAV_VEC__)
    Common::InitVectorMask();
    if (n == 0 || n > ELEMENTS_PER_TILE) {
        return;
    }
    uint32_t sample_offset = 0;
    uint32_t samples_to_process = 0;
    const uint32_t num_cores = get_block_num() * get_subblockdim();
    const uint32_t vid = get_block_idx() * get_subblockdim() + get_subblockid();
    if (!Common::ResolveCoreWork(batch, num_cores, vid, sample_offset, samples_to_process)) {
        return;
    }
    Common::InitPipeEvents();
    RunHadamardTiles(x, y, sample_offset, samples_to_process, n, log2_n);
    Common::DrainPipeEvents();
#endif
}

} // namespace

// ---- Kernel launch entry (<<<>>> takes raw pointers + scalars, like layernorm_stride) ----
__global__ __aicore__ void FastHadamardKernel(__gm__ uint8_t* x, __gm__ uint8_t* y, uint32_t batch, uint32_t n,
                                              uint32_t logN)
{
    runTFastHadamard<half>((__gm__ half*)x, (__gm__ half*)y, batch, n, logN);
}

void FastHadamardKernelLaunch(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint32_t batch, uint32_t n,
                              uint32_t logN)
{
    FastHadamardKernel<<<blockDim, nullptr, stream>>>(x, y, batch, n, logN);
}

// ---- Torch op: validates, derives (batch, n, logN), launches on the current stream ----
constexpr int64_t MAX_HADAMARD_N = 16384; // matches FAST_HADAMARD_BATCHED_CASES
constexpr uint32_t AIV_NUM = 40;          // vector cores on Atlas A2 (Ascend910B)

int64_t FastHadamardNpu(torch::Tensor& x, torch::Tensor& y)
{
    TORCH_CHECK(torch_npu::utils::is_npu(x), "x must be on the NPU device");
    TORCH_CHECK(torch_npu::utils::is_npu(y), "out must be on the NPU device");
    TORCH_CHECK(x.scalar_type() == at::kHalf, "x dtype must be float16");
    TORCH_CHECK(y.scalar_type() == at::kHalf, "out dtype must be float16");
    TORCH_CHECK(x.is_contiguous() && y.is_contiguous(), "x and out must be contiguous");
    TORCH_CHECK(x.sizes() == y.sizes(), "out must have the same shape as x");
    TORCH_CHECK(x.dim() >= 1, "x rank must be >= 1");

    const int64_t n = x.size(x.dim() - 1);
    const int64_t totalElems = x.numel();
    TORCH_CHECK(n > 1 && n <= MAX_HADAMARD_N, "last dim n=", n, " out of supported range (2, ", MAX_HADAMARD_N, "]");
    TORCH_CHECK((n & (n - 1)) == 0, "last dim n=", n, " is not a power of two");
    TORCH_CHECK(totalElems % n == 0, "total elements not divisible by n");

    uint32_t logN = 0;
    for (int64_t v = n; v > 1; v >>= 1) {
        ++logN;
    }
    const uint32_t batch = static_cast<uint32_t>(totalElems / n);
    uint32_t blockDim = (batch < AIV_NUM) ? batch : AIV_NUM;
    if (blockDim == 0) {
        blockDim = 1;
    }

    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    at_npu::native::OpCommand::RunOpApi("fast_hadamard", [=]() -> int {
        FastHadamardKernelLaunch(blockDim, stream, (uint8_t*)x.data_ptr(), (uint8_t*)y.data_ptr(), batch,
                                 static_cast<uint32_t>(n), logN);
        return 0;
    });
    return 0;
}

TORCH_LIBRARY_FRAGMENT(EXTENSION_MODULE_NAME, m) { m.def("fast_hadamard(Tensor x, Tensor out) -> int"); }

TORCH_LIBRARY_IMPL(EXTENSION_MODULE_NAME, PrivateUse1, m) { m.impl("fast_hadamard", FastHadamardNpu); }

} // namespace FastHadamard
} // namespace ascend_ops
