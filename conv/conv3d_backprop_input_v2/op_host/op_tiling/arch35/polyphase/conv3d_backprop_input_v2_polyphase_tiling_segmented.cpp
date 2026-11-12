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
 * \file conv3d_backprop_input_v2_polyphase_tiling_segmented.cpp
 * \brief PHASED dense、串行和并行 depthwise 分段搜索。
 */

#include "conv3d_backprop_input_v2_polyphase_tiling.h"
#include <algorithm>
#include <array>

namespace Ops {
namespace NN {
namespace Conv {
namespace Polyphase {

constexpr uint32_t DENSE_TILE_M = 32U;
constexpr uint32_t DENSE_PHASE_GROUP = 8U;
constexpr uint32_t MIN_PHASE_GROUPS = 2U;
constexpr uint32_t MIN_PARALLEL_PHASES = 8U;
constexpr uint32_t GATHER_INDEX_LIMIT = static_cast<uint32_t>(UINT16_MAX) + 1U;
constexpr uint32_t PHASE_MAX_STRIDE = 63U;
constexpr uint32_t SEGMENT_CAPACITY = 64U;
constexpr uint32_t PARALLEL_RANK_SIZE = 3U;
constexpr uint32_t DEPTHWISE_RANK_SIZE = 6U;
constexpr uint32_t SHAPE_TUPLE_SIZE = 3U;

using ParallelRank = std::array<uint64_t, PARALLEL_RANK_SIZE>;
using DepthwiseRank = std::array<uint64_t, DEPTHWISE_RANK_SIZE>;

bool SegmentedProduct(uint64_t left, uint64_t right, uint64_t& result)
{
    if (right != 0U && left > UINT64_MAX / right) {
        return false;
    }
    result = left * right;
    return true;
}

uint64_t SegmentedCost(uint64_t left, uint64_t right)
{
    uint64_t result = 0U;
    return SegmentedProduct(left, right, result) ? result : UINT64_MAX;
}

bool IsSegmentedDepthwise(const Conv3dBpInputV2RunInfo& runInfo, bool bias)
{
    if (bias || !IsDepthwise(runInfo) || runInfo.batch_n < 1 || runInfo.groups < 1 || runInfo.dedy_w < 1 ||
        runInfo.dedx_w < 1 || runInfo.kernel_w < 1 || runInfo.stride_w < 1 || runInfo.stride_w > PHASE_MAX_STRIDE ||
        runInfo.dilation_w != 1 || runInfo.pad_l < 0 || runInfo.pad_r < 0) {
        return false;
    }
    const int64_t wi = runInfo.dedy_w;
    const int64_t kw = runInfo.kernel_w;
    const int64_t stride = runInfo.stride_w;
    if (wi - 1 > (INT64_MAX - kw) / stride) {
        return false;
    }
    const int64_t extent = (wi - 1) * stride + kw;
    if (runInfo.pad_l >= extent || runInfo.pad_r >= extent - runInfo.pad_l ||
        extent - runInfo.pad_l - runInfo.pad_r != runInfo.dedx_w) {
        return false;
    }
    // 字节偏移与有符号 halo 坐标单独校验，不混入排序成本。
    uint64_t nc = 0U;
    uint64_t elements = 0U;
    if (!SegmentedProduct(runInfo.batch_n, runInfo.groups, nc)) {
        return false;
    }
    for (uint64_t width : {static_cast<uint64_t>(runInfo.dedy_w), static_cast<uint64_t>(runInfo.dedx_w)}) {
        if (!SegmentedProduct(nc, width, elements) ||
            elements > static_cast<uint64_t>(INT64_MAX) / runInfo.a_dtype_bytes) {
            return false;
        }
    }
    return SegmentedProduct(runInfo.groups, runInfo.kernel_w, elements) &&
           elements <= static_cast<uint64_t>(INT64_MAX) / runInfo.a_dtype_bytes;
}

bool SegmentedFits(const Conv3dBpInputV2RunInfo& runInfo, const Resources& resources, uint64_t m, uint64_t cap)
{
    const uint64_t a1 = (BASIC_BLOCK_SIZE_16 * resources.dataBytes) * (m + cap - 1U);
    const uint64_t b1 = (DB_ON * (BASIC_BLOCK_SIZE_16 * BASIC_BLOCK_SIZE_16 * resources.dataBytes)) * cap;
    const uint64_t l0a = DB_ON * Ops::Base::CeilAlign<uint64_t>(
                                     (BASIC_BLOCK_SIZE_16 * resources.dataBytes) * m * cap,
                                     (BASIC_BLOCK_SIZE_16 * BASIC_BLOCK_SIZE_16 * resources.dataBytes));
    const uint64_t l0b = DB_ON * Ops::Base::CeilAlign<uint64_t>(
                                     (BASIC_BLOCK_SIZE_16 * BASIC_BLOCK_SIZE_16 * resources.dataBytes) * cap,
                                     (BASIC_BLOCK_SIZE_16 * BASIC_BLOCK_SIZE_16 * resources.dataBytes));
    const uint64_t l0c = Ops::Base::CeilAlign<uint64_t>(
        (BASIC_BLOCK_SIZE_16 * resources.accBytes) * m,
        (BASIC_BLOCK_SIZE_16 * BASIC_BLOCK_SIZE_16 * resources.accBytes));
    const uint64_t row = Ops::Base::CeilAlign<uint64_t>(static_cast<uint64_t>(runInfo.stride_w) * m,
                                                        (BASIC_BLOCK_SIZE_256 / resources.dataBytes));
    const uint64_t aiv0 = DB_ON * ((BASIC_BLOCK_SIZE_16 * resources.dataBytes) * runInfo.stride_w * m +
                                   ((BASIC_BLOCK_SIZE_16 + 1U) * resources.dataBytes) * row);
    const uint64_t aiv1 = (FILTER_PACK_BUFFERS * (BASIC_BLOCK_SIZE_16 * BASIC_BLOCK_SIZE_16 * resources.dataBytes)) *
                              cap +
                          BASIC_BLOCK_SIZE_256;
    const uint64_t ub = resources.ub;
    return a1 + b1 <= resources.l1 && l0a <= resources.l0a && l0b <= resources.l0b && l0c <= resources.l0c &&
           aiv0 <= ub && aiv1 <= ub &&
           static_cast<uint64_t>(runInfo.stride_w) * BASIC_BLOCK_SIZE_16 * m <= GATHER_INDEX_LIMIT;
}

uint64_t SegmentedPhaseBoundary(uint64_t stride, uint64_t active, uint64_t pad, uint64_t groups, uint64_t group)
{
    if (group == 0U) {
        return 0U;
    }
    if (group == groups) {
        return stride;
    }
    const uint64_t ordinal = Ops::Base::CeilDiv(group * active, groups);
    if (active == stride) {
        return ordinal;
    }
    const uint64_t first = (stride - pad % stride) % stride;
    const uint64_t wrapped = first + active > stride ? first + active - stride : 0U;
    return ordinal < wrapped ? ordinal : first + ordinal - wrapped;
}

bool SegmentedParallelFits(const Conv3dBpInputV2RunInfo& runInfo, const Resources& resources, const Plan& plan,
                           uint64_t width)
{
    const uint64_t m = plan.baseM;
    const uint64_t pitch = Ops::Base::CeilAlign<uint64_t>(width, BASIC_BLOCK_SIZE_16);
    const uint64_t row = Ops::Base::CeilAlign<uint64_t>(m * pitch, (BASIC_BLOCK_SIZE_256 / resources.dataBytes));
    return SegmentedFits(runInfo, resources, m, plan.ka) && width != 0U &&
           width <= static_cast<uint64_t>(runInfo.stride_w) && BASIC_BLOCK_SIZE_16 * m * width <= GATHER_INDEX_LIMIT &&
           DB_ON * ((BASIC_BLOCK_SIZE_16 * resources.dataBytes) * width * m +
                    ((BASIC_BLOCK_SIZE_16 + 1U) * resources.dataBytes) * row) <=
               resources.ub;
}

void RefineSegmentedParallel(const Conv3dBpInputV2RunInfo& runInfo, const Resources& resources, Plan& plan)
{
    const uint64_t stride = runInfo.stride_w;
    const uint64_t active = std::min(static_cast<uint64_t>(runInfo.kernel_w), stride);
    uint64_t tasks = 0U;
    if (active < MIN_PARALLEL_PHASES || static_cast<uint64_t>(runInfo.dedx_w) <= stride ||
        !SegmentedProduct(plan.mDim, plan.nDim, tasks) || !SegmentedProduct(tasks, runInfo.batch_n, tasks) ||
        tasks == 0U || resources.cores / tasks < MIN_PHASE_GROUPS) {
        return;
    }
    const uint64_t limit = std::min(active, resources.cores / tasks);
    bool found = false;
    ParallelRank best{};
    for (uint64_t groups = MIN_PHASE_GROUPS; groups <= limit; ++groups) {
        uint64_t width = 0U;
        uint64_t previous = 0U;
        for (uint64_t group = 1U; group <= groups; ++group) {
            const uint64_t boundary = SegmentedPhaseBoundary(stride, active, runInfo.pad_l, groups, group);
            width = std::max(width, boundary - previous);
            previous = boundary;
        }
        if (!SegmentedParallelFits(runInfo, resources, plan, width)) {
            continue;
        }
        const ParallelRank rank{Ops::Base::CeilDiv(active, groups),
                                Ops::Base::CeilAlign<uint64_t>(width, BASIC_BLOCK_SIZE_16), groups};
        if (!found || rank < best) {
            found = true;
            best = rank;
            plan.strategy = Strategy::PARALLEL_DEPTHWISE;
            plan.phaseGroups = static_cast<uint32_t>(groups);
            plan.phaseWidth = static_cast<uint32_t>(width);
            plan.coreNum = static_cast<uint32_t>(tasks * groups);
        }
    }
}

void SetDepthwiseSegmentedPlan(uint64_t m, uint64_t cap, uint64_t mDim, uint64_t nDim, uint64_t cores, Plan& plan)
{
    plan = Plan{};
    plan.strategy = Strategy::SERIAL_DEPTHWISE;
    plan.baseM = m;
    plan.baseK = BASIC_BLOCK_SIZE_16 * cap;
    plan.singleCoreM = m;
    plan.mDim = mDim;
    plan.nDim = nDim;
    plan.coreNum = cores;
    plan.ka = cap;
    plan.residency = SINGLE_PHASE_B1;
    SetSegments(plan.baseK, plan.baseK, plan);
}

uint64_t CountDepthwiseSegments(const Conv3dBpInputV2RunInfo& runInfo, uint64_t m, uint64_t cap)
{
    const uint64_t stride = runInfo.stride_w;
    const uint64_t padPhase = static_cast<uint64_t>(runInfo.pad_l) % stride;
    uint64_t segments = 0U;
    for (uint64_t r = 0U; r < std::min(stride, static_cast<uint64_t>(runInfo.dedx_w)); ++r) {
        const uint64_t k0 = (r + padPhase) % stride;
        if (k0 < static_cast<uint64_t>(runInfo.kernel_w)) {
            const uint64_t q = 1U + (static_cast<uint64_t>(runInfo.kernel_w) - 1U - k0) / stride;
            segments += Ops::Base::CeilDiv(std::min(q, static_cast<uint64_t>(runInfo.dedy_w) + m - 1U), cap);
        }
    }
    return segments;
}

bool BuildSegmentedDepthwise(const Conv3dBpInputV2RunInfo& runInfo, bool bias, const Resources& resources, Plan& plan)
{
    if (!IsSegmentedDepthwise(runInfo, bias) || resources.cores == 0U) {
        return false;
    }
    const uint64_t stride = runInfo.stride_w;
    const uint64_t width = Ops::Base::CeilDiv<uint64_t>(runInfo.dedx_w, stride);
    const uint64_t nDim = Ops::Base::CeilDiv<uint64_t>(runInfo.groups, BASIC_BLOCK_SIZE_16);
    const uint64_t maxQ = Ops::Base::CeilDiv<uint64_t>(runInfo.kernel_w, stride);
    bool found = false;
    DepthwiseRank best{};
    for (uint64_t m = BASIC_BLOCK_SIZE_16; m <= BASIC_BLOCK_SIZE_256; m += BASIC_BLOCK_SIZE_16) {
        const uint64_t cap = std::min(
            {maxQ, static_cast<uint64_t>(runInfo.dedy_w) + m - 1U, static_cast<uint64_t>(SEGMENT_CAPACITY),
             resources.l0a / ((BASIC_BLOCK_SIZE_16 * resources.accBytes) * m),
             resources.l0b / (DB_ON * (BASIC_BLOCK_SIZE_16 * BASIC_BLOCK_SIZE_16 * resources.dataBytes))});
        if (!SegmentedFits(runInfo, resources, m, cap)) {
            continue;
        }
        const uint64_t mDim = Ops::Base::CeilDiv(width, m);
        uint64_t tasks = 0U;
        if (mDim > UINT32_MAX || nDim > UINT32_MAX || !SegmentedProduct(nDim, mDim, tasks) ||
            !SegmentedProduct(tasks, runInfo.batch_n, tasks) || tasks == 0U) {
            continue;
        }
        const uint64_t cores = std::min(tasks, resources.cores);
        const uint64_t waves = Ops::Base::CeilDiv(tasks, cores);
        const uint64_t segments = CountDepthwiseSegments(runInfo, m, cap);
        const uint64_t work = SegmentedCost(tasks, segments);
        const DepthwiseRank rank{
            SegmentedCost(waves, segments),
            SegmentedCost(waves, stride * m),
            SegmentedCost(work, (BASIC_BLOCK_SIZE_16 * BASIC_BLOCK_SIZE_16 * resources.dataBytes) * m * cap),
            SegmentedCost(work,
                          (BASIC_BLOCK_SIZE_16 * resources.dataBytes) * (m + cap - 1U) +
                              ((4U * BASIC_BLOCK_SIZE_16 + 1U) * BASIC_BLOCK_SIZE_16 * resources.dataBytes) * cap),
            tasks,
            UINT64_MAX - m};
        if (!found || rank < best) {
            found = true;
            best = rank;
            SetDepthwiseSegmentedPlan(m, cap, mDim, nDim, cores, plan);
        }
    }
    if (found) {
        RefineSegmentedParallel(runInfo, resources, plan);
    }
    return found;
}

bool IsExpandedDense(const Conv3dBpInputV2RunInfo& runInfo, bool bias)
{
    if (!bias || runInfo.groups != 1 || IsDepthwise(runInfo) || runInfo.batch_n < 1 || runInfo.dedy_w < 1 ||
        runInfo.dedx_w < 1 || runInfo.dedy_cout < 1 || runInfo.dedy_cout > MAX_CHANNELS || runInfo.dedx_cin < 1 ||
        runInfo.dedx_cin > MAX_CHANNELS || runInfo.stride_w < 1 || runInfo.stride_w > PHASE_MAX_STRIDE ||
        runInfo.kernel_w < 1 || runInfo.kernel_w % runInfo.stride_w != 0 || runInfo.dilation_w != 1 ||
        runInfo.pad_l < 0 || runInfo.pad_r < 0) {
        return false;
    }
    const int64_t wi = runInfo.dedy_w;
    const int64_t kw = runInfo.kernel_w;
    const int64_t stride = runInfo.stride_w;
    if (wi - 1 > (INT64_MAX - kw) / stride) {
        return false;
    }
    const int64_t extent = (wi - 1) * stride + kw;
    if (runInfo.pad_l >= extent || runInfo.pad_r >= extent - runInfo.pad_l ||
        extent - runInfo.pad_l - runInfo.pad_r != runInfo.dedx_w) {
        return false;
    }
    uint64_t elements = 0U;
    for (const auto& shape :
         {std::array<uint64_t, SHAPE_TUPLE_SIZE>{static_cast<uint64_t>(runInfo.batch_n),
                                                 static_cast<uint64_t>(runInfo.dedy_cout), static_cast<uint64_t>(wi)},
          std::array<uint64_t, SHAPE_TUPLE_SIZE>{static_cast<uint64_t>(runInfo.batch_n),
                                                 static_cast<uint64_t>(runInfo.dedx_cin),
                                                 static_cast<uint64_t>(runInfo.dedx_w)},
          std::array<uint64_t, SHAPE_TUPLE_SIZE>{static_cast<uint64_t>(runInfo.dedy_cout),
                                                 static_cast<uint64_t>(runInfo.dedx_cin), static_cast<uint64_t>(kw)}}) {
        if (!SegmentedProduct(shape[0], shape[1], elements) || !SegmentedProduct(elements, shape[2], elements) ||
            elements > static_cast<uint64_t>(INT64_MAX) / runInfo.a_dtype_bytes) {
            return false;
        }
    }
    return true;
}

bool DenseSegmentedFits(const Resources& resources, uint64_t m, uint64_t channels, uint64_t cap, uint64_t baseK,
                        uint64_t phases)
{
    const uint64_t a1 = Ops::Base::CeilAlign<uint64_t>(resources.dataBytes * channels * (m + cap - 1U), BYTE_BLOCK);
    const uint64_t b1 = resources.dataBytes * channels * cap * BASIC_BLOCK_SIZE_16;
    const uint64_t row = m * BASIC_BLOCK_SIZE_16;
    const uint64_t bank = Ops::Base::CeilAlign<uint64_t>(
        (resources.accBytes + resources.dataBytes) * phases * BASIC_BLOCK_SIZE_16 * m +
            resources.dataBytes * BASIC_BLOCK_SIZE_16 * row + BASIC_BLOCK_SIZE_16 * resources.accBytes,
        BYTE_BLOCK);
    const uint64_t ub = resources.ub;
    return (m == DENSE_TILE_M || m == BASIC_BLOCK_SIZE_16) && channels != 0U && channels <= MAX_CHANNELS &&
           channels % BASIC_BLOCK_SIZE_16 == 0U && cap != 0U && cap <= SEGMENT_CAPACITY && baseK != 0U &&
           baseK <= MAX_MMAD_K && phases != 0U && phases <= DENSE_PHASE_GROUP && a1 + DB_ON * b1 <= resources.l1 &&
           (DB_ON * resources.dataBytes) * m * baseK <= resources.l0a &&
           (DB_ON * resources.dataBytes) * BASIC_BLOCK_SIZE_16 * baseK <= resources.l0b &&
           resources.accBytes * m * BASIC_BLOCK_SIZE_16 <= resources.l0c && b1 <= ub &&
           DB_ON * bank + Ops::Base::CeilAlign<uint64_t>(resources.dataBytes * row, BYTE_BLOCK) <= ub;
}

void SetDenseSegmentedPlan(uint64_t m, uint64_t baseK, uint64_t channels, uint64_t cap, uint64_t mDim, uint64_t nDim,
                           uint64_t dDim, uint64_t phases, uint64_t tasks, const Resources& resources, Plan& plan)
{
    plan = Plan{};
    plan.strategy = Strategy::DENSE_SEGMENTED;
    plan.baseM = static_cast<uint32_t>(m);
    plan.singleCoreM = plan.baseM;
    plan.baseK = static_cast<uint32_t>(baseK);
    plan.channels = static_cast<uint32_t>(channels);
    plan.ka = static_cast<uint32_t>(cap);
    plan.mDim = static_cast<uint32_t>(mDim);
    plan.nDim = static_cast<uint32_t>(nDim);
    plan.phaseGroups = static_cast<uint32_t>(dDim);
    plan.phaseWidth = static_cast<uint32_t>(phases);
    plan.coreNum = static_cast<uint32_t>(std::min(tasks, resources.cores));
    plan.residency = SINGLE_PHASE_B1;
    const uint64_t filterBytes = resources.dataBytes * channels * cap * BASIC_BLOCK_SIZE_16;
    plan.bulkFilter = FILTER_PACK_BUFFERS * filterBytes + BASIC_BLOCK_SIZE_256 <= resources.ub;
}

bool BuildSegmentedDense(const Conv3dBpInputV2RunInfo& runInfo, bool bias, const Resources& resources, Plan& plan)
{
    if (!IsExpandedDense(runInfo, bias) || resources.cores == 0U) {
        return false;
    }
    const uint64_t channels = Ops::Base::CeilAlign<uint64_t>(runInfo.dedy_cout, BASIC_BLOCK_SIZE_16);
    const uint64_t stride = runInfo.stride_w;
    const uint64_t phases = std::min(stride, static_cast<uint64_t>(DENSE_PHASE_GROUP));
    const uint64_t dDim = Ops::Base::CeilDiv(std::min(stride, static_cast<uint64_t>(runInfo.dedx_w)), phases);
    const uint64_t nDim = Ops::Base::CeilDiv<uint64_t>(runInfo.dedx_cin, BASIC_BLOCK_SIZE_16);
    const uint64_t width = Ops::Base::CeilDiv<uint64_t>(runInfo.dedx_w, stride);
    for (uint64_t m : {DENSE_TILE_M, BASIC_BLOCK_SIZE_16}) {
        const uint64_t kLimit = std::min({static_cast<uint64_t>(MAX_MMAD_K),
                                          resources.l0a / ((DB_ON * resources.dataBytes) * m),
                                          resources.l0b / ((DB_ON * resources.dataBytes) * BASIC_BLOCK_SIZE_16)});
        for (uint64_t cap = std::min<uint64_t>(runInfo.kernel_w / stride, SEGMENT_CAPACITY); cap != 0U; --cap) {
            const uint64_t channelTile = std::min(channels, kLimit / (BASIC_BLOCK_SIZE_16 * cap) * BASIC_BLOCK_SIZE_16);
            const uint64_t baseK = channelTile * cap;
            if (!DenseSegmentedFits(resources, m, channels, cap, baseK, phases)) {
                continue;
            }
            const uint64_t mDim = Ops::Base::CeilDiv(width, m);
            uint64_t tasks = 0U;
            if (mDim > UINT32_MAX || !SegmentedProduct(mDim, nDim, tasks) || !SegmentedProduct(tasks, dDim, tasks) ||
                !SegmentedProduct(tasks, runInfo.batch_n, tasks) || tasks == 0U) {
                return false;
            }
            SetDenseSegmentedPlan(m, baseK, channels, cap, mDim, nDim, dDim, phases, tasks, resources, plan);
            return SetSegments(channels * cap, baseK, plan);
        }
    }
    return false;
}

} // namespace Polyphase
} // namespace Conv
} // namespace NN
} // namespace Ops
