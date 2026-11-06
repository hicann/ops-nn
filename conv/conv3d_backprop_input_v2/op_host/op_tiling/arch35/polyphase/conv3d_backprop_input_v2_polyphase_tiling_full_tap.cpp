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
 * \file conv3d_backprop_input_v2_polyphase_tiling_full_tap.cpp
 * \brief PHASED 全 tap 搜索与性能准入。
 */

#include "conv3d_backprop_input_v2_polyphase_tiling.h"
#include <algorithm>
#include <array>

namespace Ops {
namespace NN {
namespace Conv {
namespace Polyphase {

constexpr uint32_t INTERLEAVE_STRIDE = 2U;
constexpr uint8_t ALL_PHASE_B1 = 1U;
constexpr uint32_t MAX_LOAD_WIDTH = 65535U;

using PhasedScore = std::array<uint64_t, 12>;

bool AivFits(uint64_t stride, uint64_t outputM, bool dense, const Resources& resources)
{
    const uint64_t phase = stride * BASIC_BLOCK_SIZE_16 * outputM;
    const uint64_t row = stride * outputM;
    const uint64_t stored = Ops::Base::CeilAlign<uint64_t>(row, stride == INTERLEAVE_STRIDE ?
                                                                    static_cast<uint64_t>(BASIC_BLOCK_SIZE_16) :
                                                                    BASIC_BLOCK_SIZE_256 / resources.dataBytes);
    if (row > UINT16_MAX || phase > UINT16_MAX) {
        return false;
    }
    uint64_t bytes = phase * (dense ? resources.accBytes : resources.dataBytes);
    if (dense) {
        bytes += phase * resources.dataBytes;
    }
    bytes += BASIC_BLOCK_SIZE_16 * stored * resources.dataBytes;
    bytes += dense ? BASIC_BLOCK_SIZE_16 * resources.accBytes : 0U;
    bytes += stride == INTERLEAVE_STRIDE ? 0U :
                                           Ops::Base::CeilAlign<uint64_t>(stored * resources.dataBytes, BYTE_BLOCK);
    return Ops::Base::CeilAlign<uint64_t>(bytes, BYTE_BLOCK) <= resources.ub / DB_ON;
}

bool PhasedFits(const Conv3dBpInputV2RunInfo& runInfo, const Resources& resources, bool dense, uint8_t residency,
                uint64_t baseM, uint64_t baseK, uint64_t stripe)
{
    const uint64_t q = runInfo.kernel_w / runInfo.stride_w;
    const uint64_t alignedCout = dense ? runInfo.dedy_cout : BASIC_BLOCK_SIZE_16;
    const uint64_t l0K = dense ? baseK / q : baseK;
    const uint64_t a1 = Ops::Base::CeilAlign<uint64_t>(alignedCout * (stripe + q - 1U) * resources.dataBytes,
                                                       static_cast<uint64_t>(BYTE_BLOCK));
    const uint64_t phaseB1 = BASIC_BLOCK_SIZE_16 * alignedCout * q * resources.dataBytes;
    const uint64_t resident = residency == ALL_PHASE_B1 ? runInfo.stride_w : 1U;
    const uint64_t l1 = a1 + DB_ON * phaseB1 * resident;
    if (l1 > resources.l1) {
        return false;
    }
    const uint64_t l0a = Ops::Base::CeilAlign<uint64_t>(
        baseM * l0K * resources.dataBytes, (BASIC_BLOCK_SIZE_16 * BASIC_BLOCK_SIZE_16 * resources.dataBytes));
    const uint64_t l0b = Ops::Base::CeilAlign<uint64_t>(
        BASIC_BLOCK_SIZE_16 * l0K * resources.dataBytes,
        (BASIC_BLOCK_SIZE_16 * BASIC_BLOCK_SIZE_16 * resources.dataBytes));
    const uint64_t l0c = Ops::Base::CeilAlign<uint64_t>(
        baseM * BASIC_BLOCK_SIZE_16 * resources.accBytes,
        (BASIC_BLOCK_SIZE_16 * BASIC_BLOCK_SIZE_16 * resources.accBytes));
    const uint64_t aiv1 = dense ? phaseB1 : FILTER_PACK_BUFFERS * phaseB1 + BASIC_BLOCK_SIZE_256;
    const uint64_t outputM = residency == ALL_PHASE_B1 ? baseM : stripe;
    return DB_ON * l0a <= resources.l0a && DB_ON * l0b <= resources.l0b && l0c <= resources.l0c &&
           aiv1 <= resources.ub && AivFits(runInfo.stride_w, outputM, dense, resources);
}

struct PhasedSearch {
    const Conv3dBpInputV2RunInfo& runInfo;
    const Resources& resources;
    bool dense;
    bool found = false;
    PhasedScore best{};
    Plan plan{};
};

void SetPhasedPlan(uint64_t baseM, uint64_t baseK, uint64_t stripe, uint64_t mDim, uint64_t nDim, uint64_t cores,
                   uint8_t residency, Plan& plan)
{
    plan = {};
    plan.baseM = baseM;
    plan.baseK = baseK;
    plan.singleCoreM = stripe;
    plan.mDim = mDim;
    plan.nDim = nDim;
    plan.coreNum = cores;
    plan.residency = residency;
}

void RankPhased(PhasedSearch& search, uint8_t residency, uint64_t baseM, uint64_t baseK, uint64_t stripe)
{
    const auto& runInfo = search.runInfo;
    const auto& resources = search.resources;
    const bool dense = search.dense;
    const uint64_t q = runInfo.kernel_w / runInfo.stride_w;
    const uint64_t alignedCout = dense ? runInfo.dedy_cout : BASIC_BLOCK_SIZE_16;
    const uint64_t fullK = alignedCout * q;
    const uint64_t outputTiles = dense ? runInfo.dedx_cin : runInfo.groups;
    const uint64_t nDim = Ops::Base::CeilDiv(outputTiles, static_cast<uint64_t>(BASIC_BLOCK_SIZE_16));
    const uint64_t width = runInfo.dedy_w + q - 1U;
    const uint64_t available = resources.cores;
    const uint64_t mDim = Ops::Base::CeilDiv(width, stripe);
    const uint64_t workPerBatch = nDim * mDim;
    if (workPerBatch == 0U || workPerBatch > UINT32_MAX ||
        static_cast<uint64_t>(runInfo.batch_n) > UINT32_MAX / workPerBatch) {
        return;
    }
    const uint64_t work = static_cast<uint64_t>(runInfo.batch_n) * workPerBatch;
    const uint64_t cores = std::min(work, available);
    const uint64_t waves = Ops::Base::CeilDiv(work, available);
    const uint64_t slots = waves * available;
    const uint64_t filterBytes = work * runInfo.stride_w * BASIC_BLOCK_SIZE_16 * q * resources.dataBytes;
    const uint64_t inputBytes = work * alignedCout * (stripe + q - 1U) * resources.dataBytes;
    const uint64_t overlapBytes = work * alignedCout * (q - 1U) * resources.dataBytes;
    const uint64_t innerTiles = work * Ops::Base::CeilDiv(stripe, baseM);
    const uint64_t innerOps = innerTiles * runInfo.stride_w * Ops::Base::CeilDiv(fullK, baseK);
    const PhasedScore score{waves,
                            slots - work,
                            innerOps,
                            innerTiles * runInfo.stride_w,
                            filterBytes,
                            inputBytes,
                            overlapBytes,
                            nDim * BASIC_BLOCK_SIZE_16 - outputTiles,
                            mDim * stripe - width,
                            static_cast<uint64_t>(residency == SINGLE_PHASE_B1),
                            UINT64_MAX - baseK,
                            UINT64_MAX - baseM};
    if (!search.found || score < search.best) {
        search.found = true;
        search.best = score;
        SetPhasedPlan(baseM, baseK, stripe, mDim, nDim, cores, residency, search.plan);
    }
}

namespace {
struct CoreCost {
    uint64_t mmads = 0U;
    uint64_t l0bCalls = 0U;
    uint64_t fixpipes = 0U;
    uint64_t macBlocks = 0U;
    uint64_t transferBytes = 0U;
};

uint64_t CountTails(uint64_t repeats, uint64_t mDim, uint64_t cores, uint64_t core)
{
    uint64_t tails = 0U;
    for (uint64_t j = 0U; j < std::min(repeats, cores); ++j) {
        if ((mDim - 1U + j * mDim) % cores == core) {
            tails += 1U + (repeats - 1U - j) / cores;
        }
    }
    return tails;
}

CoreCost CalcCoreCost(const Conv3dBpInputV2RunInfo& runInfo, uint64_t baseM, uint64_t baseK, uint64_t stripe,
                      uint64_t tailWidth, uint64_t count, uint64_t tails)
{
    const uint64_t q = runInfo.kernel_w / runInfo.stride_w;
    const uint64_t channels = runInfo.dedy_cout;
    const uint64_t inner = (count - tails) * (stripe / baseM) + tails * Ops::Base::CeilDiv(tailWidth, baseM);
    const uint64_t rows = (count - tails) * (stripe / BASIC_BLOCK_SIZE_16) +
                          tails * Ops::Base::CeilDiv(tailWidth, static_cast<uint64_t>(BASIC_BLOCK_SIZE_16));
    CoreCost cost;
    cost.fixpipes = inner * runInfo.stride_w;
    cost.mmads = cost.fixpipes * q * Ops::Base::CeilDiv(channels, baseK / q);
    cost.l0bCalls = cost.fixpipes * q * (channels / BASIC_BLOCK_SIZE_16);
    cost.macBlocks = rows * runInfo.stride_w * q * (channels / BASIC_BLOCK_SIZE_16);
    cost.transferBytes = (rows * BASIC_BLOCK_SIZE_16 + count * (q - 1U)) * channels * runInfo.a_dtype_bytes +
                         count * runInfo.stride_w * BASIC_BLOCK_SIZE_16 * channels * q * runInfo.a_dtype_bytes;
    return cost;
}

void RankDense(PhasedSearch& search, uint64_t baseM, uint64_t baseK, uint64_t stripe)
{
    const auto& runInfo = search.runInfo;
    const uint64_t q = runInfo.kernel_w / runInfo.stride_w;
    const uint64_t width = runInfo.dedy_w + q - 1U;
    const uint64_t nDim = Ops::Base::CeilDiv<uint64_t>(runInfo.dedx_cin, BASIC_BLOCK_SIZE_16);
    const uint64_t mDim = Ops::Base::CeilDiv(width, stripe);
    const uint64_t workPerBatch = nDim * mDim;
    if (workPerBatch == 0U || workPerBatch > UINT32_MAX ||
        static_cast<uint64_t>(runInfo.batch_n) > UINT32_MAX / workPerBatch) {
        return;
    }
    const uint64_t work = static_cast<uint64_t>(runInfo.batch_n) * workPerBatch;
    const uint64_t cores = std::min(work, search.resources.cores);
    const uint64_t waves = Ops::Base::CeilDiv(work, search.resources.cores);
    const uint64_t repeats = static_cast<uint64_t>(runInfo.batch_n) * nDim;
    const uint64_t tailWidth = width - (mDim - 1U) * stripe;
    CoreCost peak;
    uint64_t totalMmads = 0U;
    for (uint64_t core = 0U; core < cores; ++core) {
        const uint64_t count = 1U + (work - 1U - core) / cores;
        const uint64_t tails = CountTails(repeats, mDim, cores, core);
        const auto cost = CalcCoreCost(runInfo, baseM, baseK, stripe, tailWidth, count, tails);
        peak.mmads = std::max(peak.mmads, cost.mmads);
        peak.l0bCalls = std::max(peak.l0bCalls, cost.l0bCalls);
        peak.fixpipes = std::max(peak.fixpipes, cost.fixpipes);
        peak.macBlocks = std::max(peak.macBlocks, cost.macBlocks);
        peak.transferBytes = std::max(peak.transferBytes, cost.transferBytes);
        totalMmads += cost.mmads;
    }
    const PhasedScore score{waves,
                            peak.mmads,
                            peak.l0bCalls,
                            peak.fixpipes,
                            peak.macBlocks,
                            peak.transferBytes,
                            work * runInfo.stride_w,
                            totalMmads,
                            waves * search.resources.cores - work,
                            mDim * stripe - width,
                            UINT64_MAX - baseK,
                            UINT64_MAX - baseM};
    if (!search.found || score < search.best) {
        search.found = true;
        search.best = score;
        SetPhasedPlan(baseM, baseK, stripe, mDim, nDim, cores, SINGLE_PHASE_B1, search.plan);
    }
}

} // namespace

bool BuildPhased(const Conv3dBpInputV2RunInfo& runInfo, const Resources& resources, bool dense, bool transpose,
                 Plan& plan)
{
    const uint64_t q = runInfo.kernel_w / runInfo.stride_w;
    const uint64_t channels = dense ? runInfo.dedy_cout : BASIC_BLOCK_SIZE_16;
    const uint64_t fullK = channels * q;
    const uint64_t kStep = q * BASIC_BLOCK_SIZE_16;
    const uint64_t outputTiles = dense ? runInfo.dedx_cin : runInfo.groups;
    const uint64_t nDim = Ops::Base::CeilDiv<uint64_t>(outputTiles, BASIC_BLOCK_SIZE_16);
    const uint64_t width = runInfo.dedy_w + q - 1U;
    if (q == 0 || fullK > MAX_MMAD_K || nDim == 0 || resources.cores == 0 || q - 1 > MAX_LOAD_WIDTH) {
        return false;
    }
    PhasedSearch search{runInfo, resources, dense};
    for (uint8_t residency : {ALL_PHASE_B1, SINGLE_PHASE_B1}) {
        if (transpose && residency == ALL_PHASE_B1) {
            continue;
        }
        for (uint64_t baseM = BASIC_BLOCK_SIZE_256; baseM >= BASIC_BLOCK_SIZE_16; baseM -= BASIC_BLOCK_SIZE_16) {
            const uint64_t stripeEnd = std::min(Ops::Base::CeilAlign<uint64_t>(width, baseM),
                                                (MAX_LOAD_WIDTH - (q - 1U)) / baseM * baseM);
            for (uint64_t baseK = kStep; baseK <= fullK; baseK += kStep) {
                for (uint64_t stripe = baseM; stripe <= stripeEnd; stripe += baseM) {
                    if (!PhasedFits(runInfo, resources, dense, residency, baseM, baseK, stripe)) {
                        break;
                    }
                    if (transpose && dense) {
                        RankDense(search, baseM, baseK, stripe);
                    } else {
                        RankPhased(search, residency, baseM, baseK, stripe);
                    }
                }
            }
        }
    }
    plan = search.plan;
    return search.found && SetSegments(fullK, plan.baseK, plan);
}

bool AdmitPhasedPerformance(const Conv3dBpInputV2RunInfo& runInfo, const Plan& plan, uint64_t cores)
{
    constexpr uint64_t fewTaps = 2U;
    constexpr uint64_t amortizedTaps = 4U;
    constexpr uint64_t smallRows = 2U * BASIC_BLOCK_SIZE_16;
    constexpr uint64_t phasePackBudget = BASIC_BLOCK_SIZE_16 * smallRows;
    const uint64_t taps = Ops::Base::CeilDiv<uint64_t>(runInfo.kernel_w, runInfo.stride_w);
    const uint64_t tasks = static_cast<uint64_t>(runInfo.batch_n) * plan.mDim * plan.nDim * plan.phaseGroups;
    const bool multipleWaves = tasks > cores;
    const bool segmented = plan.strategy == Strategy::SERIAL_DEPTHWISE || plan.strategy == Strategy::PARALLEL_DEPTHWISE;
    return segmented ? (taps >= amortizedTaps || (plan.phaseGroups > 1U && taps > fewTaps) ||
                        (runInfo.stride_w < smallRows && runInfo.dedy_w >= BASIC_BLOCK_SIZE_16 &&
                         (plan.baseM > smallRows || !multipleWaves))) :
                       (plan.strategy == Strategy::DENSE_SEGMENTED || IsDepthwise(runInfo) ||
                        static_cast<uint64_t>(runInfo.dedy_cout) * runInfo.kernel_w <= phasePackBudget);
}

} // namespace Polyphase
} // namespace Conv
} // namespace NN
} // namespace Ops
