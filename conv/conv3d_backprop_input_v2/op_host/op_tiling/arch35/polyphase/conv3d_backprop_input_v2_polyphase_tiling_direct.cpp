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
 * \file conv3d_backprop_input_v2_polyphase_tiling_direct.cpp
 * \brief DIRECT_DENSE 搜索、分组打包与性能准入。
 */

#include "conv3d_backprop_input_v2_polyphase_tiling.h"
#include <algorithm>
#include <limits>
#include <numeric>

namespace Ops {
namespace NN {
namespace Conv {
namespace Polyphase {

constexpr uint32_t RESIDENCY_CHOICES = AB_FULL + 1U;
constexpr uint32_t COMPUTE_RESIDENT_CHANNEL_BLOCKS = 48U;
constexpr uint32_t REFINE_MAX_OVERHEAD_NUMERATOR = 5U;
constexpr uint32_t REFINE_RATIO_DENOMINATOR = 4U;
constexpr uint32_t NDDMA_GM_STRIDE_BITS = 40U;
constexpr uint32_t NDDMA_UB_STRIDE_BITS = 20U;
constexpr uint32_t DIRECT_MAX_STRIDE = 64U;
constexpr uint32_t DIRECT_MAX_WIDTH = 65520U;
constexpr uint32_t DIRECT_MAX_N = 1024U;
constexpr uint32_t DIRECT_MAX_M = 16384U;
constexpr uint32_t MAC_OPERATIONS = 2U;
constexpr uint32_t MIN_GROUP_ENLARGE = 2U;
constexpr long double MODEL_CUBE_THROUGHPUT = 8192.0L;
constexpr long double MODEL_TRANSFER_THROUGHPUT = 128.0L;
constexpr long double MODEL_EPILOGUE_THROUGHPUT = 64.0L;
constexpr long double MODEL_TRANSFER_SETUP = 8.0L;
constexpr long double MODEL_K_TILE_SETUP = 2.0L;
constexpr long double MODEL_PACK_OPERATIONS = 14.0L;
constexpr long double MODEL_PACK_SETUP = 32.0L;
constexpr long double GROUP_REFINE_MAX_COST_RATIO = 0.95L;
constexpr uint32_t MODEL_VECTOR_SETUP = 6U;
constexpr uint32_t MODEL_BIAS_SETUP = 2U;
constexpr uint32_t MODEL_TAP_SETUP = 5U;
constexpr uint32_t MODEL_PHASE_SETUP = 8U;
constexpr uint32_t MODEL_GROUP_TRANSFER_SETUP = 8U;
constexpr uint32_t MODEL_TASK_SETUP = 16U;

uint64_t DirectTasks(const Conv3dBpInputV2RunInfo& runInfo, uint64_t mDim, uint64_t nDim, uint64_t enlarge)
{
    if (runInfo.batch_n <= 0 || runInfo.groups <= 0 || enlarge == 0U) {
        return 0U;
    }
    uint64_t tasks = 1U;
    for (uint64_t factor :
         {static_cast<uint64_t>(runInfo.batch_n), Ops::Base::CeilDiv<uint64_t>(runInfo.groups, enlarge), mDim, nDim}) {
        if (factor == 0U || tasks > UINT32_MAX / factor) {
            return 0U;
        }
        tasks *= factor;
    }
    return tasks;
}

bool TensorFits(uint64_t first, uint64_t second, uint64_t third, uint64_t bytes)
{
    for (uint64_t extent : {first, second, third}) {
        if (extent == 0U || bytes > static_cast<uint64_t>(INT64_MAX) / extent) {
            return false;
        }
        bytes *= extent;
    }
    return true;
}

struct DirectSearch {
    const Conv3dBpInputV2RunInfo& runInfo;
    const Resources& resources;
    long double best = std::numeric_limits<long double>::max();
    Plan plan{};
};

struct DirectTile {
    uint64_t channels;
    uint64_t tile;
    uint64_t m;
    uint64_t n;
    uint64_t mDim;
    uint64_t nDim;
    uint64_t tasks;
    uint64_t pBytes;
};

bool IsDirectInput(const Conv3dBpInputV2RunInfo& runInfo)
{
    if (runInfo.batch_n <= 0 || runInfo.groups <= 0 || runInfo.dedy_cout <= 0 || runInfo.dedx_cin <= 0 ||
        runInfo.dedy_w <= 0 || runInfo.dedx_w <= 0 || runInfo.kernel_w <= 0 || runInfo.stride_w <= 0 ||
        runInfo.dilation_w <= 0 || runInfo.dedy_cout % runInfo.groups != 0 || runInfo.dedx_cin % runInfo.groups != 0) {
        return false;
    }
    const bool extended = runInfo.batch_n > 1 || runInfo.groups > 1;
    if (extended &&
        (runInfo.dedy_cout > MAX_CHANNELS || runInfo.dedx_cin > MAX_CHANNELS || runInfo.groups > MAX_CHANNELS ||
         runInfo.stride_w > DIRECT_MAX_STRIDE || runInfo.dilation_w != 1 ||
         static_cast<int64_t>(runInfo.dedx_w) != (static_cast<int64_t>(runInfo.dedy_w) - 1) * runInfo.stride_w +
                                                     runInfo.kernel_w - runInfo.pad_l - runInfo.pad_r)) {
        return false;
    }
    return true;
}

void RankDirect(DirectSearch& search, const DirectTile& tileInfo, uint64_t k0, uint64_t chunk, bool requestA,
                bool requestB)
{
    const auto& runInfo = search.runInfo;
    const auto& resources = search.resources;
    const uint64_t fullK = Ops::Base::CeilAlign<uint64_t>(runInfo.dedy_cout / runInfo.groups, BASIC_BLOCK_SIZE_16);
    const uint64_t ci = runInfo.dedy_cout / runInfo.groups;
    const uint64_t kw = runInfo.kernel_w;
    const uint64_t ka = requestA ? fullK : chunk;
    const uint64_t kb = requestB ? fullK : chunk;
    const bool aFull = requestA || ka == fullK;
    const bool bFull = requestB || kb == fullK;
    if (ka > UINT32_MAX || kb > UINT32_MAX || k0 == 0U || k0 > UINT32_MAX) {
        return;
    }
    const uint64_t l1 = tileInfo.m * ka * resources.dataBytes * (aFull ? 1U : DB_ON) +
                        tileInfo.n * kb * resources.dataBytes * (bFull ? 1U : DB_ON);
    if (l1 > resources.l1) {
        return;
    }
    const uint64_t blocks = std::min(resources.cores, tileInfo.tasks);
    const uint64_t waves = Ops::Base::CeilDiv(tileInfo.tasks, blocks);
    const long double cube = static_cast<long double>(tileInfo.m) * tileInfo.n * fullK * MAC_OPERATIONS;
    const long double gm = static_cast<long double>(tileInfo.m + tileInfo.channels * kw) * ci * resources.dataBytes;
    const long double epilogue = static_cast<long double>(tileInfo.channels) * tileInfo.tile * kw / runInfo.stride_w;
    const uint64_t transfers = Ops::Base::CeilDiv(fullK, ka) + Ops::Base::CeilDiv(fullK, kb);
    const long double cost = waves * (cube / MODEL_CUBE_THROUGHPUT + gm / MODEL_TRANSFER_THROUGHPUT +
                                      tileInfo.pBytes / MODEL_TRANSFER_THROUGHPUT +
                                      epilogue / MODEL_EPILOGUE_THROUGHPUT + transfers * MODEL_TRANSFER_SETUP +
                                      static_cast<long double>(fullK) / k0 * MODEL_K_TILE_SETUP);
    if (cost < search.best) {
        search.best = cost;
        search.plan.baseM = tileInfo.m;
        search.plan.baseN = tileInfo.n;
        search.plan.baseK = static_cast<uint32_t>(k0);
        search.plan.singleCoreM = tileInfo.tile;
        search.plan.channels = tileInfo.channels;
        search.plan.ka = static_cast<uint32_t>(ka);
        search.plan.kb = static_cast<uint32_t>(kb);
        search.plan.mDim = tileInfo.mDim;
        search.plan.nDim = tileInfo.nDim;
        search.plan.coreNum = blocks;
        search.plan.residency = static_cast<uint8_t>((aFull ? A_FULL : 0U) | (bFull ? B_FULL : 0U));
    }
}

void SearchDirectK(DirectSearch& search, const DirectTile& tile)
{
    const auto& resources = search.resources;
    if (resources.dataBytes == 0U || tile.m == 0U || tile.n == 0U) {
        return;
    }
    const uint64_t fullK = Ops::Base::CeilAlign<uint64_t>(search.runInfo.dedy_cout / search.runInfo.groups,
                                                          BASIC_BLOCK_SIZE_16);
    const uint64_t k0Max = std::min(fullK, std::min(resources.l0a / DB_ON / resources.dataBytes / tile.m,
                                                    resources.l0b / DB_ON / resources.dataBytes / tile.n));
    for (uint64_t k0 = BASIC_BLOCK_SIZE_16; k0 <= k0Max; k0 += BASIC_BLOCK_SIZE_16) {
        if (fullK % k0 != 0U) {
            continue;
        }
        for (uint32_t residency = 0U; residency < RESIDENCY_CHOICES; ++residency) {
            const bool requestA = (residency & A_FULL) != 0U;
            const bool requestB = (residency & B_FULL) != 0U;
            for (uint64_t chunk = k0; chunk <= fullK; chunk += k0) {
                if (requestA && requestB && chunk != k0) {
                    break;
                }
                RankDirect(search, tile, k0, chunk, requestA, requestB);
            }
        }
    }
}

void SearchDirectTile(DirectSearch& search, uint64_t channels, uint64_t tile, uint64_t n)
{
    const auto& runInfo = search.runInfo;
    const auto& resources = search.resources;
    if (tile == 0U || runInfo.kernel_w <= 0 || runInfo.dilation_w <= 0) {
        return;
    }
    const uint64_t stride = runInfo.stride_w;
    const uint64_t receptive = (static_cast<uint64_t>(runInfo.kernel_w) - 1U) * runInfo.dilation_w;
    if (receptive > UINT64_MAX - (tile - 1U)) {
        return;
    }
    const uint64_t span = tile - 1U + receptive;
    const uint64_t localDilation = runInfo.dilation_w / std::gcd(stride, static_cast<uint64_t>(runInfo.dilation_w));
    const uint64_t upper = Ops::Base::CeilDiv(span, stride) + (BASIC_BLOCK_SIZE_256 / resources.accBytes);
    const uint64_t lower = Ops::Base::CeilDiv(receptive, stride) + localDilation;
    if (upper > INT32_MAX || lower > INT32_MAX) {
        return;
    }
    const uint64_t m = Ops::Base::CeilAlign<uint64_t>(std::min<uint64_t>(runInfo.dedy_w, span / stride + 1U),
                                                      BASIC_BLOCK_SIZE_16);
    const uint64_t mDim = Ops::Base::CeilDiv<uint64_t>(runInfo.dedx_w, tile);
    const uint64_t nDim = Ops::Base::CeilDiv<uint64_t>(runInfo.dedx_cin / runInfo.groups, channels);
    const uint64_t tasks = DirectTasks(runInfo, mDim, nDim);
    const uint64_t outputBytes = Ops::Base::CeilAlign<uint64_t>(
        channels * Ops::Base::CeilAlign<uint64_t>(tile, BASIC_BLOCK_SIZE_16) * resources.dataBytes, BYTE_BLOCK);
    const uint64_t biasBytes = Ops::Base::CeilAlign<uint64_t>(channels * resources.accBytes, BYTE_BLOCK);
    const uint64_t pBytes = Ops::Base::CeilAlign<uint64_t>(m * n * resources.accBytes, BYTE_BLOCK);
    const uint64_t ub = DB_ON * pBytes + DB_ON * outputBytes + biasBytes;
    const uint64_t l0c = m * n * resources.accBytes;
    if (m == 0U || m > DIRECT_MAX_M || tasks == 0U || tasks > UINT32_MAX || ub > resources.ub || l0c > resources.l0c) {
        return;
    }
    SearchDirectK(search, {channels, tile, m, n, mDim, nDim, tasks, pBytes});
}

bool BuildDirect(const Conv3dBpInputV2RunInfo& runInfo, const Resources& resources, Plan& plan)
{
    plan = {};
    if (!IsDirectInput(runInfo)) {
        return false;
    }
    const uint64_t ci = runInfo.dedy_cout / runInfo.groups;
    const uint64_t co = runInfo.dedx_cin / runInfo.groups;
    const uint64_t wo = runInfo.dedx_w;
    const uint64_t kw = runInfo.kernel_w;
    const uint64_t fullK = Ops::Base::CeilAlign<uint64_t>(ci, BASIC_BLOCK_SIZE_16);
    if (resources.cores == 0U || resources.dataBytes == 0U || resources.accBytes == 0U || co * kw > UINT32_MAX ||
        ci > UINT32_MAX - (BASIC_BLOCK_SIZE_16 - 1U) ||
        !TensorFits(runInfo.batch_n, runInfo.dedy_cout, runInfo.dedy_w, resources.dataBytes) ||
        !TensorFits(runInfo.batch_n, runInfo.dedx_cin, wo, resources.dataBytes) ||
        !TensorFits(runInfo.dedy_cout, co, kw, resources.dataBytes) ||
        !TensorFits(1U, 1U, runInfo.dedx_cin, resources.accBytes)) {
        return false;
    }
    const uint64_t channelLimit = std::min<uint64_t>(co, DIRECT_MAX_N / kw);
    DirectSearch search{runInfo, resources};
    for (uint64_t channels = 1U; channels <= channelLimit; ++channels) {
        const uint64_t n = Ops::Base::CeilAlign<uint64_t>(channels * kw, BASIC_BLOCK_SIZE_16);
        const uint64_t maxTile = std::min<uint64_t>(
            DIRECT_MAX_WIDTH, std::min<uint64_t>(wo, resources.ub / (resources.accBytes * channels)));
        for (uint64_t ordinal = 0U; ordinal <= maxTile / BASIC_BLOCK_SIZE_16; ++ordinal) {
            const uint64_t tile = ordinal == 0U ? maxTile : ordinal * BASIC_BLOCK_SIZE_16;
            if (tile == 0U) {
                continue;
            }
            SearchDirectTile(search, channels, tile, n);
        }
    }
    plan = search.plan;
    return plan.coreNum != 0U && SetSegments(fullK, plan.baseK, plan);
}

// E=1 沿用原搜索，仅细化多波次的分组任务。
bool DirectGroupFits(const Conv3dBpInputV2RunInfo& runInfo, const Resources& resources, const Plan& plan)
{
    const uint64_t ci = runInfo.dedy_cout / runInfo.groups;
    const uint64_t co = runInfo.dedx_cin / runInfo.groups;
    const uint64_t e = plan.enlarge;
    const uint64_t c = e * co;
    const uint64_t k = Ops::Base::CeilAlign<uint64_t>(e * ci, BASIC_BLOCK_SIZE_16);
    const uint64_t n = Ops::Base::CeilAlign<uint64_t>(c * runInfo.kernel_w, BASIC_BLOCK_SIZE_16);
    const uint64_t m = plan.baseM;
    const uint64_t w = plan.singleCoreM;
    if (e <= 1U || e > std::min<uint64_t>(BASIC_BLOCK_SIZE_16, runInfo.groups) || c * runInfo.kernel_w > DIRECT_MAX_N ||
        plan.channels != c || plan.fullK != k || plan.baseN != n || plan.nDim != 1U || plan.ka != k || plan.kb != k ||
        plan.residency != AB_FULL || plan.baseK == 0U || k % plan.baseK != 0U || m == 0U || m > DIRECT_MAX_M ||
        w == 0U || w > DIRECT_MAX_WIDTH || k * n > UINT16_MAX ||
        n * (BASIC_BLOCK_SIZE_16 - 1U) + ((BASIC_BLOCK_SIZE_256 / resources.dataBytes / BASIC_BLOCK_SIZE_16) - 1U) >
            UINT16_MAX ||
        ci * n + co * runInfo.kernel_w >= (1U << NDDMA_UB_STRIDE_BITS) ||
        ci * co * runInfo.kernel_w >= (1ULL << NDDMA_GM_STRIDE_BITS)) {
        return false;
    }
    const uint64_t pBytes = Ops::Base::CeilAlign<uint64_t>(resources.accBytes * m * n, BYTE_BLOCK);
    const uint64_t yBytes = Ops::Base::CeilAlign<uint64_t>(
        resources.dataBytes * c * Ops::Base::CeilAlign<uint64_t>(w, BASIC_BLOCK_SIZE_16), BYTE_BLOCK);
    const uint64_t packBytes = Ops::Base::CeilAlign<uint64_t>(resources.dataBytes * k * n, BYTE_BLOCK);
    const uint64_t ub = DB_ON * pBytes + DB_ON * yBytes +
                        Ops::Base::CeilAlign<uint64_t>(resources.accBytes * c, BYTE_BLOCK) +
                        FILTER_PACK_BUFFERS * packBytes + BASIC_BLOCK_SIZE_256;
    return ub <= resources.ub && resources.dataBytes * (m + n) * k <= resources.l1 &&
           (DB_ON * resources.dataBytes) * m * plan.baseK <= resources.l0a &&
           (DB_ON * resources.dataBytes) * n * plan.baseK <= resources.l0b &&
           resources.accBytes * m * n <= resources.l0c;
}

long double ScoreDirectGroup(const Conv3dBpInputV2RunInfo& runInfo, bool bias, const Resources& resources,
                             const Plan& plan)
{
    const uint64_t tasks = DirectTasks(runInfo, plan.mDim, plan.nDim, plan.enlarge);
    const uint64_t blocks = std::min<uint64_t>(tasks, resources.cores);
    const uint64_t waves = Ops::Base::CeilDiv(tasks, blocks);
    const bool dual = tasks <= blocks;
    const uint64_t ci = runInfo.dedy_cout / runInfo.groups;
    const uint64_t co = runInfo.dedx_cin / runInfo.groups;
    const uint64_t c = plan.channels;
    const uint64_t w = plan.singleCoreM;
    const uint64_t k = plan.fullK;
    const uint64_t stride = runInfo.stride_w;
    const uint64_t kw = runInfo.kernel_w;
    const uint64_t laneC = dual ? Ops::Base::CeilDiv(c, static_cast<uint64_t>(resources.vectorRatio)) : c;
    const uint64_t phase = std::min(w, stride);
    const uint64_t vectors = w / ((BASIC_BLOCK_SIZE_256 / resources.accBytes) * stride) * stride +
                             std::min(w % ((BASIC_BLOCK_SIZE_256 / resources.accBytes) * stride), stride);
    const uint64_t taps = Ops::Base::CeilDiv(w, (BASIC_BLOCK_SIZE_256 / resources.accBytes) * stride) * kw;
    const long double matrix = static_cast<long double>(plan.baseM) * plan.baseN;
    const long double gm = static_cast<long double>(runInfo.a_dtype_bytes) * plan.baseM * plan.enlarge * ci +
                           (plan.enlarge == 1U ?
                                static_cast<long double>(runInfo.a_dtype_bytes) * ci * c * kw :
                                static_cast<long double>(runInfo.a_dtype_bytes) * plan.enlarge * ci * co * kw);
    const long double pack = plan.enlarge > 1U ?
                                 MODEL_PACK_OPERATIONS * k * plan.baseN / MODEL_TRANSFER_THROUGHPUT + MODEL_PACK_SETUP :
                                 0.0L;
    return waves * (MAC_OPERATIONS * matrix * k / MODEL_CUBE_THROUGHPUT + gm / MODEL_TRANSFER_THROUGHPUT +
                    static_cast<long double>(runInfo.c_dtype_bytes) * matrix * (dual ? resources.vectorRatio : 1U) /
                        MODEL_TRANSFER_THROUGHPUT +
                    static_cast<long double>(runInfo.a_dtype_bytes) * c * w / MODEL_TRANSFER_THROUGHPUT +
                    (MODEL_VECTOR_SETUP + (bias ? MODEL_BIAS_SETUP : 0U)) * laneC * vectors +
                    MODEL_TAP_SETUP * laneC * taps + MODEL_PHASE_SETUP * phase + MODEL_TASK_SETUP +
                    MODEL_GROUP_TRANSFER_SETUP *
                        (Ops::Base::CeilDiv<uint64_t>(k, plan.ka) + Ops::Base::CeilDiv<uint64_t>(k, plan.kb)) +
                    MODEL_K_TILE_SETUP * k / plan.baseK + pack);
}

Plan MakeGroupPlan(uint64_t e, uint64_t c, uint64_t k, uint64_t n, uint64_t m, uint64_t tile, uint64_t mDim,
                   uint64_t cores, uint64_t tasks, uint64_t k0)
{
    Plan candidate{};
    if (m == 0U || m > DIRECT_MAX_M) {
        return candidate;
    }
    candidate.baseM = static_cast<uint32_t>(m);
    candidate.baseN = n;
    candidate.baseK = k0;
    candidate.singleCoreM = tile;
    candidate.channels = c;
    candidate.ka = k;
    candidate.kb = k;
    candidate.mDim = mDim;
    candidate.nDim = 1U;
    candidate.coreNum = std::min(cores, tasks);
    candidate.residency = AB_FULL;
    candidate.enlarge = e;
    SetSegments(k, k0, candidate);
    return candidate;
}

void SearchGroupTile(const Conv3dBpInputV2RunInfo& runInfo, bool bias, const Resources& resources, uint64_t e,
                     uint64_t tile, long double& best, Plan& plan)
{
    const uint64_t ci = runInfo.dedy_cout / runInfo.groups;
    const uint64_t c = e * (runInfo.dedx_cin / runInfo.groups);
    const uint64_t k = Ops::Base::CeilAlign<uint64_t>(e * ci, BASIC_BLOCK_SIZE_16);
    const uint64_t n = Ops::Base::CeilAlign<uint64_t>(c * runInfo.kernel_w, BASIC_BLOCK_SIZE_16);
    const uint64_t receptive = static_cast<uint64_t>(runInfo.kernel_w - 1) * runInfo.dilation_w;
    const uint64_t m = Ops::Base::CeilAlign<uint64_t>(
        std::min<uint64_t>(runInfo.dedy_w, (tile - 1U + receptive) / runInfo.stride_w + 1U), BASIC_BLOCK_SIZE_16);
    const uint64_t mDim = Ops::Base::CeilDiv<uint64_t>(runInfo.dedx_w, tile);
    const uint64_t tasks = DirectTasks(runInfo, mDim, 1U, e);
    if (tasks == 0U || m == 0U || n == 0U || resources.dataBytes == 0U) {
        return;
    }
    const uint64_t k0Max = std::min(
        k, std::min(resources.l0a / DB_ON / resources.dataBytes / m, resources.l0b / DB_ON / resources.dataBytes / n));
    for (uint64_t k0 = BASIC_BLOCK_SIZE_16; k0 <= k0Max; k0 += BASIC_BLOCK_SIZE_16) {
        if (k % k0 != 0U) {
            continue;
        }
        Plan candidate = MakeGroupPlan(e, c, k, n, m, tile, mDim, resources.cores, tasks, k0);
        if (!DirectGroupFits(runInfo, resources, candidate)) {
            continue;
        }
        const long double score = ScoreDirectGroup(runInfo, bias, resources, candidate);
        if (score < best) {
            best = score;
            plan = candidate;
        }
    }
}

void EnlargeGroups(const Conv3dBpInputV2RunInfo& runInfo, bool bias, const Resources& resources, Plan& plan)
{
    if (runInfo.groups <= 1 || DirectTasks(runInfo, plan.mDim, plan.nDim) <= resources.cores) {
        return;
    }
    const uint64_t ci = runInfo.dedy_cout / runInfo.groups;
    const uint64_t co = runInfo.dedx_cin / runInfo.groups;
    const uint64_t eMax = std::min<uint64_t>(
        runInfo.groups, std::lcm(BASIC_BLOCK_SIZE_16 / std::gcd<uint64_t>(ci, BASIC_BLOCK_SIZE_16),
                                 BASIC_BLOCK_SIZE_16 / std::gcd<uint64_t>(co, BASIC_BLOCK_SIZE_16)));
    const uint64_t baseK = Ops::Base::CeilAlign<uint64_t>(ci, BASIC_BLOCK_SIZE_16);
    const uint64_t baseN = Ops::Base::CeilAlign<uint64_t>(co * runInfo.kernel_w, BASIC_BLOCK_SIZE_16);
    long double best = GROUP_REFINE_MAX_COST_RATIO * ScoreDirectGroup(runInfo, bias, resources, plan);
    for (uint64_t e = MIN_GROUP_ENLARGE; e <= eMax; ++e) {
        const uint64_t c = e * co;
        const uint64_t k = Ops::Base::CeilAlign<uint64_t>(e * ci, BASIC_BLOCK_SIZE_16);
        const uint64_t n = Ops::Base::CeilAlign<uint64_t>(c * runInfo.kernel_w, BASIC_BLOCK_SIZE_16);
        if (c * runInfo.kernel_w > DIRECT_MAX_N ||
            REFINE_RATIO_DENOMINATOR * k * n > REFINE_MAX_OVERHEAD_NUMERATOR * e * baseK * baseN) {
            continue;
        }
        const uint64_t maxTile = std::min<uint64_t>(
            DIRECT_MAX_WIDTH, std::min<uint64_t>(runInfo.dedx_w, resources.ub / (resources.accBytes * c)));
        for (uint64_t ordinal = 0U; ordinal <= maxTile / BASIC_BLOCK_SIZE_16; ++ordinal) {
            const uint64_t tile = ordinal == 0U ? maxTile : ordinal * BASIC_BLOCK_SIZE_16;
            if (tile == 0U) {
                continue;
            }
            SearchGroupTile(runInfo, bias, resources, e, tile, best, plan);
        }
    }
}

bool AdmitDirectMultiWave(const Conv3dBpInputV2RunInfo& runInfo)
{
    constexpr uint64_t computeResidentChannels = COMPUTE_RESIDENT_CHANNEL_BLOCKS * BASIC_BLOCK_SIZE_16;
    return static_cast<uint64_t>(runInfo.dedy_cout) >= computeResidentChannels ||
           (runInfo.kernel_w == 1 && runInfo.dedx_cin == 1);
}

bool AdmitDirectPerformance(const Conv3dBpInputV2RunInfo& runInfo, const Plan& plan, uint64_t cores)
{
    constexpr uint64_t fewTaps = 2U;
    constexpr uint64_t amortizedTaps = 4U;
    constexpr uint64_t smallRows = 2U * BASIC_BLOCK_SIZE_16;
    constexpr uint64_t fanoutChannels = 8U * BASIC_BLOCK_SIZE_16;
    constexpr uint64_t maxShortFanout = 4U;
    constexpr uint64_t smallRowKSegments = 4U;
    const uint64_t ci = static_cast<uint64_t>(runInfo.dedy_cout / runInfo.groups);
    const uint64_t co = static_cast<uint64_t>(runInfo.dedx_cin / runInfo.groups);
    const uint64_t taps = Ops::Base::CeilDiv<uint64_t>(runInfo.kernel_w, runInfo.stride_w);
    const bool multipleWaves = DirectTasks(runInfo, plan.mDim, plan.nDim, plan.enlarge) > cores;
    if (plan.enlarge > 1U) {
        return !multipleWaves && (plan.enlarge < BASIC_BLOCK_SIZE_16 / 2U || runInfo.batch_n == 1 ||
                                  runInfo.stride_w == 1 || runInfo.kernel_w == runInfo.stride_w);
    }
    const bool shortFanout = co >= fanoutChannels && runInfo.dedy_w < BASIC_BLOCK_SIZE_16 &&
                             (co > maxShortFanout * ci || multipleWaves);
    const bool smallWork = (runInfo.stride_w > 1 && runInfo.kernel_w <= runInfo.stride_w &&
                            runInfo.dedx_w < BASIC_BLOCK_SIZE_16) ||
                           (ci <= BASIC_BLOCK_SIZE_16 && co <= smallRows && multipleWaves) ||
                           (plan.baseM <= smallRows && plan.kSegments > smallRowKSegments);
    const bool sparseSetup = ci < BASIC_BLOCK_SIZE_16 && co < BASIC_BLOCK_SIZE_16 &&
                             runInfo.stride_w >= BASIC_BLOCK_SIZE_16 && plan.baseM <= smallRows &&
                             taps <= amortizedTaps;
    return !(taps <= fewTaps && (shortFanout || smallWork)) && !sparseSetup;
}

bool RejectDirectSetup(const Conv3dBpInputV2RunInfo& runInfo, uint64_t live, uint64_t tile,
                       optiling::Conv3DBackpropInputArch35TilingData& data, bool bias, uint64_t inputBankLimit)
{
    constexpr uint64_t phaseBudget = 32U;             // 每个 C0^3 矩阵块的经验启动预算。
    constexpr uint64_t phaseSpanBudget = 2U;          // phase 复用保护的经验 C0 跨度。
    constexpr uint64_t shortMatrixBudget = 3U;        // 短输入和归约的经验 C0 跨度。
    constexpr uint64_t amortizedReductionBlocks = 4U; // 非重叠 E=1 的经验归约复用块数。
    constexpr uint64_t outputPerMatrixBudget = 32U;   // 每个 C0^3 MAC 的经验输出预算。
    const uint64_t c0 = data.get_c0();
    const uint64_t stride = runInfo.stride_w;
    const uint64_t kernel = runInfo.kernel_w;
    const uint64_t tapCount = std::min(kernel, stride);
    const uint64_t fullK = data.get_kValueSegment();
    const uint64_t matrixUnits = (data.get_baseM() / c0) * (data.get_baseN() / c0) * (fullK / c0);
    const uint64_t phaseWork = tapCount * (tapCount + 1U) / 2U + (stride - tapCount) * tapCount +
                               (data.get_baseM() > c0 ? stride + kernel : 0U);
    // 带 bias 的短周期、未扩组的深归约属于实测保护区；2/4 个 C0 是经验界。
    const bool phaseReuse = bias ? stride <= phaseSpanBudget * c0 :
                                   stride < phaseSpanBudget * c0 && kernel <= stride && data.get_enlarge() == 1U &&
                                       fullK >= amortizedReductionBlocks * c0;
    // 大展开输入回退代价高，超过 L1 双缓冲单 bank 的经验尺度则保留 DIRECT；不改资源合法性。
    const bool phaseSetup = stride > c0 && kernel <= stride + stride && !phaseReuse && live <= inputBankLimit &&
                            phaseWork >= phaseBudget * matrixUnits;
    // 单 tap 短输入、浅归约难摊销搬运与收集，3 个 C0 为经验边界。
    const bool shortInput = !bias && kernel == 1U && data.get_mDim() > 1U && runInfo.dedx_cin > runInfo.groups &&
                            live <= shortMatrixBudget * c0 && fullK <= shortMatrixBudget * c0;
    const uint64_t inputChannels = static_cast<uint64_t>(runInfo.dedy_cout / runInfo.groups);
    const uint64_t matrixWork = uint64_t{data.get_baseM()} * data.get_baseN() * inputChannels;
    const uint64_t outputWork = tile * data.get_singleCoreCin() * c0 * c0 * c0;
    // 多 tap 浅归约按输出/矩阵工作量比较，32 为经验预算，不保证任意规格收益。
    const bool outputSetup = !bias && kernel > 1U && kernel <= stride + 1U && data.get_enlarge() == 1U &&
                             data.get_singleCoreCin() > 1U && fullK <= c0 &&
                             outputWork >= outputPerMatrixBudget * matrixWork;
    // phase 扫描和输出预算仅用于包含完整 stride 周期的 tile；短输入独立判断。
    return shortInput || (tile >= stride && (phaseSetup || outputSetup));
}

} // namespace Polyphase

} // namespace Conv
} // namespace NN
} // namespace Ops
