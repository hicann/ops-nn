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
 * \file transpose_batch_mat_mul_w8a8_tiling.h
 * \brief W8A8 host tiling 入口与选块。
 */

#ifndef TRANSPOSE_BATCH_MAT_MUL_W8A8_TILING_H
#define TRANSPOSE_BATCH_MAT_MUL_W8A8_TILING_H

#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

#include "exe_graph/runtime/tiling_context.h"
#include "../../op_kernel/transpose_batch_mat_mul_tiling_data.h"

#ifndef PP_W8A8_CONSTANTS_H
#define PP_W8A8_CONSTANTS_H
namespace PpW8A8 {
constexpr unsigned L1_K = 512;
constexpr unsigned L1_K_ND = 256;
constexpr unsigned L1_SLOTS_MIN = 2;
constexpr unsigned L1_SLOTS_MAX = 8;
constexpr unsigned L0_K = 128;
constexpr unsigned L0C_SLOT_BYTES = 65536;
} // namespace PpW8A8
#endif

namespace PpW8A8 {

inline constexpr unsigned L1Slots(unsigned tileM, unsigned tileN, unsigned l1)
{
    const unsigned nPad = (tileN + 31U) / 32U * 32U;
    const unsigned slotBytes = (tileM + nPad) * L1_K;
    if (slotBytes == 0 || l1 / slotBytes < L1_SLOTS_MIN) {
        return 0;
    }
    const unsigned slots = l1 / slotBytes;
    return slots > L1_SLOTS_MAX ? L1_SLOTS_MAX : slots;
}

struct Capacity {
    uint32_t cores;
    uint64_t l1, l0a, l0b, l0c, ub;
};
struct Tile {
    uint32_t m = 0, n = 0, ubM = 0, cores = 0, l0cBuffers = 0;
    double cost = std::numeric_limits<double>::infinity();
};
inline uint64_t DivUp(uint64_t a, uint64_t b) { return (a - 1) / b + 1; }
inline uint64_t Gcd(uint64_t a, uint64_t b)
{
    while (b) {
        const uint64_t t = a % b;
        a = b;
        b = t;
    }
    return a;
}
inline uint32_t VectorRows(uint32_t m, uint32_t n, uint64_t ub)
{
    uint32_t rows = std::min(m, 64U);
    while (rows) {
        const uint64_t alignedRows = (rows + 7U) / 8U * 8U;
        const uint64_t bytes = 20ULL * rows * n + 32768 + 8ULL * n + 8ULL * alignedRows + 2048;
        if (bytes <= ub) {
            return rows;
        }
        rows /= 2;
    }
    return 0;
}
inline bool Fits(uint32_t m, uint32_t n, const Capacity& cap, uint32_t buffers = 1)
{
    const uint64_t limit = buffers == 2 ? L0C_SLOT_BYTES : std::min<uint64_t>(cap.l0c, 2ULL * L0C_SLOT_BYTES);
    return (buffers == 1 || buffers == 2) && cap.cores && (buffers == 1 || cap.l0c >= 2ULL * L0C_SLOT_BYTES) &&
           4ULL * m * n <= limit && 1ULL * m * L0_K <= 32768 && 32768ULL + m * L0_K <= cap.l0a &&
           1ULL * n * L0_K <= 32768 && 32768ULL + n * L0_K <= cap.l0b &&
           L1Slots(m, n, static_cast<unsigned>(cap.l1)) >= L1_SLOTS_MIN;
}
inline double TaskCost(uint32_t m, uint32_t n, uint32_t k, uint32_t ubM, double aWeight = 2.0)
{
    const uint64_t mr = (m + 15U) / 16U * 16U;
    constexpr double UNALIGNED_OUTPUT_COST = 24.0;
    const double outputIssueCost = n % 32 ? UNALIGNED_OUTPUT_COST * m * n : 0.0;
    return double(k) * (aWeight * m + n) + 10.0 * m * n + outputIssueCost + 4.0 * n * DivUp(m, ubM) + 4.0 * m +
           double(mr) * n * k / 128.0 + 256.0 * DivUp(k, L0_K) + 4096.0;
}
inline Tile Evaluate(uint32_t m, uint32_t b, uint32_t k, uint32_t n, uint32_t tm, uint32_t tn, const Capacity& cap,
                     uint32_t buffers = 1)
{
    Tile tile;
    if (!Fits(tm, tn, cap, buffers)) {
        return tile;
    }
    tile.ubM = VectorRows(std::min(m, tm), std::min(n, tn), cap.ub);
    if (!tile.ubM) {
        return tile;
    }
    const uint64_t mc = DivUp(m, tm);
    const uint64_t nc = DivUp(n, tn);
    const uint64_t tasks = uint64_t(b) * mc * nc;
    tile.m = tm;
    tile.n = tn;
    tile.l0cBuffers = buffers;
    tile.cores = std::min<uint64_t>(cap.cores, tasks);
    std::vector<double> load(tile.cores, 0.0);
    const double aWeight = b > 1 ? 2.0 : 1.0;
    const uint64_t period = tile.cores / Gcd(mc * nc, tile.cores);
    for (uint64_t bi = 0; bi < std::min<uint64_t>(b, period); ++bi) {
        const uint64_t copies = b / period + (bi < b % period);
        for (uint64_t mi = 0; mi < mc; ++mi) {
            const uint32_t cm = std::min<uint64_t>(tm, m - mi * tm);
            const double full = TaskCost(cm, tn, k, tile.ubM, aWeight) * copies;
            const uint64_t start = ((bi * mc + mi) * nc) % tile.cores;
            const uint64_t q = (nc - 1) / tile.cores;
            const uint64_t rem = (nc - 1) % tile.cores;
            for (uint32_t c = 0; c < tile.cores; ++c) {
                load[c] += q * full;
            }
            for (uint32_t i = 0; i < rem; ++i) {
                load[(start + i) % tile.cores] += full;
            }
            const uint32_t tailN = n - (nc - 1) * tn;
            load[(start + nc - 1) % tile.cores] += TaskCost(cm, tailN, k, tile.ubM, aWeight) * copies;
        }
    }
    tile.cost = *std::max_element(load.begin(), load.end()) +
                std::accumulate(load.begin(), load.end(), 0.0) / tile.cores;
    return tile;
}
inline Tile Select(uint32_t m, uint32_t b, uint32_t k, uint32_t n, const Capacity& cap)
{
    Tile best;
    const uint32_t maxTileM = std::min<uint64_t>((uint64_t(m) + 15U) / 16U * 16U, 32768U / L0_K);
    for (uint32_t tm = 16; tm <= maxTileM; tm += 16) {
        for (uint32_t tn = 16; tn <= std::min(n, 256U); tn += 16) {
            Tile candidate = Evaluate(m, b, k, n, tm, tn, cap, 1);
            if (candidate.cost < best.cost) {
                best = candidate;
            }
        }
    }
    return best;
}

inline Tile SelectNd(uint32_t m, uint32_t b, uint32_t k, uint32_t n, const Capacity& cap)
{
    (void)k;
    Tile best;
    const uint32_t aic = cap.cores ? cap.cores : 1U;
    const uint64_t l0c = cap.l0c ? cap.l0c : 131072ULL;
    const uint64_t ub = cap.ub ? cap.ub : 196608ULL;

    auto pack = [&](uint32_t tm, uint32_t tn, uint32_t buffers) -> Tile {
        Tile tile;
        if (tm == 0 || tn == 0 || buffers == 0) {
            return tile;
        }
        if (4ULL * tm * tn * buffers > l0c) {
            return tile;
        }
        if (tm * L0_K > 32768U || tn * L0_K > 32768U) {
            return tile;
        }
        tile.ubM = VectorRows(std::min(m, tm), std::min(n, tn), ub);
        if (!tile.ubM) {
            return tile;
        }
        const uint64_t mc = DivUp(m, tm);
        const uint64_t nc = DivUp(n, tn);
        const uint64_t groups = DivUp(nc, buffers);
        const uint64_t total = std::max<uint64_t>(1, static_cast<uint64_t>(b) * mc * groups);
        tile.m = tm;
        tile.n = tn;
        tile.l0cBuffers = buffers;
        tile.cores = static_cast<uint32_t>(std::min<uint64_t>(aic, total));
        const uint64_t q = total / tile.cores;
        const uint64_t r = total % tile.cores;
        const uint64_t maxTasks = r ? q + 1 : q;
        const uint32_t copyN = static_cast<uint32_t>(std::min<uint64_t>(n, static_cast<uint64_t>(buffers) * tn));
        tile.cost = static_cast<double>(maxTasks) * 1e18 + static_cast<double>(mc) * 1e15 +
                    (256.0 - static_cast<double>(copyN)) * 1e12 + static_cast<double>(r) * 1e6;
        return tile;
    };

    auto take = [&](const Tile& candidate) {
        if (candidate.m != 0 && candidate.cost < best.cost) {
            best = candidate;
        }
    };

    const uint32_t tmFull = std::max(16U, (m + 15U) / 16U * 16U);
    const uint32_t nBuf[] = {2U, 1U};
    for (uint32_t buffers : nBuf) {
        take(pack(tmFull, 256U, buffers));
    }
    const uint32_t mPair[] = {64U, 48U, 80U, 96U};
    for (uint32_t tm : mPair) {
        take(pack(tm, 256U, 2U));
    }
    const uint32_t mOnce[] = {128U};
    for (uint32_t tm : mOnce) {
        take(pack(tm, 256U, 1U));
    }
    if (best.m == 0) {
        take(pack(tmFull, std::min(n, 256U), 1U));
    }
    return best;
}
} // namespace PpW8A8

namespace optiling {
namespace transpose_batch_mat_mul {

constexpr uint32_t W8A8_BUFFER_NUM = 2;
constexpr uint32_t W8A8_WORKSPACE_LIMIT = 80 * 1024 * 1024;
constexpr uint32_t W8A8_INT8_K_ALIGN = 32;
constexpr uint32_t W8A8_INT8_N_ALIGN = 16;
constexpr uint32_t W8A8_M_ALIGN = 16;
constexpr uint32_t W8A8_A2_BATCH_K_LIMIT = 65536;

bool IsW8A8Mode(gert::TilingContext* context);
ge::graphStatus TransposeBatchMatMulW8A8DoTiling(gert::TilingContext* context);

} // namespace transpose_batch_mat_mul
} // namespace optiling

#endif
