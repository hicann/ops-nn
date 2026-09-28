/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file gemm_syrk_base_tiling.cpp
 * \brief GemmSyrk base tiling template: DoOpTiling runs the inherited
 * BatchMatMulV3 ASW basic tiling, then clamps the block geometry to one
 * symmetric square via SyrkFloorSqrt16 (hardware caps) and
 * SearchBestBlock (calibrated cost-model scan).
 */

#include "gemm_syrk_base_tiling.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "gemm_syrk_tiling_key.h"
#include "gemm_syrk_tiling_strategy.h"
#include "matmul/common/op_host/math_util_nn.h"
#include "matmul/mat_mul_v3/op_host/op_tiling/arch35/matmul_tiling_registry.h"

namespace optiling {
namespace gemm_syrk {
using namespace strategy;

namespace {
constexpr uint64_t ATTR_TRANSPOSE_X_IDX = 2;
constexpr uint64_t SYRK_COST_SCALE = 32UL;       // cycle scale (k/32 per unit/row)
constexpr uint64_t SYRK_FETCH_FIX_CYC = 512UL;   // per GM->L1 nd2nz descriptor, per k-chunk
constexpr uint64_t SYRK_SLOT_FIX_CYC = 6000UL;   // per slot: dual fixpipe + transpose + handshake
constexpr uint64_t SYRK_WAVE_PENALTY_PCT = 15UL; // multi-round tail-imbalance premium (percent)
constexpr uint64_t SYRK_PCT_BASE = 100UL;        // percent basis for the wave premium
// Cap on the K-axis GM->L1->L0 pipeline depth (baseK * stepKa/stepKb stages)
// folded into the kIters estimate: the ASW kL1 window never exceeds this.
constexpr uint64_t SYRK_MAX_K_L1_STAGES = 4UL;

uint64_t SyrkFloorSqrt16(uint64_t capElems)
{
    uint64_t root = static_cast<uint64_t>(std::sqrt(static_cast<double>(capElems)));
    if (root > 0UL && root * root > capElems) {
        --root; // sqrt rounded up: step back onto the floor
    }
    if ((root + 1UL) * (root + 1UL) <= capElems) {
        ++root; // sqrt rounded down: step up onto the floor
    }
    return std::max(ops::FloorAlign(root, BASIC_BLOCK_SIZE_16), BASIC_BLOCK_SIZE_16);
}

uint64_t SearchBestBlock(uint64_t mValue, uint64_t kValue, uint64_t batchCnt, uint64_t hardCap, uint64_t coreBudget,
                         uint64_t kIters)
{
    uint64_t bestCost = UINT64_MAX;
    uint64_t bestBlock = BASIC_BLOCK_SIZE_16;
    bool haveOneRoundRes = false;
    thread_local std::vector<uint64_t> fractBuf;
    for (uint64_t cand = BASIC_BLOCK_SIZE_16; cand <= hardCap; cand += BASIC_BLOCK_SIZE_16) {
        const uint64_t numBlocks = ops::CeilDiv(mValue, cand);
        const uint64_t slotsPerBatch = numBlocks * (numBlocks + 1UL) / NUM_TWO;
        const uint64_t slots = slotsPerBatch * batchCnt;
        const bool isSingleRound = slots <= coreBudget;
        if (haveOneRoundRes && !isSingleRound) {
            continue;
        }
        haveOneRoundRes = haveOneRoundRes || isSingleRound;
        // Fractal row counts u_r = ceil16(rows_r) plus their running sums.
        fractBuf.assign(numBlocks, 0UL);
        uint64_t sumU = 0UL;
        uint64_t sumU2 = 0UL;
        for (uint64_t r = 0UL; r < numBlocks; ++r) {
            const uint64_t rows = std::min(cand, mValue - r * cand);
            fractBuf[r] = ops::CeilDiv(rows, BASIC_BLOCK_SIZE_16);
            sumU += fractBuf[r];
            sumU2 += fractBuf[r] * fractBuf[r];
        }
        uint64_t cost;
        if (isSingleRound) {
            // Every slot gets its own core: the makespan is exactly the
            // heaviest slot cost. The diagonal (0, 0) maximizes the
            // k-proportional term but an off-diagonal carries one more fetch
            // descriptor, so scan all slots.
            cost = 0UL;
            for (uint64_t i = 0UL; i < numBlocks; ++i) {
                for (uint64_t j = i; j < numBlocks; ++j) {
                    const uint64_t units = fractBuf[i] * fractBuf[j];
                    const uint64_t fetchRows = BASIC_BLOCK_SIZE_16 * (fractBuf[i] + (i == j ? 0UL : fractBuf[j]));
                    const uint64_t fixed = ((i == j) ? 1UL : NUM_TWO) * SYRK_FETCH_FIX_CYC * kIters + SYRK_SLOT_FIX_CYC;
                    cost = std::max(cost, std::max(units, fetchRows) * kValue + fixed * SYRK_COST_SCALE);
                }
            }
        } else {
            // Multi-round closed-form totals over the upper triangle
            // (u_r = ceil16 rows):
            //   totalUnits  = ((sumU^2 + sumU2) / 2)
            //   totalFetch  = 16 * (numBlocks * sumU + (sumU^2 - sumU2) / 2)
            //     (each diagonal fetches its own block once, each off-diagonal
            //      slot fetches BOTH sides)
            //   totalFetchOps = diagSlots + 2 * offSlots descriptors
            // With slots past the core count, LPT approaches
            // max(heaviest slot, total/cores); the premium covers the tail
            // imbalance across waves.
            const uint64_t offSlots = slotsPerBatch - numBlocks;
            const uint64_t totalUnits = (sumU * sumU + sumU2) / NUM_TWO;
            const uint64_t totalFetchRows = BASIC_BLOCK_SIZE_16 * (numBlocks * sumU + (sumU * sumU - sumU2) / NUM_TWO);
            const uint64_t totalFetchOps = (numBlocks + NUM_TWO * offSlots) * batchCnt;
            const uint64_t totalFixed = totalFetchOps * SYRK_FETCH_FIX_CYC * kIters + slots * SYRK_SLOT_FIX_CYC;
            const uint64_t totalCost = std::max(totalUnits, totalFetchRows) * kValue + totalFixed * SYRK_COST_SCALE;
            const uint64_t maxSlotUnits = fractBuf[0] * fractBuf[0];
            const uint64_t maxSlotFetch = BASIC_BLOCK_SIZE_16 * NUM_TWO * fractBuf[0];
            const uint64_t maxSlotFixed = ((numBlocks == 1UL) ? 1UL : NUM_TWO) * SYRK_FETCH_FIX_CYC * kIters +
                                          SYRK_SLOT_FIX_CYC;
            const uint64_t maxSlotCost = std::max(maxSlotUnits, maxSlotFetch) * kValue + maxSlotFixed * SYRK_COST_SCALE;
            cost = std::max(maxSlotCost, totalCost / std::max<uint64_t>(coreBudget, 1UL));
            cost += cost * SYRK_WAVE_PENALTY_PCT / SYRK_PCT_BASE;
        }
        if (cost < bestCost || (cost == bestCost && cand > bestBlock)) {
            bestCost = cost;
            bestBlock = cand;
        }
    }
    return std::max(bestBlock, BASIC_BLOCK_SIZE_16);
}
} // namespace

MM_REGISTER_TILING_TEMPLATE(GemmSyrk, GemmSyrkBaseTiling, DAV_3510, SYRK_BASE);

bool GemmSyrkBaseTiling::IsCapable()
{
    if (compileInfo_.aicNum == 0UL || compileInfo_.aivNum != compileInfo_.aicNum * NUM_TWO) {
        CUBE_INNER_ERR_REPORT(context_->GetNodeName(),
                              "GemmSyrk requires aicNum:aivNum = 1:2 (MIX), aicNum=%lu, aivNum=%lu",
                              compileInfo_.aicNum, compileInfo_.aivNum);
        return false;
    }
    return true;
}

ge::graphStatus GemmSyrkBaseTiling::DoOpTiling()
{
    const ge::graphStatus ret = BatchMatMulV3AswBasicTiling::DoOpTiling();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    // Clamp the K-axis L1 pipeline depth before deriving the row budgets: the
    // ASW stepKa (tuned for generic matmul) inflates the kL1 window
    // (baseK * stepKa) and squeezes the symmetric-block row budget below the
    // single-round block on long-k cases (e.g. m=n=512, k=4096 with
    // baseK=128/stepKa=6 caps l1BlockMax at 80, forcing two waves). The cost
    // model evaluates kIters at a depth of at most SYRK_MAX_K_L1_STAGES
    // anyway, so clamping here unlocks the row budget without adding any k
    // iteration; the kernel consumes the clamped kL1 via tilingData.
    runInfo_.stepKa = std::min<uint64_t>(runInfo_.stepKa, SYRK_MAX_K_L1_STAGES);

    // Symmetric square caps from the hardware budgets and the ASW basic runInfo.
    const uint64_t mAlign = ops::CeilAlign(args_.mValue, BASIC_BLOCK_SIZE_16);
    const uint64_t nAlign = ops::CeilAlign(args_.nValue, BASIC_BLOCK_SIZE_16);
    const uint64_t l0cBlockMax = SyrkFloorSqrt16(compileInfo_.l0CSize / NUM_TWO / sizeof(float));
    const uint64_t kL1 = std::max<uint64_t>(runInfo_.baseK * runInfo_.stepKa, BASIC_BLOCK_SIZE_16);
    uint64_t l1BlockMax = compileInfo_.l1Size / DB_SIZE / (NUM_TWO * args_.aDtypeSize * kL1);
    l1BlockMax = (l1BlockMax / BASIC_BLOCK_SIZE_16) * BASIC_BLOCK_SIZE_16;
    uint64_t l0BlockMax = compileInfo_.l0ASize / NUM_TWO / args_.aDtypeSize / runInfo_.baseK;
    l0BlockMax = (l0BlockMax / BASIC_BLOCK_SIZE_16) * BASIC_BLOCK_SIZE_16;

    const uint64_t hardCap = std::min({mAlign, nAlign, l0cBlockMax, l1BlockMax, l0BlockMax});
    const uint64_t coreBudget = std::min<uint64_t>(runInfo_.usedCoreNum, compileInfo_.aicNum);
    const uint64_t kL1Max = runInfo_.baseK *
                            std::min<uint64_t>({runInfo_.stepKa, runInfo_.stepKb, SYRK_MAX_K_L1_STAGES});
    const uint64_t kIters = ops::CeilDiv(args_.kValue, std::max<uint64_t>(kL1Max, 1UL));
    const uint64_t batchCnt = (args_.batchInfo == nullptr) ? 1UL : std::max<uint64_t>(args_.batchInfo->batchC, 1UL);

    const uint64_t syrkBase = SearchBestBlock(args_.mValue, args_.kValue, batchCnt, hardCap, coreBudget, kIters);
    OP_LOGI("GemmSyrk",
            "syrk clamp inputs: baseM=%lu baseN=%lu stepM=%lu stepN=%lu baseK=%lu kL1=%lu l0cBlockMax=%lu "
            "l1BlockMax=%lu l0BlockMax=%lu coreBudget=%lu kIters=%lu -> baseBlock=%lu",
            runInfo_.baseM, runInfo_.baseN, runInfo_.stepM, runInfo_.stepN, runInfo_.baseK, kL1, l0cBlockMax,
            l1BlockMax, l0BlockMax, coreBudget, kIters, syrkBase);
    runInfo_.baseM = syrkBase;
    runInfo_.baseN = syrkBase;
    return ge::GRAPH_SUCCESS;
}

uint64_t GemmSyrkBaseTiling::GetTilingKey() const
{
    const auto* attrs = context_->GetAttrs();
    const auto* transposeX = attrs == nullptr ? nullptr : attrs->GetAttrPointer<bool>(ATTR_TRANSPOSE_X_IDX);
    return GemmSyrkTilingKey().SetTrans(transposeX != nullptr && *transposeX).GetTilingKey();
}

} // namespace gemm_syrk
} // namespace optiling
