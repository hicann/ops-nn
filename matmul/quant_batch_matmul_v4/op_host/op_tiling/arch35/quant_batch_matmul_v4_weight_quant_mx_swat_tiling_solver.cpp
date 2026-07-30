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
 * \file quant_batch_matmul_v4_weight_quant_mx_swat_tiling_solver.cpp
 * \brief
 */

#include "quant_batch_matmul_v4_weight_quant_mx_swat_tiling_solver.h"

#include <algorithm>
#include <limits>

#include "quant_batch_matmul_v4_tiling.h"

namespace {
constexpr uint64_t BLOCK_CUBE_SIZE = 16UL;
constexpr uint64_t NUM_TWO = 2UL;
constexpr uint64_t DB_SIZE = 2UL;
constexpr uint64_t L1_FOUR_BUFFER = 4UL;
constexpr uint64_t STEPK_THRESHOLD = 4UL;
constexpr uint64_t BASEM_BASEN_RATIO = 2UL;
constexpr uint64_t BASEK_LIMIT = 4095UL;
constexpr uint64_t BASIC_BLOCK_SIZE_128 = 128UL;
constexpr uint64_t BASIC_BLOCK_SIZE_256 = 256UL;
constexpr uint64_t MTE2_CACHELINE_SIZE = 128UL;
constexpr uint64_t TAIL_WINDOW_LEN = 4UL;
constexpr uint64_t SCALE_FACTOR_MAX = 4UL;
constexpr uint64_t L1_HALF_SIZE = 256UL * 1024UL;
constexpr uint64_t DATA_SIZE_UINT8 = 1UL;
constexpr uint64_t DATA_SIZE_FP32 = 4UL;
constexpr uint64_t VECTOR_REG_BYTES = 256UL;
constexpr uint64_t INT4_PACK_NUM = 2UL;
constexpr uint64_t K_ALIGN_SIZE_MX_L1 = 64UL;
constexpr uint64_t K_ALIGN_SIZE_MX_BLOCK = 32UL;
constexpr uint32_t ORDER_N = 1U;

static uint64_t CeilDiv(uint64_t lhs, uint64_t rhs) { return rhs == 0UL ? 0UL : (lhs + rhs - 1UL) / rhs; }

static uint64_t Align(uint64_t value, uint64_t align) { return align == 0UL ? value : CeilDiv(value, align) * align; }

static uint64_t FloorAlign(uint64_t value, uint64_t align) { return align == 0UL ? value : value / align * align; }
} // namespace

namespace optiling {
using SwatTilingData = qbmmv4_tiling::QuantBatchMatmulV4WeightQuantMxSwatTilingData;

WeightQuantMxSwatTilingSolver::WeightQuantMxSwatTilingSolver(uint64_t targetL1BufferNum, bool allowConservativeFallback)
    : targetL1BufferNum_(targetL1BufferNum), allowConservativeFallback_(allowConservativeFallback)
{}

bool WeightQuantMxSwatTilingSolver::Solve(const WeightQuantMxSwatPlatformParam& platform,
                                          const WeightQuantMxSwatShapeParam& shape, uint64_t groupSize, bool hasBias,
                                          bool hasX1Scale, bool hasX2Scale, bool weightNz, ge::DataType yDtype,
                                          SwatTilingData& tilingData, std::string& reason)
{
    if (!Init(platform, shape, reason)) {
        return false;
    }
    hasBias_ = hasBias;
    weightNz_ = weightNz;
    biasDataSize_ = GetBiasDataSize(yDtype);
    bool foundTiling = CalcBasicBlock(reason);
    if (foundTiling) {
        OptimizeEdgeBasicBlock();
        CalcTailBasicBlock();
        foundTiling = CalcPathSpecificL1(reason) && ValidateTilingResult();
    }
    if (!foundTiling && (!allowConservativeFallback_ || !CalcConservativeTiling(reason))) {
        return false;
    }
    BuildTilingData(groupSize, hasBias, hasX1Scale, hasX2Scale, weightNz, yDtype, tilingData);
    return true;
}

bool WeightQuantMxSwatTilingSolver::CalcConservativeTiling(std::string& reason)
{
    runInfo_ = {};
    runInfo_.baseM = BLOCK_CUBE_SIZE;
    runInfo_.baseN = BLOCK_CUBE_SIZE;
    runInfo_.baseK = std::min(shape_.k, K_ALIGN_SIZE_MX_L1);
    if (!IsL0Feasible(runInfo_.baseM, runInfo_.baseN, runInfo_.baseK)) {
        reason = "minimum SWAT base shape exceeds L0A/L0B/L0C capacity";
        return false;
    }

    runInfo_.mBlockCnt = CeilDiv(shape_.m, runInfo_.baseM);
    runInfo_.nBlockCnt = CeilDiv(shape_.n, runInfo_.baseN);
    runInfo_.totalBlockCnt = runInfo_.mBlockCnt * runInfo_.nBlockCnt;
    runInfo_.tailBlockCnt = runInfo_.totalBlockCnt % platform_.aicNum;
    runInfo_.mTailSize = shape_.m - (runInfo_.mBlockCnt - 1UL) * runInfo_.baseM;
    runInfo_.nTailSize = shape_.n - (runInfo_.nBlockCnt - 1UL) * runInfo_.baseN;
    CalcTailBasicBlock();
    if (!CalcPathSpecificL1(reason) || !ValidateTilingResult()) {
        reason = "minimum SWAT base shape cannot satisfy L1/UB capacity";
        return false;
    }
    return true;
}

bool WeightQuantMxSwatTilingSolver::Init(const WeightQuantMxSwatPlatformParam& platform,
                                         const WeightQuantMxSwatShapeParam& shape, std::string& reason)
{
    if (targetL1BufferNum_ != DB_SIZE && targetL1BufferNum_ != L1_FOUR_BUFFER) {
        reason = "target L1 buffer count must be 2 or 4";
        return false;
    }
    if (shape.m == 0UL || shape.n == 0UL || shape.k == 0UL) {
        reason = "m, n, and k must be greater than zero";
        return false;
    }
    if ((shape.k % matmul_v4::K_ALIGN_SIZE_MX) != 0UL) {
        reason = "k must be aligned to 8";
        return false;
    }
    if (shape.m > std::numeric_limits<uint32_t>::max() || shape.n > std::numeric_limits<uint32_t>::max() ||
        shape.k > std::numeric_limits<uint32_t>::max()) {
        reason = "m, n, and k must not exceed UINT32_MAX";
        return false;
    }

    if (platform.aicNum == 0UL) {
        reason = "AI Core count must be greater than zero";
        return false;
    }
    if (platform.ubSize == 0UL || platform.l1Size == 0UL || platform.l0aSize == 0UL || platform.l0bSize == 0UL ||
        platform.l0cSize == 0UL) {
        reason = "UB, L1, L0A, L0B, and L0C sizes must be greater than zero";
        return false;
    }

    shape_ = shape;
    platform_ = platform;
    runInfo_ = {};
    return true;
}

bool WeightQuantMxSwatTilingSolver::CalcBasicBlock(std::string& reason)
{
    runInfo_.baseM = Align(std::min(shape_.m, BASIC_BLOCK_SIZE_256), BLOCK_CUBE_SIZE);
    runInfo_.baseN = Align(std::min(shape_.n, BASIC_BLOCK_SIZE_256), BLOCK_CUBE_SIZE);
    runInfo_.baseK = Align(std::min(shape_.k, BASIC_BLOCK_SIZE_128), matmul_v4::K_ALIGN_SIZE_MX);

    uint64_t blockNum = CeilDiv(shape_.m, runInfo_.baseM) * CeilDiv(shape_.n, runInfo_.baseN);
    if (blockNum < platform_.aicNum) {
        AdjustBasicBlock();
    }

    if (runInfo_.baseM == 0UL || runInfo_.baseN == 0UL || runInfo_.baseK == 0UL) {
        reason = "baseM, baseN, and baseK must be non-zero";
        return false;
    }
    if (!IsL0Feasible(runInfo_.baseM, runInfo_.baseN, runInfo_.baseK)) {
        reason = "base shape exceeds L0A/L0B/L0C capacity";
        return false;
    }

    runInfo_.mBlockCnt = CeilDiv(shape_.m, runInfo_.baseM);
    runInfo_.nBlockCnt = CeilDiv(shape_.n, runInfo_.baseN);
    runInfo_.totalBlockCnt = runInfo_.mBlockCnt * runInfo_.nBlockCnt;
    runInfo_.tailBlockCnt = runInfo_.totalBlockCnt % platform_.aicNum;
    runInfo_.mTailSize = shape_.m - (runInfo_.mBlockCnt - 1UL) * runInfo_.baseM;
    runInfo_.nTailSize = shape_.n - (runInfo_.nBlockCnt - 1UL) * runInfo_.baseN;
    return true;
}

void WeightQuantMxSwatTilingSolver::AdjustBasicBlock()
{
    uint64_t mMaxTile = CeilDiv(shape_.m, BLOCK_CUBE_SIZE);
    uint64_t nMaxTile = CeilDiv(shape_.n, BLOCK_CUBE_SIZE);
    uint64_t tempBaseM = runInfo_.baseM;
    uint64_t tempBaseN = runInfo_.baseN;

    uint64_t mCnt = std::max<uint64_t>(1UL, CeilDiv(shape_.m, runInfo_.baseM));
    uint64_t nCnt = std::max<uint64_t>(1UL, CeilDiv(shape_.n, runInfo_.baseN));
    if (mMaxTile > nMaxTile) {
        tempBaseN = Align(CeilDiv(shape_.n, nCnt), BLOCK_CUBE_SIZE);
        nCnt = std::max<uint64_t>(1UL, CeilDiv(shape_.n, tempBaseN));
        mCnt = std::max<uint64_t>(1UL, platform_.aicNum / nCnt);
        tempBaseM = Align(CeilDiv(shape_.m, mCnt), BLOCK_CUBE_SIZE);
    } else {
        tempBaseM = Align(CeilDiv(shape_.m, mCnt), BLOCK_CUBE_SIZE);
        mCnt = std::max<uint64_t>(1UL, CeilDiv(shape_.m, tempBaseM));
        nCnt = std::max<uint64_t>(1UL, platform_.aicNum / mCnt);
        tempBaseN = Align(CeilDiv(shape_.n, nCnt), BLOCK_CUBE_SIZE);
    }

    mCnt = std::max<uint64_t>(1UL, CeilDiv(shape_.m, tempBaseM));
    nCnt = std::max<uint64_t>(1UL, CeilDiv(shape_.n, tempBaseN));
    while (tempBaseN > tempBaseM * BASEM_BASEN_RATIO && nCnt < platform_.aicNum / NUM_TWO &&
           tempBaseN != BLOCK_CUBE_SIZE) {
        nCnt *= NUM_TWO;
        mCnt = std::max<uint64_t>(1UL, platform_.aicNum / nCnt);
        tempBaseM = Align(CeilDiv(shape_.m, mCnt), BLOCK_CUBE_SIZE);
        tempBaseN = Align(CeilDiv(shape_.n, nCnt), BLOCK_CUBE_SIZE);
        mCnt = std::max<uint64_t>(1UL, CeilDiv(shape_.m, tempBaseM));
        nCnt = std::max<uint64_t>(1UL, CeilDiv(shape_.n, tempBaseN));
    }
    while (tempBaseM >= tempBaseN * BASEM_BASEN_RATIO && mCnt < platform_.aicNum / NUM_TWO &&
           tempBaseM != BLOCK_CUBE_SIZE) {
        mCnt *= NUM_TWO;
        nCnt = std::max<uint64_t>(1UL, platform_.aicNum / mCnt);
        tempBaseM = Align(CeilDiv(shape_.m, mCnt), BLOCK_CUBE_SIZE);
        tempBaseN = Align(CeilDiv(shape_.n, nCnt), BLOCK_CUBE_SIZE);
        mCnt = std::max<uint64_t>(1UL, CeilDiv(shape_.m, tempBaseM));
        nCnt = std::max<uint64_t>(1UL, CeilDiv(shape_.n, tempBaseN));
    }

    uint64_t kAlignValue = Align(shape_.k, BASIC_BLOCK_SIZE_128);
    uint64_t kMaxValue = (platform_.l0aSize / DB_SIZE) / std::max(tempBaseM, tempBaseN);
    kMaxValue = FloorAlign(kMaxValue, BASIC_BLOCK_SIZE_128);
    if (kMaxValue >= BASIC_BLOCK_SIZE_128 && IsL0Feasible(tempBaseM, tempBaseN, std::min(kAlignValue, kMaxValue))) {
        runInfo_.baseM = tempBaseM;
        runInfo_.baseN = tempBaseN;
        runInfo_.baseK = std::min(kAlignValue, kMaxValue);
        runInfo_.baseK = runInfo_.baseK > BASEK_LIMIT ? Align(runInfo_.baseK / NUM_TWO, BASIC_BLOCK_SIZE_256) :
                                                        runInfo_.baseK;
    }
}

void WeightQuantMxSwatTilingSolver::OptimizeEdgeBasicBlock()
{
    if (runInfo_.mBlockCnt == 1UL && runInfo_.nBlockCnt == 1UL) {
        return;
    }

    bool isInnerAxisAlign = (shape_.k * DATA_SIZE_UINT8) % MTE2_CACHELINE_SIZE == 0UL;
    uint64_t mTailSize = shape_.m % runInfo_.baseM;
    if (runInfo_.mBlockCnt > 1UL && mTailSize > 0UL && isInnerAxisAlign) {
        uint64_t baseTailCntMax = std::min((runInfo_.baseM - mTailSize) / BLOCK_CUBE_SIZE, runInfo_.mBlockCnt);
        uint64_t windowSize = std::min(TAIL_WINDOW_LEN, runInfo_.mBlockCnt);
        uint64_t mainWindowNum = runInfo_.mBlockCnt / windowSize - 1UL;
        uint64_t tailWindowSize = runInfo_.mBlockCnt - mainWindowNum * windowSize;
        uint64_t perfRes = (mainWindowNum + 1UL) * runInfo_.baseM;
        uint64_t mergeWindowNum = 1UL;
        for (uint64_t mergeLen = tailWindowSize - 1UL; mergeLen < baseTailCntMax;
             mergeLen += windowSize, ++mergeWindowNum) {
            uint64_t newTailMain = Align(CeilDiv(mergeLen * runInfo_.baseM + mTailSize, mergeLen + 1UL),
                                         BLOCK_CUBE_SIZE);
            uint64_t curPerf = (mainWindowNum + 1UL - mergeWindowNum) * runInfo_.baseM + mergeWindowNum * newTailMain;
            if (curPerf <= perfRes) {
                perfRes = curPerf;
                runInfo_.mTailMain = newTailMain;
                runInfo_.mBaseTailSplitCnt = mergeLen + 1UL;
            }
        }
    }

    uint64_t nTailSize = shape_.n % runInfo_.baseN;
    if (runInfo_.nBlockCnt > 1UL && nTailSize > 0UL && isInnerAxisAlign) {
        uint64_t baseTailCntMax = std::min((runInfo_.baseN - nTailSize) / BLOCK_CUBE_SIZE, runInfo_.nBlockCnt);
        uint64_t windowSize = std::min(TAIL_WINDOW_LEN, runInfo_.nBlockCnt);
        uint64_t mainWindowNum = runInfo_.nBlockCnt / windowSize - 1UL;
        uint64_t tailWindowSize = runInfo_.nBlockCnt - mainWindowNum * windowSize;
        uint64_t perfRes = (mainWindowNum + 1UL) * runInfo_.baseN;
        uint64_t mergeWindowNum = 1UL;
        for (uint64_t mergeLen = tailWindowSize - 1UL; mergeLen < baseTailCntMax;
             mergeLen += windowSize, ++mergeWindowNum) {
            uint64_t newTailMain = Align(CeilDiv(mergeLen * runInfo_.baseN + nTailSize, mergeLen + 1UL),
                                         BLOCK_CUBE_SIZE);
            uint64_t curPerf = (mainWindowNum + 1UL - mergeWindowNum) * runInfo_.baseN + mergeWindowNum * newTailMain;
            if (curPerf <= perfRes) {
                perfRes = curPerf;
                runInfo_.nTailMain = newTailMain;
                runInfo_.nBaseTailSplitCnt = mergeLen + 1UL;
            }
        }
    }
}

void WeightQuantMxSwatTilingSolver::CalcTailBasicBlock()
{
    if (runInfo_.tailBlockCnt == 0UL) {
        return;
    }

    uint64_t mTile = 1UL;
    uint64_t nTile = 1UL;
    uint64_t preSplit = 1UL;
    uint64_t secSplit = 1UL;
    uint64_t& preSplitValid = runInfo_.mTailSize >= runInfo_.nTailSize ? mTile : nTile;
    uint64_t& secSplitValid = runInfo_.mTailSize >= runInfo_.nTailSize ? nTile : mTile;
    uint64_t mTileMax = CeilDiv(runInfo_.baseM, BLOCK_CUBE_SIZE);
    uint64_t nTileMax = CeilDiv(runInfo_.baseN, BLOCK_CUBE_SIZE);
    uint64_t preSplitMax = runInfo_.mTailSize >= runInfo_.nTailSize ? mTileMax : nTileMax;
    uint64_t secSplitMax = runInfo_.mTailSize >= runInfo_.nTailSize ? nTileMax : mTileMax;
    bool splitMFirst = runInfo_.mTailSize >= runInfo_.nTailSize;
    bool updated = true;
    while (updated) {
        updated = false;
        uint64_t currentUsedCoreNum = CalUsedCoreNum(mTile, nTile);
        uint64_t preCandidateM = splitMFirst ? preSplit + 1UL : secSplit;
        uint64_t preCandidateN = splitMFirst ? secSplit : preSplit + 1UL;
        uint64_t preCandidateUsedCoreNum = CalUsedCoreNum(preCandidateM, preCandidateN);
        if (preSplit < preSplitMax && preCandidateUsedCoreNum <= platform_.aicNum &&
            preCandidateUsedCoreNum > currentUsedCoreNum) {
            preSplitValid = ++preSplit;
            updated = true;
            currentUsedCoreNum = preCandidateUsedCoreNum;
        }
        uint64_t secCandidateM = splitMFirst ? preSplit : secSplit + 1UL;
        uint64_t secCandidateN = splitMFirst ? secSplit + 1UL : preSplit;
        uint64_t secCandidateUsedCoreNum = CalUsedCoreNum(secCandidateM, secCandidateN);
        if (secSplit < secSplitMax && secCandidateUsedCoreNum <= platform_.aicNum &&
            secCandidateUsedCoreNum > currentUsedCoreNum) {
            secSplitValid = ++secSplit;
            updated = true;
        }
    }

    runInfo_.mTailTile = mTile;
    runInfo_.nTailTile = nTile;
}

bool WeightQuantMxSwatTilingSolver::CalcPathSpecificL1(std::string& reason)
{
    uint64_t maxStepK = std::min(STEPK_THRESHOLD, CeilDiv(shape_.k, runInfo_.baseK));
    for (uint64_t stepK = maxStepK; stepK > 0UL; --stepK) {
        uint64_t kBl1Size = std::min(shape_.k, stepK * runInfo_.baseK);
        uint64_t nBl1Size = std::min(shape_.n, runInfo_.baseN);
        uint64_t nBubSize = 0UL;
        uint64_t kBubSize = 0UL;
        if (!FindBubSize(nBl1Size, kBl1Size, nBubSize, kBubSize)) {
            continue;
        }

        uint64_t maxScaleFactor = CalcMaxScaleFactor(stepK);
        for (uint64_t scaleFactor = maxScaleFactor; scaleFactor > 0UL; --scaleFactor) {
            uint64_t tileShapeKL1 = stepK * runInfo_.baseK;
            uint64_t tileShapeScaleKL1 = tileShapeKL1 * scaleFactor;
            if (IsL1Feasible(tileShapeKL1, tileShapeScaleKL1)) {
                runInfo_.tileShapeKL1 = tileShapeKL1;
                runInfo_.tileShapeScaleKL1 = tileShapeScaleKL1;
                runInfo_.nBubSize = nBubSize;
                runInfo_.kBubSize = kBubSize;
                return true;
            }
        }
    }
    reason = "cannot satisfy L1 and path-specific UB capacity constraints";
    return false;
}

uint64_t WeightQuantMxSwatTilingSolver::CalcMaxScaleFactor(uint64_t stepK) const
{
    uint64_t kL1Size = stepK * runInfo_.baseK;
    return std::max<uint64_t>(1UL, std::min(SCALE_FACTOR_MAX, CeilDiv(shape_.k, kL1Size)));
}

bool WeightQuantMxSwatTilingSolver::ValidateTilingResult() const
{
    bool hasValidTileShape = runInfo_.baseM > 0UL && runInfo_.baseN > 0UL && runInfo_.baseK > 0UL &&
                             runInfo_.baseK % matmul_v4::K_ALIGN_SIZE_MX == 0UL && runInfo_.tileShapeKL1 > 0UL &&
                             runInfo_.tileShapeKL1 % runInfo_.baseK == 0UL && runInfo_.tileShapeScaleKL1 > 0UL &&
                             runInfo_.tileShapeScaleKL1 % runInfo_.tileShapeKL1 == 0UL;
    if (!hasValidTileShape) {
        return false;
    }
    uint64_t nBl1Size = std::min(shape_.n, runInfo_.baseN);
    uint64_t kBl1Size = std::min(shape_.k, runInfo_.tileShapeKL1);
    uint64_t expectedN = 0UL;
    uint64_t expectedK = 0UL;
    bool hasValidBub = FindBubSize(nBl1Size, kBl1Size, expectedN, expectedK);
    return IsL0Feasible(runInfo_.baseM, runInfo_.baseN, runInfo_.baseK) &&
           IsL1Feasible(runInfo_.tileShapeKL1, runInfo_.tileShapeScaleKL1) && hasValidBub &&
           runInfo_.nBubSize == expectedN && runInfo_.kBubSize == expectedK;
}

bool WeightQuantMxSwatTilingSolver::IsL0Feasible(uint64_t baseM, uint64_t baseN, uint64_t baseK) const
{
    uint64_t a2Size = baseM * baseK * DB_SIZE;
    uint64_t b2Size = baseN * baseK * DB_SIZE;
    uint64_t cSize = baseM * baseN * DATA_SIZE_FP32;
    return a2Size <= platform_.l0aSize && b2Size <= platform_.l0bSize && cSize <= platform_.l0cSize;
}

bool WeightQuantMxSwatTilingSolver::IsL1Feasible(uint64_t tileShapeKL1, uint64_t tileShapeScaleKL1) const
{
    if (platform_.l1Size < L1_HALF_SIZE) {
        return false;
    }
    uint64_t kL1SizeAligned = Align(tileShapeKL1, K_ALIGN_SIZE_MX_L1);
    uint64_t scaleKL1SizeAligned = Align(tileShapeScaleKL1, K_ALIGN_SIZE_MX_L1);
    uint64_t aL1Size = runInfo_.baseM * kL1SizeAligned * DATA_SIZE_UINT8;
    uint64_t bL1Size = runInfo_.baseN * kL1SizeAligned * DATA_SIZE_UINT8;
    uint64_t scaleAL1Size = runInfo_.baseM * scaleKL1SizeAligned * DATA_SIZE_UINT8 / matmul_v4::MX_GROUP_SIZE;
    uint64_t scaleBL1Size = runInfo_.baseN * scaleKL1SizeAligned * DATA_SIZE_UINT8 / matmul_v4::MX_GROUP_SIZE;
    uint64_t biasL1Size = hasBias_ ? Align(runInfo_.baseN, BLOCK_CUBE_SIZE) * biasDataSize_ : 0UL;
    uint64_t buffersPerHalf = targetL1BufferNum_ / NUM_TWO;
    uint64_t halfL1Use = buffersPerHalf * (aL1Size + bL1Size + biasL1Size) + scaleAL1Size + scaleBL1Size;
    // Kernel buffer 1 starts at the fixed 256 KiB boundary, so the shorter physical half sets the limit.
    uint64_t halfL1Limit = std::min(L1_HALF_SIZE, platform_.l1Size - L1_HALF_SIZE);
    return halfL1Use <= halfL1Limit;
}

bool WeightQuantMxSwatTilingSolver::IsBubTilingValid(uint64_t nBubSize, uint64_t kBubSize) const
{
    return nBubSize > 0UL && kBubSize > 0UL && kBubSize % matmul_v4::K_ALIGN_SIZE_MX == 0UL &&
           GetBubSize(targetL1BufferNum_, nBubSize, kBubSize) <= platform_.ubSize;
}

void WeightQuantMxSwatTilingSolver::BuildTilingData(uint64_t groupSize, bool hasBias, bool hasX1Scale, bool hasX2Scale,
                                                    bool weightNz, ge::DataType yDtype,
                                                    SwatTilingData& tilingData) const
{
    tilingData = {};
    tilingData.m = static_cast<uint32_t>(shape_.m);
    tilingData.n = static_cast<uint32_t>(shape_.n);
    tilingData.k = static_cast<uint32_t>(shape_.k);
    tilingData.baseM = static_cast<uint32_t>(runInfo_.baseM);
    tilingData.baseN = static_cast<uint32_t>(runInfo_.baseN);
    tilingData.baseK = static_cast<uint32_t>(runInfo_.baseK);
    tilingData.tileShapeKL1 = static_cast<uint32_t>(runInfo_.tileShapeKL1);
    tilingData.tileShapeScaleKL1 = static_cast<uint32_t>(runInfo_.tileShapeScaleKL1);
    tilingData.usedCoreNum = static_cast<uint32_t>(runInfo_.totalBlockCnt >= platform_.aicNum ?
                                                       platform_.aicNum :
                                                       CalUsedCoreNum(runInfo_.mTailTile, runInfo_.nTailTile));
    tilingData.cubeNumBlocksM = static_cast<uint32_t>(runInfo_.mBlockCnt);
    tilingData.cubeNumBlocksN = static_cast<uint32_t>(runInfo_.nBlockCnt);
    tilingData.iterateOrder = ORDER_N;
    tilingData.mTailTile = static_cast<uint32_t>(runInfo_.mTailTile);
    tilingData.nTailTile = static_cast<uint32_t>(runInfo_.nTailTile);
    tilingData.mBaseTailSplitCnt = static_cast<uint32_t>(runInfo_.mBaseTailSplitCnt);
    tilingData.nBaseTailSplitCnt = static_cast<uint32_t>(runInfo_.nBaseTailSplitCnt);
    tilingData.mTailMain = static_cast<uint32_t>(runInfo_.mTailMain);
    tilingData.nTailMain = static_cast<uint32_t>(runInfo_.nTailMain);
    tilingData.nBubSize = static_cast<uint32_t>(runInfo_.nBubSize);
    tilingData.kBubSize = static_cast<uint32_t>(runInfo_.kBubSize);
    tilingData.groupSize = static_cast<uint32_t>(groupSize);
    tilingData.hasBias = static_cast<uint32_t>(hasBias);
    tilingData.hasX1Scale = static_cast<uint32_t>(hasX1Scale);
    tilingData.hasX2Scale = static_cast<uint32_t>(hasX2Scale);
    tilingData.weightNz = static_cast<uint32_t>(weightNz);
    tilingData.yDtype = static_cast<uint32_t>(yDtype);
    tilingData.l1BufferNum = static_cast<uint32_t>(targetL1BufferNum_);
}

bool WeightQuantMxSwatTilingSolver::FindBubSize(uint64_t nBl1Size, uint64_t kBl1Size, uint64_t& nBubSize,
                                                uint64_t& kBubSize) const
{
    nBubSize = nBl1Size;
    if (weightNz_) {
        kBubSize = FindKOnlyBubSize(nBubSize, kBl1Size);
        return kBubSize > 0UL;
    }

    kBubSize = kBl1Size;
    if (targetL1BufferNum_ == L1_FOUR_BUFFER && nBl1Size > BLOCK_CUBE_SIZE) {
        nBubSize = Align(CeilDiv(nBl1Size, NUM_TWO), BLOCK_CUBE_SIZE);
    }
    return IsBubTilingValid(nBubSize, kBubSize);
}

uint64_t WeightQuantMxSwatTilingSolver::FindKOnlyBubSize(uint64_t nBubSize, uint64_t kBl1Size) const
{
    if (kBl1Size <= K_ALIGN_SIZE_MX_L1) {
        return IsBubTilingValid(nBubSize, kBl1Size) ? kBl1Size : 0UL;
    }
    uint64_t minK = Align(CeilDiv(kBl1Size, NUM_TWO), K_ALIGN_SIZE_MX_L1);
    if (minK >= kBl1Size || (minK % matmul_v4::K_ALIGN_SIZE_MX) != 0UL) {
        return 0UL;
    }
    return IsBubTilingValid(nBubSize, minK) ? minK : 0UL;
}

uint64_t WeightQuantMxSwatTilingSolver::GetBubSize(uint64_t bufferNum, uint64_t nDimSize, uint64_t kDimSize) const
{
    uint64_t nDimAlign = Align(nDimSize, BLOCK_CUBE_SIZE);
    uint64_t kDimBlockAlign = Align(kDimSize, K_ALIGN_SIZE_MX_BLOCK);
    uint64_t kDimL1Align = Align(kDimSize, K_ALIGN_SIZE_MX_L1);
    uint64_t sizeWeightIn = 0UL;
    uint64_t sizeWeightOut = 0UL;
    if (weightNz_) {
        sizeWeightIn = bufferNum * DATA_SIZE_UINT8 * nDimAlign * kDimBlockAlign / INT4_PACK_NUM;
        sizeWeightOut = bufferNum * DATA_SIZE_UINT8 * nDimAlign * kDimL1Align;
    } else {
        sizeWeightIn = bufferNum * DATA_SIZE_UINT8 * nDimSize * kDimL1Align / INT4_PACK_NUM;
        sizeWeightOut = bufferNum * DATA_SIZE_UINT8 * (nDimAlign + 1UL) * kDimL1Align;
    }
    uint64_t sizeBias = 0UL;
    if (hasBias_) {
        uint64_t biasVectorElements = VECTOR_REG_BYTES / biasDataSize_;
        // The prologue keeps a complete baseN bias tile even when ND weight conversion is split along N.
        uint64_t singleBiasBufferSize = Align(runInfo_.baseN, biasVectorElements) * biasDataSize_;
        sizeBias = NUM_TWO * bufferNum * singleBiasBufferSize;
    }
    return sizeWeightIn + sizeWeightOut + sizeBias;
}

uint64_t WeightQuantMxSwatTilingSolver::GetBiasDataSize(ge::DataType yDtype)
{
    switch (yDtype) {
        case ge::DT_FLOAT:
            return DATA_SIZE_FP32;
        case ge::DT_FLOAT16:
        case ge::DT_BF16:
            return DATA_SIZE_UINT8 * NUM_TWO;
        default:
            return DATA_SIZE_FP32;
    }
}

uint64_t WeightQuantMxSwatTilingSolver::CalUsedCoreNum(uint64_t mTile, uint64_t nTile) const
{
    uint64_t usedCoreNum = 0UL;
    uint64_t baseRoundTileNum = runInfo_.totalBlockCnt - runInfo_.tailBlockCnt;
    for (uint64_t tailIdx = 0UL; tailIdx < runInfo_.tailBlockCnt; ++tailIdx) {
        uint64_t mTileIdx = 0UL;
        uint64_t nTileIdx = 0UL;
        GetLogicalTileCoord(baseRoundTileNum + tailIdx, mTileIdx, nTileIdx);
        usedCoreNum += CalcValidSplitCount(GetSingleCoreM(mTileIdx), GetSingleCoreN(nTileIdx), mTile, nTile);
    }
    return usedCoreNum;
}

uint64_t WeightQuantMxSwatTilingSolver::CalcValidSplitCount(uint64_t singleCoreM, uint64_t singleCoreN, uint64_t mTile,
                                                            uint64_t nTile) const
{
    uint64_t singleCoreMSplit = Align(CeilDiv(singleCoreM, mTile), BLOCK_CUBE_SIZE);
    uint64_t singleCoreNSplit = Align(CeilDiv(singleCoreN, nTile), BLOCK_CUBE_SIZE);
    uint64_t validM = std::min(mTile, CeilDiv(singleCoreM, singleCoreMSplit));
    uint64_t validN = std::min(nTile, CeilDiv(singleCoreN, singleCoreNSplit));
    return validM * validN;
}

void WeightQuantMxSwatTilingSolver::GetLogicalTileCoord(uint64_t tileIdx, uint64_t& mTileIdx, uint64_t& nTileIdx) const
{
    uint64_t mCoreNum = std::min(TAIL_WINDOW_LEN, runInfo_.mBlockCnt);
    uint64_t mainRow = runInfo_.mBlockCnt / mCoreNum - 1UL;
    uint64_t mTailCoreNum = runInfo_.mBlockCnt - mCoreNum * mainRow;
    uint64_t rowIdx = tileIdx / (mCoreNum * runInfo_.nBlockCnt);
    if (rowIdx < mainRow) {
        uint64_t localTileIdx = tileIdx - rowIdx * mCoreNum * runInfo_.nBlockCnt;
        mTileIdx = rowIdx * mCoreNum + localTileIdx % mCoreNum;
        nTileIdx = (localTileIdx / mCoreNum) % runInfo_.nBlockCnt;
    } else {
        rowIdx = mainRow;
        uint64_t tailIdx = tileIdx - mainRow * mCoreNum * runInfo_.nBlockCnt;
        mTileIdx = mainRow * mCoreNum + tailIdx % mTailCoreNum;
        nTileIdx = (tailIdx / mTailCoreNum) % runInfo_.nBlockCnt;
    }
    if ((rowIdx & 1UL) != 0UL) {
        nTileIdx = runInfo_.nBlockCnt - 1UL - nTileIdx;
    }
}

uint64_t WeightQuantMxSwatTilingSolver::GetSingleCoreM(uint64_t mTileIdx) const
{
    uint64_t mBaseNormCnt = runInfo_.mBlockCnt - runInfo_.mBaseTailSplitCnt;
    if (mTileIdx >= mBaseNormCnt) {
        uint64_t mMergeSize = shape_.m - mBaseNormCnt * runInfo_.baseM;
        uint64_t mBaseTailMain = runInfo_.mBaseTailSplitCnt == 1UL ? mMergeSize : runInfo_.mTailMain;
        uint64_t mBaseTailLast = mMergeSize - (runInfo_.mBaseTailSplitCnt - 1UL) * mBaseTailMain;
        return mTileIdx < runInfo_.mBlockCnt - 1UL ? mBaseTailMain : mBaseTailLast;
    }
    return runInfo_.baseM;
}

uint64_t WeightQuantMxSwatTilingSolver::GetSingleCoreN(uint64_t nTileIdx) const
{
    uint64_t nBaseNormCnt = runInfo_.nBlockCnt - runInfo_.nBaseTailSplitCnt;
    if (nTileIdx >= nBaseNormCnt) {
        uint64_t nMergeSize = shape_.n - nBaseNormCnt * runInfo_.baseN;
        uint64_t nBaseTailMain = runInfo_.nBaseTailSplitCnt == 1UL ? nMergeSize : runInfo_.nTailMain;
        uint64_t nBaseTailLast = nMergeSize - (runInfo_.nBaseTailSplitCnt - 1UL) * nBaseTailMain;
        return nTileIdx < runInfo_.nBlockCnt - 1UL ? nBaseTailMain : nBaseTailLast;
    }
    return runInfo_.baseN;
}
} // namespace optiling
