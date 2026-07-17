/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file matmul_v3_sc_splitk_al1_fullload_tiling.cpp
 * \brief Single-core split-K + A L1 full load tiling helpers (tilingKey 65569).
 */

#include "matmul_v3_sc_splitk_al1_fullload_tiling.h"
#include "log/log.h"
#include "matmul/common/op_host/math_util.h"

namespace optiling {
namespace matmul_v3 {
namespace {

constexpr uint64_t kL1RpcReserve = 256UL;
constexpr uint64_t kBaseKAlign = BASIC_ALIGN_32;
constexpr uint64_t kMinBaseK = kBaseKAlign;
constexpr uint64_t kMaxBaseK = BASIC_BLOCK_SIZE_256;
constexpr uint64_t kL1AdjustMax = 3UL;
constexpr uint64_t kBaseNCandidates[] = {
    BASIC_ALIGN_512, 384UL, BASIC_BLOCK_SIZE_256, BASIC_BLOCK_SIZE_128, BASIC_BLOCK_SIZE_64, 32UL,
};
constexpr uint64_t kBTransKgtNMaxBaseN = BASIC_BLOCK_SIZE_128;
constexpr uint64_t kRefBaseN = BASIC_BLOCK_SIZE_256;
constexpr uint64_t kRefBaseK = BASIC_BLOCK_SIZE_128;

inline uint64_t CeilDiv(uint64_t dividend, uint64_t divisor)
{
    return Ops::NN::MathUtil::CeilDivision(dividend, divisor);
}

inline bool IsL0CubeFit(const MatmulV3CompileInfo& compileInfo, uint64_t baseM, uint64_t baseN, uint64_t baseK,
                        uint64_t aDtypeSize, uint64_t bDtypeSize)
{
    return baseK > 0 && baseN > 0 && baseM * baseK * aDtypeSize * DB_SIZE <= compileInfo.l0ASize &&
           baseN * baseK * bDtypeSize * DB_SIZE <= compileInfo.l0BSize;
}

inline bool IsL0CFit(const MatmulV3CompileInfo& compileInfo, uint64_t baseM, uint64_t baseN, uint64_t dbL0c)
{
    return baseM * baseN * dbL0c * DATA_SIZE_FP32 <= compileInfo.l0CSize;
}

inline uint64_t CalcMaxBaseNByL0C(const MatmulV3CompileInfo& compileInfo, uint64_t baseM, uint64_t dbL0c)
{
    return ops::FloorDiv(compileInfo.l0CSize, baseM * dbL0c * DATA_SIZE_FP32);
}

inline uint64_t CalcBiasL1Size(uint64_t baseN, bool hasBias) { return hasBias ? baseN * DATA_SIZE_FP32 : 0UL; }

inline uint64_t CalcAL1LoopCount(uint64_t kValue, uint64_t stepKa, uint64_t baseK)
{
    return (kValue == 0 || stepKa == 0 || baseK == 0) ? UINT64_MAX : CeilDiv(kValue, stepKa * baseK);
}

inline uint64_t CalcBkTileAlignScore(uint64_t bkTileBytes)
{
    if (bkTileBytes == 0) {
        return 0;
    }
    if (bkTileBytes % 512UL == 0) {
        return 2UL;
    }
    return (bkTileBytes % 256UL == 0) ? 1UL : 0UL;
}

inline bool IsPerfectNTileSplit(uint64_t nTileCnt, uint64_t usedCoreNum)
{
    return usedCoreNum > 0 && nTileCnt % usedCoreNum == 0;
}

inline uint64_t CalcBlockRoundCount(uint64_t singleCoreN, uint64_t singleCoreK, uint64_t baseN, uint64_t baseK)
{
    if (baseN == 0 || baseK == 0) {
        return UINT64_MAX;
    }
    return CeilDiv(singleCoreN, baseN) * CeilDiv(singleCoreK, baseK);
}

// Return 1 if cur is better, -1 if best is better, 0 if tie.
// Prefer rounds <= ref (baseN=256, baseK=128); among those, closer to ref from below; otherwise minimize excess.
int CompareBlockRounds(uint64_t curSingleCoreN, uint64_t curBaseN, uint64_t curBaseK, uint64_t bestSingleCoreN,
                       uint64_t bestBaseN, uint64_t bestBaseK, uint64_t singleCoreK)
{
    uint64_t curRounds = CalcBlockRoundCount(curSingleCoreN, singleCoreK, curBaseN, curBaseK);
    uint64_t bestRounds = CalcBlockRoundCount(bestSingleCoreN, singleCoreK, bestBaseN, bestBaseK);
    uint64_t curRef = CalcBlockRoundCount(curSingleCoreN, singleCoreK, kRefBaseN, kRefBaseK);
    uint64_t bestRef = CalcBlockRoundCount(bestSingleCoreN, singleCoreK, kRefBaseN, kRefBaseK);
    bool curLeRef = curRounds <= curRef;
    bool bestLeRef = bestRounds <= bestRef;
    if (curLeRef != bestLeRef) {
        return curLeRef ? 1 : -1;
    }
    if (curLeRef) {
        if (curRounds * bestRef > bestRounds * curRef) {
            return 1;
        }
        if (curRounds * bestRef < bestRounds * curRef) {
            return -1;
        }
        return 0;
    }
    uint64_t curExcess = curRounds - curRef;
    uint64_t bestExcess = bestRounds - bestRef;
    if (curExcess * bestRef < bestExcess * curRef) {
        return 1;
    }
    if (curExcess * bestRef > bestExcess * curRef) {
        return -1;
    }
    return 0;
}

void CalcNTileSplit(uint64_t nValue, uint64_t baseN, uint64_t aicNum, uint64_t& nTileCnt, uint64_t& usedCoreNum,
                    uint64_t& headCoreNum, uint64_t& tailCoreNum)
{
    nTileCnt = usedCoreNum = headCoreNum = tailCoreNum = 0;
    if (baseN == 0 || aicNum == 0) {
        return;
    }
    nTileCnt = CeilDiv(nValue, baseN);
    usedCoreNum = std::min(nTileCnt, aicNum);
    if (usedCoreNum == 0) {
        return;
    }
    uint64_t remainder = nTileCnt % usedCoreNum;
    headCoreNum = (remainder == 0) ? usedCoreNum : remainder;
    tailCoreNum = usedCoreNum - headCoreNum;
}

uint64_t CalcPolicyBaseK(const MatmulV3CompileInfo& compileInfo, uint64_t baseM, uint64_t baseN, uint64_t aDtypeSize,
                         uint64_t bDtypeSize, uint64_t dbL0c)
{
    if (!IsL0CFit(compileInfo, baseM, baseN, dbL0c)) {
        return 0UL;
    }
    uint64_t maxKa = ops::FloorDiv(compileInfo.l0ASize, DB_SIZE * aDtypeSize * baseM);
    uint64_t maxKb = ops::FloorDiv(compileInfo.l0BSize, DB_SIZE * bDtypeSize * baseN);
    uint64_t maxBaseK = ops::FloorAlign(std::min(maxKa, maxKb), kBaseKAlign);
    maxBaseK = std::min(maxBaseK, kMaxBaseK);
    if (maxBaseK < kMinBaseK) {
        return 0UL;
    }
    for (uint64_t tryBaseK = maxBaseK; tryBaseK >= kMinBaseK; tryBaseK -= kBaseKAlign) {
        if (IsL0CubeFit(compileInfo, baseM, baseN, tryBaseK, aDtypeSize, bDtypeSize)) {
            return tryBaseK;
        }
        if (tryBaseK == kMinBaseK) {
            break;
        }
    }
    return 0UL;
}

uint64_t EstMinAL1LoopCount(uint64_t totalL1Size, uint64_t baseM, uint64_t stepM, uint64_t baseN, uint64_t baseK,
                            uint64_t kValue, uint64_t aDtypeSize, uint64_t bDtypeSize, bool hasBias)
{
    uint64_t biasL1Size = CalcBiasL1Size(baseN, hasBias);
    if (totalL1Size <= biasL1Size) {
        return UINT64_MAX;
    }
    uint64_t availL1Size = totalL1Size - biasL1Size;
    uint64_t blockASize = baseM * baseK * aDtypeSize;
    uint64_t blockBSize = baseN * baseK * bDtypeSize;
    uint64_t minBL1Size = DB_SIZE * blockBSize;
    if (minBL1Size >= availL1Size || blockASize == 0 || baseK == 0) {
        return UINT64_MAX;
    }
    uint64_t maxStepKa = std::min((availL1Size - minBL1Size) / (stepM * blockASize), kValue / baseK);
    return maxStepKa >= 1 ? CalcAL1LoopCount(kValue, maxStepKa, baseK) : UINT64_MAX;
}

bool IsBetterL1Step(uint64_t kValue, uint64_t baseK, uint64_t bDtypeSize, bool isBTrans, uint64_t curStepKa,
                    uint64_t curStepKb, uint64_t curBL1Size, uint64_t bestStepKa, uint64_t bestStepKb,
                    uint64_t bestBL1Size)
{
    if (bestStepKa == 0) {
        return true;
    }
    if (isBTrans) {
        uint64_t curScore = CalcBkTileAlignScore(curStepKb * baseK * bDtypeSize);
        uint64_t bestScore = CalcBkTileAlignScore(bestStepKb * baseK * bDtypeSize);
        if (curScore != bestScore) {
            return curScore > bestScore;
        }
    }
    uint64_t curLoops = CalcAL1LoopCount(kValue, curStepKa, baseK);
    uint64_t bestLoops = CalcAL1LoopCount(kValue, bestStepKa, baseK);
    if (curLoops != bestLoops) {
        return curLoops < bestLoops;
    }
    if (curBL1Size != bestBL1Size) {
        return curBL1Size > bestBL1Size;
    }
    return curStepKa > bestStepKa;
}

struct L1StepSelection {
    uint64_t stepKa = 0;
    uint64_t stepKb = 0;
    uint64_t stepN = 0;
    uint64_t depthA1 = 0;
    uint64_t depthB1 = 0;
};

bool SearchBestKbForStepN(uint64_t tryStepN, uint64_t singleCoreN, uint64_t baseN, uint64_t kValue, uint64_t baseK,
                          uint64_t bDtypeSize, bool isBTrans, uint64_t availL1Size, uint64_t biasL1Size, uint64_t stepM,
                          uint64_t blockASize, uint64_t blockBSize, L1StepSelection& selection)
{
    if (baseK == 0 || stepM == 0 || blockASize == 0) {
        return false;
    }
    if (tryStepN * baseN > singleCoreN) {
        return false;
    }
    uint64_t bBlockUnit = tryStepN * DB_SIZE * blockBSize;
    if (bBlockUnit == 0) {
        return false;
    }
    uint64_t maxStepKb = std::min(availL1Size > biasL1Size ? (availL1Size - biasL1Size) / bBlockUnit : 0UL,
                                  kValue / baseK);
    if (maxStepKb < 1) {
        return false;
    }
    L1StepSelection best;
    uint64_t bestBL1Size = 0;
    for (uint64_t tryStepKb = 1; tryStepKb <= maxStepKb; ++tryStepKb) {
        uint64_t bL1Size = tryStepN * tryStepKb * DB_SIZE * blockBSize;
        if (bL1Size + biasL1Size > availL1Size) {
            continue;
        }
        uint64_t aBlockUnit = stepM * blockASize;
        uint64_t maxStepKa = std::min((availL1Size - bL1Size - biasL1Size) / aBlockUnit, kValue / baseK);
        if (maxStepKa < tryStepKb) {
            continue;
        }
        uint64_t tryStepKa = (maxStepKa / tryStepKb) * tryStepKb;
        if (tryStepKa < tryStepKb) {
            continue;
        }
        if (IsBetterL1Step(kValue, baseK, bDtypeSize, isBTrans, tryStepKa, tryStepKb, bL1Size, best.stepKa, best.stepKb,
                           bestBL1Size)) {
            best.stepKa = tryStepKa;
            best.stepKb = tryStepKb;
            best.stepN = tryStepN;
            best.depthA1 = stepM * tryStepKa;
            best.depthB1 = tryStepN * tryStepKb * DB_SIZE;
            bestBL1Size = bL1Size;
        }
    }
    if (best.stepKa < 1) {
        return false;
    }
    selection = best;
    return true;
}

bool SelectL1Steps(uint64_t baseM, uint64_t stepM, uint64_t kValue, uint64_t aDtypeSize, uint64_t bDtypeSize,
                   bool isBTrans, uint64_t availL1Size, uint64_t biasL1Size, uint64_t singleCoreN, uint64_t baseN,
                   uint64_t baseK, uint64_t& stepKa, uint64_t& stepKb, uint64_t& stepN, uint64_t& depthA1,
                   uint64_t& depthB1)
{
    stepKa = stepKb = stepN = depthA1 = depthB1 = 0;
    if (baseN == 0 || baseK == 0) {
        return false;
    }
    uint64_t blockASize = baseM * baseK * aDtypeSize;
    uint64_t blockBSize = baseN * baseK * bDtypeSize;
    if (blockASize == 0 || blockBSize == 0) {
        return false;
    }
    L1StepSelection selection;
    for (uint64_t tryStepN = 1, maxStepN = std::max(1UL, CeilDiv(singleCoreN, baseN)); tryStepN <= maxStepN;
         ++tryStepN) {
        if (!SearchBestKbForStepN(tryStepN, singleCoreN, baseN, kValue, baseK, bDtypeSize, isBTrans, availL1Size,
                                  biasL1Size, stepM, blockASize, blockBSize, selection)) {
            continue;
        }
        stepKa = selection.stepKa;
        stepKb = selection.stepKb;
        stepN = selection.stepN;
        depthA1 = selection.depthA1;
        depthB1 = selection.depthB1;
        return true;
    }
    return false;
}

bool IsBetterBaseN(uint64_t baseM, uint64_t stepM, uint64_t kValue, uint64_t totalL1Size, uint64_t aDtypeSize,
                   uint64_t bDtypeSize, bool hasBias, bool isBTrans, uint64_t curBaseN, uint64_t curBaseK,
                   uint64_t curBkAlignScore, uint64_t curSingleCoreN, uint64_t curNTileCnt, uint64_t curUsedCoreNum,
                   uint64_t curHeadCoreNum, uint64_t curTailCoreNum, uint64_t bestBaseN, uint64_t bestBaseK,
                   uint64_t bestBkAlignScore, uint64_t bestSingleCoreN, uint64_t bestNTileCnt, uint64_t bestUsedCoreNum,
                   uint64_t bestHeadCoreNum, uint64_t bestTailCoreNum)
{
    if (bestBaseN == 0) {
        return true;
    }
    if (!isBTrans) {
        bool curPerfect = IsPerfectNTileSplit(curNTileCnt, curUsedCoreNum);
        bool bestPerfect = IsPerfectNTileSplit(bestNTileCnt, bestUsedCoreNum);
        if (curPerfect != bestPerfect) {
            return curPerfect;
        }
    }
    if (curUsedCoreNum != bestUsedCoreNum) {
        return curUsedCoreNum > bestUsedCoreNum;
    }
    if (!isBTrans) {
        bool curPerfect = IsPerfectNTileSplit(curNTileCnt, curUsedCoreNum);
        if (!curPerfect && curHeadCoreNum - curTailCoreNum != bestHeadCoreNum - bestTailCoreNum) {
            return curHeadCoreNum - curTailCoreNum < bestHeadCoreNum - bestTailCoreNum;
        }
    }
    if (isBTrans && curBkAlignScore != bestBkAlignScore) {
        return curBkAlignScore > bestBkAlignScore;
    }
    int blockRoundCmp = CompareBlockRounds(curSingleCoreN, curBaseN, curBaseK, bestSingleCoreN, bestBaseN, bestBaseK,
                                           kValue);
    if (blockRoundCmp != 0) {
        return blockRoundCmp > 0;
    }
    if (curBaseN != bestBaseN) {
        return curBaseN > bestBaseN;
    }
    if (curBaseK != bestBaseK) {
        return curBaseK > bestBaseK;
    }
    uint64_t curLoops = EstMinAL1LoopCount(totalL1Size, baseM, stepM, curBaseN, curBaseK, kValue, aDtypeSize,
                                           bDtypeSize, hasBias);
    uint64_t bestLoops = EstMinAL1LoopCount(totalL1Size, baseM, stepM, bestBaseN, bestBaseK, kValue, aDtypeSize,
                                            bDtypeSize, hasBias);
    if (curLoops != bestLoops) {
        return curLoops < bestLoops;
    }
    if (curBaseK == 0 || bestBaseK == 0) {
        return curBaseK > bestBaseK;
    }
    return baseM * curBaseN * bestBaseK * (baseM + bestBaseN) > baseM * bestBaseN * curBaseK * (baseM + curBaseN);
}

struct BaseBlockBest {
    uint64_t baseN = 0;
    uint64_t baseK = 0;
    uint64_t bkAlignScore = 0;
    uint64_t singleCoreN = 0;
    uint64_t nTileCnt = 0;
    uint64_t usedCoreNum = 0;
    uint64_t headCoreNum = 0;
    uint64_t tailCoreNum = 0;
};

bool CalcBkAlignScoreForBaseK(uint64_t baseM, uint64_t stepM, uint64_t kValue, uint64_t totalL1Size,
                              uint64_t aDtypeSize, uint64_t bDtypeSize, bool hasBias, bool isBTrans,
                              uint64_t singleCoreN, uint64_t tryBaseN, uint64_t tryBaseK, uint64_t& bkAlignScore)
{
    if (!isBTrans) {
        bkAlignScore = 0;
        return true;
    }
    uint64_t biasL1Size = CalcBiasL1Size(tryBaseN, hasBias);
    if (totalL1Size <= biasL1Size) {
        return false;
    }
    uint64_t estStepKa = 0;
    uint64_t estStepKb = 0;
    uint64_t estStepN = 0;
    uint64_t estDepthA1 = 0;
    uint64_t estDepthB1 = 0;
    if (!SelectL1Steps(baseM, stepM, kValue, aDtypeSize, bDtypeSize, isBTrans, totalL1Size - biasL1Size, biasL1Size,
                       singleCoreN, tryBaseN, tryBaseK, estStepKa, estStepKb, estStepN, estDepthA1, estDepthB1)) {
        return false;
    }
    bkAlignScore = CalcBkTileAlignScore(estStepKb * tryBaseK * bDtypeSize);
    return true;
}

void TryUpdateBaseBlockBest(uint64_t baseM, uint64_t stepM, uint64_t kValue, uint64_t totalL1Size, uint64_t aDtypeSize,
                            uint64_t bDtypeSize, bool hasBias, bool isBTrans, uint64_t tryBaseN, uint64_t tryBaseK,
                            uint64_t bkAlignScore, uint64_t singleCoreN, uint64_t nTileCnt, uint64_t usedCoreNum,
                            uint64_t headCoreNum, uint64_t tailCoreNum, BaseBlockBest& best)
{
    if (!IsBetterBaseN(baseM, stepM, kValue, totalL1Size, aDtypeSize, bDtypeSize, hasBias, isBTrans, tryBaseN, tryBaseK,
                       bkAlignScore, singleCoreN, nTileCnt, usedCoreNum, headCoreNum, tailCoreNum, best.baseN,
                       best.baseK, best.bkAlignScore, best.singleCoreN, best.nTileCnt, best.usedCoreNum,
                       best.headCoreNum, best.tailCoreNum)) {
        return;
    }
    best.baseN = tryBaseN;
    best.baseK = tryBaseK;
    best.bkAlignScore = bkAlignScore;
    best.singleCoreN = singleCoreN;
    best.nTileCnt = nTileCnt;
    best.usedCoreNum = usedCoreNum;
    best.headCoreNum = headCoreNum;
    best.tailCoreNum = tailCoreNum;
}

void SearchBaseKForBaseN(const MatmulV3CompileInfo& compileInfo, uint64_t baseM, uint64_t stepM, uint64_t kValue,
                         uint64_t totalL1Size, uint64_t aDtypeSize, uint64_t bDtypeSize, bool hasBias, bool isBTrans,
                         uint64_t tryBaseN, uint64_t policyBaseK, uint64_t singleCoreN, uint64_t nTileCnt,
                         uint64_t usedCoreNum, uint64_t headCoreNum, uint64_t tailCoreNum, BaseBlockBest& best)
{
    for (uint64_t tryBaseK = policyBaseK; tryBaseK >= kMinBaseK; tryBaseK -= kBaseKAlign) {
        if (!IsL0CubeFit(compileInfo, baseM, tryBaseN, tryBaseK, aDtypeSize, bDtypeSize)) {
            if (tryBaseK == kMinBaseK) {
                break;
            }
            continue;
        }
        uint64_t bkAlignScore = 0;
        if (!CalcBkAlignScoreForBaseK(baseM, stepM, kValue, totalL1Size, aDtypeSize, bDtypeSize, hasBias, isBTrans,
                                      singleCoreN, tryBaseN, tryBaseK, bkAlignScore)) {
            continue;
        }
        TryUpdateBaseBlockBest(baseM, stepM, kValue, totalL1Size, aDtypeSize, bDtypeSize, hasBias, isBTrans, tryBaseN,
                               tryBaseK, bkAlignScore, singleCoreN, nTileCnt, usedCoreNum, headCoreNum, tailCoreNum,
                               best);
        if (tryBaseK == kMinBaseK) {
            break;
        }
    }
}

void SearchBaseBlockCandidates(const MatmulV3CompileInfo& compileInfo, uint64_t baseM, uint64_t stepM, uint64_t kValue,
                               uint64_t nValue, uint64_t totalL1Size, uint64_t aDtypeSize, uint64_t bDtypeSize,
                               bool hasBias, bool isBTrans, uint64_t dbL0c, uint64_t maxBaseN, BaseBlockBest& best)
{
    for (uint64_t tryBaseN : kBaseNCandidates) {
        if (tryBaseN > maxBaseN) {
            continue;
        }
        uint64_t nTileCnt = 0;
        uint64_t usedCoreNum = 0;
        uint64_t headCoreNum = 0;
        uint64_t tailCoreNum = 0;
        CalcNTileSplit(nValue, tryBaseN, compileInfo.aicNum, nTileCnt, usedCoreNum, headCoreNum, tailCoreNum);
        if (usedCoreNum == 0 || (tailCoreNum > 0 && headCoreNum < tailCoreNum)) {
            continue;
        }
        uint64_t policyBaseK = CalcPolicyBaseK(compileInfo, baseM, tryBaseN, aDtypeSize, bDtypeSize, dbL0c);
        if (policyBaseK < kMinBaseK) {
            continue;
        }
        uint64_t singleCoreN = tryBaseN * CeilDiv(nTileCnt, usedCoreNum);
        SearchBaseKForBaseN(compileInfo, baseM, stepM, kValue, totalL1Size, aDtypeSize, bDtypeSize, hasBias, isBTrans,
                            tryBaseN, policyBaseK, singleCoreN, nTileCnt, usedCoreNum, headCoreNum, tailCoreNum, best);
    }
}

bool TryBaseBlockFallback(const char* opName, const MatmulV3CompileInfo& compileInfo, uint64_t baseM,
                          uint64_t aDtypeSize, uint64_t bDtypeSize, uint64_t dbL0c, uint64_t maxBaseN, uint64_t& baseN,
                          uint64_t& baseK)
{
    for (uint64_t tryBaseN : kBaseNCandidates) {
        if (tryBaseN > maxBaseN) {
            continue;
        }
        uint64_t policyBaseK = CalcPolicyBaseK(compileInfo, baseM, tryBaseN, aDtypeSize, bDtypeSize, dbL0c);
        if (policyBaseK >= kMinBaseK) {
            baseN = tryBaseN;
            baseK = policyBaseK;
            OP_LOGI(opName, "ScSplitKAl1FullLoad fallback baseN=%lu baseK=%lu", baseN, baseK);
            return true;
        }
    }
    return false;
}

bool SelectBaseBlock(const char* opName, const MatmulV3CompileInfo& compileInfo, uint64_t baseM, uint64_t stepM,
                     uint64_t kValue, uint64_t nValue, uint64_t totalL1Size, uint64_t aDtypeSize, uint64_t bDtypeSize,
                     bool hasBias, bool isBTrans, uint64_t dbL0c, uint64_t& baseN, uint64_t& baseK)
{
    baseN = baseK = 0;
    uint64_t maxBaseN = CalcMaxBaseNByL0C(compileInfo, baseM, dbL0c);
    if (isBTrans && kValue > nValue) {
        maxBaseN = std::min(maxBaseN, kBTransKgtNMaxBaseN);
    }
    BaseBlockBest best;
    SearchBaseBlockCandidates(compileInfo, baseM, stepM, kValue, nValue, totalL1Size, aDtypeSize, bDtypeSize, hasBias,
                              isBTrans, dbL0c, maxBaseN, best);
    if (best.baseN != 0) {
        baseN = best.baseN;
        baseK = best.baseK;
        OP_LOGI(opName, "ScSplitKAl1FullLoad baseN=%lu baseK=%lu usedCore=%lu perfect=%d", baseN, baseK,
                best.usedCoreNum, static_cast<int>(IsPerfectNTileSplit(best.nTileCnt, best.usedCoreNum)));
        return true;
    }
    if (TryBaseBlockFallback(opName, compileInfo, baseM, aDtypeSize, bDtypeSize, dbL0c, maxBaseN, baseN, baseK)) {
        return true;
    }
    OP_LOGI(opName, "ScSplitKAl1FullLoad SelectBaseBlock failed baseM=%lu isBTrans=%d", baseM,
            static_cast<int>(isBTrans));
    return false;
}

bool AdjustL1Overflow(const char* opName, uint64_t totalL1Size, uint64_t baseM, uint64_t stepM, uint64_t aDtypeSize,
                      uint64_t bDtypeSize, uint64_t biasL1Size, MatmulV3RunInfo& runInfo)
{
    uint64_t actualAL1Size = runInfo.baseM * runInfo.baseK * runInfo.depthA1 * aDtypeSize;
    uint64_t actualBL1Size = runInfo.baseN * runInfo.baseK * runInfo.depthB1 * bDtypeSize;
    uint64_t totalUsed = actualAL1Size + actualBL1Size + biasL1Size;
    if (totalUsed <= totalL1Size) {
        return true;
    }
    OP_LOGI(opName, "ScSplitKAl1FullLoad L1 overflow, adjust stepKa/stepN");
    for (uint64_t i = 0; i < kL1AdjustMax && runInfo.stepKa > runInfo.stepKb && totalUsed > totalL1Size; ++i) {
        runInfo.stepKa -= runInfo.stepKb;
        runInfo.stepKa = (runInfo.stepKb > 0) ? (runInfo.stepKa / runInfo.stepKb) * runInfo.stepKb : 0;
        if (runInfo.stepKa < runInfo.stepKb) {
            break;
        }
        runInfo.depthA1 = stepM * runInfo.stepKa;
        actualAL1Size = baseM * runInfo.baseK * runInfo.depthA1 * aDtypeSize;
        totalUsed = actualAL1Size + actualBL1Size + biasL1Size;
    }
    for (uint64_t i = 0; i < kL1AdjustMax && runInfo.stepN > 1 && totalUsed > totalL1Size; ++i) {
        runInfo.stepN /= 2; // reduce stepN by half
        runInfo.depthB1 = runInfo.stepN * runInfo.stepKb * DB_SIZE;
        actualBL1Size = runInfo.baseN * runInfo.baseK * runInfo.depthB1 * bDtypeSize;
        totalUsed = actualAL1Size + actualBL1Size + biasL1Size;
    }
    if (totalUsed > totalL1Size || runInfo.stepKa < runInfo.stepKb || runInfo.stepKa % runInfo.stepKb != 0) {
        return false;
    }
    return true;
}

} // namespace

bool MatmulV3ScSplitKAl1FullLoadTiling::DoTiling(const char* opName, const MatmulV3Args& args,
                                                 const MatmulV3CompileInfo& compileInfo, uint64_t aDtypeSize,
                                                 uint64_t bDtypeSize, MatmulV3RunInfo& runInfo)
{
    runInfo.baseM = ops::CeilAlign(args.mValue, BASIC_ALIGN_16);
    runInfo.dbL0c = 1;
    uint64_t stepM = CeilDiv(args.mValue, runInfo.baseM);
    uint64_t totalL1Size = compileInfo.l1Size + kL1RpcReserve;

    if (!SelectBaseBlock(opName, compileInfo, runInfo.baseM, stepM, args.kValue, args.nValue, totalL1Size, aDtypeSize,
                         bDtypeSize, args.hasBias, args.isBTrans, runInfo.dbL0c, runInfo.baseN, runInfo.baseK)) {
        return false;
    }
    if (runInfo.baseK == 0) {
        runInfo.baseK = CalcPolicyBaseK(compileInfo, runInfo.baseM, runInfo.baseN, aDtypeSize, bDtypeSize,
                                        runInfo.dbL0c);
        if (runInfo.baseK < kMinBaseK) {
            OP_LOGI(opName, "ScSplitKAl1FullLoad baseK failed baseM=%lu baseN=%lu", runInfo.baseM, runInfo.baseN);
            return false;
        }
    }

    uint64_t biasL1Size = CalcBiasL1Size(runInfo.baseN, args.hasBias);
    uint64_t nTileCnt = CeilDiv(args.nValue, runInfo.baseN);
    runInfo.usedCoreNum = std::min(nTileCnt, compileInfo.aicNum);
    uint64_t singleCoreN = runInfo.baseN * CeilDiv(nTileCnt, runInfo.usedCoreNum);
    if (!SelectL1Steps(runInfo.baseM, stepM, args.kValue, aDtypeSize, bDtypeSize, args.isBTrans,
                       totalL1Size - biasL1Size, biasL1Size, singleCoreN, runInfo.baseN, runInfo.baseK, runInfo.stepKa,
                       runInfo.stepKb, runInfo.stepN, runInfo.depthA1, runInfo.depthB1)) {
        OP_LOGI(opName, "ScSplitKAl1FullLoad SelectL1Steps failed");
        return false;
    }

    runInfo.stepM = stepM;
    runInfo.singleCoreM = args.mValue;
    runInfo.singleCoreN = singleCoreN;
    runInfo.singleCoreK = args.kValue;
    runInfo.iterateOrder = 0;

    if (!AdjustL1Overflow(opName, totalL1Size, runInfo.baseM, stepM, aDtypeSize, bDtypeSize, biasL1Size, runInfo)) {
        OP_LOGI(opName, "ScSplitKAl1FullLoad L1 still overflow after adjustment");
        return false;
    }
    OP_LOGI(opName,
            "ScSplitKAl1FullLoad tiling: baseM=%lu baseN=%lu baseK=%lu stepKa=%lu stepN=%lu stepKb=%lu "
            "usedCore=%lu singleCoreN=%lu",
            runInfo.baseM, runInfo.baseN, runInfo.baseK, runInfo.stepKa, runInfo.stepN, runInfo.stepKb,
            runInfo.usedCoreNum, runInfo.singleCoreN);
    return true;
}

bool MatmulV3ScSplitKAl1FullLoadTiling::CheckTilingOk(const char* opName, const MatmulV3RunInfo& runInfo,
                                                      const MatmulV3CompileInfo& compileInfo, uint64_t aDtypeSize,
                                                      uint64_t bDtypeSize)
{
    const bool basicDimsValid = runInfo.usedCoreNum >= 1 && runInfo.singleCoreK >= runInfo.baseK &&
                                runInfo.singleCoreN >= 1 && runInfo.depthA1 >= 1 && runInfo.depthB1 >= 1;
    if (!basicDimsValid) {
        OP_LOGI(opName, "ScSplitKAl1FullLoad check failed: basic dims invalid");
        return false;
    }

    const bool stepNWithinSingleCoreN = runInfo.stepN * runInfo.baseN <= runInfo.singleCoreN;
    const bool stepKaWithinSingleCoreK = runInfo.stepKa * runInfo.baseK <= runInfo.singleCoreK;
    const bool stepKaGeStepKb = runInfo.stepKa >= runInfo.stepKb;
    const bool stepKaMultipleOfStepKb = runInfo.stepKa % runInfo.stepKb == 0;
    const bool depthB1Consistent = runInfo.depthB1 == runInfo.stepN * runInfo.stepKb * DB_SIZE;
    const bool stepRelationValid = stepNWithinSingleCoreN && stepKaWithinSingleCoreK && stepKaGeStepKb &&
                                   stepKaMultipleOfStepKb && depthB1Consistent;
    if (!stepRelationValid) {
        OP_LOGI(opName, "ScSplitKAl1FullLoad check failed: step relation invalid");
        return false;
    }

    const bool baseKAligned = runInfo.baseK >= kMinBaseK && runInfo.baseK % kBaseKAlign == 0;
    if (!baseKAligned) {
        OP_LOGI(opName, "ScSplitKAl1FullLoad check failed: baseK=%lu invalid", runInfo.baseK);
        return false;
    }

    const bool l0CubeFits = IsL0CubeFit(compileInfo, runInfo.baseM, runInfo.baseN, runInfo.baseK, aDtypeSize,
                                        bDtypeSize);
    const bool l0CFits = IsL0CFit(compileInfo, runInfo.baseM, runInfo.baseN, runInfo.dbL0c);
    const bool l0ResourceFits = l0CubeFits && l0CFits;
    if (!l0ResourceFits) {
        OP_LOGI(opName, "ScSplitKAl1FullLoad check failed: L0 overflow");
        return false;
    }
    return true;
}

} // namespace matmul_v3
} // namespace optiling
