/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file base_block_calculator.cpp
 * \brief
 */
#include "base_block_calculator.h"

#include <algorithm>
#include "error_util.h"
#include "quant_batch_matmul_v3_tiling_util.h"
#include "util/math_util.h"

namespace {
constexpr uint64_t PER_BLOCK_BASE_SIZE_256 = 256UL;
// If adjusted baseN loses 128B alignment and K is below this threshold, keep the original base block.
constexpr uint64_t LOAD_BALANCE_BASE_N_128_ALIGN_K_THRESHOLD = 2560UL;
// Rebalance M/N split when one base dimension is at least twice the other.
constexpr uint64_t BASEM_BASEN_RATIO = 2UL;
// Oversized baseK candidates are halved above this supported tiling range.
constexpr uint64_t BASEK_LIMIT = 4095UL;
constexpr uint32_t DOUBLE_CORE_NUM = 2U;
// Epsilon for comparing score ratios during base-block search.
constexpr double SCORE_COMPARE_EPS = 1e-12;
constexpr uint32_t SMALL_MN_EXPAND_RATIO = 3U;

uint64_t GetShapeWithDataType(uint64_t size, ge::DataType dtype)
{
    if (dtype == ge::DT_INT4 || dtype == ge::DT_FLOAT4_E2M1 || dtype == ge::DT_FLOAT4_E1M2) {
        return size + size;
    }
    uint64_t dtypeSize = static_cast<uint64_t>(ge::GetSizeByDataType(dtype));
    return dtypeSize == 0UL ? 0UL : size / dtypeSize;
}

uint64_t GetNextLoadBalanceBase(uint64_t curBase, uint64_t baseAlign)
{
    if (curBase <= baseAlign) {
        return 0UL;
    }
    return curBase - baseAlign;
}

bool IsScoreEqual(double lhs, double rhs) { return lhs + SCORE_COMPARE_EPS >= rhs && rhs + SCORE_COMPARE_EPS >= lhs; }

double GetBaseShapeScore(uint64_t baseM, uint64_t baseN)
{
    // Smaller ratio means the M/N split is closer to square.
    return static_cast<double>(std::max(baseM, baseN)) / std::min(baseM, baseN);
}

double GetMemoryComputeScore(uint64_t baseM, uint64_t baseN)
{
    // This is (baseM + baseN) / (baseM * baseN); lower means smaller baseM + baseN for the same output area.
    return (static_cast<double>(baseM) + baseN) / (static_cast<double>(baseM) * baseN);
}
} // namespace

namespace optiling {

using Ops::NN::MathUtil;

BaseBlockCalculator::BaseBlockCalculator(const QuantBatchMatmulInfo& inputParams,
                                         const QuantBatchMatmulV3CompileInfo& compileInfo, uint64_t batchCoreCnt)
    : inputParams_(inputParams), compileInfo_(compileInfo), batchCoreCnt_(batchCoreCnt)
{}

bool BaseBlockCalculator::Compute(BaseBlockMode mode)
{
    baseBlockRes_ = BaseBlockRes();
    if (!ValidateInput()) {
        return false;
    }
    switch (mode) {
        case BaseBlockMode::PERBLOCK:
            ComputeBaseBlockPerblock();
            break;
        case BaseBlockMode::MMAD_S8S4:
            ComputeBaseBlockMmadS8S4();
            break;
        case BaseBlockMode::STREAMK:
            if (!ComputeBaseBlockStreamK()) {
                return false;
            }
            break;
        case BaseBlockMode::DEFAULT:
        default:
            ComputeBaseBlockDefault();
            break;
    }
    // StreamK base block is bound to split-K scheduling. ASWT base optimization can change baseM/baseN without
    // recomputing streamKCnt and singleCoreK, so skip those adjustments and use the common validation below.
    if (mode != BaseBlockMode::STREAMK) {
        if (!OptimizeBaseBlockForCoreUtilization(mode)) {
            OP_LOGE(inputParams_.opName, "Failed to adjust base block.");
            return false;
        }
        OptimizeBaseBlockForLoadBalance();
    }
    return ValidateBaseBlock();
}

const BaseBlockRes& BaseBlockCalculator::GetOutput() const { return baseBlockRes_; }

bool BaseBlockCalculator::ValidateInput() const
{
    OP_TILING_CHECK(
        compileInfo_.aicNum == 0UL || batchCoreCnt_ == 0UL,
        CUBE_INNER_ERR_REPORT(
            inputParams_.opName,
            "Invalid BaseBlockCalculator input: aicNum(%lu) and batchCoreCnt(%lu) should be greater than 0.",
            static_cast<uint64_t>(compileInfo_.aicNum), batchCoreCnt_),
        return false);
    uint64_t baseMAlignSize = GetBaseMAlignSize();
    uint64_t baseNAlignSize = GetBaseNAlignSize();
    uint64_t baseKAlignSize = GetBaseKAlignSize();
    OP_TILING_CHECK(
        baseMAlignSize == 0UL || baseNAlignSize == 0UL || baseKAlignSize == 0UL,
        CUBE_INNER_ERR_REPORT(
            inputParams_.opName,
            "Invalid BaseBlockCalculator divisor: baseMAlignSize(%lu), baseNAlignSize(%lu) and baseKAlignSize(%lu) "
            "should be greater than 0.",
            baseMAlignSize, baseNAlignSize, baseKAlignSize),
        return false);
    return true;
}

bool BaseBlockCalculator::ValidateBaseBlock() const
{
    OP_TILING_CHECK(baseBlockRes_.baseM == 0UL || baseBlockRes_.baseN == 0UL || baseBlockRes_.baseK == 0UL,
                    CUBE_INNER_ERR_REPORT(
                        inputParams_.opName,
                        "BaseM, baseN and baseK should be greater than 0, but baseM: %lu, baseN: %lu, baseK: %lu.",
                        baseBlockRes_.baseM, baseBlockRes_.baseN, baseBlockRes_.baseK),
                    return false);
    return true;
}

void BaseBlockCalculator::ComputeBaseBlockDefault()
{
    baseBlockRes_.baseM = ops::CeilAlign(std::min(inputParams_.mSize, qmmv3_tiling_const::BASIC_BLOCK_SIZE_256),
                                         GetBaseMAlignSize());
    baseBlockRes_.baseN = ops::CeilAlign(std::min(inputParams_.nSize, qmmv3_tiling_const::BASIC_BLOCK_SIZE_256),
                                         GetBaseNAlignSize());

    uint64_t baseKDefaultSize = GetShapeWithDataType(qmmv3_tiling_const::BASIC_BLOCK_SIZE_128, inputParams_.aDtype);
    baseBlockRes_.baseK = ops::CeilAlign(std::min(baseKDefaultSize, inputParams_.kSize), GetBaseKAlignSize());
}

void BaseBlockCalculator::ComputeBaseBlockPerblock()
{
    if (inputParams_.mSize <= qmmv3_tiling_const::PER_BLOCK_SIZE ||
        inputParams_.mSize % qmmv3_tiling_const::PER_BLOCK_SIZE != 0UL) {
        baseBlockRes_.baseM = inputParams_.groupSizeM == 1UL &&
                                      inputParams_.mSize < qmmv3_tiling_const::PER_BLOCK_SIZE ?
                                  ops::CeilAlign(inputParams_.mSize, GetBaseMAlignSize()) :
                                  qmmv3_tiling_const::PER_BLOCK_SIZE;
        baseBlockRes_.baseN = inputParams_.nSize % qmmv3_tiling_const::PER_BLOCK_SIZE == 0UL ?
                                  PER_BLOCK_BASE_SIZE_256 :
                                  qmmv3_tiling_const::PER_BLOCK_SIZE;
    } else {
        baseBlockRes_.baseM = PER_BLOCK_BASE_SIZE_256;
        baseBlockRes_.baseN = qmmv3_tiling_const::PER_BLOCK_SIZE;
    }
    baseBlockRes_.baseK = qmmv3_tiling_const::PER_BLOCK_SIZE;
}

void BaseBlockCalculator::ComputeBaseBlockMmadS8S4()
{
    baseBlockRes_.baseM = ops::CeilAlign(std::min(inputParams_.mSize, qmmv3_tiling_const::BASIC_BLOCK_SIZE_256),
                                         GetBaseMAlignSize());
    baseBlockRes_.baseN = ops::CeilAlign(std::min(inputParams_.nSize, qmmv3_tiling_const::BASIC_BLOCK_SIZE_256),
                                         GetBaseNAlignSize());

    uint64_t basicBlockSizeA = qmmv3_tiling_const::BASIC_BLOCK_SIZE_128;
    uint64_t basicBlockSizeB = qmmv3_tiling_const::BASIC_BLOCK_SIZE_128;
    if (baseBlockRes_.baseM != 0UL &&
        compileInfo_.l0aSize / qmmv3_tiling_const::DOUBLE_BUFFER_NUM / baseBlockRes_.baseM >=
            qmmv3_tiling_const::BASIC_BLOCK_SIZE_256) {
        basicBlockSizeA = qmmv3_tiling_const::BASIC_BLOCK_SIZE_256;
    }
    if (baseBlockRes_.baseN != 0UL &&
        compileInfo_.l0bSize / qmmv3_tiling_const::DOUBLE_BUFFER_NUM / baseBlockRes_.baseN >=
            qmmv3_tiling_const::BASIC_BLOCK_SIZE_256) {
        basicBlockSizeB = qmmv3_tiling_const::BASIC_BLOCK_SIZE_256;
    }
    uint64_t minBaseK = std::min(std::min(GetShapeWithDataType(basicBlockSizeA, inputParams_.aDtype),
                                          GetShapeWithDataType(basicBlockSizeB, inputParams_.bDtype)),
                                 inputParams_.kSize);
    uint64_t maxAlignSize = std::max(GetBaseKAlignSize(),
                                     GetShapeWithDataType(qmmv3_tiling_const::CUBE_REDUCE_BLOCK, inputParams_.bDtype));
    baseBlockRes_.baseK = ops::CeilAlign(minBaseK, maxAlignSize);
}

bool BaseBlockCalculator::InitStreamKBaseBlock()
{
    baseBlockRes_.baseM = ops::CeilAlign(std::min(inputParams_.mSize, qmmv3_tiling_const::BASIC_BLOCK_SIZE_256),
                                         GetBaseMAlignSize());
    baseBlockRes_.baseN = ops::CeilAlign(std::min(inputParams_.nSize, qmmv3_tiling_const::BASIC_BLOCK_SIZE_256),
                                         GetBaseNAlignSize());
    OP_TILING_CHECK(baseBlockRes_.baseM == 0UL || baseBlockRes_.baseN == 0UL,
                    CUBE_INNER_ERR_REPORT(inputParams_.opName, "Invalid StreamK base divisor: baseM(%lu), baseN(%lu).",
                                          baseBlockRes_.baseM, baseBlockRes_.baseN),
                    return false);

    baseBlockRes_.mCnt = MathUtil::CeilDivision(inputParams_.mSize, baseBlockRes_.baseM);
    baseBlockRes_.nCnt = MathUtil::CeilDivision(inputParams_.nSize, baseBlockRes_.baseN);
    baseBlockRes_.preSplitKBlockCnt = batchCoreCnt_ * baseBlockRes_.mCnt * baseBlockRes_.nCnt;
    return true;
}

bool BaseBlockCalculator::UpdateSmallMnStreamKBase()
{
    if (baseBlockRes_.mCnt > compileInfo_.aicNum / SMALL_MN_EXPAND_RATIO &&
        baseBlockRes_.mCnt < compileInfo_.aicNum / qmmv3_tiling_const::NUM_HALF) {
        baseBlockRes_.mCnt = compileInfo_.aicNum / qmmv3_tiling_const::NUM_HALF;
    }
    if (baseBlockRes_.nCnt > compileInfo_.aicNum / SMALL_MN_EXPAND_RATIO &&
        baseBlockRes_.nCnt < compileInfo_.aicNum / qmmv3_tiling_const::NUM_HALF) {
        baseBlockRes_.nCnt = compileInfo_.aicNum / qmmv3_tiling_const::NUM_HALF;
    }
    baseBlockRes_.baseM = ops::CeilAlign(MathUtil::CeilDivision(inputParams_.mSize, baseBlockRes_.mCnt),
                                         GetBaseMAlignSize());
    baseBlockRes_.baseN = ops::CeilAlign(MathUtil::CeilDivision(inputParams_.nSize, baseBlockRes_.nCnt),
                                         GetBaseNAlignSize());
    baseBlockRes_.preSplitKBlockCnt = batchCoreCnt_ * baseBlockRes_.mCnt * baseBlockRes_.nCnt;
    OP_TILING_CHECK(baseBlockRes_.preSplitKBlockCnt == 0UL,
                    CUBE_INNER_ERR_REPORT(
                        inputParams_.opName, "Invalid StreamK preSplitKBlockCnt should be greater than 0."),
                    return false);
    baseBlockRes_.streamKCnt = std::max(compileInfo_.aicNum / baseBlockRes_.preSplitKBlockCnt, 1UL);
    baseBlockRes_.singleCoreK = MathUtil::CeilDivision(inputParams_.kSize, baseBlockRes_.streamKCnt);
    return true;
}

void BaseBlockCalculator::UpdateTailStreamKBase()
{
    uint64_t tailMnCnt = baseBlockRes_.preSplitKBlockCnt % compileInfo_.aicNum;
    if (tailMnCnt == 0UL) {
        baseBlockRes_.streamKCnt = 1UL;
        baseBlockRes_.singleCoreK = inputParams_.kSize;
        return;
    }
    baseBlockRes_.streamKCnt = std::max(compileInfo_.aicNum / tailMnCnt, 1UL);
    uint64_t skSingleCoreK = MathUtil::CeilDivision(inputParams_.kSize, baseBlockRes_.streamKCnt);
    baseBlockRes_.streamKCnt = MathUtil::CeilDivision(inputParams_.kSize, skSingleCoreK);
    baseBlockRes_.singleCoreK = skSingleCoreK;
}

bool BaseBlockCalculator::FinalizeStreamKBaseK()
{
    uint64_t baseKAlignValue = GetBaseKAlignSize();
    baseBlockRes_.singleCoreK = ops::CeilAlign(baseBlockRes_.singleCoreK, baseKAlignValue);
    OP_TILING_CHECK(baseBlockRes_.singleCoreK == 0UL,
                    CUBE_INNER_ERR_REPORT(inputParams_.opName, "Invalid StreamK singleCoreK should be greater than 0."),
                    return false);
    baseBlockRes_.streamKCnt = MathUtil::CeilDivision(inputParams_.kSize, baseBlockRes_.singleCoreK);

    uint64_t kValueMax = GetShapeWithDataType(compileInfo_.l0aSize / qmmv3_tiling_const::DOUBLE_BUFFER_NUM,
                                              inputParams_.aDtype) /
                         std::max(baseBlockRes_.baseM, baseBlockRes_.baseN);
    kValueMax = ops::FloorAlign(kValueMax, baseKAlignValue);
    OP_TILING_CHECK(
        kValueMax == 0UL,
        CUBE_INNER_ERR_REPORT(inputParams_.opName,
                              "Invalid StreamK L0A capacity: l0aSize(%lu), baseM(%lu), baseN(%lu), baseKAlign(%lu).",
                              compileInfo_.l0aSize, baseBlockRes_.baseM, baseBlockRes_.baseN, baseKAlignValue),
        return false);
    uint64_t kValueAlign = ops::CeilAlign(inputParams_.kSize, baseKAlignValue);
    baseBlockRes_.baseK = std::min(std::min(baseBlockRes_.singleCoreK, kValueMax), kValueAlign);
    baseBlockRes_.useTailWinLogic = false;
    return true;
}

bool BaseBlockCalculator::ComputeBaseBlockStreamK()
{
    if (!InitStreamKBaseBlock()) {
        return false;
    }
    if (baseBlockRes_.preSplitKBlockCnt <= compileInfo_.aicNum / qmmv3_tiling_const::NUM_HALF) {
        return UpdateSmallMnStreamKBase() && FinalizeStreamKBaseK();
    }
    UpdateTailStreamKBase();
    return FinalizeStreamKBaseK();
}

uint64_t BaseBlockCalculator::GetBaseMAlignSize() const
{
    return inputParams_.transA ? GetShapeWithDataType(qmmv3_tiling_const::L1_ALIGN_SIZE, inputParams_.aDtype) :
                                 qmmv3_tiling_const::CUBE_BLOCK;
}

uint64_t BaseBlockCalculator::GetBaseNAlignSize() const
{
    return inputParams_.transB ? qmmv3_tiling_const::CUBE_BLOCK :
                                 GetShapeWithDataType(qmmv3_tiling_const::L1_ALIGN_SIZE, inputParams_.bDtype);
}

uint64_t BaseBlockCalculator::GetBaseKAlignSize() const
{
    return inputParams_.isMxPerGroup ? qmmv3_tiling_const::MXFP_DIVISOR_SIZE :
                                       GetShapeWithDataType(qmmv3_tiling_const::CUBE_REDUCE_BLOCK, inputParams_.aDtype);
}

bool BaseBlockCalculator::OptimizeBaseBlockForCoreUtilization(BaseBlockMode mode)
{
    uint64_t oriBlock = batchCoreCnt_ * ops::CeilDiv(inputParams_.mSize, baseBlockRes_.baseM) *
                        ops::CeilDiv(inputParams_.nSize, baseBlockRes_.baseN);
    if (oriBlock >= compileInfo_.aicNum) {
        return true;
    }
    switch (compileInfo_.npuArch) {
        case NpuArch::DAV_3510:
            return mode == BaseBlockMode::PERBLOCK ? AdjustBaseBlockPerblock() : AdjustBaseBlockDefault();
        case NpuArch::DAV_RESV:
            return AdjustBaseBlockMmadS8S4(oriBlock);
        default:
            OP_LOGE(inputParams_.opName,
                    "Failed to find the AdjustBaseBlock function for the current NPU architecture.");
            return false;
    }
}

void BaseBlockCalculator::OptimizeBaseBlockForLoadBalance()
{
    if (!inputParams_.isMxPerGroup || compileInfo_.npuArch != NpuArch::DAV_3510) {
        return;
    }
    uint64_t roundLimit = GetSingleCoreMaxRound(baseBlockRes_.baseM, baseBlockRes_.baseN);
    // Rebalance only two- or three-round cases; keep the default block otherwise.
    if (roundLimit == 1UL || roundLimit > 3UL) {
        return;
    }
    uint64_t originLastRoundBlockCnt = GetLastRoundBlockCnt(baseBlockRes_.baseM, baseBlockRes_.baseN);
    if (originLastRoundBlockCnt == 0UL) {
        return;
    }
    const uint64_t originLastRoundUsedCore = originLastRoundBlockCnt *
                                             (static_cast<uint64_t>(compileInfo_.aicNum) / originLastRoundBlockCnt);
    const double originMemoryComputeScore = GetMemoryComputeScore(baseBlockRes_.baseM, baseBlockRes_.baseN);
    uint64_t bestBaseM = baseBlockRes_.baseM;
    uint64_t bestBaseN = baseBlockRes_.baseN;
    SearchLoadBalanceBaseBlock(roundLimit, originLastRoundUsedCore, originMemoryComputeScore, bestBaseM, bestBaseN);
    TryApplyLoadBalanceBase(bestBaseM, bestBaseN);
}

void BaseBlockCalculator::SearchLoadBalanceBaseBlock(uint64_t roundLimit, uint64_t originLastRoundUsedCore,
                                                     double originMemoryComputeScore, uint64_t& bestBaseM,
                                                     uint64_t& bestBaseN) const
{
    double balanceRate = 0.0;
    double memoryComputeScore = 0.0;
    uint64_t baseMAlignNum = inputParams_.transA ?
                                 GetShapeWithDataType(qmmv3_tiling_const::L2_ALIGN_SIZE, inputParams_.aDtype) :
                                 qmmv3_tiling_const::CUBE_BLOCK;
    uint64_t baseNAlignNum = inputParams_.transB ?
                                 qmmv3_tiling_const::CUBE_BLOCK :
                                 GetShapeWithDataType(qmmv3_tiling_const::L2_ALIGN_SIZE, inputParams_.bDtype);
    uint64_t searchBaseM = ops::CeilAlign(baseBlockRes_.baseM, baseMAlignNum);
    uint64_t searchBaseN = ops::CeilAlign(baseBlockRes_.baseN, baseNAlignNum);
    bool hasCandidate = false;
    for (uint64_t curBaseM = searchBaseM; curBaseM != 0UL; curBaseM = GetNextLoadBalanceBase(curBaseM, baseMAlignNum)) {
        for (uint64_t curBaseN = searchBaseN; curBaseN != 0UL;
             curBaseN = GetNextLoadBalanceBase(curBaseN, baseNAlignNum)) {
            uint64_t curRound = GetSingleCoreMaxRound(curBaseM, curBaseN);
            // Decreasing baseN only increases tile count in this loop, so later candidates will not recover.
            if (curRound > roundLimit) {
                break;
            }
            if (ShouldSkipLoadBalanceCandidate(curBaseM, curBaseN, originLastRoundUsedCore, originMemoryComputeScore)) {
                continue;
            }
            double curBalanceRate = GetBalanceRate(curBaseM, curBaseN);
            double curMemoryComputeScore = GetMemoryComputeScore(curBaseM, curBaseN);
            // Pick higher curBalanceRate first; use memory-compute score only as a tie-breaker.
            bool balanceBetter = curBalanceRate > balanceRate + SCORE_COMPARE_EPS;
            bool memoryComputeBetter = IsScoreEqual(curBalanceRate, balanceRate) &&
                                       curMemoryComputeScore + SCORE_COMPARE_EPS < memoryComputeScore;
            if (!hasCandidate || balanceBetter || memoryComputeBetter) {
                bestBaseM = curBaseM;
                bestBaseN = curBaseN;
                balanceRate = curBalanceRate;
                memoryComputeScore = curMemoryComputeScore;
                hasCandidate = true;
            }
        }
    }
}

bool BaseBlockCalculator::ShouldSkipLoadBalanceCandidate(uint64_t curBaseM, uint64_t curBaseN,
                                                         uint64_t originLastRoundUsedCore,
                                                         double originMemoryComputeScore) const
{
    bool isOriginBase = curBaseM == baseBlockRes_.baseM && curBaseN == baseBlockRes_.baseN;
    uint64_t curLastRoundBlockCnt = GetLastRoundBlockCnt(curBaseM, curBaseN);
    double curMemoryComputeScore = GetMemoryComputeScore(curBaseM, curBaseN);
    bool lastRoundBlockCntLess = curLastRoundBlockCnt < originLastRoundUsedCore;
    bool memoryComputeWorseWithoutCoreGain = curLastRoundBlockCnt == originLastRoundUsedCore &&
                                             curMemoryComputeScore > originMemoryComputeScore + SCORE_COMPARE_EPS;
    return isOriginBase || lastRoundBlockCntLess || memoryComputeWorseWithoutCoreGain;
}

void BaseBlockCalculator::TryApplyLoadBalanceBase(uint64_t bestBaseM, uint64_t bestBaseN)
{
    uint64_t baseN128AlignFallbackKThreshold = GetShapeWithDataType(LOAD_BALANCE_BASE_N_128_ALIGN_K_THRESHOLD,
                                                                    inputParams_.aDtype);
    // For small K, keep the original base if the candidate baseN is not 128-aligned.
    bool shouldFallbackToOriginBase = bestBaseN % qmmv3_tiling_const::BASIC_BLOCK_SIZE_128 != 0UL &&
                                      inputParams_.kSize < baseN128AlignFallbackKThreshold;
    if (shouldFallbackToOriginBase) {
        return;
    }
    baseBlockRes_.baseM = bestBaseM;
    baseBlockRes_.baseN = bestBaseN;
    if (baseBlockRes_.baseM > inputParams_.mSize) {
        baseBlockRes_.baseM = ops::CeilAlign(inputParams_.mSize, GetBaseMAlignSize());
    }
    if (baseBlockRes_.baseN > inputParams_.nSize) {
        baseBlockRes_.baseN = ops::CeilAlign(inputParams_.nSize, GetBaseNAlignSize());
    }
}

uint64_t BaseBlockCalculator::GetSingleCoreMaxRound(uint64_t baseM, uint64_t baseN) const
{
    uint64_t blockCnt = batchCoreCnt_ * ops::CeilDiv(inputParams_.mSize, baseM) *
                        ops::CeilDiv(inputParams_.nSize, baseN);
    return ops::CeilDiv(blockCnt, static_cast<uint64_t>(compileInfo_.aicNum));
}

double BaseBlockCalculator::GetBalanceRate(uint64_t baseM, uint64_t baseN) const
{
    uint64_t round = GetSingleCoreMaxRound(baseM, baseN);
    return static_cast<double>(batchCoreCnt_) * inputParams_.mSize * inputParams_.nSize / compileInfo_.aicNum /
           (static_cast<double>(round) * baseM * baseN);
}

uint64_t BaseBlockCalculator::GetLastRoundBlockCnt(uint64_t baseM, uint64_t baseN) const
{
    uint64_t aicNum = static_cast<uint64_t>(compileInfo_.aicNum);
    uint64_t blockCnt = batchCoreCnt_ * ops::CeilDiv(inputParams_.mSize, baseM) *
                        ops::CeilDiv(inputParams_.nSize, baseN);
    uint64_t tailBlockCnt = blockCnt % aicNum;
    return tailBlockCnt == 0UL ? aicNum : tailBlockCnt;
}

bool BaseBlockCalculator::AdjustBaseBlockDefault()
{
    uint64_t baseMAlignNum = inputParams_.transA ?
                                 GetShapeWithDataType(qmmv3_tiling_const::L2_ALIGN_SIZE, inputParams_.aDtype) :
                                 qmmv3_tiling_const::CUBE_BLOCK;
    uint64_t baseNAlignNum = inputParams_.transB ?
                                 qmmv3_tiling_const::CUBE_BLOCK :
                                 GetShapeWithDataType(qmmv3_tiling_const::L2_ALIGN_SIZE, inputParams_.bDtype);
    uint64_t baseKAlignNum = (inputParams_.transA && !inputParams_.transB) ?
                                 GetShapeWithDataType(qmmv3_tiling_const::BASIC_BLOCK_SIZE_32, inputParams_.aDtype) :
                                 GetShapeWithDataType(qmmv3_tiling_const::L2_ALIGN_SIZE, inputParams_.aDtype);
    if (IsMxBackwardTrans()) {
        baseKAlignNum = GetShapeWithDataType(qmmv3_tiling_const::MXFP_DIVISOR_SIZE, inputParams_.aDtype);
    }
    OP_TILING_CHECK(
        baseMAlignNum == 0UL || baseNAlignNum == 0UL || baseKAlignNum == 0UL,
        CUBE_INNER_ERR_REPORT(
            inputParams_.opName,
            "Invalid base block alignment: baseMAlignNum(%lu), baseNAlignNum(%lu), baseKAlignNum(%lu) should be "
            "greater than 0.",
            baseMAlignNum, baseNAlignNum, baseKAlignNum),
        return false);
    uint64_t mMaxtile = MathUtil::CeilDivision(inputParams_.mSize, baseMAlignNum);
    uint64_t nMaxtile = MathUtil::CeilDivision(inputParams_.nSize, baseNAlignNum);
    uint64_t tempBaseM = baseBlockRes_.baseM;
    uint64_t tempBaseN = baseBlockRes_.baseN;
    uint64_t coreNumMN = compileInfo_.aicNum / batchCoreCnt_;
    if (mMaxtile * nMaxtile < coreNumMN && (inputParams_.transA || !inputParams_.transB)) {
        return true;
    }

    uint64_t mCore = MathUtil::CeilDivision(inputParams_.mSize, baseBlockRes_.baseM);
    uint64_t nCore = MathUtil::CeilDivision(inputParams_.nSize, baseBlockRes_.baseN);
    if (mMaxtile < nMaxtile || (mMaxtile == nMaxtile && baseNAlignNum == qmmv3_tiling_const::CUBE_BLOCK)) {
        tempBaseM = ops::CeilAlign(MathUtil::CeilDivision(inputParams_.mSize, mCore), baseMAlignNum);
        mCore = MathUtil::CeilDivision(inputParams_.mSize, tempBaseM);
        nCore = coreNumMN / mCore;
        tempBaseN = ops::CeilAlign(MathUtil::CeilDivision(inputParams_.nSize, nCore), baseNAlignNum);
    } else {
        tempBaseN = ops::CeilAlign(MathUtil::CeilDivision(inputParams_.nSize, nCore), baseNAlignNum);
        nCore = MathUtil::CeilDivision(inputParams_.nSize, tempBaseN);
        mCore = coreNumMN / nCore;
        tempBaseM = ops::CeilAlign(MathUtil::CeilDivision(inputParams_.mSize, mCore), baseMAlignNum);
    }

    while (tempBaseN >= tempBaseM * BASEM_BASEN_RATIO && nCore < coreNumMN / qmmv3_tiling_const::NUM_HALF &&
           tempBaseN != baseNAlignNum) {
        nCore = nCore * DOUBLE_CORE_NUM;
        mCore = coreNumMN / nCore;
        tempBaseM = ops::CeilAlign(MathUtil::CeilDivision(inputParams_.mSize, mCore), baseMAlignNum);
        tempBaseN = ops::CeilAlign(MathUtil::CeilDivision(inputParams_.nSize, nCore), baseNAlignNum);
        mCore = MathUtil::CeilDivision(inputParams_.mSize, tempBaseM);
        nCore = MathUtil::CeilDivision(inputParams_.nSize, tempBaseN);
    }
    while (tempBaseM >= tempBaseN * BASEM_BASEN_RATIO && mCore < coreNumMN / qmmv3_tiling_const::NUM_HALF &&
           tempBaseM != baseMAlignNum) {
        mCore = mCore * DOUBLE_CORE_NUM;
        nCore = coreNumMN / mCore;
        tempBaseM = ops::CeilAlign(MathUtil::CeilDivision(inputParams_.mSize, mCore), baseMAlignNum);
        tempBaseN = ops::CeilAlign(MathUtil::CeilDivision(inputParams_.nSize, nCore), baseNAlignNum);
        mCore = MathUtil::CeilDivision(inputParams_.mSize, tempBaseM);
        nCore = MathUtil::CeilDivision(inputParams_.nSize, tempBaseN);
    }
    TrySwapBaseMNForMxFalseTrue(tempBaseM, tempBaseN);
    uint64_t kValueAlign = ops::CeilAlign(inputParams_.kSize, baseKAlignNum);
    uint64_t kValueMax = GetShapeWithDataType(compileInfo_.l0aSize / qmmv3_tiling_const::DOUBLE_BUFFER_NUM,
                                              inputParams_.aDtype) /
                         std::max(tempBaseM, tempBaseN);
    if (kValueMax >= baseKAlignNum) {
        baseBlockRes_.baseM = tempBaseM;
        baseBlockRes_.baseN = tempBaseN;
        kValueMax = ops::FloorAlign(kValueMax, baseKAlignNum);
        baseBlockRes_.baseK = std::min(kValueAlign, kValueMax);
        baseBlockRes_.baseK = baseBlockRes_.baseK > BASEK_LIMIT ?
                                  ops::CeilAlign(baseBlockRes_.baseK / qmmv3_tiling_const::NUM_HALF, baseKAlignNum) :
                                  baseBlockRes_.baseK;
        baseBlockRes_.useTailWinLogic = false;
    }
    return true;
}

void BaseBlockCalculator::TrySwapBaseMNForMxFalseTrue(uint64_t& baseM, uint64_t& baseN) const
{
    // MX false/true can end up with a narrow unaligned N tile. In single-round cases, try swapping M/N so baseN
    // becomes 128-aligned without reducing core utilization.
    if (!inputParams_.isMxPerGroup || inputParams_.transA || !inputParams_.transB ||
        baseN % qmmv3_tiling_const::BASIC_BLOCK_SIZE_128 == 0UL) {
        return;
    }
    uint64_t swapBaseM = baseN;
    uint64_t swapBaseN = baseM;
    uint64_t swapRound = GetSingleCoreMaxRound(swapBaseM, swapBaseN);
    // Only adjust single-round cases here; multi-round shapes are handled by load-balance search.
    if (swapRound != 1UL) {
        return;
    }
    uint64_t curMCore = MathUtil::CeilDivision(inputParams_.mSize, baseM);
    uint64_t curNCore = MathUtil::CeilDivision(inputParams_.nSize, baseN);
    uint64_t swapMCore = MathUtil::CeilDivision(inputParams_.mSize, swapBaseM);
    uint64_t swapNCore = MathUtil::CeilDivision(inputParams_.nSize, swapBaseN);
    uint64_t curUsedCore = curMCore * curNCore;
    uint64_t swapUsedCore = swapMCore * swapNCore;
    double curCoreShapeScore = GetBaseShapeScore(curMCore, curNCore);
    double swapCoreShapeScore = GetBaseShapeScore(swapMCore, swapNCore);
    bool swapCoreShapeBetter = swapCoreShapeScore + SCORE_COMPARE_EPS < curCoreShapeScore;
    bool swapBaseNIsFriendly = IsScoreEqual(swapCoreShapeScore, curCoreShapeScore) &&
                               swapBaseN % qmmv3_tiling_const::BASIC_BLOCK_SIZE_128 == 0UL;
    if (swapUsedCore >= curUsedCore && (swapCoreShapeBetter || swapBaseNIsFriendly)) {
        baseM = swapBaseM;
        baseN = swapBaseN;
    }
}

bool BaseBlockCalculator::AdjustBaseBlockPerblock()
{
    uint64_t coreNumMN = compileInfo_.aicNum / batchCoreCnt_;
    if (inputParams_.groupSizeM == 1UL) {
        return AdjustBaseBlockPertile(coreNumMN);
    } else if (baseBlockRes_.baseM == PER_BLOCK_BASE_SIZE_256 &&
               ops::CeilDiv(inputParams_.mSize, qmmv3_tiling_const::PER_BLOCK_SIZE) *
                       ops::CeilDiv(inputParams_.nSize, baseBlockRes_.baseN) <=
                   coreNumMN) {
        baseBlockRes_.baseM = qmmv3_tiling_const::PER_BLOCK_SIZE;
    } else if (baseBlockRes_.baseN == PER_BLOCK_BASE_SIZE_256 &&
               ops::CeilDiv(inputParams_.mSize, baseBlockRes_.baseM) *
                       ops::CeilDiv(inputParams_.nSize, qmmv3_tiling_const::PER_BLOCK_SIZE) <=
                   coreNumMN) {
        baseBlockRes_.baseN = qmmv3_tiling_const::PER_BLOCK_SIZE;
    }
    return true;
}

bool BaseBlockCalculator::AdjustBaseBlockPertile(uint64_t coreNumMN)
{
    uint64_t baseMAlignNum = inputParams_.transA ?
                                 GetShapeWithDataType(qmmv3_tiling_const::L2_ALIGN_SIZE, inputParams_.aDtype) :
                                 qmmv3_tiling_const::CUBE_BLOCK;
    uint64_t baseNAlignNum = !inputParams_.transB ?
                                 GetShapeWithDataType(qmmv3_tiling_const::L2_ALIGN_SIZE, inputParams_.bDtype) :
                                 qmmv3_tiling_const::CUBE_BLOCK;
    uint64_t adjustBaseM = baseBlockRes_.baseM;
    uint64_t adjustBaseN = baseBlockRes_.baseN;
    uint64_t adjustMCore = MathUtil::CeilDivision(inputParams_.mSize, adjustBaseM);
    uint64_t adjustNCore = MathUtil::CeilDivision(inputParams_.nSize, adjustBaseN);

    adjustMCore = coreNumMN / adjustNCore;
    adjustBaseM = ops::CeilAlign(MathUtil::CeilDivision(inputParams_.mSize, adjustMCore), baseMAlignNum);
    adjustMCore = MathUtil::CeilDivision(inputParams_.mSize, adjustBaseM);

    while (adjustBaseN / qmmv3_tiling_const::NUM_HALF >= baseNAlignNum &&
           adjustMCore * MathUtil::CeilDivision(inputParams_.nSize, adjustBaseN / qmmv3_tiling_const::NUM_HALF) <=
               coreNumMN) {
        adjustBaseN = adjustBaseN / qmmv3_tiling_const::NUM_HALF;
        adjustNCore = MathUtil::CeilDivision(inputParams_.nSize, adjustBaseN);
    }

    while (adjustBaseN > adjustBaseM * BASEM_BASEN_RATIO &&
           adjustBaseN / qmmv3_tiling_const::NUM_HALF >= baseNAlignNum) {
        uint64_t tempBaseN = adjustBaseN / qmmv3_tiling_const::NUM_HALF;
        uint64_t tempNCore = MathUtil::CeilDivision(inputParams_.nSize, tempBaseN);
        if (tempNCore == 0UL || tempNCore > coreNumMN) {
            break;
        }
        uint64_t tempMCore = coreNumMN / tempNCore;
        uint64_t tempBaseM = ops::CeilAlign(MathUtil::CeilDivision(inputParams_.mSize, tempMCore), baseMAlignNum);
        tempMCore = MathUtil::CeilDivision(inputParams_.mSize, tempBaseM);
        uint64_t tempUsedCoreNum = tempMCore * tempNCore;

        if (tempUsedCoreNum > coreNumMN) {
            break;
        }
        adjustBaseM = tempBaseM;
        adjustBaseN = tempBaseN;
        adjustMCore = tempMCore;
        adjustNCore = tempNCore;
    }

    baseBlockRes_.baseM = adjustBaseM;
    baseBlockRes_.baseN = adjustBaseN;
    return true;
}

bool BaseBlockCalculator::AdjustBaseBlockMmadS8S4(uint64_t oriBlock)
{
    uint64_t baseMAlignNum = inputParams_.transA ?
                                 GetShapeWithDataType(qmmv3_tiling_const::L2_ALIGN_SIZE, inputParams_.aDtype) :
                                 qmmv3_tiling_const::CUBE_BLOCK;
    uint64_t baseNAlignNum = qmmv3_tiling_const::L2_ALIGN_SIZE;
    uint64_t baseKAlignNum = (inputParams_.transA && !inputParams_.transB) ?
                                 GetShapeWithDataType(qmmv3_tiling_const::BASIC_BLOCK_SIZE_32, inputParams_.aDtype) :
                                 GetShapeWithDataType(qmmv3_tiling_const::L2_ALIGN_SIZE, inputParams_.aDtype);
    baseKAlignNum = std::min(baseKAlignNum, inputParams_.kSize);
    OP_TILING_CHECK(
        baseMAlignNum == 0UL || baseNAlignNum == 0UL || baseKAlignNum == 0UL,
        CUBE_INNER_ERR_REPORT(
            inputParams_.opName,
            "Invalid MMAD S8S4 alignment: baseMAlignNum(%lu), baseNAlignNum(%lu), baseKAlignNum(%lu) should be "
            "greater than 0.",
            baseMAlignNum, baseNAlignNum, baseKAlignNum),
        return false);
    uint64_t mMaxtile = MathUtil::CeilDivision(inputParams_.mSize, baseMAlignNum);
    uint64_t nMaxtile = MathUtil::CeilDivision(inputParams_.nSize, baseNAlignNum);
    uint64_t tempBaseM = baseMAlignNum;
    uint64_t tempBaseN = baseNAlignNum;
    uint64_t coreNumMN = static_cast<uint64_t>(compileInfo_.aicNum) / batchCoreCnt_;
    bool optimalFound = false;
    if (mMaxtile * nMaxtile < (oriBlock / batchCoreCnt_) && mMaxtile != 1UL && nMaxtile != 1UL) {
        return true;
    }
    if (mMaxtile == 1UL || nMaxtile == 1UL) {
        tempBaseM = mMaxtile == 1UL ?
                        baseMAlignNum :
                        std::max(baseMAlignNum,
                                 ops::CeilAlign(MathUtil::CeilDivision(inputParams_.mSize, coreNumMN), baseMAlignNum));
        tempBaseN = nMaxtile == 1UL ?
                        baseNAlignNum :
                        std::max(baseNAlignNum,
                                 ops::CeilAlign(MathUtil::CeilDivision(inputParams_.nSize, coreNumMN), baseNAlignNum));
        optimalFound = true;
    } else {
        optimalFound = CalculateOptimalSplit(tempBaseM, tempBaseN, baseMAlignNum, baseNAlignNum, baseKAlignNum);
    }
    uint64_t kValueAlign = ops::CeilAlign(inputParams_.kSize, baseKAlignNum);
    uint64_t kValueMax = GetShapeWithDataType(compileInfo_.l0aSize / qmmv3_tiling_const::DOUBLE_BUFFER_NUM,
                                              inputParams_.aDtype) /
                         std::max(tempBaseM, tempBaseN);
    if (kValueMax >= baseKAlignNum && optimalFound) {
        baseBlockRes_.baseM = tempBaseM;
        baseBlockRes_.baseN = tempBaseN;
        kValueMax = ops::FloorAlign(kValueMax, baseKAlignNum);
        baseBlockRes_.baseK = std::min(kValueAlign, kValueMax);
        baseBlockRes_.baseK = baseBlockRes_.baseK > BASEK_LIMIT ? baseBlockRes_.baseK / qmmv3_tiling_const::NUM_HALF :
                                                                  baseBlockRes_.baseK;
        uint64_t maxAlignSize = std::max(
            GetShapeWithDataType(qmmv3_tiling_const::CUBE_REDUCE_BLOCK, inputParams_.aDtype),
            GetShapeWithDataType(qmmv3_tiling_const::CUBE_REDUCE_BLOCK, inputParams_.bDtype));
        OP_TILING_CHECK(maxAlignSize == 0UL,
                        CUBE_INNER_ERR_REPORT(
                            inputParams_.opName,
                            "Invalid MMAD S8S4 K alignment: maxAlignSize(%lu) should be greater than 0.", maxAlignSize),
                        return false);
        baseBlockRes_.baseK = ops::CeilAlign(baseBlockRes_.baseK, maxAlignSize);
        baseBlockRes_.useTailWinLogic = false;
    }
    return true;
}

bool BaseBlockCalculator::CalculateOptimalSplit(uint64_t& baseM, uint64_t& baseN, uint64_t baseMAlignNum,
                                                uint64_t baseNAlignNum, uint64_t baseKAlignNum) const
{
    uint64_t mMaxtile = MathUtil::CeilDivision(inputParams_.mSize, baseMAlignNum);
    uint64_t nMaxtile = MathUtil::CeilDivision(inputParams_.nSize, baseNAlignNum);
    uint64_t maxUsedCore = MathUtil::CeilDivision(inputParams_.mSize, baseBlockRes_.baseM) *
                           MathUtil::CeilDivision(inputParams_.nSize, baseBlockRes_.baseN);
    uint64_t maxDiff = UINT64_MAX;
    uint64_t iterMSplite = std::min(mMaxtile, static_cast<uint64_t>(compileInfo_.aicNum / batchCoreCnt_));
    uint64_t iterNSplite = std::min(nMaxtile, static_cast<uint64_t>(compileInfo_.aicNum / batchCoreCnt_));
    uint64_t l0aHalfShape = GetShapeWithDataType(compileInfo_.l0aSize / qmmv3_tiling_const::DOUBLE_BUFFER_NUM,
                                                 inputParams_.aDtype);
    bool optimalFound = false;
    for (uint64_t mFactor = 1UL; mFactor <= iterMSplite; ++mFactor) {
        for (uint64_t nFactor = 1UL; nFactor <= iterNSplite; ++nFactor) {
            uint64_t tempMBase = mFactor * baseMAlignNum;
            uint64_t mCore = MathUtil::CeilDivision(inputParams_.mSize, tempMBase);
            uint64_t tempNBase = nFactor * baseNAlignNum;
            uint64_t nCore = MathUtil::CeilDivision(inputParams_.nSize, tempNBase);
            uint64_t usedCore = mCore * nCore;
            uint64_t diff = (tempMBase >= tempNBase) ? tempMBase - tempNBase : tempNBase - tempMBase;
            uint64_t kValueMax = l0aHalfShape / std::max(tempMBase, tempNBase);
            if (usedCore > compileInfo_.aicNum / batchCoreCnt_) {
                continue;
            }
            if ((usedCore > maxUsedCore || (usedCore == maxUsedCore && diff < maxDiff)) && kValueMax >= baseKAlignNum) {
                maxUsedCore = usedCore;
                maxDiff = diff;
                baseM = tempMBase;
                baseN = tempNBase;
                optimalFound = true;
            }
        }
    }
    return optimalFound;
}

bool BaseBlockCalculator::IsMxBackwardTrans() const
{
    return inputParams_.scaleDtype == ge::DT_FLOAT8_E8M0 && (inputParams_.transA || !inputParams_.transB);
}

} // namespace optiling
