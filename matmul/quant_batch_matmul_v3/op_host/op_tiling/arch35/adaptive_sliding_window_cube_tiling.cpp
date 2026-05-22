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
 * \file adaptive_sliding_window_cube_tiling.cpp
 * \brief
 */
#include <vector>

#include "common/op_host/op_tiling/tiling_type.h"
#include "log/log.h"
#include "error_util.h"
#include "op_host/tiling_templates_registry.h"
#include "adaptive_sliding_window_cube_tiling.h"
#include "base_block_calculator.h"
#include "l1_tiling_data_calculator.h"
#include "quant_batch_matmul_v3_tiling_strategy.h"

namespace {
constexpr uint64_t CUBE_BLOCK = 16;
constexpr uint64_t CUBE_REDUCE_BLOCK = 32;
constexpr uint64_t BASIC_BLOCK_SIZE_128 = 128UL;
constexpr uint32_t DB_SIZE = 2;
constexpr uint32_t DATA_SIZE_L0C = 4;
constexpr uint32_t NUM_HALF = 2;
constexpr uint64_t MIN_CARRY_DATA_SIZE_32K = 32 * 1024UL;
constexpr uint64_t FULL_LOAD_DATA_SIZE_64K = 64 * 1024UL;
constexpr uint64_t CACHE_LINE_512B = 512UL;
constexpr uint32_t MAX_STEPK_With_BL1_FULL = 8U;

const std::vector<int32_t> supportedNpuArch = {
    static_cast<int32_t>(NpuArch::DAV_3510), static_cast<int32_t>(NpuArch::DAV_RESV)};
constexpr int32_t TILING_PRIORITY = optiling::strategy::CUBE_ASW;
} // namespace

namespace optiling {

AdaptiveSlidingWindowCubeTiling::AdaptiveSlidingWindowCubeTiling(gert::TilingContext* context)
    : AdaptiveSlidingWindowTiling(context)
{
    Reset();
    tilingDataSize_ = sizeof(DequantBmm::QuantBatchMatmulV3TilingDataParams);
}

AdaptiveSlidingWindowCubeTiling::AdaptiveSlidingWindowCubeTiling(
    gert::TilingContext* context, DequantBmm::QuantBatchMatmulV3TilingDataParams* out)
    : AdaptiveSlidingWindowTiling(context, out)
{
    Reset();
    InitCompileInfo();
    inputParams_.Reset();
    tilingDataSize_ = sizeof(DequantBmm::QuantBatchMatmulV3TilingDataParams);
}

bool AdaptiveSlidingWindowCubeTiling::IsCapable()
{
    bool isCubePerTensor =
        inputParams_.isPerTensor &&
        ((inputParams_.aDtype == ge::DT_INT8 && inputParams_.biasDtype == ge::DT_INT32 && !inputParams_.isPertoken) ||
         ((inputParams_.aDtype == ge::DT_FLOAT8_E4M3FN || inputParams_.aDtype == ge::DT_FLOAT8_E5M2 ||
           inputParams_.aDtype == ge::DT_HIFLOAT8) &&
          inputParams_.scaleDtype == ge::DT_UINT64));
    bool isCubePerChannel =
        inputParams_.isPerChannel && (inputParams_.scaleDtype == ge::DT_UINT64 ||
                                      inputParams_.scaleDtype == ge::DT_INT64 || inputParams_.cDtype == ge::DT_INT32);
    return compileInfo_.supportMmadS8S4 ||
           (((inputParams_.isDoubleScale && !inputParams_.isPerChannel) || isCubePerTensor || isCubePerChannel) &&
            inputParams_.bFormat == ge::FORMAT_FRACTAL_NZ);
}

bool AdaptiveSlidingWindowCubeTiling::CalcBasicBlock()
{
    BaseBlockMode mode = compileInfo_.supportMmadS8S4 ? BaseBlockMode::MMAD_S8S4 : BaseBlockMode::DEFAULT;
    BaseBlockCalculator calculator(inputParams_, compileInfo_, GetBatchCoreCnt());
    if (!calculator.Compute(mode)) {
        return false;
    }
    const BaseBlockRes& baseBlockRes = calculator.GetOutput();
    adaptiveWin_.baseM = baseBlockRes.baseM;
    adaptiveWin_.baseN = baseBlockRes.baseN;
    adaptiveWin_.baseK = baseBlockRes.baseK;
    adaptiveWin_.useTailWinLogic = baseBlockRes.useTailWinLogic;
    return true;
}

bool AdaptiveSlidingWindowCubeTiling::CalL1Tiling()
{
    basicTiling_.usedCoreNum = CalUsedCoreNum();
    OP_LOGD(inputParams_.opName, "CoreNum: %u", basicTiling_.usedCoreNum);
    basicTiling_.baseM = adaptiveWin_.baseM;
    basicTiling_.baseN = adaptiveWin_.baseN;
    basicTiling_.baseK = adaptiveWin_.baseK;

    basicTiling_.stepM = 1U;
    basicTiling_.stepN = 1U;
    basicTiling_.singleCoreM = std::min(inputParams_.mSize, static_cast<uint64_t>(basicTiling_.baseM));
    basicTiling_.singleCoreN = std::min(inputParams_.nSize, static_cast<uint64_t>(basicTiling_.baseN));
    basicTiling_.singleCoreK = inputParams_.kSize;

    basicTiling_.iterateOrder = 0U;
    basicTiling_.dbL0c =
        ((basicTiling_.baseM * basicTiling_.baseN * DATA_SIZE_L0C * DB_SIZE <= aicoreParams_.l0cSize) &&
         CheckBiasAndScale(basicTiling_.baseN, DB_SIZE)) ?
            DB_SIZE :
            1U;

    switch (compileInfo_.npuArch) {
        case NpuArch::DAV_3510:
            return CalL1Tiling3510();
        case NpuArch::DAV_RESV:
            return CalL1TilingMmadS8S4();
        default:
            OP_LOGE(inputParams_.opName, "Failed to find the CalL1Tiling function for the current NPU architecture.");
            return false;
    }
}

bool AdaptiveSlidingWindowCubeTiling::CalL1Tiling3510()
{
    L1TilingMode mode = isAFullLoad_ ? L1TilingMode::A_L1_FULL_LOAD : L1TilingMode::DEFAULT;
    L1TilingDataCalculator l1Calculator(
        inputParams_, compileInfo_, basicTiling_.baseM, basicTiling_.baseN, basicTiling_.baseK);
    if (!l1Calculator.Compute(mode)) {
        return false;
    }
    const L1TilingData& l1TilingData = l1Calculator.GetOutput();
    basicTiling_.depthA1 = static_cast<uint32_t>(l1TilingData.depthKa_);
    basicTiling_.depthB1 = static_cast<uint32_t>(l1TilingData.depthKb_);
    basicTiling_.stepKa = static_cast<uint32_t>(l1TilingData.stepKa_);
    basicTiling_.stepKb = static_cast<uint32_t>(l1TilingData.stepKb_);
    return true;
}

bool AdaptiveSlidingWindowCubeTiling::CalL1TilingMmadS8S4()
{
    if (!IsCalL1TilingDepth4MmadS8S4()) {
        return CalL1Tiling3510();
    }
    uint64_t biasDtypeSize = ge::GetSizeByDataType(inputParams_.biasDtype);
    uint64_t scaleDtypeSize = ge::GetSizeByDataType(inputParams_.scaleDtype);
    if (inputParams_.cDtype == ge::DT_INT32) {
        scaleDtypeSize = 0;
    }
    uint64_t singleCoreBiasSize = inputParams_.hasBias ? basicTiling_.baseN * biasDtypeSize : 0UL;
    uint64_t singleCoreScaleSize = inputParams_.isPerChannel ? basicTiling_.baseN * scaleDtypeSize : 0UL;
    OP_TILING_CHECK(
        aicoreParams_.l1Size < singleCoreBiasSize ||
            aicoreParams_.l1Size - singleCoreBiasSize < singleCoreScaleSize,
        CUBE_INNER_ERR_REPORT(
            inputParams_.opName,
            "L1 size is insufficient after reserving bias and scale. L1Size: %lu, BiasSize: %lu, ScaleSize: %lu.",
            aicoreParams_.l1Size, singleCoreBiasSize, singleCoreScaleSize),
        return false);
    uint64_t leftL1Size = aicoreParams_.l1Size - singleCoreBiasSize - singleCoreScaleSize;
    CalL1TilingDepth4MmadS8S4(leftL1Size);
    return true;
}

bool AdaptiveSlidingWindowCubeTiling::IsCalL1TilingDepth4MmadS8S4() const
{
    bool a8w8Flag = inputParams_.aDtype == ge::DT_INT8 && inputParams_.bDtype == ge::DT_INT8;
    bool a8w4Flag = inputParams_.aDtype == ge::DT_INT8 && inputParams_.bDtype == ge::DT_INT4;
    bool a4w4Flag = inputParams_.aDtype == ge::DT_INT4 && inputParams_.bDtype == ge::DT_INT4;
    bool quantModeSupport = inputParams_.isPerChannel || inputParams_.isPerTensor;
    return compileInfo_.supportMmadS8S4 && (a8w8Flag || a8w4Flag || a4w4Flag) && quantModeSupport;
}

void AdaptiveSlidingWindowCubeTiling::UpdateAFullLoadStatus()
{
    if (inputParams_.batchA != 1UL || IsMxKOdd() || IsMxBackwardTrans() || isTilingOut_ || inputParams_.isPerBlock ||
        inputParams_.cDtype == ge::DT_INT32) {
        isAFullLoad_ = false;
        return;
    }
    uint64_t realBaseMSize = compileInfo_.supportMmadS8S4 || adaptiveWin_.mBaseTailSplitCnt == 1UL ?
                                 adaptiveWin_.baseM :
                                 adaptiveWin_.mTailMain;
    singleCoreASizeWithFullLoad_ =
        realBaseMSize *
        (inputParams_.transA ?
             GetSizeWithDataType(ops::CeilAlign(inputParams_.kSize, CUBE_BLOCK), inputParams_.aDtype) :
             ops::CeilAlign(GetSizeWithDataType(inputParams_.kSize, inputParams_.aDtype), CUBE_REDUCE_BLOCK));
    constexpr uint64_t A_L1_LOAD_THRESHOLD = 256 * 1024UL;
    bool blockCntCmp = (adaptiveWin_.mBlockCnt <= adaptiveWin_.nBlockCnt);
    if (compileInfo_.supportMmadS8S4) {
        blockCntCmp = true;
    }
    isAFullLoad_ =
        singleCoreASizeWithFullLoad_ <= A_L1_LOAD_THRESHOLD && blockCntCmp &&
        (((adaptiveWin_.mBlockCnt == 1UL || adaptiveWin_.mBlockCnt == 2UL || adaptiveWin_.mBlockCnt == 4UL) &&
          aicoreParams_.aicNum >= adaptiveWin_.mBlockCnt && aicoreParams_.aicNum % adaptiveWin_.mBlockCnt == 0) ||
         adaptiveWin_.mBlockCnt * adaptiveWin_.nBlockCnt <= aicoreParams_.aicNum);
    if (isAFullLoad_ && adaptiveWin_.baseM != realBaseMSize) {
        adaptiveWin_.baseM = realBaseMSize;
        adaptiveWin_.mBaseTailSplitCnt = 1UL;
        adaptiveWin_.mTailMain = 0UL;
    }
}

void AdaptiveSlidingWindowCubeTiling::UpdateBFullLoadStatus()
{
    if (!compileInfo_.supportMmadS8S4 || inputParams_.batchA != 1 || inputParams_.batchB != 1 ||
        inputParams_.aDtype != ge::DT_INT8 || inputParams_.bDtype != ge::DT_INT8 ||
        !(inputParams_.cDtype == ge::DT_INT8 || inputParams_.cDtype == ge::DT_FLOAT16) ||
        !(inputParams_.isPerTensor || inputParams_.isPerChannel) || isAFullLoad_ || isTilingOut_ ||
        inputParams_.cDtype == ge::DT_INT32) {
        isBFullLoad_ = false;
        return;
    }
    singleCoreBSizeWithFullLoad_ =
        adaptiveWin_.baseN *
        (inputParams_.transB ?
             ops::CeilAlign(GetSizeWithDataType(inputParams_.kSize, inputParams_.bDtype), CUBE_REDUCE_BLOCK) :
             GetSizeWithDataType(ops::CeilAlign(inputParams_.kSize, CUBE_BLOCK), inputParams_.bDtype));

    isBFullLoad_ =
        singleCoreBSizeWithFullLoad_ <= aicoreParams_.l1Size / NUM_HALF &&
        adaptiveWin_.mBlockCnt > adaptiveWin_.nBlockCnt &&
        (((adaptiveWin_.nBlockCnt == 1UL || adaptiveWin_.nBlockCnt == 2UL || adaptiveWin_.nBlockCnt == 4UL) &&
          aicoreParams_.aicNum >= adaptiveWin_.nBlockCnt && aicoreParams_.aicNum % adaptiveWin_.nBlockCnt == 0) ||
         adaptiveWin_.mBlockCnt * adaptiveWin_.nBlockCnt <= aicoreParams_.aicNum);
}

void AdaptiveSlidingWindowCubeTiling::UpdateABFullLoadStatus()
{
    if (inputParams_.batchA != 1 || inputParams_.batchB != 1 || isTilingOut_ || inputParams_.isPerBlock ||
        !compileInfo_.supportMmadS8S4 || inputParams_.isPertoken || inputParams_.aDtype != ge::DT_INT8 ||
        !(inputParams_.cDtype == ge::DT_INT8 || inputParams_.cDtype == ge::DT_FLOAT16)) {
        isABFullLoad_ = false;
        return;
    }
    singleCoreASizeWithFullLoad_ =
        adaptiveWin_.baseM *
        (inputParams_.transA ?
             GetSizeWithDataType(ops::CeilAlign(inputParams_.kSize, CUBE_BLOCK), inputParams_.aDtype) :
             ops::CeilAlign(GetSizeWithDataType(inputParams_.kSize, inputParams_.aDtype), CUBE_REDUCE_BLOCK));
    singleCoreBSizeWithFullLoad_ =
        adaptiveWin_.baseN *
        (inputParams_.transB ?
             ops::CeilAlign(GetSizeWithDataType(inputParams_.kSize, inputParams_.bDtype), CUBE_REDUCE_BLOCK) :
             GetSizeWithDataType(ops::CeilAlign(inputParams_.kSize, CUBE_BLOCK), inputParams_.bDtype));

    uint64_t biasDtypeSize = ge::GetSizeByDataType(inputParams_.biasDtype);
    uint64_t scaleDtypeSize = ge::GetSizeByDataType(inputParams_.scaleDtype);
    uint64_t singleCoreBiasSize = inputParams_.hasBias ? adaptiveWin_.baseN * biasDtypeSize : 0UL;
    uint64_t singleCoreScaleSize = inputParams_.isPerChannel ? adaptiveWin_.baseN * scaleDtypeSize : 0UL;
    constexpr uint64_t AB_L1_SINGLE_LOAD_THRESHOLD = 64 * 1024UL;
    constexpr uint64_t AB_L1_BOTH_LOAD_THRESHOLD = 272 * 1024UL;
    bool hasEnoughL1 =
        aicoreParams_.l1Size >= singleCoreASizeWithFullLoad_ &&
        aicoreParams_.l1Size - singleCoreASizeWithFullLoad_ >= singleCoreBSizeWithFullLoad_ &&
        aicoreParams_.l1Size - singleCoreASizeWithFullLoad_ - singleCoreBSizeWithFullLoad_ >= singleCoreBiasSize &&
        aicoreParams_.l1Size - singleCoreASizeWithFullLoad_ - singleCoreBSizeWithFullLoad_ - singleCoreBiasSize >=
            singleCoreScaleSize;
    bool isABSizeInRange = AB_L1_BOTH_LOAD_THRESHOLD >= singleCoreASizeWithFullLoad_ &&
                           AB_L1_BOTH_LOAD_THRESHOLD - singleCoreASizeWithFullLoad_ >= singleCoreBSizeWithFullLoad_;

    isABFullLoad_ = hasEnoughL1 && adaptiveWin_.mBlockCnt * adaptiveWin_.nBlockCnt <= aicoreParams_.aicNum &&
                    (singleCoreASizeWithFullLoad_ <= AB_L1_SINGLE_LOAD_THRESHOLD ||
                     singleCoreBSizeWithFullLoad_ <= AB_L1_SINGLE_LOAD_THRESHOLD) &&
                    isABSizeInRange;
}

void AdaptiveSlidingWindowCubeTiling::AnalyseFullLoadInfo()
{
    UpdateABFullLoadStatus();
    if (isABFullLoad_) {
        adaptiveWin_.useTailWinLogic = false;
        return;
    }
    UpdateAFullLoadStatus();
    UpdateBFullLoadStatus();
}

void AdaptiveSlidingWindowCubeTiling::CalcTailRoundBasicBlockSplit()
{
    if (!adaptiveWin_.useTailWinLogic) {
        return;
    }
    if (isAFullLoad_) {
        CalcTailBasicBlockAfullLoad();
    } else if (isBFullLoad_) {
        CalcTailBasicBlockBfullLoad();
    } else if (compileInfo_.supportMmadS8S4) {
        CalcTailBasicBlock4MmadS8S4();
    } else {
        CalcTailBasicBlock();
    }
}

void AdaptiveSlidingWindowCubeTiling::CalcTailBasicBlockBfullLoad()
{
    adaptiveWin_.nTailTile = 1UL;

    uint64_t mTile = 1UL;
    constexpr uint64_t MIN_BASEN_PER_TILE = 16UL;

    if (adaptiveWin_.tailWinBlockCnt != 0UL) {
        while (CalUsedCoreNum((mTile + 1UL), adaptiveWin_.nTailTile) <= aicoreParams_.aicNum &&
               adaptiveWin_.baseM / (mTile + 1UL) >= MIN_BASEN_PER_TILE) {
            mTile += 1UL;
        }
    }
    adaptiveWin_.mTailTile = mTile;
}

void AdaptiveSlidingWindowCubeTiling::CalcTailBasicBlock4MmadS8S4()
{
    if (adaptiveWin_.tailWinBlockCnt == 0UL) {
        return;
    }

    uint64_t mTile = 1UL;
    uint64_t nTile = 1UL;
    uint64_t preSplit = 1UL;
    uint64_t secSplit = 1UL;
    auto& preSplitValid = adaptiveWin_.mTail >= adaptiveWin_.nTail ? mTile : nTile;
    auto& secSplitValid = adaptiveWin_.mTail >= adaptiveWin_.nTail ? nTile : mTile;
    while (CalUsedCoreNum(preSplit + 1UL, secSplit) <= aicoreParams_.aicNum) {
        preSplit += 1UL;
        preSplitValid = !IsInValidWeighNzTailSplit(preSplit, true) ? preSplit : preSplitValid;
        if (CalUsedCoreNum(preSplit, secSplit + 1UL) <= aicoreParams_.aicNum) {
            secSplit += 1UL;
            secSplitValid = !IsInValidWeighNzTailSplit(secSplit, false) ? secSplit : secSplitValid;
        }
    }
    adaptiveWin_.mTailTile = mTile;
    adaptiveWin_.nTailTile = nTile;
}

bool AdaptiveSlidingWindowCubeTiling::CheckL1Size(uint64_t leftL1Size, uint64_t tempStepKa, uint64_t tempStepKb) const
{
    uint64_t singleCoreASize =
        tempStepKa *
        GetSizeWithDataType(static_cast<uint64_t>(basicTiling_.baseM) * basicTiling_.baseK, inputParams_.aDtype) *
        DB_SIZE;
    uint64_t singleCoreBSize =
        tempStepKb *
        GetSizeWithDataType(static_cast<uint64_t>(basicTiling_.baseN) * basicTiling_.baseK, inputParams_.bDtype) *
        DB_SIZE;

    if (isAFullLoad_) {
        singleCoreASize = singleCoreASizeWithFullLoad_;
    } else if (isBFullLoad_) {
        singleCoreBSize = singleCoreBSizeWithFullLoad_;
    }
    return leftL1Size >= singleCoreASize + singleCoreBSize;
}

void AdaptiveSlidingWindowCubeTiling::AdjustStepK(
    uint64_t leftL1Size, uint64_t& tempStepKa, uint64_t& tempStepKb, bool isStepKa) const
{
    uint64_t oneBaseADataSize =
        GetSizeWithDataType(static_cast<uint64_t>(basicTiling_.baseM) * basicTiling_.baseK, inputParams_.aDtype) *
        DB_SIZE;
    uint64_t oneBaseBDataSize =
        GetSizeWithDataType(static_cast<uint64_t>(basicTiling_.baseN) * basicTiling_.baseK, inputParams_.bDtype) *
        DB_SIZE;
    if (isStepKa) {
        uint64_t singleCoreBSize = tempStepKb * oneBaseBDataSize;
        if (isBFullLoad_) {
            singleCoreBSize = singleCoreBSizeWithFullLoad_;
        }
        if (leftL1Size < singleCoreBSize + oneBaseADataSize) {
            return;
        }
        tempStepKa = (leftL1Size - singleCoreBSize) / oneBaseADataSize;
    } else {
        uint64_t singleCoreASize = tempStepKa * oneBaseADataSize;
        if (isAFullLoad_) {
            singleCoreASize = singleCoreASizeWithFullLoad_;
        }
        if (leftL1Size < singleCoreASize + oneBaseBDataSize) {
            return;
        }
        tempStepKb = (leftL1Size - singleCoreASize) / oneBaseBDataSize;
    }
}

void AdaptiveSlidingWindowCubeTiling::CarryDataSizePass(uint64_t leftL1Size, uint64_t maxStepK)
{
    uint64_t tempStepKa = static_cast<uint64_t>(basicTiling_.stepKa);
    uint64_t tempStepKb = static_cast<uint64_t>(basicTiling_.stepKb);
    if (!isAFullLoad_) {
        tempStepKa = ops::CeilDiv(
            MIN_CARRY_DATA_SIZE_32K,
            GetSizeWithDataType(static_cast<uint64_t>(basicTiling_.baseM) * basicTiling_.baseK, inputParams_.aDtype));
        tempStepKa = std::min(tempStepKa, maxStepK);
        if (!CheckL1Size(leftL1Size, tempStepKa, tempStepKb)) {
            AdjustStepK(leftL1Size, tempStepKa, tempStepKb, true);
        }
    }
    if (!isBFullLoad_) {
        tempStepKb = ops::CeilDiv(
            MIN_CARRY_DATA_SIZE_32K,
            GetSizeWithDataType(static_cast<uint64_t>(basicTiling_.baseN) * basicTiling_.baseK, inputParams_.bDtype));
        tempStepKb = std::min(tempStepKb, maxStepK);
        if (!CheckL1Size(leftL1Size, tempStepKa, tempStepKb)) {
            AdjustStepK(leftL1Size, tempStepKa, tempStepKb, false);
        }
    }

    if (tempStepKa < static_cast<uint64_t>(basicTiling_.stepKa) ||
        tempStepKb < static_cast<uint64_t>(basicTiling_.stepKb)) {
        return;
    }
    basicTiling_.stepKa = static_cast<uint32_t>(tempStepKa);
    basicTiling_.stepKb = static_cast<uint32_t>(tempStepKb);
    OP_LOGD(inputParams_.opName, "CarryDataSizePass: stepKa stepKb %u %u", basicTiling_.stepKa, basicTiling_.stepKb);
}

void AdaptiveSlidingWindowCubeTiling::BalanceStepKPass(uint64_t leftL1Size)
{
    if (isAFullLoad_ || isBFullLoad_) {
        return;
    }
    if (basicTiling_.stepKa == basicTiling_.stepKb) {
        return;
    }
    uint64_t biggerStepK = static_cast<uint64_t>(std::max(basicTiling_.stepKa, basicTiling_.stepKb));
    if (CheckL1Size(leftL1Size, biggerStepK, biggerStepK)) {
        basicTiling_.stepKa = biggerStepK;
        basicTiling_.stepKb = biggerStepK;
        return;
    }

    uint64_t tempStepKa = static_cast<uint64_t>(basicTiling_.stepKa);
    uint64_t tempStepKb = static_cast<uint64_t>(basicTiling_.stepKb);
    if (tempStepKa > tempStepKb && tempStepKa % tempStepKb == 0UL) {
        uint64_t bestStepKb = tempStepKa;
        for (; bestStepKb >= tempStepKb; --bestStepKb) {
            if (tempStepKa % bestStepKb == 0UL && CheckL1Size(leftL1Size, tempStepKa, bestStepKb)) {
                break;
            }
        }
        tempStepKb = bestStepKb;
    } else if (tempStepKb > tempStepKa && tempStepKb % tempStepKa == 0UL) {
        uint64_t bestStepKa = tempStepKb;
        for (; bestStepKa >= tempStepKa; --bestStepKa) {
            if (tempStepKb % bestStepKa == 0UL && CheckL1Size(leftL1Size, bestStepKa, tempStepKb)) {
                break;
            }
        }
        tempStepKa = bestStepKa;
    } else {
        return;
    }
    basicTiling_.stepKa = static_cast<uint32_t>(tempStepKa);
    basicTiling_.stepKb = static_cast<uint32_t>(tempStepKb);
    OP_LOGD(inputParams_.opName, "BalanceStepKPass: stepKa stepKb %u %u", basicTiling_.stepKa, basicTiling_.stepKb);
}

void AdaptiveSlidingWindowCubeTiling::L1FullLoadCacheLinePass(
    uint64_t& tempStepKa, uint64_t& tempStepKb, uint64_t aCacheLine, uint64_t bCacheLine)
{
    uint32_t maxStepKWithSmallCase = 4U * DB_SIZE;
    if (ops::CeilDiv(basicTiling_.singleCoreK, basicTiling_.baseK) < maxStepKWithSmallCase) {
        return;
    }
    bool isEnableA = CACHE_LINE_512B % aCacheLine == 0;
    bool isEnableB = CACHE_LINE_512B % bCacheLine == 0;
    if (isAFullLoad_) {
        if (inputParams_.transB && isEnableB && (basicTiling_.baseN * bCacheLine <= FULL_LOAD_DATA_SIZE_64K)) {
            tempStepKb *= (CACHE_LINE_512B / bCacheLine);
        }
    } else {
        if (ops::CeilDiv(basicTiling_.singleCoreK, basicTiling_.baseK) < MAX_STEPK_With_BL1_FULL) {
            return;
        }
        if (!inputParams_.transA && isEnableA && (basicTiling_.baseM * aCacheLine <= FULL_LOAD_DATA_SIZE_64K)) {
            tempStepKa *= (CACHE_LINE_512B / aCacheLine);
        }
    }
}

void AdaptiveSlidingWindowCubeTiling::NONL1FullLoadCacheLinePass(
    uint64_t& tempStepKa, uint64_t& tempStepKb, uint64_t aCacheLine, uint64_t bCacheLine)
{
    bool isEnableA = CACHE_LINE_512B % aCacheLine == 0;
    bool isEnableB = CACHE_LINE_512B % bCacheLine == 0;
    uint64_t aDataSize =
        GetSizeWithDataType(static_cast<uint64_t>(basicTiling_.baseM) * basicTiling_.baseK, inputParams_.aDtype) *
        tempStepKa;
    uint64_t bDataSize =
        GetSizeWithDataType(static_cast<uint64_t>(basicTiling_.baseN) * basicTiling_.baseK, inputParams_.bDtype) *
        tempStepKb;
    uint64_t factor = 1UL;
    if (inputParams_.transA && inputParams_.transB) {
        factor = isEnableB ? CACHE_LINE_512B / bCacheLine : 1UL;
    } else if (!inputParams_.transA && !inputParams_.transB) {
        factor = isEnableA ? CACHE_LINE_512B / aCacheLine : 1UL;
    } else {
        if (aDataSize > bDataSize && aCacheLine > bCacheLine && isEnableA) {
            factor = CACHE_LINE_512B / aCacheLine;
        } else if (aDataSize < bDataSize && aCacheLine < bCacheLine && isEnableB) {
            factor = CACHE_LINE_512B / bCacheLine;
        }
    }
    tempStepKa *= factor;
    tempStepKb *= factor;
}

void AdaptiveSlidingWindowCubeTiling::PostCacheLinePass(uint64_t leftL1Size, uint64_t maxStepK)
{
    if (inputParams_.transA && !inputParams_.transB) {
        return;
    }
    uint64_t tempStepKa = static_cast<uint64_t>(basicTiling_.stepKa);
    uint64_t tempStepKb = static_cast<uint64_t>(basicTiling_.stepKb);
    uint64_t aCacheLine =
        GetSizeWithDataType(static_cast<uint64_t>(basicTiling_.baseK), inputParams_.aDtype) * tempStepKa;
    uint64_t bCacheLine =
        GetSizeWithDataType(static_cast<uint64_t>(basicTiling_.baseK), inputParams_.bDtype) * tempStepKb;

    if (isAFullLoad_ || isBFullLoad_) {
        L1FullLoadCacheLinePass(tempStepKa, tempStepKb, aCacheLine, bCacheLine);
    } else {
        NONL1FullLoadCacheLinePass(tempStepKa, tempStepKb, aCacheLine, bCacheLine);
    }

    if (!CheckL1Size(leftL1Size, tempStepKa, tempStepKb) || tempStepKa > maxStepK || tempStepKb > maxStepK) {
        return;
    }
    basicTiling_.stepKa = static_cast<uint32_t>(tempStepKa);
    basicTiling_.stepKb = static_cast<uint32_t>(tempStepKb);
    OP_LOGD(inputParams_.opName, "PostCacheLinePass: stepKa stepKb %u %u", basicTiling_.stepKa, basicTiling_.stepKb);
}

void AdaptiveSlidingWindowCubeTiling::CalL1TilingDepth4MmadS8S4(uint64_t leftL1Size)
{
    basicTiling_.stepKa = 1U;
    basicTiling_.stepKb = 1U;
    basicTiling_.depthA1 = 1U;
    basicTiling_.depthB1 = 1U;
    if (isABFullLoad_) {
        return;
    }

    uint64_t maxStepK = ops::CeilDiv(inputParams_.kSize, static_cast<uint64_t>(basicTiling_.baseK));
    CarryDataSizePass(leftL1Size, maxStepK);
    BalanceStepKPass(leftL1Size);
    PostCacheLinePass(leftL1Size, maxStepK);

    basicTiling_.stepKa = isAFullLoad_ ? 1U : basicTiling_.stepKa;
    basicTiling_.depthA1 = isAFullLoad_ ? basicTiling_.stepKa : basicTiling_.stepKa * DB_SIZE;
    basicTiling_.stepKb = isBFullLoad_ ? 1U : basicTiling_.stepKb;
    basicTiling_.depthB1 = isBFullLoad_ ? basicTiling_.stepKb : basicTiling_.stepKb * DB_SIZE;
}

REGISTER_TILING_TEMPLATE_WITH_ARCH(
    QuantBatchMatmulV3, AdaptiveSlidingWindowCubeTiling, supportedNpuArch, TILING_PRIORITY);
} // namespace optiling
