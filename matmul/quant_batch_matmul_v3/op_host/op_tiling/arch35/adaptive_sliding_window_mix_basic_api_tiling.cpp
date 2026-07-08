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
 * \file adaptive_sliding_window_mix_basic_api_tiling.cpp
 * \brief Mix template with BasicAPI (tensorApi) tiling paradigm - batch version (batchC > 1).
 */

#include "common/op_host/op_tiling/tiling_type.h"
#include "log/log.h"
#include "error_util.h"
#include "op_host/tiling_templates_registry.h"
#include "adaptive_sliding_window_mix_basic_api_tiling.h"
#include "base_block_calculator.h"
#include "l1_tiling_data_calculator.h"
#include "quant_batch_matmul_v3_tiling_strategy.h"
#include "util/math_util.h"

namespace {
constexpr uint32_t DB_SIZE = 2;
constexpr uint64_t WINDOW_LEN = 4UL;
constexpr uint32_t DATA_SIZE_L0C = 4;
constexpr uint8_t L1_TWO_BUFFER = 2;
constexpr uint8_t L1_FOUR_BUFFER = 4;
constexpr uint32_t CORE_RATIO = 2U;
constexpr uint64_t AFULLLOAD_SINGLE_CORE_A_SCALER = 2UL;
constexpr uint64_t MIXFP_DIVISOR_SIZE = 64UL;
constexpr uint64_t CUBE_BLOCK = 16UL;
constexpr uint64_t CUBE_REDUCE_BLOCK = 32UL;
constexpr uint64_t BASEM_BASEN_RATIO = 2UL;
constexpr uint32_t NUM_HALF = 2;
constexpr uint64_t L1_ALIGN_SIZE = 32UL;
constexpr uint64_t L2_ALIGN_SIZE = 128UL;
constexpr uint64_t BASIC_BLOCK_SIZE_32 = 32UL;
constexpr uint64_t BASEK_LIMIT = 4095UL;

const std::vector<int32_t> supportedNpuArch = {static_cast<int32_t>(NpuArch::DAV_3510)};
constexpr int32_t TILING_PRIORITY = optiling::strategy::MIX_BASIC_API_ASW;

uint64_t GetShapeWithDataType(uint64_t size, ge::DataType dtype)
{
    uint64_t dtypeSize = static_cast<uint64_t>(ge::GetSizeByDataType(dtype));
    return dtypeSize == 0UL ? 0UL : size / dtypeSize;
}
} // namespace

namespace optiling {

AdaptiveSlidingWindowMixBasicAPITiling::AdaptiveSlidingWindowMixBasicAPITiling(gert::TilingContext* context)
    : AdaptiveSlidingWindowTiling(context), tilingData_(tilingDataSelf_)
{
    Reset();
    tilingDataSize_ = sizeof(DequantBmm::QuantBatchMatmulV3BasicAPITilingData);
}

AdaptiveSlidingWindowMixBasicAPITiling::AdaptiveSlidingWindowMixBasicAPITiling(
    gert::TilingContext* context, DequantBmm::QuantBatchMatmulV3BasicAPITilingData* out)
    : AdaptiveSlidingWindowTiling(context, nullptr), tilingData_(*out)
{
    Reset();
    InitCompileInfo();
    inputParams_.Reset();
    tilingDataSize_ = sizeof(DequantBmm::QuantBatchMatmulV3BasicAPITilingData);
}

void AdaptiveSlidingWindowMixBasicAPITiling::Reset()
{
    if (!isTilingOut_) {
        tilingData_ = DequantBmm::QuantBatchMatmulV3BasicAPITilingData();
    }
    withoutBatchTilingData_ = DequantBmm::QuantBatchMatmulV3TensorAPIWithoutBatchTilingData();
    useWithoutBatchTilingData_ = false;
    tilingDataSize_ = sizeof(DequantBmm::QuantBatchMatmulV3BasicAPITilingData);
}

bool AdaptiveSlidingWindowMixBasicAPITiling::IsWithoutBatchTilingData() const
{
    return !isTilingOut_ && inputParams_.batchC == 1UL;
}

uint64_t AdaptiveSlidingWindowMixBasicAPITiling::GetApiLevel(NpuArch) const
{
    return static_cast<uint64_t>(QMMApiLevel::BLAZE_LEVEL);
}

bool AdaptiveSlidingWindowMixBasicAPITiling::IsCapable()
{
    SetBf16Compat();
    bool isScaleVecPostProcess = inputParams_.isPerChannel &&
                                 !(inputParams_.scaleDtype == ge::DT_UINT64 || inputParams_.scaleDtype == ge::DT_INT64);
    bool isFp8OrHif8TTBiasMix = IsFp8OrHif8TTFloatBiasMix(inputParams_);
    bool isMixType = ((inputParams_.aDtype == ge::DT_INT8 && inputParams_.cDtype == ge::DT_BF16) || 
                inputParams_.aDtype == ge::DT_FLOAT8_E4M3FN || inputParams_.aDtype == ge::DT_HIFLOAT8) && 
                inputParams_.aDtype == inputParams_.bDtype;
    bool capable = (isScaleVecPostProcess || inputParams_.isPertoken || isBf16Mix_ || isFp8OrHif8TTBiasMix) && 
                    inputParams_.transA == 0 && isMixType && inputParams_.cDtype != ge::DT_INT32 &&
                    inputParams_.bFormat == ge::FORMAT_FRACTAL_NZ && IsTensorapiCapable();
    return capable;
}

bool AdaptiveSlidingWindowMixBasicAPITiling::CheckCoreNum() const
{
    if (compileInfo_.aivNum != CORE_RATIO * compileInfo_.aicNum) {
        OP_LOGE(
            inputParams_.opName, "For mix template, aicNum:aivNum should be 1:2, actual aicNum: %u, aivNum: %u.",
            compileInfo_.aicNum, compileInfo_.aivNum);
        return false;
    }
    return true;
}

ge::graphStatus AdaptiveSlidingWindowMixBasicAPITiling::GetWorkspaceSize()
{
    workspaceSize_ = inputParams_.libApiWorkSpaceSize;
    return ge::GRAPH_SUCCESS;
}

uint64_t AdaptiveSlidingWindowMixBasicAPITiling::GetBatchCoreCnt() const
{
    return inputParams_.batchC;
}

const void* AdaptiveSlidingWindowMixBasicAPITiling::GetTilingData() const
{
    return useWithoutBatchTilingData_ ? static_cast<const void*>(&withoutBatchTilingData_) :
                                        static_cast<const void*>(&tilingData_);
}

uint64_t AdaptiveSlidingWindowMixBasicAPITiling::GetBaseMAlignSize() const
{
    return CUBE_BLOCK;
}

uint64_t AdaptiveSlidingWindowMixBasicAPITiling::GetBaseNAlignSize() const
{
    return inputParams_.transB ? CUBE_BLOCK : GetShapeWithDataType(L1_ALIGN_SIZE, inputParams_.bDtype);
}

bool AdaptiveSlidingWindowMixBasicAPITiling::InitBaseBlockOptimizeInfo(BaseBlockOptimizeInfo &info)
{
    info.baseMAlign = CUBE_BLOCK;
    info.baseNAlign = inputParams_.transB ? CUBE_BLOCK :
                                            GetShapeWithDataType(L2_ALIGN_SIZE, inputParams_.bDtype);
    info.baseKAlign = (inputParams_.transA && !inputParams_.transB) ?
        GetShapeWithDataType(BASIC_BLOCK_SIZE_32, inputParams_.aDtype) :
        GetShapeWithDataType(L2_ALIGN_SIZE, inputParams_.aDtype);

    if (info.baseMAlign == 0UL || info.baseNAlign == 0UL || info.baseKAlign == 0UL ||
        adaptiveWin_.baseM == 0UL || adaptiveWin_.baseN == 0UL || adaptiveWin_.baseK == 0UL) {
        return false;
    }

    info.mCore = ops::CeilDiv(inputParams_.mSize, adaptiveWin_.baseM);
    info.nCore = ops::CeilDiv(inputParams_.nSize, adaptiveWin_.baseN);
    const uint64_t batchCoreCnt = std::max(1UL, GetBatchCoreCnt());
    info.coreNumMN = std::max(1UL, ops::FloorDiv(static_cast<uint64_t>(compileInfo_.aicNum), batchCoreCnt));
    info.blockCnt = info.mCore * info.nCore;
    info.targetBlockCnt = std::max(info.blockCnt, info.coreNumMN);

    const uint64_t mTail = inputParams_.mSize % adaptiveWin_.baseM;
    const uint64_t nTail = inputParams_.nSize % adaptiveWin_.baseN;
    info.hasSmallMTail = inputParams_.mSize > adaptiveWin_.baseM && mTail > 0UL &&
                         mTail * NUM_HALF < adaptiveWin_.baseM;
    info.hasSmallNTail = inputParams_.nSize > adaptiveWin_.baseN && nTail > 0UL &&
                         nTail * NUM_HALF < adaptiveWin_.baseN;

    return info.hasSmallMTail || info.hasSmallNTail || info.blockCnt < info.coreNumMN;
}

// Calculate score for a baseMN configuration.
uint64_t AdaptiveSlidingWindowMixBasicAPITiling::CalcBaseBlockScore(
    uint64_t baseM, uint64_t baseN, const BaseBlockOptimizeInfo &info)
{
    const uint64_t curMCore = ops::CeilDiv(inputParams_.mSize, baseM);
    const uint64_t curNCore = ops::CeilDiv(inputParams_.nSize, baseN);
    const uint64_t curBlockCnt = curMCore * curNCore;

    const uint64_t coveredMN = curBlockCnt * baseM * baseN;
    const uint64_t usefulMN = inputParams_.mSize * inputParams_.nSize;
    const uint64_t wasteScore = coveredMN > usefulMN ? coveredMN - usefulMN : 0UL;

    const uint64_t coreScore = curBlockCnt < info.targetBlockCnt ?
        (info.targetBlockCnt - curBlockCnt) * baseM * baseN : 0UL;

    const uint64_t balanceScore =
        (baseN >= baseM * BASEM_BASEN_RATIO || baseM >= baseN * BASEM_BASEN_RATIO) ?
        baseM * baseN : 0UL;

    return wasteScore + coreScore + balanceScore;
}

// Try a candidate baseMNK configuration.
void AdaptiveSlidingWindowMixBasicAPITiling::TryUpdateBaseBlockCandidate(uint64_t candMCore, uint64_t candNCore, 
    const BaseBlockOptimizeInfo &info, uint64_t &bestBaseM, uint64_t &bestBaseN, uint64_t &bestBaseK, uint64_t &bestScore)
{
    if (candMCore == 0UL || candNCore == 0UL) {
        return;
    }

    const uint64_t candBaseM = ops::CeilAlign(ops::CeilDiv(inputParams_.mSize, candMCore), info.baseMAlign);
    const uint64_t candBaseN = ops::CeilAlign(ops::CeilDiv(inputParams_.nSize, candNCore), info.baseNAlign);
    if (candBaseM == 0UL || candBaseN == 0UL || candBaseM > inputParams_.mSize ||
        candBaseN > inputParams_.nSize || candBaseM * candBaseN * DATA_SIZE_L0C > aicoreParams_.l0cSize ||
        !CheckBiasAndScale(candBaseN, 1UL)) {
        return;
    }

    uint64_t kValueMax = GetShapeWithDataType(compileInfo_.l0aSize / DB_SIZE, inputParams_.aDtype) /
                         std::max(candBaseM, candBaseN);
    if (kValueMax < info.baseKAlign) {
        return;
    }
    kValueMax = ops::FloorAlign(kValueMax, info.baseKAlign);

    uint64_t candBaseK = adaptiveWin_.baseK;
    if (candBaseK == 0UL || candBaseK > kValueMax || candBaseK % info.baseKAlign != 0UL) {
        candBaseK = std::min(ops::CeilAlign(inputParams_.kSize, info.baseKAlign), kValueMax);
        if (candBaseK > BASEK_LIMIT) {
            candBaseK = ops::CeilAlign(candBaseK / NUM_HALF, info.baseKAlign);
        }
    }

    if (candBaseK == 0UL || candBaseK > kValueMax || candBaseK % info.baseKAlign != 0UL) {
        return;
    }

    const uint64_t score = CalcBaseBlockScore(candBaseM, candBaseN, info);
    if (score >= bestScore) {
        return;
    }

    bestScore = score;
    bestBaseM = candBaseM;
    bestBaseN = candBaseN;
    bestBaseK = candBaseK;
}

// Main function for repartitioning baseMNK.
void AdaptiveSlidingWindowMixBasicAPITiling::OptimizeBaseBlock()
{
    BaseBlockOptimizeInfo info;
    if (!InitBaseBlockOptimizeInfo(info)) {
        return;
    }

    uint64_t bestBaseM = adaptiveWin_.baseM;
    uint64_t bestBaseN = adaptiveWin_.baseN;
    uint64_t bestBaseK = adaptiveWin_.baseK;

    // Use the current baseMNK as the initial best candidate instead of UINT64_MAX.
    uint64_t bestScore = CalcBaseBlockScore(bestBaseM, bestBaseN, info);

    TryUpdateBaseBlockCandidate(info.mCore, info.nCore, info,
                                bestBaseM, bestBaseN, bestBaseK, bestScore);

    // Adjust baseMNK to fully utilize all cores when the total basic block count is less than core count.
    if (info.blockCnt < info.coreNumMN) {
        const uint64_t mCoreMax = std::min(ops::CeilDiv(inputParams_.mSize, info.baseMAlign), info.coreNumMN);
        const uint64_t nCoreMax = ops::CeilDiv(inputParams_.nSize, info.baseNAlign);

        for (uint64_t candMCore = 1UL; candMCore <= mCoreMax; ++candMCore) {
            const uint64_t candNCore = std::min(nCoreMax, std::max(1UL, info.coreNumMN / candMCore));
            TryUpdateBaseBlockCandidate(candMCore, candNCore, info,
                                        bestBaseM, bestBaseN, bestBaseK, bestScore);
        }
    }

    adaptiveWin_.baseM = bestBaseM;
    adaptiveWin_.baseN = bestBaseN;
    adaptiveWin_.baseK = bestBaseK;
}

bool AdaptiveSlidingWindowMixBasicAPITiling::CalcBasicBlock()
{
    BaseBlockCalculator calculator(inputParams_, compileInfo_, GetBatchCoreCnt());
    if (!calculator.Compute(BaseBlockMode::DEFAULT)) {
        return false;
    }
    const BaseBlockRes& baseBlockRes = calculator.GetOutput();

    adaptiveWin_.baseM = baseBlockRes.baseM;
    adaptiveWin_.baseN = baseBlockRes.baseN;
    adaptiveWin_.baseK = baseBlockRes.baseK;
    OptimizeBaseBlock();
    adaptiveWin_.useTailWinLogic = baseBlockRes.useTailWinLogic;
    return true;
}

bool AdaptiveSlidingWindowMixBasicAPITiling::CalL1Tiling()
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

ge::graphStatus AdaptiveSlidingWindowMixBasicAPITiling::DoLibApiTiling()
{
    tilingData_.matmulTiling.m = inputParams_.mSize;
    tilingData_.matmulTiling.n = inputParams_.nSize;
    tilingData_.matmulTiling.k = inputParams_.kSize;

    tilingData_.matmulTiling.baseM = basicTiling_.baseM;
    tilingData_.matmulTiling.baseN = basicTiling_.baseN;
    tilingData_.matmulTiling.baseK = basicTiling_.baseK;
    tilingData_.matmulTiling.isBias = inputParams_.hasBias ? 1UL : 0UL;
    tilingData_.matmulTiling.dbL0C = static_cast<uint8_t>(basicTiling_.dbL0c);
    CalculateNBufferNum4Mix();
    if (useWithoutBatchTilingData_) {
        SetWithoutBatchTilingData();
    }
    return ge::GRAPH_SUCCESS;
}

void AdaptiveSlidingWindowMixBasicAPITiling::CalculateNBufferNum4Mix()
{
    tilingData_.matmulTiling.stepKa = basicTiling_.stepKa;
    tilingData_.matmulTiling.stepKb = basicTiling_.stepKb;
    uint64_t kL1 = 0;
    if (isAFullLoad_) {
        kL1 = basicTiling_.stepKb * tilingData_.matmulTiling.baseK;
    } else {
        uint64_t stepK = std::min(basicTiling_.stepKa, basicTiling_.stepKb);
        kL1 = stepK * tilingData_.matmulTiling.baseK;
    }
    uint64_t usedL1Size = 0UL;
    if (isBFullLoad_) {
        uint64_t kAligned = ops::CeilAlign(inputParams_.kSize, inputParams_.transB ? CUBE_REDUCE_BLOCK : CUBE_BLOCK);
        usedL1Size = GetSizeWithDataType(basicTiling_.baseN * kAligned, inputParams_.bDtype);
    } else {
        usedL1Size = GetSizeWithDataType(basicTiling_.baseN * kL1, inputParams_.bDtype) * L1_FOUR_BUFFER;
    }
    if (inputParams_.isPerChannel) {
        usedL1Size += GetSizeWithDataType(basicTiling_.baseN, inputParams_.scaleDtype) * L1_TWO_BUFFER;
    }
    if (inputParams_.hasBias) {
        usedL1Size += GetSizeWithDataType(basicTiling_.baseN, inputParams_.biasDtype) * L1_TWO_BUFFER;
    }
    if (isAFullLoad_) {
        uint64_t kAligned = ops::CeilAlign(inputParams_.kSize, !inputParams_.transA ? CUBE_REDUCE_BLOCK : CUBE_BLOCK);
        usedL1Size += GetSizeWithDataType(basicTiling_.baseM * kAligned, inputParams_.aDtype);
    } else {
        usedL1Size += GetSizeWithDataType(basicTiling_.baseM * kL1, inputParams_.aDtype) * L1_FOUR_BUFFER;
    }
    tilingData_.matmulTiling.nBufferNum = usedL1Size < aicoreParams_.l1Size ? L1_FOUR_BUFFER : L1_TWO_BUFFER;
    if (tilingData_.matmulTiling.nBufferNum == L1_FOUR_BUFFER) {
        tilingData_.matmulTiling.stepKa = std::min(basicTiling_.stepKa, basicTiling_.stepKb);
        tilingData_.matmulTiling.stepKb = tilingData_.matmulTiling.stepKa;
    }
}

void AdaptiveSlidingWindowMixBasicAPITiling::UpdateAFullLoadStatus()
{
    uint64_t realBaseMSize = adaptiveWin_.mBaseTailSplitCnt == 1UL ? adaptiveWin_.baseM : adaptiveWin_.mTailMain;
    uint64_t singleCoreASize = GetSizeWithDataType(realBaseMSize * inputParams_.kSize, inputParams_.aDtype);
    isAFullLoad_ = singleCoreASize <= aicoreParams_.l1Size / AFULLLOAD_SINGLE_CORE_A_SCALER &&
                   adaptiveWin_.mBlockCnt < WINDOW_LEN && aicoreParams_.aicNum % adaptiveWin_.mBlockCnt == 0 &&
                   adaptiveWin_.totalBlockCnt > aicoreParams_.aicNum && inputParams_.batchC == 1;
    if (isAFullLoad_ && adaptiveWin_.baseM != realBaseMSize) {
        adaptiveWin_.baseM = realBaseMSize;
        adaptiveWin_.mBaseTailSplitCnt = 1UL;
        adaptiveWin_.mTailMain = 0UL;
    }
}

void AdaptiveSlidingWindowMixBasicAPITiling::AnalyseFullLoadInfo()
{
    isABFullLoad_ = false;
    isBFullLoad_ = false;
    UpdateAFullLoadStatus();
}

void AdaptiveSlidingWindowMixBasicAPITiling::CalcTailRoundBasicBlockSplit()
{
    if (!adaptiveWin_.useTailWinLogic) {
        return;
    }
    if (isAFullLoad_) {
        CalcTailBasicBlockAfullLoad();
    } else {
        CalcTailBasicBlock();
    }
}

void AdaptiveSlidingWindowMixBasicAPITiling::SetTilingData()
{
    useWithoutBatchTilingData_ = IsWithoutBatchTilingData();
    tilingDataSize_ = useWithoutBatchTilingData_ ?
        sizeof(DequantBmm::QuantBatchMatmulV3TensorAPIWithoutBatchTilingData) :
        sizeof(DequantBmm::QuantBatchMatmulV3BasicAPITilingData);
    QuantBatchMatMulV3TilingUtil::SetCommonTilingData(inputParams_, tilingData_);
    if (inputParams_.isDoubleScale) {
        tilingData_.params.x1QuantMode = static_cast<uint32_t>(optiling::BasicQuantMode::PERTENSOR_MODE);
    } else if (inputParams_.isPertoken) {
        tilingData_.params.x1QuantMode = static_cast<uint32_t>(optiling::BasicQuantMode::PERTOKEN_MODE);
    }
    if (inputParams_.isPerTensor) {
        tilingData_.params.x2QuantMode = static_cast<uint32_t>(optiling::BasicQuantMode::PERTENSOR_MODE);
    } else if (inputParams_.isPerChannel) {
        tilingData_.params.x2QuantMode = static_cast<uint32_t>(optiling::BasicQuantMode::PERCHANNEL_MODE);
    }
    tilingData_.adaptiveSlidingWin.mTailTile = adaptiveWin_.mTailTile;
    tilingData_.adaptiveSlidingWin.nTailTile = adaptiveWin_.nTailTile;
    tilingData_.adaptiveSlidingWin.mBaseTailSplitCnt = static_cast<uint32_t>(adaptiveWin_.mBaseTailSplitCnt);
    tilingData_.adaptiveSlidingWin.nBaseTailSplitCnt = static_cast<uint32_t>(adaptiveWin_.nBaseTailSplitCnt);
    tilingData_.adaptiveSlidingWin.mTailMain = static_cast<uint32_t>(adaptiveWin_.mTailMain);
    tilingData_.adaptiveSlidingWin.nTailMain = static_cast<uint32_t>(adaptiveWin_.nTailMain);

    if (useWithoutBatchTilingData_) {
        SetWithoutBatchTilingData();
    }
}

void AdaptiveSlidingWindowMixBasicAPITiling::SetWithoutBatchTilingData()
{
    withoutBatchTilingData_.m = static_cast<uint32_t>(inputParams_.mSize);
    withoutBatchTilingData_.n = static_cast<uint32_t>(inputParams_.nSize);
    withoutBatchTilingData_.k = static_cast<uint32_t>(inputParams_.kSize);
    withoutBatchTilingData_.groupSizeM = static_cast<uint32_t>(inputParams_.groupSizeM);
    withoutBatchTilingData_.groupSizeN = static_cast<uint32_t>(inputParams_.groupSizeN);
    withoutBatchTilingData_.groupSizeK = static_cast<uint32_t>(inputParams_.groupSizeK);
    withoutBatchTilingData_.mBaseTailSplitCnt = static_cast<uint32_t>(adaptiveWin_.mBaseTailSplitCnt);
    withoutBatchTilingData_.nBaseTailSplitCnt = static_cast<uint32_t>(adaptiveWin_.nBaseTailSplitCnt);
    withoutBatchTilingData_.mTailTile = static_cast<uint16_t>(adaptiveWin_.mTailTile);
    withoutBatchTilingData_.nTailTile = static_cast<uint16_t>(adaptiveWin_.nTailTile);
    withoutBatchTilingData_.mTailMain = static_cast<uint16_t>(adaptiveWin_.mTailMain);
    withoutBatchTilingData_.nTailMain = static_cast<uint16_t>(adaptiveWin_.nTailMain);
    withoutBatchTilingData_.baseM = static_cast<uint16_t>(basicTiling_.baseM);
    withoutBatchTilingData_.baseN = static_cast<uint16_t>(basicTiling_.baseN);
    withoutBatchTilingData_.baseK = static_cast<uint16_t>(basicTiling_.baseK);
    withoutBatchTilingData_.stepKa = tilingData_.matmulTiling.stepKa;
    withoutBatchTilingData_.stepKb = tilingData_.matmulTiling.stepKb;
    if (inputParams_.isDoubleScale) {
        withoutBatchTilingData_.x1QuantMode = static_cast<uint32_t>(optiling::BasicQuantMode::PERTENSOR_MODE);
    } else if (inputParams_.isPertoken) {
        withoutBatchTilingData_.x1QuantMode = static_cast<uint32_t>(optiling::BasicQuantMode::PERTOKEN_MODE);
    }
    if (inputParams_.isPerTensor) {
        withoutBatchTilingData_.x2QuantMode = static_cast<uint32_t>(optiling::BasicQuantMode::PERTENSOR_MODE);
    } else if (inputParams_.isPerChannel) {
        withoutBatchTilingData_.x2QuantMode = static_cast<uint32_t>(optiling::BasicQuantMode::PERCHANNEL_MODE);
    }
    withoutBatchTilingData_.nBufferNum = tilingData_.matmulTiling.nBufferNum;
    withoutBatchTilingData_.dbL0C = tilingData_.matmulTiling.dbL0C;
    withoutBatchTilingData_.isBias = tilingData_.matmulTiling.isBias;
    withoutBatchTilingData_.biasDtype = static_cast<uint8_t>(inputParams_.biasDtype);
}
REGISTER_TILING_TEMPLATE_WITH_ARCH(
    QuantBatchMatmulV3, AdaptiveSlidingWindowMixBasicAPITiling, supportedNpuArch, TILING_PRIORITY);
} // namespace optiling
