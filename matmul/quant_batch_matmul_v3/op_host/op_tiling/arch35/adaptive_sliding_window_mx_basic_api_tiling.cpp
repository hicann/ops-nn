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
 * \file adaptive_sliding_window_mx_basic_api_tiling.cpp
 * \brief
 */

#include "common/op_host/op_tiling/tiling_type.h"
#include "log/log.h"
#include "error_util.h"
#include "op_host/tiling_templates_registry.h"
#include "adaptive_sliding_window_mx_basic_api_tiling.h"
#include "base_block_calculator.h"
#include "l1_tiling_data_calculator.h"
#include "quant_batch_matmul_v3_tiling_strategy.h"

namespace {
constexpr uint32_t STEP_K_TWO = 2U;
// Keep A full-load if repeated A reads exceed this share of non-full-load GM traffic.
constexpr double REPEAT_A_LOAD_RATIO_THRESHOLD = 0.20;

const std::vector<int32_t> supportedNpuArch = {static_cast<int32_t>(NpuArch::DAV_3510)};
constexpr int32_t TILING_PRIORITY = optiling::strategy::MX_BASIC_API_ASW;

uint64_t GetSingleRoundTailSplitBase(uint64_t axisSize, uint64_t baseSize, uint64_t tailTile, uint64_t alignSize)
{
    if (tailTile <= 1UL) {
        return baseSize;
    }
    // A single-window tail split can be represented as a smaller aligned base block.
    uint64_t splitBase = ops::CeilDiv(std::min(axisSize, baseSize), tailTile);
    return ops::CeilAlign(splitBase, alignSize);
}

} // namespace

namespace optiling {

AdaptiveSlidingWindowMXBasicAPITiling::AdaptiveSlidingWindowMXBasicAPITiling(gert::TilingContext* context)
    : AdaptiveSlidingWindowTiling(context), tilingData_(tilingDataSelf_)
{
    Reset();
    tilingDataSize_ = sizeof(DequantBmm::QuantBatchMatmulV3BasicAPITilingData);
}

AdaptiveSlidingWindowMXBasicAPITiling::AdaptiveSlidingWindowMXBasicAPITiling(
    gert::TilingContext* context, DequantBmm::QuantBatchMatmulV3BasicAPITilingData* out)
    : AdaptiveSlidingWindowTiling(context, nullptr), tilingData_(*out)
{
    Reset();
    InitCompileInfo();
    inputParams_.Reset();
    tilingDataSize_ = sizeof(DequantBmm::QuantBatchMatmulV3BasicAPITilingData);
}

void AdaptiveSlidingWindowMXBasicAPITiling::Reset()
{
    if (!isTilingOut_) {
        tilingData_ = DequantBmm::QuantBatchMatmulV3BasicAPITilingData();
    }
    withoutBatchTilingData_ = DequantBmm::QuantBatchMatmulV3TensorAPIWithoutBatchTilingData();
    useWithoutBatchTilingData_ = false;
    tilingDataSize_ = sizeof(DequantBmm::QuantBatchMatmulV3BasicAPITilingData);
}

bool AdaptiveSlidingWindowMXBasicAPITiling::IsWithoutBatchTilingData() const
{
    return IsTensorapiCapable() && inputParams_.batchC == 1UL;
}

bool AdaptiveSlidingWindowMXBasicAPITiling::IsCapable()
{
    return IsMxBasicApiCapable(inputParams_);
}

uint64_t AdaptiveSlidingWindowMXBasicAPITiling::GetBatchCoreCnt() const
{
    return inputParams_.batchC;
}

const void* AdaptiveSlidingWindowMXBasicAPITiling::GetTilingData() const
{
    return useWithoutBatchTilingData_ ? static_cast<const void*>(&withoutBatchTilingData_) :
                                        static_cast<const void*>(&tilingData_);
}

uint64_t AdaptiveSlidingWindowMXBasicAPITiling::GetApiLevel(NpuArch npuArch) const
{
    return IsTensorapiCapable() ? static_cast<uint64_t>(QMMApiLevel::BLAZE_LEVEL) :
                                  static_cast<uint64_t>(QMMApiLevel::BASIC_LEVEL);
}

bool AdaptiveSlidingWindowMXBasicAPITiling::CalcBasicBlock()
{
    BaseBlockCalculator calculator(inputParams_, compileInfo_, GetBatchCoreCnt());
    if (!calculator.Compute(BaseBlockMode::DEFAULT)) {
        return false;
    }
    const BaseBlockRes& baseBlockRes = calculator.GetOutput();
    adaptiveWin_.baseM = baseBlockRes.baseM;
    adaptiveWin_.baseN = baseBlockRes.baseN;
    adaptiveWin_.baseK = baseBlockRes.baseK;
    adaptiveWin_.useTailWinLogic = baseBlockRes.useTailWinLogic;
    return true;
}

bool AdaptiveSlidingWindowMXBasicAPITiling::CalL1Tiling()
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
        ((basicTiling_.baseM * basicTiling_.baseN * qmmv3_tiling_const::DATA_SIZE_L0C * qmmv3_tiling_const::DOUBLE_BUFFER_NUM <= aicoreParams_.l0cSize) &&
         CheckBiasAndScale(basicTiling_.baseN, qmmv3_tiling_const::DOUBLE_BUFFER_NUM)) ?
            qmmv3_tiling_const::DOUBLE_BUFFER_NUM :
            1U;

    L1TilingMode mode = isAFullLoad_ ? L1TilingMode::A_L1_FULL_LOAD : L1TilingMode::DEFAULT;
    L1TilingDataCalculator l1Calculator(inputParams_, compileInfo_, basicTiling_.baseM, basicTiling_.baseN,
        basicTiling_.baseK);
    if (!l1Calculator.Compute(mode)) {
        return false;
    }
    const L1TilingData& l1TilingData = l1Calculator.GetOutput();
    basicTiling_.depthA1 = static_cast<uint32_t>(l1TilingData.depthKa_);
    basicTiling_.depthB1 = static_cast<uint32_t>(l1TilingData.depthKb_);
    basicTiling_.stepKa = static_cast<uint32_t>(l1TilingData.stepKa_);
    basicTiling_.stepKb = static_cast<uint32_t>(l1TilingData.stepKb_);
    basicTiling_.scaleFactorA = static_cast<uint32_t>(l1TilingData.scaleFactorA_);
    basicTiling_.scaleFactorB = static_cast<uint32_t>(l1TilingData.scaleFactorB_);
    return true;
}

ge::graphStatus AdaptiveSlidingWindowMXBasicAPITiling::DoLibApiTiling()
{
    tilingData_.matmulTiling.m = inputParams_.mSize;
    tilingData_.matmulTiling.n = inputParams_.nSize;
    tilingData_.matmulTiling.k = inputParams_.kSize;

    tilingData_.matmulTiling.baseM = basicTiling_.baseM;
    tilingData_.matmulTiling.baseN = basicTiling_.baseN;
    tilingData_.matmulTiling.baseK = basicTiling_.baseK;
    tilingData_.matmulTiling.isBias = inputParams_.hasBias ? 1UL : 0UL;
    tilingData_.matmulTiling.dbL0C = static_cast<uint8_t>(basicTiling_.dbL0c);

    tilingData_.matmulTiling.scaleKL1 = std::min(basicTiling_.scaleFactorA * basicTiling_.stepKa * basicTiling_.baseK,
                                                 basicTiling_.scaleFactorB * basicTiling_.stepKb * basicTiling_.baseK);
    CalculateNBufferNum4MX();
    tilingData_.matmulTiling.scaleFactorA = basicTiling_.scaleFactorA;
    tilingData_.matmulTiling.scaleFactorB = basicTiling_.scaleFactorB;
    if (useWithoutBatchTilingData_) {
        SetWithoutBatchTilingData();
    }
    return ge::GRAPH_SUCCESS;
}

void AdaptiveSlidingWindowMXBasicAPITiling::CalculateNBufferNum4MX()
{
    uint32_t stepK = std::min(basicTiling_.stepKa, basicTiling_.stepKb);
    uint64_t kL1 = static_cast<uint64_t>(stepK) * tilingData_.matmulTiling.baseK;
    tilingData_.matmulTiling.stepKa = stepK;
    tilingData_.matmulTiling.stepKb = stepK;
    uint64_t scaleKL1 = GetHalfKFallbackScaleKL1(kL1);
    uint64_t usedL1Size = CalcFourBufferUsedL1Size4MX(
        kL1, scaleKL1, tilingData_.matmulTiling.baseM, tilingData_.matmulTiling.baseN, isAFullLoad_);
    if (usedL1Size <= aicoreParams_.l1Size) {
        // Once four-buffer fits, expand scaleKL1 to cover full K if the extra scale buffer still fits L1.
        scaleKL1 = GetFullCoverScaleKL1IfPossible(kL1, scaleKL1);
        tilingData_.matmulTiling.scaleKL1 = static_cast<uint32_t>(scaleKL1);
        tilingData_.matmulTiling.nBufferNum = qmmv3_tiling_const::L1_FOUR_BUFFER;
        return;
    }
    const uint64_t stepKTwoKL1 = static_cast<uint64_t>(STEP_K_TWO) * tilingData_.matmulTiling.baseK;
    // If stepK 3/4 blocks four-buffer from fitting L1 while two-buffer still cannot cover K,
    // try stepK 2 to reduce per-round L1 usage and leave room for four-buffer.
    if ((stepK == 3U || stepK == 4U) && CanReduceStepKToTwo(stepK, stepKTwoKL1)) {
        kL1 = stepKTwoKL1;
        tilingData_.matmulTiling.stepKa = STEP_K_TWO;
        tilingData_.matmulTiling.stepKb = STEP_K_TWO;
        uint64_t candidateScaleKL1 = GetHalfKFallbackScaleKL1(kL1);
        usedL1Size = CalcFourBufferUsedL1Size4MX(
            kL1, candidateScaleKL1, tilingData_.matmulTiling.baseM, tilingData_.matmulTiling.baseN, isAFullLoad_);
        if (usedL1Size <= aicoreParams_.l1Size) {
            candidateScaleKL1 = GetFullCoverScaleKL1IfPossible(kL1, candidateScaleKL1);
            tilingData_.matmulTiling.scaleKL1 = static_cast<uint32_t>(candidateScaleKL1);
            tilingData_.matmulTiling.nBufferNum = qmmv3_tiling_const::L1_FOUR_BUFFER;
            return;
        }
        kL1 = static_cast<uint64_t>(stepK) * tilingData_.matmulTiling.baseK;
        tilingData_.matmulTiling.stepKa = stepK;
        tilingData_.matmulTiling.stepKb = stepK;
    }
    tilingData_.matmulTiling.scaleKL1 = static_cast<uint32_t>(scaleKL1);
    tilingData_.matmulTiling.nBufferNum = qmmv3_tiling_const::L1_TWO_BUFFER;
}

bool AdaptiveSlidingWindowMXBasicAPITiling::CanReduceStepKToTwo(uint32_t stepK, uint64_t stepKTwoKL1) const
{
    const uint64_t currentTwoBufferKL1 =
        static_cast<uint64_t>(stepK) * tilingData_.matmulTiling.baseK * qmmv3_tiling_const::L1_TWO_BUFFER;
    const bool isCurrentTwoBufferNotOverK = currentTwoBufferKL1 < inputParams_.kSize;

    const bool isAInnerKAligned = inputParams_.transA ||
                                  GetSizeWithDataType(inputParams_.kSize, inputParams_.aDtype) %
                                  qmmv3_tiling_const::MTE2_ADDRESS_ALIGN_SIZE == 0UL;
    const bool isBInnerKAligned = !inputParams_.transB ||
                                  GetSizeWithDataType(inputParams_.kSize, inputParams_.bDtype) %
                                  qmmv3_tiling_const::MTE2_ADDRESS_ALIGN_SIZE == 0UL;

    const bool isStepKTwoAInnerKAligned = inputParams_.transA ||
                                          GetSizeWithDataType(stepKTwoKL1, inputParams_.aDtype) %
                                          qmmv3_tiling_const::BASIC_BLOCK_SIZE_256 == 0UL;
    const bool isStepKTwoBInnerKAligned = !inputParams_.transB ||
                                          GetSizeWithDataType(stepKTwoKL1, inputParams_.bDtype) %
                                          qmmv3_tiling_const::BASIC_BLOCK_SIZE_256 == 0UL;

    return isCurrentTwoBufferNotOverK && isAInnerKAligned && isBInnerKAligned &&
           isStepKTwoAInnerKAligned && isStepKTwoBInnerKAligned;
}

uint64_t AdaptiveSlidingWindowMXBasicAPITiling::GetHalfKFallbackScaleKL1(uint64_t kL1) const
{
    uint64_t scaleKL1 = tilingData_.matmulTiling.scaleKL1;
    if (scaleKL1 % qmmv3_tiling_const::ESTIMATED_SCALE_K == 0UL) {
        return scaleKL1;
    }
    // If scaleKL1 is between half-K and full-K, shrink it toward half-K to free L1 for four-buffer,
    // while keeping scaleFactor an integer multiple of kL1.
    uint64_t halfK = ops::CeilDiv(inputParams_.kSize, 2UL);
    if (scaleKL1 > halfK && scaleKL1 < inputParams_.kSize) {
        uint64_t adjustedScaleKL1 = ops::CeilAlign(halfK, kL1);
        scaleKL1 = adjustedScaleKL1 < scaleKL1 ? adjustedScaleKL1 : scaleKL1;
    }
    return scaleKL1;
}

uint64_t AdaptiveSlidingWindowMXBasicAPITiling::GetFullCoverScaleKL1IfPossible(
    uint64_t kL1, uint64_t scaleKL1) const
{
    uint64_t fullCoverScaleKL1 = ops::CeilAlign(inputParams_.kSize, kL1);
    if (fullCoverScaleKL1 <= scaleKL1) {
        return scaleKL1;
    }
    uint64_t usedL1Size = CalcFourBufferUsedL1Size4MX(
        kL1, fullCoverScaleKL1, tilingData_.matmulTiling.baseM, tilingData_.matmulTiling.baseN, isAFullLoad_);
    return usedL1Size <= aicoreParams_.l1Size ? fullCoverScaleKL1 : scaleKL1;
}

uint64_t AdaptiveSlidingWindowMXBasicAPITiling::CalcMxFullKLoadSize(
    uint64_t outerSize, ge::DataType dataDtype, ge::DataType scaleDtype) const
{
    uint64_t kAligned = ops::CeilAlign(inputParams_.kSize, qmmv3_tiling_const::MXFP_DIVISOR_SIZE);
    uint64_t scaleK = ops::CeilDiv(inputParams_.kSize, qmmv3_tiling_const::MXFP_DIVISOR_SIZE) *
                      qmmv3_tiling_const::MXFP_MULTI_BASE_SIZE;
    return GetSizeWithDataType(outerSize * kAligned, dataDtype) +
           GetSizeWithDataType(outerSize * scaleK, scaleDtype);
}

uint64_t AdaptiveSlidingWindowMXBasicAPITiling::CalcFourBufferUsedL1Size4MX(
    uint64_t kL1, uint64_t scaleKL1, uint64_t baseM, uint64_t baseN, bool isAFullLoad) const
{
    uint64_t usedL1Size = GetSizeWithDataType(baseN * kL1, inputParams_.bDtype) * qmmv3_tiling_const::L1_FOUR_BUFFER;
    // B-side MX scale follows the scaleKL1 window and is double-buffered separately from B data.
    usedL1Size +=
        GetSizeWithDataType(baseN * ops::CeilDiv(scaleKL1, qmmv3_tiling_const::MX_GROUP_SIZE), inputParams_.scaleDtype) *
        qmmv3_tiling_const::L1_TWO_BUFFER;
    if (inputParams_.hasBias) {
        usedL1Size += GetSizeWithDataType(baseN, inputParams_.biasDtype) * qmmv3_tiling_const::L1_TWO_BUFFER;
    }
    if (isAFullLoad) {
        usedL1Size += CalcMxFullKLoadSize(baseM, inputParams_.aDtype, inputParams_.perTokenScaleDtype);
    } else {
        usedL1Size += GetSizeWithDataType(baseM * kL1, inputParams_.aDtype) * qmmv3_tiling_const::L1_FOUR_BUFFER;
        usedL1Size += GetSizeWithDataType(
                          baseM * ops::CeilDiv(scaleKL1, qmmv3_tiling_const::MX_GROUP_SIZE), inputParams_.perTokenScaleDtype) *
                      qmmv3_tiling_const::L1_TWO_BUFFER;
    }
    return usedL1Size;
}

bool AdaptiveSlidingWindowMXBasicAPITiling::CanOpenFourBufferByL1Estimate(
    bool isAFullLoad, uint64_t baseM, uint64_t baseN) const
{
    if (!isAFullLoad) {
        const bool isAInnerKAligned = inputParams_.transA ||
            GetSizeWithDataType(inputParams_.kSize, inputParams_.aDtype) %
                qmmv3_tiling_const::MTE2_ADDRESS_ALIGN_SIZE == 0UL;
        const bool isBInnerKAligned = !inputParams_.transB ||
            GetSizeWithDataType(inputParams_.kSize, inputParams_.bDtype) %
                qmmv3_tiling_const::MTE2_ADDRESS_ALIGN_SIZE == 0UL;
        // Use max stepK=4 for a conservative two-buffer K-coverage estimate.
        const bool isEstimatedTwoBufferNotOverK =
            4UL * adaptiveWin_.baseK * qmmv3_tiling_const::L1_TWO_BUFFER <
            inputParams_.kSize;
        if (!isEstimatedTwoBufferNotOverK || !isAInnerKAligned || !isBInnerKAligned) {
            return false;
        }
    }
    // This runs before final L1 tiling. Estimate with stepK=2 and capped scaleKL1 to decide whether
    // isAFullLoad_ still leaves room for four-buffer.
    uint64_t estimatedKL1 = static_cast<uint64_t>(STEP_K_TWO) * adaptiveWin_.baseK;
    uint64_t estimatedScaleKL1 =
        ops::CeilAlign(std::min(inputParams_.kSize, qmmv3_tiling_const::ESTIMATED_SCALE_K), estimatedKL1);
    uint64_t usedL1Size =
        CalcFourBufferUsedL1Size4MX(estimatedKL1, estimatedScaleKL1, baseM, baseN, isAFullLoad);
    return usedL1Size <= aicoreParams_.l1Size;
}

bool AdaptiveSlidingWindowMXBasicAPITiling::ShouldKeepAFullLoadByRepeatLoadRatio() const
{
    if (adaptiveWin_.nBlockCnt <= 1UL || adaptiveWin_.mBlockCnt == 0UL) {
        return false;
    }
    double singleRoundABytes = static_cast<double>(
        CalcMxFullKLoadSize(inputParams_.mSize, inputParams_.aDtype, inputParams_.perTokenScaleDtype));
    double repeatABytes = singleRoundABytes * static_cast<double>(adaptiveWin_.nBlockCnt - 1UL);
    double nonFullLoadABytes = singleRoundABytes * static_cast<double>(adaptiveWin_.nBlockCnt);
    double nonFullLoadBBytes =
        static_cast<double>(CalcMxFullKLoadSize(
            inputParams_.nSize, inputParams_.bDtype, inputParams_.scaleDtype)) *
        static_cast<double>(adaptiveWin_.mBlockCnt);
    double totalLoadBytes = nonFullLoadABytes + nonFullLoadBBytes;
    if (totalLoadBytes <= 0.0) {
        return false;
    }
    return repeatABytes / totalLoadBytes > REPEAT_A_LOAD_RATIO_THRESHOLD;
}

void AdaptiveSlidingWindowMXBasicAPITiling::UpdateAFullLoadStatus()
{
    uint64_t realBaseMSize = adaptiveWin_.mBaseTailSplitCnt == 1UL ? adaptiveWin_.baseM : adaptiveWin_.mTailMain;
    uint64_t kAligned = ops::CeilAlign(inputParams_.kSize, qmmv3_tiling_const::MXFP_DIVISOR_SIZE);
    uint64_t singleCoreASize = GetSizeWithDataType(realBaseMSize * kAligned, inputParams_.aDtype);
    bool isAFullLoadCandidate = singleCoreASize <= aicoreParams_.l1Size / qmmv3_tiling_const::AFULLLOAD_SINGLE_CORE_A_SCALER &&
                                adaptiveWin_.mBlockCnt < qmmv3_tiling_const::WINDOW_LEN &&
                                aicoreParams_.aicNum % adaptiveWin_.mBlockCnt == 0 &&
                                adaptiveWin_.totalBlockCnt > aicoreParams_.aicNum && inputParams_.batchC == 1;
    isAFullLoad_ = false;
    if (!isAFullLoadCandidate) {
        return;
    }
    // Prefer A full-load when it still leaves enough L1 to enable four-buffer.
    if (CanOpenFourBufferByL1Estimate(true, realBaseMSize, adaptiveWin_.baseN)) {
        isAFullLoad_ = true;
    } else if (CanOpenFourBufferByL1Estimate(false, adaptiveWin_.baseM, adaptiveWin_.baseN) &&
               !ShouldKeepAFullLoadByRepeatLoadRatio()) {
        // Keep A full-load disabled only when repeated A load is a small part of non-full-load GM traffic.
        return;
    } else {
        isAFullLoad_ = true;
    }
    if (isAFullLoad_ && adaptiveWin_.baseM != realBaseMSize) {
        adaptiveWin_.baseM = realBaseMSize;
        adaptiveWin_.mBaseTailSplitCnt = 1UL;
        adaptiveWin_.mTailMain = 0UL;
    }
}

void AdaptiveSlidingWindowMXBasicAPITiling::AnalyseFullLoadInfo()
{
    isABFullLoad_ = false;
    isBFullLoad_ = false;
    UpdateAFullLoadStatus();
}

void AdaptiveSlidingWindowMXBasicAPITiling::CalcTailRoundBasicBlockSplit()
{
    if (!adaptiveWin_.useTailWinLogic) {
        return;
    }
    if (isAFullLoad_) {
        CalcTailBasicBlockAfullLoad();
    } else {
        CalcTailBasicBlock();
    }
    NormalizeSingleRoundTailSplitBasicBlock();
}

void AdaptiveSlidingWindowMXBasicAPITiling::NormalizeSingleRoundTailSplitBasicBlock()
{
    // Only single-window tail-split cases are normalized here; MX A-L1 full-load has totalWinCnt > 1.
    if (adaptiveWin_.totalWinCnt != 1UL || adaptiveWin_.tailWinBlockCnt == 0UL ||
        (adaptiveWin_.mTailTile == 1UL && adaptiveWin_.nTailTile == 1UL)) {
        return;
    }

    const uint64_t baseMAlignSize =
        inputParams_.transA ? GetShapeWithDataType(qmmv3_tiling_const::L1_ALIGN_SIZE, inputParams_.aDtype) : qmmv3_tiling_const::CUBE_BLOCK;
    const uint64_t baseNAlignSize =
        inputParams_.transB ? qmmv3_tiling_const::CUBE_BLOCK : GetShapeWithDataType(qmmv3_tiling_const::L1_ALIGN_SIZE, inputParams_.bDtype);
    const uint64_t newBaseM =
        GetSingleRoundTailSplitBase(inputParams_.mSize, adaptiveWin_.baseM, adaptiveWin_.mTailTile, baseMAlignSize);
    const uint64_t newBaseN =
        GetSingleRoundTailSplitBase(inputParams_.nSize, adaptiveWin_.baseN, adaptiveWin_.nTailTile, baseNAlignSize);
    if (newBaseM == 0UL || newBaseN == 0UL || newBaseM > adaptiveWin_.baseM || newBaseN > adaptiveWin_.baseN) {
        return;
    }

    const bool isBaseUpdated = newBaseM != adaptiveWin_.baseM || newBaseN != adaptiveWin_.baseN;
    if (!isBaseUpdated) {
        return;
    }
    adaptiveWin_.baseM = newBaseM;
    adaptiveWin_.baseN = newBaseN;
    adaptiveWin_.mTailTile = 1UL;
    adaptiveWin_.nTailTile = 1UL;
    LoadBalanceDataReset();
    // baseM/baseN changed from a split tile into a real base block; recompute window counters before edge tuning.
    CalcBlockWindowInfo();
    OptimizeEdgeBasicBlock();
}

void AdaptiveSlidingWindowMXBasicAPITiling::SetTilingData()
{
    useWithoutBatchTilingData_ = IsWithoutBatchTilingData();
    tilingDataSize_ = useWithoutBatchTilingData_ ?
        sizeof(DequantBmm::QuantBatchMatmulV3TensorAPIWithoutBatchTilingData) :
        sizeof(DequantBmm::QuantBatchMatmulV3BasicAPITilingData);

    QuantBatchMatMulV3TilingUtil::SetCommonTilingData(inputParams_, tilingData_);
    tilingData_.params.x1QuantMode = static_cast<uint32_t>(optiling::BasicQuantMode::MX_PERGROUP_MODE);
    tilingData_.params.x2QuantMode = static_cast<uint32_t>(optiling::BasicQuantMode::MX_PERGROUP_MODE);
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

void AdaptiveSlidingWindowMXBasicAPITiling::SetWithoutBatchTilingData()
{
    withoutBatchTilingData_.m = static_cast<uint32_t>(inputParams_.mSize);
    withoutBatchTilingData_.n = static_cast<uint32_t>(inputParams_.nSize);
    withoutBatchTilingData_.k = static_cast<uint32_t>(inputParams_.kSize);
    withoutBatchTilingData_.scaleKL1 = tilingData_.matmulTiling.scaleKL1;
    withoutBatchTilingData_.baseM = static_cast<uint16_t>(basicTiling_.baseM);
    withoutBatchTilingData_.baseN = static_cast<uint16_t>(basicTiling_.baseN);
    withoutBatchTilingData_.baseK = static_cast<uint16_t>(basicTiling_.baseK);
    withoutBatchTilingData_.stepKa = tilingData_.matmulTiling.stepKa;
    withoutBatchTilingData_.stepKb = tilingData_.matmulTiling.stepKb;
    withoutBatchTilingData_.groupSizeM = static_cast<uint16_t>(inputParams_.groupSizeM);
    withoutBatchTilingData_.groupSizeN = static_cast<uint16_t>(inputParams_.groupSizeN);
    withoutBatchTilingData_.groupSizeK = static_cast<uint16_t>(inputParams_.groupSizeK);
    withoutBatchTilingData_.mTailTile = static_cast<uint16_t>(adaptiveWin_.mTailTile);
    withoutBatchTilingData_.nTailTile = static_cast<uint16_t>(adaptiveWin_.nTailTile);
    withoutBatchTilingData_.mBaseTailSplitCnt = static_cast<uint16_t>(adaptiveWin_.mBaseTailSplitCnt);
    withoutBatchTilingData_.nBaseTailSplitCnt = static_cast<uint16_t>(adaptiveWin_.nBaseTailSplitCnt);
    withoutBatchTilingData_.mTailMain = static_cast<uint16_t>(adaptiveWin_.mTailMain);
    withoutBatchTilingData_.nTailMain = static_cast<uint16_t>(adaptiveWin_.nTailMain);
    withoutBatchTilingData_.x1QuantMode = static_cast<uint8_t>(optiling::BasicQuantMode::MX_PERGROUP_MODE);
    withoutBatchTilingData_.x2QuantMode = static_cast<uint8_t>(optiling::BasicQuantMode::MX_PERGROUP_MODE);
    withoutBatchTilingData_.isBias = tilingData_.matmulTiling.isBias;
    withoutBatchTilingData_.biasDtype = static_cast<uint8_t>(inputParams_.biasDtype);
    withoutBatchTilingData_.nBufferNum = tilingData_.matmulTiling.nBufferNum;
    withoutBatchTilingData_.dbL0C = tilingData_.matmulTiling.dbL0C;
}

REGISTER_TILING_TEMPLATE_WITH_ARCH(
    QuantBatchMatmulV3, AdaptiveSlidingWindowMXBasicAPITiling, supportedNpuArch, TILING_PRIORITY);
} // namespace optiling
