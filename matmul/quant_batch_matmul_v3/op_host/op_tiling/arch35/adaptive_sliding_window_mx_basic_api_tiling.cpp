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

#include "adaptive_sliding_window_mx_basic_api_tiling.h"

#include <algorithm>

#include "common/op_host/op_tiling/tiling_type_mm.h"
#include "log/log.h"
#include "error_util.h"
#include "op_host/tiling_templates_registry.h"
#include "base_block_calculator.h"
#include "l1_tiling_data_calculator.h"
#include "quant_batch_matmul_v3_tiling_strategy.h"
#include "quant_batch_matmul_v3_tiling_util.h"

namespace {
constexpr uint32_t STEP_K_TWO = 2U;
// Empirical value for the L0C ping-pong template.
constexpr uint64_t MX_L0C_PINGPONG_SCALE_KL1_TARGET = 2048UL;
constexpr uint64_t MX_L0C_PINGPONG_OUTPUT_SIZE_LIMIT = 128UL * 1024UL * 1024UL;
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
    AdaptiveSlidingWindowMXBasicAPITiling::ResetTilingData();
    tilingDataSize_ = sizeof(DequantBmm::QuantBatchMatmulV3BasicAPITilingData);
}

AdaptiveSlidingWindowMXBasicAPITiling::AdaptiveSlidingWindowMXBasicAPITiling(
    gert::TilingContext* context, DequantBmm::QuantBatchMatmulV3BasicAPITilingData* out)
    : AdaptiveSlidingWindowTiling(context, nullptr), tilingData_(*out)
{
    AdaptiveSlidingWindowMXBasicAPITiling::ResetTilingData();
    InitCompileInfo();
    inputParams_.Reset();
    tilingDataSize_ = sizeof(DequantBmm::QuantBatchMatmulV3BasicAPITilingData);
}

void AdaptiveSlidingWindowMXBasicAPITiling::ResetTilingData()
{
    AdaptiveSlidingWindowTiling::ResetTilingData();
    if (!isTilingOut_) {
        tilingData_ = DequantBmm::QuantBatchMatmulV3BasicAPITilingData();
    }
    withoutBatchTilingData_ = DequantBmm::QuantBatchMatmulV3TensorAPIWithoutBatchTilingData();
    useWithoutBatchTilingData_ = false;
    tilingDataSize_ = sizeof(DequantBmm::QuantBatchMatmulV3BasicAPITilingData);
}

bool AdaptiveSlidingWindowMXBasicAPITiling::IsWithoutBatchTilingData() const
{
    return IsTensorApiEnabled() && inputParams_.batchC == 1UL;
}

void AdaptiveSlidingWindowMXBasicAPITiling::AdjustScaleFactorForL0CPingpong(uint32_t& scaleFactor, uint32_t step,
                                                                            uint32_t baseK) const
{
    uint64_t scaleKUnit = static_cast<uint64_t>(step) * baseK;
    if (scaleKUnit == 0UL) {
        return;
    }
    uint64_t adjustedScaleFactor = std::max<uint64_t>(qmmv3_tiling_const::SCALER_FACTOR_MIN,
                                                      MX_L0C_PINGPONG_SCALE_KL1_TARGET / scaleKUnit);
    scaleFactor = static_cast<uint32_t>(std::min<uint64_t>(scaleFactor, adjustedScaleFactor));
}

bool AdaptiveSlidingWindowMXBasicAPITiling::IsCapable() { return IsMxBasicApiCapable(inputParams_); }

uint64_t AdaptiveSlidingWindowMXBasicAPITiling::GetBatchCoreCnt() const { return inputParams_.batchC; }

const void* AdaptiveSlidingWindowMXBasicAPITiling::GetTilingData() const
{
    return useWithoutBatchTilingData_ ? static_cast<const void*>(&withoutBatchTilingData_) :
                                        static_cast<const void*>(&tilingData_);
}

uint64_t AdaptiveSlidingWindowMXBasicAPITiling::GetApiLevel(NpuArch npuArch) const
{
    return IsTensorApiEnabled() ? static_cast<uint64_t>(QMMApiLevel::BLAZE_LEVEL) :
                                  static_cast<uint64_t>(QMMApiLevel::BASIC_LEVEL);
}

uint64_t AdaptiveSlidingWindowMXBasicAPITiling::GetBatchMode() const
{
    return IsWithoutBatchTilingData() ? static_cast<uint64_t>(BatchMode::WITHOUT_BATCH) :
                                        static_cast<uint64_t>(BatchMode::WITH_BATCH);
}

uint64_t AdaptiveSlidingWindowMXBasicAPITiling::GetKernelType() const
{
    const bool useMxL0CPingpong = IsMxL0CPingpong(inputParams_);
    if (isAFullLoad_ && useMxL0CPingpong) {
        return static_cast<uint64_t>(QMMKernelType::NO_VEC_EPILOGUE_CUSTOM_GMTOAL1_WITH_MMAPI_MX_L0C_PINGPONG);
    }
    if (useMxL0CPingpong) {
        return static_cast<uint64_t>(QMMKernelType::NO_VEC_EPILOGUE_WITH_MMAPI_MX_L0C_PINGPONG);
    }
    if (isAFullLoad_) {
        return static_cast<uint64_t>(QMMKernelType::NO_VEC_EPILOGUE_CUSTOM_GMTOAL1_WITH_MMAPI);
    }
    return static_cast<uint64_t>(QMMKernelType::NO_VEC_EPILOGUE_WITH_MMAPI);
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
    basicTiling_.dbL0c = ((basicTiling_.baseM * basicTiling_.baseN * qmmv3_tiling_const::DATA_SIZE_L0C *
                               qmmv3_tiling_const::DOUBLE_BUFFER_NUM <=
                           aicoreParams_.l0cSize) &&
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

    // DeriveScaleKL1 shrinks basicTiling_.scaleFactorA/B in place when the L0C ping-pong template fires; those
    // adjusted factors flow into mxTypePara later, so the in-place write-back must be preserved.
    tilingData_.matmulTiling.scaleKL1 = static_cast<uint32_t>(
        DeriveScaleKL1(basicTiling_.scaleFactorA, basicTiling_.stepKa, basicTiling_.scaleFactorB, basicTiling_.stepKb,
                       basicTiling_.baseK));
    CalculateNBufferNum();
    if (useWithoutBatchTilingData_) {
        SetWithoutBatchTilingData();
    }
    return ge::GRAPH_SUCCESS;
}

void AdaptiveSlidingWindowMXBasicAPITiling::CalculateNBufferNum()
{
    const uint32_t stepK = std::min(basicTiling_.stepKa, basicTiling_.stepKb);
    const uint64_t currentKL1 = static_cast<uint64_t>(stepK) * tilingData_.matmulTiling.baseK;
    const MxL1EstimateParams currentParams = BuildL1EstimateParams(currentKL1, tilingData_.matmulTiling.scaleKL1,
                                                                   tilingData_.matmulTiling.baseM,
                                                                   tilingData_.matmulTiling.baseN, isAFullLoad_);
    const MxL1Plan plan = SelectMxL1Plan(currentParams, tilingData_.matmulTiling.baseK,
                                         tilingData_.matmulTiling.scaleKL1);
    tilingData_.matmulTiling.kAL1 = static_cast<uint32_t>(plan.kL1);
    tilingData_.matmulTiling.kBL1 = tilingData_.matmulTiling.kAL1;
    tilingData_.matmulTiling.scaleKL1 = static_cast<uint32_t>(plan.scaleKL1);
    tilingData_.matmulTiling.nBufferNum = static_cast<uint8_t>(plan.l1BufferNum);
}

AdaptiveSlidingWindowMXBasicAPITiling::MxL1EstimateParams AdaptiveSlidingWindowMXBasicAPITiling::BuildL1EstimateParams(
    uint64_t kL1, uint64_t baseScaleKL1, uint64_t baseM, uint64_t baseN, bool isAFullLoad) const
{
    return {kL1, GetHalfKFallbackScaleKL1(baseScaleKL1, kL1), baseM, baseN, isAFullLoad};
}

uint64_t AdaptiveSlidingWindowMXBasicAPITiling::DeriveScaleKL1(uint32_t& scaleFactorA, uint64_t stepKa,
                                                               uint32_t& scaleFactorB, uint64_t stepKb,
                                                               uint64_t baseK) const
{
    uint64_t scaleKL1 = std::min(static_cast<uint64_t>(scaleFactorA) * stepKa * baseK,
                                 static_cast<uint64_t>(scaleFactorB) * stepKb * baseK);
    const uint64_t outputSize = GetSizeWithDataType(inputParams_.mSize * inputParams_.nSize, inputParams_.cDtype);
    if (IsMxL0CPingpong(inputParams_) && outputSize <= MX_L0C_PINGPONG_OUTPUT_SIZE_LIMIT &&
        scaleKL1 > MX_L0C_PINGPONG_SCALE_KL1_TARGET) {
        AdjustScaleFactorForL0CPingpong(scaleFactorA, static_cast<uint32_t>(stepKa), static_cast<uint32_t>(baseK));
        AdjustScaleFactorForL0CPingpong(scaleFactorB, static_cast<uint32_t>(stepKb), static_cast<uint32_t>(baseK));
        scaleKL1 = std::min(static_cast<uint64_t>(scaleFactorA) * stepKa * baseK,
                            static_cast<uint64_t>(scaleFactorB) * stepKb * baseK);
    }
    return scaleKL1;
}

AdaptiveSlidingWindowMXBasicAPITiling::MxL1Plan AdaptiveSlidingWindowMXBasicAPITiling::MakeMxL1Plan(
    const MxL1EstimateParams& params, uint32_t l1BufferNum) const
{
    // Two-buffer keeps the (possibly half-K fallback) scaleKL1; multi-buffer tries to widen scale to full-K coverage.
    const uint64_t scaleKL1 = l1BufferNum == qmmv3_tiling_const::L1_TWO_BUFFER ?
                                  params.scaleKL1 :
                                  GetFullCoverScaleKL1IfPossible(params, l1BufferNum);
    return {params.kL1, scaleKL1, l1BufferNum};
}

AdaptiveSlidingWindowMXBasicAPITiling::MxL1Plan AdaptiveSlidingWindowMXBasicAPITiling::SelectMxL1Plan(
    const MxL1EstimateParams& currentParams, uint64_t baseK, uint64_t baseScaleKL1) const
{
    const uint64_t currentKL1 = currentParams.kL1;
    const uint64_t stepK = currentKL1 / baseK;
    // Restrict the smaller stepK=2 candidates to cases where the current double buffer cannot cover K.
    const bool isCurrentTwoBufferNotOverK = currentKL1 * qmmv3_tiling_const::L1_TWO_BUFFER < inputParams_.kSize;
    if (CanFitL1BufferNum(currentParams, qmmv3_tiling_const::L1_FOUR_BUFFER)) {
        return MakeMxL1Plan(currentParams, qmmv3_tiling_const::L1_FOUR_BUFFER);
    }

    // If stepK 3/4 blocks four-buffer from fitting L1 while two-buffer still cannot cover K,
    // try stepK 2 to reduce per-round L1 usage and leave room for four-buffer.
    const uint64_t stepKTwoKL1 = static_cast<uint64_t>(STEP_K_TWO) * baseK;
    const bool canReduceStepK = isCurrentTwoBufferNotOverK && (stepK == 3UL || stepK == 4UL) &&
                                CanReduceStepKToTwo(stepKTwoKL1);
    MxL1EstimateParams stepKTwoParams = currentParams;
    if (canReduceStepK) {
        stepKTwoParams = BuildL1EstimateParams(stepKTwoKL1, baseScaleKL1, currentParams.baseM, currentParams.baseN,
                                               currentParams.isAFullLoad);
        // Prefer four buffers globally, even when it requires a smaller stepK.
        if (CanFitL1BufferNum(stepKTwoParams, qmmv3_tiling_const::L1_FOUR_BUFFER)) {
            return MakeMxL1Plan(stepKTwoParams, qmmv3_tiling_const::L1_FOUR_BUFFER);
        }
    }

    // A full-load uses the third buffer only for the B-side pipeline. Otherwise, keep the current stepK and enable
    // triple-buffer only when the current double buffer cannot cover K and MTE2 is expected to dominate.
    const bool canUseCurrentThreeBuffer = currentParams.isAFullLoad ||
                                          (isCurrentTwoBufferNotOverK &&
                                           IsMte2Bound(qmmv3_tiling_const::ASCEND_950_MAX_HBM_BW_TBPS *
                                                           qmmv3_tiling_const::MTE2_BW_UTILIZATION,
                                                       qmmv3_tiling_const::ASCEND_950_MAX_L2_BW_TBPS *
                                                           qmmv3_tiling_const::MTE2_BW_UTILIZATION));
    if (canUseCurrentThreeBuffer && CanFitL1BufferNum(currentParams, qmmv3_tiling_const::L1_THREE_BUFFER)) {
        return MakeMxL1Plan(currentParams, qmmv3_tiling_const::L1_THREE_BUFFER);
    }
    if (canReduceStepK && canUseCurrentThreeBuffer &&
        CanFitL1BufferNum(stepKTwoParams, qmmv3_tiling_const::L1_THREE_BUFFER)) {
        return MakeMxL1Plan(stepKTwoParams, qmmv3_tiling_const::L1_THREE_BUFFER);
    }
    return MakeMxL1Plan(currentParams, qmmv3_tiling_const::L1_TWO_BUFFER);
}

bool AdaptiveSlidingWindowMXBasicAPITiling::CanReduceStepKToTwo(uint64_t stepKTwoKL1) const
{
    return AreKInnerAxesAligned() && IsKInnerKL1AlignedTo256Bytes(stepKTwoKL1);
}

uint64_t AdaptiveSlidingWindowMXBasicAPITiling::GetHalfKFallbackScaleKL1(uint64_t scaleKL1, uint64_t kL1) const
{
    if (scaleKL1 % qmmv3_tiling_const::ESTIMATED_SCALE_K == 0UL) {
        return scaleKL1;
    }
    // If scaleKL1 is between half-K and full-K, shrink it toward half-K to free L1 for multi-buffer,
    // while keeping scaleFactor an integer multiple of kL1.
    uint64_t halfK = ops::CeilDiv(inputParams_.kSize, 2UL);
    if (scaleKL1 > halfK && scaleKL1 < inputParams_.kSize) {
        uint64_t adjustedScaleKL1 = ops::CeilAlign(halfK, kL1);
        scaleKL1 = adjustedScaleKL1 < scaleKL1 ? adjustedScaleKL1 : scaleKL1;
    }
    return scaleKL1;
}

uint64_t AdaptiveSlidingWindowMXBasicAPITiling::GetFullCoverScaleKL1IfPossible(const MxL1EstimateParams& params,
                                                                               uint32_t l1BufferNum) const
{
    uint64_t fullCoverScaleKL1 = ops::CeilAlign(inputParams_.kSize, params.kL1);
    if (fullCoverScaleKL1 <= params.scaleKL1) {
        return params.scaleKL1;
    }
    MxL1EstimateParams fullCoverParams = params;
    fullCoverParams.scaleKL1 = fullCoverScaleKL1;
    uint64_t usedL1Size = CalcUsedL1Size(fullCoverParams, l1BufferNum);
    return usedL1Size <= aicoreParams_.l1Size ? fullCoverScaleKL1 : params.scaleKL1;
}

uint64_t AdaptiveSlidingWindowMXBasicAPITiling::CalcMxFullKLoadSize(uint64_t outerSize, ge::DataType dataDtype,
                                                                    ge::DataType scaleDtype) const
{
    uint64_t kAligned = ops::CeilAlign(inputParams_.kSize, qmmv3_tiling_const::MXFP_DIVISOR_SIZE);
    uint64_t scaleK = ops::CeilDiv(inputParams_.kSize, qmmv3_tiling_const::MXFP_DIVISOR_SIZE) *
                      qmmv3_tiling_const::MXFP_MULTI_BASE_SIZE;
    return GetSizeWithDataType(outerSize * kAligned, dataDtype) + GetSizeWithDataType(outerSize * scaleK, scaleDtype);
}

bool AdaptiveSlidingWindowMXBasicAPITiling::CanFitL1BufferNum(const MxL1EstimateParams& params,
                                                              uint32_t l1BufferNum) const
{
    if (l1BufferNum == qmmv3_tiling_const::L1_THREE_BUFFER && !IsTensorApiEnabled()) {
        return false;
    }
    return CalcUsedL1Size(params, l1BufferNum) <= aicoreParams_.l1Size;
}

uint64_t AdaptiveSlidingWindowMXBasicAPITiling::CalcUsedL1Size(const MxL1EstimateParams& params,
                                                               uint32_t l1BufferNum) const
{
    uint64_t usedL1Size = GetSizeWithDataType(params.baseN * params.kL1, inputParams_.bDtype) * l1BufferNum;
    // B-side MX scale follows the scaleKL1 window and is double-buffered separately from B data.
    usedL1Size += GetSizeWithDataType(params.baseN * ops::CeilDiv(params.scaleKL1, qmmv3_tiling_const::MX_GROUP_SIZE),
                                      inputParams_.scaleDtype) *
                  qmmv3_tiling_const::L1_TWO_BUFFER;
    if (inputParams_.hasBias) {
        usedL1Size += GetSizeWithDataType(params.baseN, inputParams_.biasDtype) * qmmv3_tiling_const::L1_TWO_BUFFER;
    }
    if (params.isAFullLoad) {
        usedL1Size += CalcMxFullKLoadSize(params.baseM, inputParams_.aDtype, inputParams_.perTokenScaleDtype);
    } else {
        usedL1Size += GetSizeWithDataType(params.baseM * params.kL1, inputParams_.aDtype) * l1BufferNum;
        usedL1Size += GetSizeWithDataType(
                          params.baseM * ops::CeilDiv(params.scaleKL1, qmmv3_tiling_const::MX_GROUP_SIZE),
                          inputParams_.perTokenScaleDtype) *
                      qmmv3_tiling_const::L1_TWO_BUFFER;
    }
    return usedL1Size;
}

bool AdaptiveSlidingWindowMXBasicAPITiling::CanSelectMultiBufferByL1Plan(bool isAFullLoad, uint64_t baseM,
                                                                         uint64_t baseN) const
{
    // Faithful dry-run of the final L1 tiling + N-buffer selection for this candidate, mirroring the cube path.
    const L1TilingMode mode = isAFullLoad ? L1TilingMode::A_L1_FULL_LOAD : L1TilingMode::DEFAULT;
    L1TilingDataCalculator calculator(inputParams_, compileInfo_, baseM, baseN, adaptiveWin_.baseK);
    if (!calculator.Compute(mode)) {
        return false;
    }
    const L1TilingData& l1Tiling = calculator.GetOutput();
    const uint64_t baseK = adaptiveWin_.baseK;
    const uint64_t currentKL1 = std::min(l1Tiling.stepKa_, l1Tiling.stepKb_) * baseK;
    // Run on local copies of the scale factors: the dry-run must not mutate the real basicTiling_/tilingData_ state.
    uint32_t scaleFactorA = static_cast<uint32_t>(l1Tiling.scaleFactorA_);
    uint32_t scaleFactorB = static_cast<uint32_t>(l1Tiling.scaleFactorB_);
    const uint64_t baseScaleKL1 = DeriveScaleKL1(scaleFactorA, l1Tiling.stepKa_, scaleFactorB, l1Tiling.stepKb_, baseK);
    const MxL1EstimateParams currentParams = BuildL1EstimateParams(currentKL1, baseScaleKL1, baseM, baseN, isAFullLoad);
    const MxL1Plan plan = SelectMxL1Plan(currentParams, baseK, baseScaleKL1);
    return plan.l1BufferNum > qmmv3_tiling_const::L1_TWO_BUFFER;
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
    double nonFullLoadBBytes = static_cast<double>(CalcMxFullKLoadSize(inputParams_.nSize, inputParams_.bDtype,
                                                                       inputParams_.scaleDtype)) *
                               static_cast<double>(adaptiveWin_.mBlockCnt);
    double totalLoadBytes = nonFullLoadABytes + nonFullLoadBBytes;
    if (totalLoadBytes <= 0.0) {
        return false;
    }
    return repeatABytes / totalLoadBytes > REPEAT_A_LOAD_RATIO_THRESHOLD;
}

bool AdaptiveSlidingWindowMXBasicAPITiling::IsMte2Bound(double gmBandwidthTbps, double l2BandwidthTbps) const
{
    const uint64_t usedCoreNum = CalUsedCoreNum();
    if (gmBandwidthTbps <= 0.0 || l2BandwidthTbps <= 0.0 || adaptiveWin_.baseM == 0UL || adaptiveWin_.baseN == 0UL ||
        usedCoreNum == 0UL) {
        return false;
    }

    if (!AreOperandInnerAxesAligned()) {
        return true;
    }

    return EstimateMte2TimeUs(gmBandwidthTbps, l2BandwidthTbps) > EstimateMacTimeUs();
}

double AdaptiveSlidingWindowMXBasicAPITiling::EstimateMte2TimeUs(double gmBandwidthTbps, double l2BandwidthTbps) const
{
    const double singleRoundABytes = static_cast<double>(
        CalcMxFullKLoadSize(inputParams_.mSize, inputParams_.aDtype, inputParams_.perTokenScaleDtype));
    const double singleRoundBBytes = static_cast<double>(
        CalcMxFullKLoadSize(inputParams_.nSize, inputParams_.bDtype, inputParams_.scaleDtype));
    const double singleRoundBiasBytes = inputParams_.hasBias ? static_cast<double>(GetSizeWithDataType(
                                                                   inputParams_.nSize, inputParams_.biasDtype)) :
                                                               0.0;
    const uint64_t mBlockCnt = ops::CeilDiv(inputParams_.mSize, adaptiveWin_.baseM);
    const uint64_t nBlockCnt = ops::CeilDiv(inputParams_.nSize, adaptiveWin_.baseN);
    const uint64_t batchCount = std::max<uint64_t>(1UL, inputParams_.batchC);
    // IsMte2Bound is reached only by a non-full plan, so A is loaded once for every N block.
    const uint64_t aLoadCount = batchCount * nBlockCnt;
    const uint64_t bLoadCount = batchCount * mBlockCnt;
    const uint64_t biasLoadCount = inputParams_.hasBias ? batchCount * mBlockCnt : 0UL;
    const Mte2LoadEstimate estimate = {singleRoundABytes, singleRoundBBytes, singleRoundBiasBytes,
                                       aLoadCount,        bLoadCount,        biasLoadCount};
    return EstimateMte2LoadTimeUs(estimate, gmBandwidthTbps, l2BandwidthTbps);
}

double AdaptiveSlidingWindowMXBasicAPITiling::EstimateMacTimeUs() const
{
    return EstimateMatmulMacTimeUs(qmmv3_tiling_const::MXFP_DIVISOR_SIZE);
}

void AdaptiveSlidingWindowMXBasicAPITiling::UpdateAFullLoadStatus()
{
    uint64_t realBaseMSize = adaptiveWin_.mBaseTailSplitCnt == 1UL ? adaptiveWin_.baseM : adaptiveWin_.mTailMain;
    uint64_t kAligned = ops::CeilAlign(inputParams_.kSize, qmmv3_tiling_const::MXFP_DIVISOR_SIZE);
    uint64_t singleCoreASize = GetSizeWithDataType(realBaseMSize * kAligned, inputParams_.aDtype);
    bool isAFullLoadCandidate = singleCoreASize <=
                                    aicoreParams_.l1Size / qmmv3_tiling_const::AFULLLOAD_SINGLE_CORE_A_SCALER &&
                                adaptiveWin_.mBlockCnt < qmmv3_tiling_const::WINDOW_LEN &&
                                aicoreParams_.aicNum % adaptiveWin_.mBlockCnt == 0 &&
                                adaptiveWin_.totalBlockCnt > aicoreParams_.aicNum && inputParams_.batchC == 1;
    isAFullLoad_ = false;
    if (!isAFullLoadCandidate) {
        return;
    }
    // Prefer A full-load when it still leaves enough L1 to enable 4-buffer or 3-buffer.
    if (CanSelectMultiBufferByL1Plan(true, realBaseMSize, adaptiveWin_.baseN)) {
        isAFullLoad_ = true;
    } else if (CanSelectMultiBufferByL1Plan(false, adaptiveWin_.baseM, adaptiveWin_.baseN) &&
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

    const uint64_t baseMAlignSize = inputParams_.transA ?
                                        GetShapeWithDataType(qmmv3_tiling_const::L1_ALIGN_SIZE, inputParams_.aDtype) :
                                        qmmv3_tiling_const::CUBE_BLOCK;
    const uint64_t baseNAlignSize = GetBaseNAlignSize(qmmv3_tiling_const::L1_ALIGN_SIZE);
    const uint64_t newBaseM = GetSingleRoundTailSplitBase(inputParams_.mSize, adaptiveWin_.baseM,
                                                          adaptiveWin_.mTailTile, baseMAlignSize);
    const uint64_t newBaseN = GetSingleRoundTailSplitBase(inputParams_.nSize, adaptiveWin_.baseN,
                                                          adaptiveWin_.nTailTile, baseNAlignSize);
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
    tilingData_.matmulTiling.weightMustHitL2 = static_cast<uint8_t>(
        IsWeightMustHitL2(inputParams_, basicTiling_.baseM));
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
    withoutBatchTilingData_.kAL1 = tilingData_.matmulTiling.kAL1;
    withoutBatchTilingData_.kBL1 = tilingData_.matmulTiling.kBL1;
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
    withoutBatchTilingData_.weightMustHitL2 = tilingData_.matmulTiling.weightMustHitL2;
}

REGISTER_TILING_TEMPLATE_WITH_ARCH(QuantBatchMatmulV3, AdaptiveSlidingWindowMXBasicAPITiling, supportedNpuArch,
                                   TILING_PRIORITY);
} // namespace optiling
