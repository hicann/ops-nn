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
 * \file adaptive_sliding_window_cube_basic_api_tiling.cpp
 * \brief
 */

#include <algorithm>

#include "common/op_host/op_tiling/tiling_type_mm.h"
#include "log/log.h"
#include "error_util.h"
#include "op_host/tiling_templates_registry.h"
#include "adaptive_sliding_window_cube_basic_api_tiling.h"
#include "base_block_calculator.h"
#include "l1_tiling_data_calculator.h"
#include "quant_batch_matmul_v3_tiling_strategy.h"

namespace {
constexpr uint64_t MIN_REDUCED_STEP_K = 2UL;
constexpr uint64_t MAX_REDUCED_STEP_K = 8UL;
// Keep A full-load if repeated A reads exceed this share of estimated non-full-load A/B traffic.
constexpr double REPEAT_A_LOAD_RATIO_THRESHOLD = 0.20;

const std::vector<int32_t> supportedNpuArch = {static_cast<int32_t>(NpuArch::DAV_3510),
                                               static_cast<int32_t>(NpuArch::DAV_RESV)};
constexpr int32_t TILING_PRIORITY = optiling::strategy::CUBE_BASIC_API_ASW;
} // namespace

namespace optiling {

AdaptiveSlidingWindowCubeBasicAPITiling::AdaptiveSlidingWindowCubeBasicAPITiling(gert::TilingContext* context)
    : AdaptiveSlidingWindowTiling(context), tilingData_(tilingDataSelf_)
{
    AdaptiveSlidingWindowCubeBasicAPITiling::ResetTilingData();
    tilingDataSize_ = sizeof(DequantBmm::QuantBatchMatmulV3BasicAPITilingData);
}

AdaptiveSlidingWindowCubeBasicAPITiling::AdaptiveSlidingWindowCubeBasicAPITiling(
    gert::TilingContext* context, DequantBmm::QuantBatchMatmulV3BasicAPITilingData* out)
    : AdaptiveSlidingWindowTiling(context, nullptr), tilingData_(*out)
{
    AdaptiveSlidingWindowCubeBasicAPITiling::ResetTilingData();
    InitCompileInfo();
    inputParams_.Reset();
    tilingDataSize_ = sizeof(DequantBmm::QuantBatchMatmulV3BasicAPITilingData);
}

void AdaptiveSlidingWindowCubeBasicAPITiling::ResetTilingData()
{
    AdaptiveSlidingWindowTiling::ResetTilingData();
    if (!isTilingOut_) {
        tilingData_ = DequantBmm::QuantBatchMatmulV3BasicAPITilingData();
    }
    withoutBatchTilingData_ = DequantBmm::QuantBatchMatmulV3TensorAPIWithoutBatchTilingData();
    useWithoutBatchTilingData_ = false;
    tilingDataSize_ = sizeof(DequantBmm::QuantBatchMatmulV3BasicAPITilingData);
}

bool AdaptiveSlidingWindowCubeBasicAPITiling::IsWithoutBatchTilingData() const
{
    // The software S4S4 fallback needs AIV preprocessing and the full tiling data layout.
    return compileInfo_.npuArch == NpuArch::DAV_3510 && IsTensorApiEnabled() && inputParams_.batchC == 1UL &&
           !isSupportS4S4_;
}

bool AdaptiveSlidingWindowCubeBasicAPITiling::IsCapable()
{
    // Mandatory input descs have been validated by GetShapeAttrsInfo before IsCapable.
    if (compileInfo_.npuArch == NpuArch::DAV_RESV) {
        return inputParams_.aDtype == ge::DT_INT8 && inputParams_.bDtype == ge::DT_INT8 &&
               inputParams_.bFormat == ge::FORMAT_ND;
    }
    isSupportS4S4_ = inputParams_.aDtype == ge::DT_INT4 && inputParams_.bDtype == ge::DT_INT4 &&
                     !compileInfo_.supportMmadS8S4;
    const auto originADtype = inputParams_.aDtype;
    const auto originBDtype = inputParams_.bDtype;
    if (isSupportS4S4_) {
        inputParams_.aDtype = ge::DT_INT8;
        inputParams_.bDtype = ge::DT_INT8;
    }
    bool isCubeBasicApiCapable = IsCubeBasicApiCapable(inputParams_);
    bool isFp8OrHif8TTBiasMix = IsFp8OrHif8TTFloatBiasMix(inputParams_);
    bool capable = !isFp8OrHif8TTBiasMix && ((isCubeBasicApiCapable && inputParams_.bFormat == ge::FORMAT_ND) ||
                                             IsWeightNzNonMxCubeBasicApiCapable(inputParams_));
    if (!capable && isSupportS4S4_) {
        inputParams_.aDtype = originADtype;
        inputParams_.bDtype = originBDtype;
    }
    return capable;
}

ge::graphStatus AdaptiveSlidingWindowCubeBasicAPITiling::GetWorkspaceSize()
{
    workspaceSize_ = inputParams_.libApiWorkSpaceSize;
    if (isSupportS4S4_) {
        uint64_t aInt8Size = inputParams_.batchC * inputParams_.mSize * inputParams_.kSize;
        uint64_t bInt8Size = inputParams_.kSize * inputParams_.nSize;
        workspaceSize_ += ops::CeilAlign(aInt8Size, qmmv3_tiling_const::BASIC_BLOCK_SIZE_128) +
                          ops::CeilAlign(bInt8Size, qmmv3_tiling_const::BASIC_BLOCK_SIZE_128);
    }
    return ge::GRAPH_SUCCESS;
}

uint64_t AdaptiveSlidingWindowCubeBasicAPITiling::GetBatchCoreCnt() const { return inputParams_.batchC; }

const void* AdaptiveSlidingWindowCubeBasicAPITiling::GetTilingData() const
{
    return useWithoutBatchTilingData_ ? static_cast<const void*>(&withoutBatchTilingData_) :
                                        static_cast<const void*>(&tilingData_);
}

uint64_t AdaptiveSlidingWindowCubeBasicAPITiling::GetKernelType() const
{
    if (isAFullLoad_) {
        return static_cast<uint64_t>(QMMKernelType::NO_VEC_EPILOGUE_CUSTOM_GMTOAL1_WITH_MMAPI);
    }
    if (compileInfo_.npuArch == NpuArch::DAV_RESV && isBFullLoad_) {
        return static_cast<uint64_t>(QMMKernelType::NO_VEC_EPILOGUE_CUSTOM_GMTOBL1_WITH_MMAPI);
    }
    return static_cast<uint64_t>(QMMKernelType::NO_VEC_EPILOGUE_WITH_MMAPI);
}

uint64_t AdaptiveSlidingWindowCubeBasicAPITiling::GetApiLevel(NpuArch npuArch) const
{
    switch (npuArch) {
        case NpuArch::DAV_3510: {
            const bool tensorApiEnabled = IsTensorApiEnabled();
            if (inputParams_.bFormat == ge::FORMAT_FRACTAL_NZ && !tensorApiEnabled) {
                return static_cast<uint64_t>(QMMApiLevel::HIGH_LEVEL);
            }
            return tensorApiEnabled ? static_cast<uint64_t>(QMMApiLevel::BLAZE_LEVEL) :
                                      static_cast<uint64_t>(QMMApiLevel::BASIC_LEVEL);
        }
        case NpuArch::DAV_RESV:
            if (inputParams_.aDtype == ge::DT_INT8 && inputParams_.bDtype == ge::DT_INT8 &&
                inputParams_.bFormat == ge::FORMAT_ND) {
                return static_cast<uint64_t>(QMMApiLevel::BASIC_LEVEL);
            }
            return static_cast<uint64_t>(QMMApiLevel::HIGH_LEVEL);
        default:
            return static_cast<uint64_t>(QMMApiLevel::HIGH_LEVEL);
    }
}

uint64_t AdaptiveSlidingWindowCubeBasicAPITiling::GetBatchMode() const
{
    return IsWithoutBatchTilingData() ? static_cast<uint64_t>(BatchMode::WITHOUT_BATCH) :
                                        static_cast<uint64_t>(BatchMode::WITH_BATCH);
}

bool AdaptiveSlidingWindowCubeBasicAPITiling::CalcBasicBlock()
{
    if (compileInfo_.npuArch != NpuArch::DAV_3510) {
        return AdaptiveSlidingWindowTiling::CalcBasicBlock();
    }
    BaseBlockMode mode = compileInfo_.supportMmadS8S4 ? BaseBlockMode::MMAD_S8S4 : BaseBlockMode::CUBE_BASIC;
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

bool AdaptiveSlidingWindowCubeBasicAPITiling::CalL1Tiling()
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

    L1TilingMode mode = L1TilingMode::DEFAULT;
    if (isAFullLoad_) {
        mode = L1TilingMode::A_L1_FULL_LOAD;
    } else if (isBFullLoad_) {
        mode = L1TilingMode::B_L1_FULL_LOAD;
    } else if (compileInfo_.npuArch == NpuArch::DAV_RESV) {
        mode = L1TilingMode::PASS_OPTIMIZED;
    }
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
    return true;
}

ge::graphStatus AdaptiveSlidingWindowCubeBasicAPITiling::DoLibApiTiling()
{
    tilingData_.matmulTiling.m = inputParams_.mSize;
    tilingData_.matmulTiling.n = inputParams_.nSize;
    tilingData_.matmulTiling.k = inputParams_.kSize;

    tilingData_.matmulTiling.baseM = basicTiling_.baseM;
    tilingData_.matmulTiling.baseN = basicTiling_.baseN;
    tilingData_.matmulTiling.baseK = basicTiling_.baseK;
    tilingData_.matmulTiling.isBias = inputParams_.hasBias ? 1UL : 0UL;
    tilingData_.matmulTiling.dbL0C = static_cast<uint8_t>(basicTiling_.dbL0c);
    CalculateNBufferNum();
    if (useWithoutBatchTilingData_) {
        SetWithoutBatchTilingData();
    }
    return ge::GRAPH_SUCCESS;
}

void AdaptiveSlidingWindowCubeBasicAPITiling::CalculateNBufferNum()
{
    if (compileInfo_.npuArch != NpuArch::DAV_3510) {
        CalculateLegacyNBufferNum();
        return;
    }

    const uint64_t currentKAL1 = static_cast<uint64_t>(basicTiling_.stepKa) * tilingData_.matmulTiling.baseK;
    const uint64_t currentKBL1 = static_cast<uint64_t>(basicTiling_.stepKb) * tilingData_.matmulTiling.baseK;
    const CubeL1Context ctx = {tilingData_.matmulTiling.baseM, tilingData_.matmulTiling.baseN,
                               tilingData_.matmulTiling.baseK, isAFullLoad_};
    const CubeL1Plan plan = SelectCubeL1Plan(ctx, currentKAL1, currentKBL1);
    tilingData_.matmulTiling.kAL1 = static_cast<uint32_t>(plan.kAL1);
    tilingData_.matmulTiling.kBL1 = static_cast<uint32_t>(plan.kBL1);
    tilingData_.matmulTiling.nBufferNum = static_cast<uint8_t>(plan.l1BufferNum);
}

void AdaptiveSlidingWindowCubeBasicAPITiling::CalculateLegacyNBufferNum()
{
    tilingData_.matmulTiling.kAL1 = basicTiling_.stepKa * tilingData_.matmulTiling.baseK;
    tilingData_.matmulTiling.kBL1 = basicTiling_.stepKb * tilingData_.matmulTiling.baseK;
    uint64_t kL1 = 0;
    if (isAFullLoad_) {
        kL1 = basicTiling_.stepKb * tilingData_.matmulTiling.baseK;
    } else {
        uint64_t stepK = std::min(basicTiling_.stepKa, basicTiling_.stepKb);
        kL1 = stepK * tilingData_.matmulTiling.baseK;
    }
    uint64_t usedL1Size = 0UL;
    if (isBFullLoad_) {
        uint64_t kAligned = ops::CeilAlign(inputParams_.kSize, inputParams_.transB ?
                                                                   qmmv3_tiling_const::CUBE_REDUCE_BLOCK :
                                                                   qmmv3_tiling_const::CUBE_BLOCK);
        usedL1Size = GetSizeWithDataType(basicTiling_.baseN * kAligned, inputParams_.bDtype);
    } else {
        usedL1Size = GetSizeWithDataType(basicTiling_.baseN * kL1, inputParams_.bDtype) *
                     qmmv3_tiling_const::L1_FOUR_BUFFER;
    }
    if (inputParams_.isPerChannel) {
        usedL1Size += GetSizeWithDataType(basicTiling_.baseN, inputParams_.scaleDtype) *
                      qmmv3_tiling_const::L1_TWO_BUFFER;
    }
    if (inputParams_.hasBias) {
        usedL1Size += GetSizeWithDataType(basicTiling_.baseN, inputParams_.biasDtype) *
                      qmmv3_tiling_const::L1_TWO_BUFFER;
    }
    if (isAFullLoad_) {
        uint64_t kAligned = ops::CeilAlign(inputParams_.kSize, !inputParams_.transA ?
                                                                   qmmv3_tiling_const::CUBE_REDUCE_BLOCK :
                                                                   qmmv3_tiling_const::CUBE_BLOCK);
        usedL1Size += GetSizeWithDataType(basicTiling_.baseM * kAligned, inputParams_.aDtype);
    } else {
        usedL1Size += GetSizeWithDataType(basicTiling_.baseM * kL1, inputParams_.aDtype) *
                      qmmv3_tiling_const::L1_FOUR_BUFFER;
    }
    tilingData_.matmulTiling.nBufferNum = usedL1Size <= aicoreParams_.l1Size ? qmmv3_tiling_const::L1_FOUR_BUFFER :
                                                                               qmmv3_tiling_const::L1_TWO_BUFFER;
    if (tilingData_.matmulTiling.nBufferNum == qmmv3_tiling_const::L1_FOUR_BUFFER) {
        tilingData_.matmulTiling.kAL1 = std::min(basicTiling_.stepKa, basicTiling_.stepKb) *
                                        tilingData_.matmulTiling.baseK;
        tilingData_.matmulTiling.kBL1 = tilingData_.matmulTiling.kAL1;
    }
}

AdaptiveSlidingWindowCubeBasicAPITiling::CubeL1Plan AdaptiveSlidingWindowCubeBasicAPITiling::SelectCubeL1Plan(
    const CubeL1Context& ctx, uint64_t currentKAL1, uint64_t currentKBL1) const
{
    // Fallback: keep the caller's current (possibly asymmetric) two-buffer plan.
    CubeL1Plan plan = {currentKAL1, currentKBL1, qmmv3_tiling_const::L1_TWO_BUFFER};

    // Both callers supply positive L1 steps with kL1 = stepK * baseK. Collapse them onto the shared minimum step so a
    // promoted plan uses one symmetric kL1 for A and B.
    const uint64_t baseK = ctx.baseK;
    const uint64_t currentStepK = ctx.isAFullLoad ? currentKBL1 / baseK : std::min(currentKAL1, currentKBL1) / baseK;
    const uint64_t currentKL1 = currentStepK * baseK;

    // Candidate 1: four buffers at the current step. Unconditional (legacy behavior); it only needs to fit L1.
    if (CanFitL1BufferNum(ctx, currentKL1, qmmv3_tiling_const::L1_FOUR_BUFFER)) {
        return {currentKL1, currentKL1, qmmv3_tiling_const::L1_FOUR_BUFFER};
    }

    // Candidates 2-4 only pay off when the roofline is MTE2-bound; A-full-load is always treated as bound.
    const bool isMte2Bound = ctx.isAFullLoad || IsMte2Bound(qmmv3_tiling_const::ASCEND_950_MAX_HBM_BW_TBPS *
                                                                qmmv3_tiling_const::MTE2_BW_UTILIZATION,
                                                            qmmv3_tiling_const::ASCEND_950_MAX_L2_BW_TBPS *
                                                                qmmv3_tiling_const::MTE2_BW_UTILIZATION);
    if (!isMte2Bound) {
        return plan;
    }

    // Candidate 2: four buffers at a reduced step.
    if (TryApplyReducedStepK(plan, ctx, currentKL1, qmmv3_tiling_const::L1_FOUR_BUFFER)) {
        return plan;
    }

    // Candidate 3: three buffers at the current step. Collapsing an asymmetric plan onto the shared step needs the
    // same K-inner alignment as the reduced-step candidates; a symmetric plan is always allowed.
    const bool isDirectPromoteAllowed = currentKAL1 == currentKBL1 || AreKInnerAxesAligned();
    if (isDirectPromoteAllowed && CanFitL1BufferNum(ctx, currentKL1, qmmv3_tiling_const::L1_THREE_BUFFER)) {
        return {currentKL1, currentKL1, qmmv3_tiling_const::L1_THREE_BUFFER};
    }

    // Candidate 4: three buffers at a reduced step; otherwise keep the two-buffer fallback.
    (void)TryApplyReducedStepK(plan, ctx, currentKL1, qmmv3_tiling_const::L1_THREE_BUFFER);
    return plan;
}

bool AdaptiveSlidingWindowCubeBasicAPITiling::TryApplyReducedStepK(CubeL1Plan& plan, const CubeL1Context& ctx,
                                                                   uint64_t currentKL1, uint32_t l1BufferNum) const
{
    const uint64_t baseK = ctx.baseK;
    const uint64_t currentStepK = currentKL1 / baseK;
    const bool isCurrentTwoBufferNotOverK = currentKL1 * qmmv3_tiling_const::L1_TWO_BUFFER < inputParams_.kSize;
    if (currentStepK <= MIN_REDUCED_STEP_K || !isCurrentTwoBufferNotOverK || !AreKInnerAxesAligned()) {
        return false;
    }
    const uint64_t reducedStepK = SelectReducedStepK(ctx, currentKL1, l1BufferNum);
    if (reducedStepK == 0UL) {
        return false;
    }
    const uint64_t reducedKL1 = reducedStepK * baseK;
    if (!CanFitL1BufferNum(ctx, reducedKL1, l1BufferNum)) {
        return false;
    }
    plan = {reducedKL1, reducedKL1, l1BufferNum};
    return true;
}

uint64_t AdaptiveSlidingWindowCubeBasicAPITiling::SelectReducedStepK(const CubeL1Context& ctx, uint64_t currentKL1,
                                                                     uint32_t l1BufferNum) const
{
    // SelectCubeL1Plan has already checked the reduced-step eligibility.
    const uint64_t baseK = ctx.baseK;
    const uint64_t currentStepK = currentKL1 / baseK;
    // baseK has already been bounded by the L0A/L0B double-buffer capacity. stepK only groups baseK tiles in L1, so
    // the L0 max-baseK formula is not a stepK limit. Bound this reduced search by the remaining K tiles, the current
    // step and the issue-queue search cap instead.
    const uint64_t kStepCount = ops::CeilDiv(inputParams_.kSize, baseK);
    const uint64_t maxCandidateStepK = std::min(std::min(currentStepK - 1UL, kStepCount), MAX_REDUCED_STEP_K);
    if (maxCandidateStepK < MIN_REDUCED_STEP_K) {
        return 0UL;
    }

    // Search from large to small. When K is an operand's inner axis, its candidate kL1 must be 256-byte aligned.
    for (uint64_t stepK = maxCandidateStepK; stepK >= MIN_REDUCED_STEP_K; --stepK) {
        const uint64_t kL1 = baseK * stepK;
        if (IsKInnerKL1AlignedTo256Bytes(kL1) && CanFitL1BufferNum(ctx, kL1, l1BufferNum)) {
            return stepK;
        }
        if (stepK == MIN_REDUCED_STEP_K) {
            break;
        }
    }
    return 0UL;
}

bool AdaptiveSlidingWindowCubeBasicAPITiling::CanFitL1BufferNum(const CubeL1Context& ctx, uint64_t kL1,
                                                                uint32_t l1BufferNum) const
{
    if (l1BufferNum == qmmv3_tiling_const::L1_THREE_BUFFER && !IsTensorApiEnabled()) {
        return false;
    }
    return CalcUsedL1Size(ctx, kL1, l1BufferNum) <= aicoreParams_.l1Size;
}

uint64_t AdaptiveSlidingWindowCubeBasicAPITiling::CalcUsedL1Size(const CubeL1Context& ctx, uint64_t kL1,
                                                                 uint32_t l1BufferNum) const
{
    const uint64_t bL1OneBuffer = GetSizeWithDataType(ctx.baseN * kL1, inputParams_.bDtype);
    // TT passes a scalar scale directly to Fixpipe. TC keeps baseN scale values in each of its two scale slots.
    const uint64_t scaleL1OneBuffer = inputParams_.isPerChannel ?
                                          GetSizeWithDataType(ctx.baseN, inputParams_.scaleDtype) :
                                          0UL;
    const uint64_t biasL1OneBuffer = inputParams_.hasBias ? GetSizeWithDataType(ctx.baseN, inputParams_.biasDtype) :
                                                            0UL;
    const uint64_t aL1OneBuffer = ctx.isAFullLoad ? CalcAFullLoadL1Size(ctx.baseM) :
                                                    GetSizeWithDataType(ctx.baseM * kL1, inputParams_.aDtype);
    return bL1OneBuffer * l1BufferNum + scaleL1OneBuffer * qmmv3_tiling_const::L1_TWO_BUFFER +
           biasL1OneBuffer * qmmv3_tiling_const::L1_TWO_BUFFER +
           (ctx.isAFullLoad ? aL1OneBuffer : aL1OneBuffer * l1BufferNum);
}

uint64_t AdaptiveSlidingWindowCubeBasicAPITiling::CalcAFullLoadL1Size(uint64_t baseM) const
{
    if (inputParams_.transA) {
        return baseM * GetSizeWithDataType(ops::CeilAlign(inputParams_.kSize, qmmv3_tiling_const::CUBE_BLOCK),
                                           inputParams_.aDtype);
    }
    return baseM * ops::CeilAlign(GetSizeWithDataType(inputParams_.kSize, inputParams_.aDtype),
                                  qmmv3_tiling_const::CUBE_REDUCE_BLOCK);
}

// Use the padded L1 layout as a proxy for full-K data traffic.
uint64_t AdaptiveSlidingWindowCubeBasicAPITiling::CalcAFullKLoadSize(uint64_t mSize) const
{
    const uint64_t aReduceAlign = GetShapeWithDataType(qmmv3_tiling_const::CUBE_REDUCE_BLOCK, inputParams_.aDtype);
    const uint64_t alignedM = ops::CeilAlign(mSize,
                                             inputParams_.transA ? aReduceAlign : qmmv3_tiling_const::CUBE_BLOCK);
    const uint64_t fullKa = ops::CeilAlign(inputParams_.kSize,
                                           inputParams_.transA ? qmmv3_tiling_const::CUBE_BLOCK : aReduceAlign);
    return GetSizeWithDataType(alignedM * fullKa, inputParams_.aDtype);
}

uint64_t AdaptiveSlidingWindowCubeBasicAPITiling::CalcBFullKLoadSize(uint64_t nSize) const
{
    const uint64_t bReduceAlign = GetShapeWithDataType(qmmv3_tiling_const::CUBE_REDUCE_BLOCK, inputParams_.bDtype);
    const uint64_t alignedN = ops::CeilAlign(nSize,
                                             inputParams_.transB ? qmmv3_tiling_const::CUBE_BLOCK : bReduceAlign);
    const uint64_t fullKb = ops::CeilAlign(inputParams_.kSize,
                                           inputParams_.transB ? bReduceAlign : qmmv3_tiling_const::CUBE_BLOCK);
    return GetSizeWithDataType(alignedN * fullKb, inputParams_.bDtype);
}

bool AdaptiveSlidingWindowCubeBasicAPITiling::ShouldKeepAFullLoadByRepeatLoadRatio() const
{
    if (adaptiveWin_.nBlockCnt <= 1UL) {
        return false;
    }
    const double singleRoundABytes = static_cast<double>(CalcAFullKLoadSize(inputParams_.mSize));
    const double repeatABytes = singleRoundABytes * static_cast<double>(adaptiveWin_.nBlockCnt - 1UL);
    const double nonFullLoadABytes = singleRoundABytes * static_cast<double>(adaptiveWin_.nBlockCnt);
    double singleRoundBBytes = static_cast<double>(CalcBFullKLoadSize(inputParams_.nSize));
    if (inputParams_.isPerChannel) {
        singleRoundBBytes += static_cast<double>(GetSizeWithDataType(inputParams_.nSize, inputParams_.scaleDtype));
    }
    const double nonFullLoadBBytes = singleRoundBBytes * static_cast<double>(adaptiveWin_.mBlockCnt);
    const double totalLoadBytes = nonFullLoadABytes + nonFullLoadBBytes;
    return repeatABytes / totalLoadBytes > REPEAT_A_LOAD_RATIO_THRESHOLD;
}

bool AdaptiveSlidingWindowCubeBasicAPITiling::IsMte2Bound(double gmBandwidthTbps, double l2BandwidthTbps) const
{
    return !AreOperandInnerAxesAligned() || EstimateMte2TimeUs(gmBandwidthTbps, l2BandwidthTbps) > EstimateMacTimeUs();
}

double AdaptiveSlidingWindowCubeBasicAPITiling::EstimateMte2TimeUs(double gmBandwidthTbps, double l2BandwidthTbps) const
{
    const double singleRoundABytes = static_cast<double>(CalcAFullKLoadSize(inputParams_.mSize));
    double singleRoundBBytes = static_cast<double>(CalcBFullKLoadSize(inputParams_.nSize));
    if (inputParams_.isPerChannel) {
        singleRoundBBytes += static_cast<double>(GetSizeWithDataType(inputParams_.nSize, inputParams_.scaleDtype));
    }
    const double singleRoundBiasBytes = inputParams_.hasBias ? static_cast<double>(GetSizeWithDataType(
                                                                   inputParams_.nSize, inputParams_.biasDtype)) :
                                                               0.0;
    const uint64_t batchCount = std::max<uint64_t>(1UL, inputParams_.batchC);
    // This estimate is used only for non-full plans, which reload A for every N block.
    const uint64_t aLoadCount = batchCount * adaptiveWin_.nBlockCnt;
    const uint64_t bLoadCount = batchCount * adaptiveWin_.mBlockCnt;
    const uint64_t biasLoadCount = inputParams_.hasBias ? batchCount * adaptiveWin_.mBlockCnt : 0UL;
    const Mte2LoadEstimate estimate = {singleRoundABytes, singleRoundBBytes, singleRoundBiasBytes,
                                       aLoadCount,        bLoadCount,        biasLoadCount};
    return EstimateMte2LoadTimeUs(estimate, gmBandwidthTbps, l2BandwidthTbps);
}

double AdaptiveSlidingWindowCubeBasicAPITiling::EstimateMacTimeUs() const
{
    return EstimateMatmulMacTimeUs(qmmv3_tiling_const::CUBE_REDUCE_BLOCK);
}

bool AdaptiveSlidingWindowCubeBasicAPITiling::CanSelectMultiBufferByL1Plan(bool isAFullLoad, uint64_t baseM,
                                                                           uint64_t baseN) const
{
    const L1TilingMode mode = isAFullLoad ? L1TilingMode::A_L1_FULL_LOAD : L1TilingMode::DEFAULT;
    L1TilingDataCalculator calculator(inputParams_, compileInfo_, baseM, baseN, adaptiveWin_.baseK);
    if (!calculator.Compute(mode)) {
        return false;
    }
    const L1TilingData& l1Tiling = calculator.GetOutput();
    const CubeL1Context ctx = {baseM, baseN, adaptiveWin_.baseK, isAFullLoad};
    const CubeL1Plan plan = SelectCubeL1Plan(ctx, l1Tiling.stepKa_ * adaptiveWin_.baseK,
                                             l1Tiling.stepKb_ * adaptiveWin_.baseK);
    return plan.l1BufferNum > qmmv3_tiling_const::L1_TWO_BUFFER;
}

void AdaptiveSlidingWindowCubeBasicAPITiling::UpdateAFullLoadStatus()
{
    const uint64_t realBaseMSize = adaptiveWin_.mBaseTailSplitCnt == 1UL ? adaptiveWin_.baseM : adaptiveWin_.mTailMain;
    const uint64_t singleCoreASize = realBaseMSize *
                                     (inputParams_.transA ?
                                          GetSizeWithDataType(
                                              ops::CeilAlign(inputParams_.kSize, qmmv3_tiling_const::CUBE_BLOCK),
                                              inputParams_.aDtype) :
                                          ops::CeilAlign(GetSizeWithDataType(inputParams_.kSize, inputParams_.aDtype),
                                                         qmmv3_tiling_const::CUBE_REDUCE_BLOCK));
    const bool isAFullLoadCandidate = singleCoreASize <=
                                          aicoreParams_.l1Size / qmmv3_tiling_const::AFULLLOAD_SINGLE_CORE_A_SCALER &&
                                      adaptiveWin_.mBlockCnt < qmmv3_tiling_const::WINDOW_LEN &&
                                      aicoreParams_.aicNum % adaptiveWin_.mBlockCnt == 0 &&
                                      adaptiveWin_.totalBlockCnt > aicoreParams_.aicNum && inputParams_.batchC == 1;
    bool selectAFullLoad = isAFullLoadCandidate;
    if (selectAFullLoad && compileInfo_.npuArch == NpuArch::DAV_3510) {
        const bool aFullSelectsMultiBuffer = CanSelectMultiBufferByL1Plan(true, realBaseMSize, adaptiveWin_.baseN);

        if (!aFullSelectsMultiBuffer) {
            const bool nonFullSelectsMultiBuffer = CanSelectMultiBufferByL1Plan(false, adaptiveWin_.baseM,
                                                                                adaptiveWin_.baseN);
            if (nonFullSelectsMultiBuffer && !ShouldKeepAFullLoadByRepeatLoadRatio()) {
                selectAFullLoad = false;
            }
        }
    }
    isAFullLoad_ = selectAFullLoad;

    if (!isAFullLoad_ || adaptiveWin_.baseM == realBaseMSize) {
        return;
    }
    adaptiveWin_.baseM = realBaseMSize;
    adaptiveWin_.mBaseTailSplitCnt = 1UL;
    adaptiveWin_.mTailMain = 0UL;
}

void AdaptiveSlidingWindowCubeBasicAPITiling::UpdateBFullLoadStatus()
{
    if (isAFullLoad_ || compileInfo_.npuArch != NpuArch::DAV_RESV) {
        isBFullLoad_ = false;
        return;
    }
    uint64_t realBaseNSize = adaptiveWin_.nBaseTailSplitCnt == 1UL ? adaptiveWin_.baseN : adaptiveWin_.nTailMain;
    uint64_t singleCoreBSize = realBaseNSize *
                               (inputParams_.transB ?
                                    ops::CeilAlign(GetSizeWithDataType(inputParams_.kSize, inputParams_.bDtype),
                                                   qmmv3_tiling_const::CUBE_REDUCE_BLOCK) :
                                    GetSizeWithDataType(
                                        ops::CeilAlign(inputParams_.kSize, qmmv3_tiling_const::CUBE_BLOCK),
                                        inputParams_.bDtype));

    isBFullLoad_ = singleCoreBSize <= aicoreParams_.l1Size / qmmv3_tiling_const::AFULLLOAD_SINGLE_CORE_B_SCALER &&
                   adaptiveWin_.nBlockCnt < qmmv3_tiling_const::WINDOW_LEN &&
                   aicoreParams_.aicNum % adaptiveWin_.nBlockCnt == 0 &&
                   adaptiveWin_.totalBlockCnt > aicoreParams_.aicNum && inputParams_.batchC == 1;
    if (isBFullLoad_ && adaptiveWin_.baseN != realBaseNSize) {
        adaptiveWin_.baseN = realBaseNSize;
        adaptiveWin_.nBaseTailSplitCnt = 1UL;
        adaptiveWin_.nTailMain = 0UL;
    }
}

void AdaptiveSlidingWindowCubeBasicAPITiling::AnalyseFullLoadInfo()
{
    isABFullLoad_ = false;
    isBFullLoad_ = false;
    UpdateAFullLoadStatus();
    UpdateBFullLoadStatus();
}

void AdaptiveSlidingWindowCubeBasicAPITiling::SetTilingData()
{
    useWithoutBatchTilingData_ = IsWithoutBatchTilingData();
    tilingDataSize_ = useWithoutBatchTilingData_ ?
                          sizeof(DequantBmm::QuantBatchMatmulV3TensorAPIWithoutBatchTilingData) :
                          sizeof(DequantBmm::QuantBatchMatmulV3BasicAPITilingData);

    QuantBatchMatMulV3TilingUtil::SetCommonTilingData(inputParams_, tilingData_);
    tilingData_.matmulTiling.weightMustHitL2 = static_cast<uint8_t>(
        IsWeightMustHitL2(inputParams_, basicTiling_.baseM));
    if (inputParams_.bFormat == ge::FORMAT_FRACTAL_NZ) {
        tilingData_.params.x1QuantMode = static_cast<uint32_t>(optiling::BasicQuantMode::DEFAULT);
        tilingData_.params.x2QuantMode = static_cast<uint32_t>(optiling::BasicQuantMode::PERCHANNEL_MODE);
        if (inputParams_.isDoubleScale) {
            tilingData_.params.x1QuantMode = static_cast<uint32_t>(optiling::BasicQuantMode::PERTENSOR_MODE);
            tilingData_.params.x2QuantMode = static_cast<uint32_t>(optiling::BasicQuantMode::PERTENSOR_MODE);
        } else if (inputParams_.isPerTensor) {
            tilingData_.params.x2QuantMode = static_cast<uint32_t>(optiling::BasicQuantMode::PERTENSOR_MODE);
        }
    } else if (inputParams_.isDoubleScale) {
        tilingData_.params.x1QuantMode = static_cast<uint32_t>(optiling::BasicQuantMode::PERTENSOR_MODE);
        tilingData_.params.x2QuantMode = static_cast<uint32_t>(optiling::BasicQuantMode::PERTENSOR_MODE);
    } else if (inputParams_.isPerTensor) {
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

void AdaptiveSlidingWindowCubeBasicAPITiling::SetWithoutBatchTilingData()
{
    withoutBatchTilingData_.m = static_cast<uint32_t>(inputParams_.mSize);
    withoutBatchTilingData_.n = static_cast<uint32_t>(inputParams_.nSize);
    withoutBatchTilingData_.k = static_cast<uint32_t>(inputParams_.kSize);
    withoutBatchTilingData_.kAL1 = tilingData_.matmulTiling.kAL1;
    withoutBatchTilingData_.kBL1 = tilingData_.matmulTiling.kBL1;
    withoutBatchTilingData_.baseM = static_cast<uint16_t>(basicTiling_.baseM);
    withoutBatchTilingData_.baseN = static_cast<uint16_t>(basicTiling_.baseN);
    withoutBatchTilingData_.baseK = static_cast<uint16_t>(basicTiling_.baseK);
    withoutBatchTilingData_.groupSizeM = static_cast<uint16_t>(inputParams_.groupSizeM);
    withoutBatchTilingData_.groupSizeN = static_cast<uint16_t>(inputParams_.groupSizeN);
    withoutBatchTilingData_.groupSizeK = static_cast<uint16_t>(inputParams_.groupSizeK);
    withoutBatchTilingData_.mTailTile = static_cast<uint16_t>(adaptiveWin_.mTailTile);
    withoutBatchTilingData_.nTailTile = static_cast<uint16_t>(adaptiveWin_.nTailTile);
    withoutBatchTilingData_.mBaseTailSplitCnt = static_cast<uint16_t>(adaptiveWin_.mBaseTailSplitCnt);
    withoutBatchTilingData_.nBaseTailSplitCnt = static_cast<uint16_t>(adaptiveWin_.nBaseTailSplitCnt);
    withoutBatchTilingData_.mTailMain = static_cast<uint16_t>(adaptiveWin_.mTailMain);
    withoutBatchTilingData_.nTailMain = static_cast<uint16_t>(adaptiveWin_.nTailMain);
    withoutBatchTilingData_.x1QuantMode = static_cast<uint8_t>(tilingData_.params.x1QuantMode);
    withoutBatchTilingData_.x2QuantMode = static_cast<uint8_t>(tilingData_.params.x2QuantMode);
    withoutBatchTilingData_.isBias = tilingData_.matmulTiling.isBias;
    withoutBatchTilingData_.biasDtype = static_cast<uint8_t>(inputParams_.biasDtype);
    withoutBatchTilingData_.nBufferNum = tilingData_.matmulTiling.nBufferNum;
    withoutBatchTilingData_.dbL0C = tilingData_.matmulTiling.dbL0C;
    withoutBatchTilingData_.weightMustHitL2 = tilingData_.matmulTiling.weightMustHitL2;
}

REGISTER_TILING_TEMPLATE_WITH_ARCH(QuantBatchMatmulV3, AdaptiveSlidingWindowCubeBasicAPITiling, supportedNpuArch,
                                   TILING_PRIORITY);
} // namespace optiling
