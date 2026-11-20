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
 * \file quant_batch_matmul_v3_pergroup_tilling.cpp
 * \brief A8W8/A8W4 X2 Per-Group tiling strategy for DAV_RESV.
 */

#include "quant_batch_matmul_v3_pergroup_tilling.h"

#include <algorithm>

#include "base_block_calculator.h"
#include "common/op_host/op_tiling/tiling_type_mm.h"
#include "error_util.h"
#include "log/log.h"
#include "op_host/tiling_templates_registry.h"
#include "quant_batch_matmul_v3_tiling_strategy.h"

using namespace QuantBatchMatmulV3Arch35TilingKey;

namespace {
// NOTE: registered for DAV_RESV only. DAV_RESV is the value the framework maps from supportMmadS8S4;
//       the runtime decision is made in IsCapable(), never compare it against GetCurNpuArch().
const std::vector<int32_t> supportedNpuArch = {static_cast<int32_t>(NpuArch::DAV_RESV)};
constexpr int32_t TILING_PRIORITY = optiling::strategy::X2_PERGROUP;
constexpr uint64_t BIAS_UB_BYTES_PER_ELEMENT = sizeof(float) + sizeof(uint16_t);

} // namespace

namespace optiling {

using X2Config = X2PerGroupConfig;

X2PerGroupTiling::X2PerGroupTiling(gert::TilingContext* context)
    : QuantBatchMatmulV3TilingBase(context, false), tilingData_(tilingDataSelf_)
{
    resetFailed_ = !Reset();
}

bool X2PerGroupTiling::Reset()
{
    bFullLoad_ = false;
    aFullLoad_ = false;
    mCnt_ = 1UL;
    nCnt_ = 1UL;
    basicTiling_ = optiling::BasicRunInfoTiling();
    if (!isTilingOut_) {
        tilingData_ = DequantBmm::QuantBatchMatmulV3BasicAPITilingData();
        if (context_ != nullptr && context_->GetRawTilingData() != nullptr &&
            context_->GetRawTilingData()->GetData() != nullptr) {
            errno_t ret = memset_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(),
                                   0, context_->GetRawTilingData()->GetCapacity());
            if (ret != EOK) {
                CUBE_INNER_ERR_REPORT("QuantBatchMatmulV3", "X2PerGroup reset raw tiling data failed, ret=%d.", ret);
                return false;
            }
        }
    }
    return true;
}

// ===================== 1. Shape and attribute parsing (first step of DoTiling) =====================

ge::graphStatus X2PerGroupTiling::GetShapeAttrsInfo()
{
    inputParams_.Reset();
    tilingDataSize_ = sizeof(DequantBmm::QuantBatchMatmulV3BasicAPITilingData);
    const ge::graphStatus status = QuantBatchMatmulV3TilingBase::GetShapeAttrsInfo();
    if (status != ge::GRAPH_SUCCESS || !inputParams_.isX2PerGroup || inputParams_.transA || inputParams_.isPertoken ||
        inputParams_.batchB != 1UL || inputParams_.aFormat != ge::FORMAT_ND ||
        inputParams_.bFormat != ge::FORMAT_FRACTAL_NZ) {
        return status;
    }
    // Preserve the measured explicit-broadcast path for BOTH singleton-B representations.
    // The common parser may fuse batch into M only for rank-2 B. Restore the
    // original A/C batch axes after all input/output/scale checks have succeeded.
    const gert::Shape x1Shape = GetX1Shape(GetX1Idx());
    if (x1Shape.GetDimNum() <= 2UL || GetBatchSize(x1Shape) <= 1UL) {
        return status;
    }
    inputParams_.mSize = static_cast<uint64_t>(x1Shape.GetDim(x1Shape.GetDimNum() - 2UL));
    inputParams_.batchA = GetBatchSize(x1Shape);
    inputParams_.batchC = inputParams_.batchA;
    AnalyzeBatchInfo(x1Shape, GetX2Shape(GetX2Idx()));
    OP_LOGD(inputParams_.opName, "X2PerGroup shared-B canonical batch: M=%lu batchA=%lu batchC=%lu.",
            inputParams_.mSize, inputParams_.batchA, inputParams_.batchC);
    return status;
}

// dtype / shape checks use the util's MMAD S8S4 arch dispatch and land in
// QuantBatchMatmulV3Checker4MmadS8S4. Do not instantiate a checker by hand here.
bool X2PerGroupTiling::CheckDtype() const
{
    return optiling::QuantBatchMatMulV3TilingUtil::CheckDtype(context_, inputParams_, compileInfo_);
}

bool X2PerGroupTiling::CheckShape(const std::vector<gert::Shape*>& mandatoryShape, const gert::StorageShape* biasShape,
                                  const gert::StorageShape* pertokenShape,
                                  const std::vector<int64_t>& dimValueOfMKN) const
{
    return optiling::QuantBatchMatMulV3TilingUtil::CheckShape(context_, inputParams_, compileInfo_, mandatoryShape,
                                                              biasShape, pertokenShape, dimValueOfMKN);
}

// ===================== 2. Platform information =====================

// Same shape as QBMMV3StreamKTiling and QuantBatchMatmulV3IterbatchTiling on this route.
// The base class already ran SetPlatformInfoForTiling() in GetShapeAttrsInfo(), so this only validates.
ge::graphStatus X2PerGroupTiling::GetPlatformInfo()
{
    OP_LOGE_IF(!SetPlatformInfoForTiling(), ge::GRAPH_FAILED, inputParams_.opName, "SetPlatformInfo failed.");
    OP_TILING_CHECK(
        aicoreParams_.aicNum == 0UL || aicoreParams_.l1Size == 0UL || aicoreParams_.l0cSize == 0UL ||
            aicoreParams_.ubSize == 0UL,
        CUBE_INNER_ERR_REPORT(inputParams_.opName,
                              "CoreNum/L1Size/L0cSize/UbSize should not be 0. CoreNum: %lu, L1Size: %lu, "
                              "L0cSize: %lu, UbSize: %lu.",
                              aicoreParams_.aicNum, aicoreParams_.l1Size, aicoreParams_.l0cSize, aicoreParams_.ubSize),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// ===================== 3. Capability gate =====================

bool X2PerGroupTiling::IsCapable()
{
    // Test supportMmadS8S4, not GetCurNpuArch() != DAV_RESV: DAV_RESV is what the framework maps from
    // that same flag, so the latter form is always false and the strategy would never be selected.
    if (!compileInfo_.supportMmadS8S4) {
        return false;
    }
    // This strategy supports non-transposed ND A and FRACTAL_NZ B.
    // The checker reports invalid inputs; capability checks only select a strategy.
    // Validate core pairing separately in DoOpTiling().
    return inputParams_.isX2PerGroup && inputParams_.aFormat == ge::FORMAT_ND &&
           inputParams_.bFormat == ge::FORMAT_FRACTAL_NZ && !inputParams_.transA;
}

// ===================== 4. Tiling computation =====================

ge::graphStatus X2PerGroupTiling::DoOpTiling()
{
    // A failed memset_s in the constructor leaves the raw tiling data partly initialized, and the
    // constructor cannot report it. Fail here before any value is written into that dirty buffer.
    if (resetFailed_) {
        CUBE_INNER_ERR_REPORT(inputParams_.opName, "X2PerGroup failed to clear raw tiling data in the constructor.");
        return ge::GRAPH_FAILED;
    }
    // The core-number check lives here rather than in IsCapable(): its semantics are "wrong environment
    // -> hard failure", matching what AdaptiveSlidingWindowTiling::DoOpTiling() does.
    if (!CheckCoreNum()) {
        return ge::GRAPH_FAILED;
    }
    if (!CalcBaseBlock()) {
        CUBE_INNER_ERR_REPORT(inputParams_.opName, "X2PerGroup CalcBaseBlock failed.");
        return ge::GRAPH_FAILED;
    }
    CalcUsedCoreNum();
    SelectReuseMode();
    if (!CalcL1Tiling()) {
        CUBE_INNER_ERR_REPORT(inputParams_.opName, "X2PerGroup CalcL1Tiling failed.");
        return ge::GRAPH_FAILED;
    }
    SetQuantMode();

    OP_LOGD(inputParams_.opName,
            "X2PerGroup tiling: G=%lu groupCnt=%lu baseM=%u baseN=%u baseK=%u stepKa=%u stepKb=%u dbL0c=%u "
            "mCnt=%lu nCnt=%lu usedCore=%u aicNum=%u aivNum=%u ubSize=%lu l1Size=%lu kernelType=%lu aFullLoad=%u "
            "bFullLoad=%u",
            inputParams_.groupSizeK, ops::CeilDiv(inputParams_.kSize, inputParams_.groupSizeK), basicTiling_.baseM,
            basicTiling_.baseN, basicTiling_.baseK, basicTiling_.stepKa, basicTiling_.stepKb, basicTiling_.dbL0c, mCnt_,
            nCnt_, basicTiling_.usedCoreNum, compileInfo_.aicNum, compileInfo_.aivNum, aicoreParams_.ubSize,
            aicoreParams_.l1Size, GetKernelType(), static_cast<uint32_t>(aFullLoad_),
            static_cast<uint32_t>(bFullLoad_));
    return ge::GRAPH_SUCCESS;
}

bool X2PerGroupTiling::CheckCoreNum() const
{
    // This strategy uses AIC:AIV = 1:1, unlike the perblock template's 1:2.
    //       The core counts come from compileInfo, never hard-code 16.
    OP_TILING_CHECK(compileInfo_.aivNum != compileInfo_.aicNum,
                    CUBE_INNER_ERR_REPORT(inputParams_.opName,
                                          "For X2 Per-Group template, aicNum:aivNum should be 1:1, "
                                          "actual aicNum: %u, aivNum: %u.",
                                          compileInfo_.aicNum, compileInfo_.aivNum),
                    return false);
    return true;
}

// Both weight types use the shared full-M CMCT Cube and Vector pipeline.
bool X2PerGroupTiling::CalcBaseBlock()
{
    optiling::BaseBlockCalculator calculator(inputParams_, compileInfo_, inputParams_.batchC);
    if (!calculator.Compute(optiling::BaseBlockMode::MMAD_S8S4)) {
        return false;
    }
    const optiling::BaseBlockRes& res = calculator.GetOutput();
    const uint64_t mAlign = qmmv3_tiling_const::CUBE_BLOCK;
    uint64_t baseM = ops::CeilAlign(std::min(res.baseM, X2Config::MAX_BASE_M), mAlign);
    // A scale row must fit FixPipeParameter even when L0C is double buffered.
    const uint64_t baseN = std::min(res.baseN, X2Config::MAX_BASE_N);
    const uint64_t asyncOutputFactor = sizeof(uint16_t);
    // Two FP16 group slots + FP32 accumulator + FP32 cast temporary.
    const uint64_t ubFactor = X2Config::UB_BYTES_PER_ELEMENT + asyncOutputFactor;
    const uint64_t biasUbSize = inputParams_.hasBias ? baseN * BIAS_UB_BYTES_PER_ELEMENT : 0UL;
    // Both types: 14 * baseM * baseN + optional FP32/FP16 bias buffers.
    // L0C must hold at least one complete compute tile before considering DB.
    while (baseM > mAlign && (ubFactor * baseM * baseN + biasUbSize > aicoreParams_.ubSize ||
                              baseM * baseN * qmmv3_tiling_const::DATA_SIZE_L0C > aicoreParams_.l0cSize)) {
        // Keep the largest 16-aligned M tile that fits UB/L0C.
        baseM -= mAlign;
    }
    const uint64_t usedUbSize = ubFactor * baseM * baseN + biasUbSize;
    const uint64_t l0cBlockSize = baseM * baseN * qmmv3_tiling_const::DATA_SIZE_L0C;
    OP_TILING_CHECK(usedUbSize > aicoreParams_.ubSize || l0cBlockSize > aicoreParams_.l0cSize,
                    CUBE_INNER_ERR_REPORT(
                        inputParams_.opName, "X2PerGroup tile exceeds UB/L0C: UB %lu/%lu, L0C %lu/%lu, M %lu N %lu.",
                        usedUbSize, aicoreParams_.ubSize, l0cBlockSize, aicoreParams_.l0cSize, baseM, baseN),
                    return false);

    basicTiling_.baseM = static_cast<uint32_t>(baseM);
    basicTiling_.baseN = static_cast<uint32_t>(baseN);
    basicTiling_.baseK = static_cast<uint32_t>(CalcBaseK(res.baseK));
    OP_TILING_CHECK(basicTiling_.baseK == 0U,
                    CUBE_INNER_ERR_REPORT(inputParams_.opName, "X2PerGroup cannot fit a 64-aligned baseK in L0A/L0B."),
                    return false);
    mCnt_ = ops::CeilDiv(inputParams_.mSize, baseM);
    nCnt_ = ops::CeilDiv(inputParams_.nSize, baseN);

    // Keep the existing conservative strict inequality for L0C ping/pong.
    basicTiling_.dbL0c = (l0cBlockSize * qmmv3_tiling_const::DOUBLE_BUFFER_NUM < aicoreParams_.l0cSize) ?
                             qmmv3_tiling_const::DOUBLE_BUFFER_NUM :
                             1U;
    return true;
}

uint64_t X2PerGroupTiling::CalcBaseK(uint64_t baseKFromCalculator) const
{
    // Select the largest 64-aligned baseK that does not exceed the common
    // calculator result, divides the actual groupSizeK, and fits L0A/L0B.
    // This keeps MMAD boundaries aligned with quant-group boundaries; for
    // example G=384 selects baseK=192 instead of an uneven 256 + 128 split.
    // A8W4 loads the full baseM into each L0A ping/pong slot.
    const uint64_t baseKAlign = X2Config::GROUP_ALIGN;
    const auto canFitBaseK = [this](uint64_t baseK) {
        const uint64_t l0aBytes = optiling::GetSizeWithDataType(static_cast<uint64_t>(basicTiling_.baseM) * baseK,
                                                                inputParams_.aDtype);
        const uint64_t l0bBytes = optiling::GetSizeWithDataType(static_cast<uint64_t>(basicTiling_.baseN) * baseK,
                                                                inputParams_.bDtype);
        return l0aBytes <= compileInfo_.l0aSize / qmmv3_tiling_const::DOUBLE_BUFFER_NUM &&
               l0bBytes <= compileInfo_.l0bSize / qmmv3_tiling_const::DOUBLE_BUFFER_NUM;
    };
    // Search every 64-aligned candidate rather than repeatedly halving, which can skip a legal
    // non-power-of-two divisor. L0B capacity is measured in packed bytes for INT4.
    uint64_t baseK = std::max(ops::FloorAlign(std::min(baseKFromCalculator, inputParams_.groupSizeK), baseKAlign),
                              baseKAlign);
    for (; baseK >= baseKAlign; baseK -= baseKAlign) {
        if (inputParams_.groupSizeK % baseK == 0UL && canFitBaseK(baseK)) {
            return baseK;
        }
    }
    return 0UL;
}

// 2D tiling over M/N times batch; the kernel maps blockIdx back to (batchIdx, mIdx, nIdx) itself.
// singleCoreM/N/K are not set here: BasicAPICubeTiling has no such field, the kernel derives them.
void X2PerGroupTiling::CalcUsedCoreNum()
{
    // Keep blockDim equal to aicNum, matching the coupled MIX launch convention.
    // Blocks beyond the actual tile count remain inactive and drain paired credits.
    // min(totalBlockCnt, aicNum) could yield an odd blockDim, breaking the required AIC/AIV pairing
    // (C:V = 1:1, one shared Scalar) and risks a cross-core flag deadlock.
    basicTiling_.usedCoreNum = static_cast<uint32_t>(aicoreParams_.aicNum);
}

bool X2PerGroupTiling::CalcL1Tiling()
{
    basicTiling_.stepM = 1U;
    basicTiling_.stepN = 1U;

    // Resident K spans are rounded only for tiling metadata. The kernel allocates/copies
    // actual K with NZ padding, not CeilDiv(K, G) * G (important for K tails).
    if (aFullLoad_) {
        basicTiling_.stepKa = static_cast<uint32_t>(
            ops::CeilDiv(inputParams_.kSize, static_cast<uint64_t>(basicTiling_.baseK)));
        basicTiling_.stepKb = static_cast<uint32_t>(CalcAFullLoadStepK());
        return basicTiling_.stepKb != 0U;
    }
    if (bFullLoad_) {
        basicTiling_.stepKa = static_cast<uint32_t>(CalcBFullLoadStepK());
        basicTiling_.stepKb = static_cast<uint32_t>(
            ops::CeilDiv(inputParams_.kSize, static_cast<uint64_t>(basicTiling_.baseK)));
        return basicTiling_.stepKa != 0U;
    }
    const uint64_t stepK = CalcL1StepK();
    OP_TILING_CHECK(stepK == 0UL,
                    CUBE_INNER_ERR_REPORT(inputParams_.opName,
                                          "L1 cannot hold one kL1 tile: baseM %u, baseN %u, baseK %u, G %lu, "
                                          "l1Size %lu.",
                                          basicTiling_.baseM, basicTiling_.baseN, basicTiling_.baseK,
                                          inputParams_.groupSizeK, aicoreParams_.l1Size),
                    return false);
    basicTiling_.stepKa = static_cast<uint32_t>(stepK);
    basicTiling_.stepKb = static_cast<uint32_t>(stepK);
    // The adapter passes independent K windows to the component-owned L1 layout.
    // nBufferNum is unused by this X2 route and does not select its loading mode.
    return true;
}

bool X2PerGroupTiling::CanUseResidentLayout() const
{
    return inputParams_.aDtype == ge::DT_INT8 &&
           (inputParams_.bDtype == ge::DT_INT4 || inputParams_.bDtype == ge::DT_INT8) && inputParams_.mSize != 0UL &&
           inputParams_.nSize != 0UL && inputParams_.kSize != 0UL && inputParams_.groupSizeK != 0UL &&
           basicTiling_.baseM != 0U && basicTiling_.baseN != 0U && basicTiling_.baseK != 0U &&
           inputParams_.nSize <= UINT32_MAX / sizeof(uint64_t);
}

X2PerGroupTiling::ResidentLayout X2PerGroupTiling::GetResidentLayout() const
{
    const uint64_t c0 = GetShapeWithDataType(X2Config::DATA_BLOCK, inputParams_.bDtype);
    const uint64_t m = ops::CeilAlign(static_cast<uint64_t>(basicTiling_.baseM), X2Config::CUBE_BLOCK);
    const uint64_t weightN = ops::CeilAlign(static_cast<uint64_t>(basicTiling_.baseN),
                                            inputParams_.transB ? X2Config::CUBE_BLOCK : c0);
    ResidentLayout result;
    result.aPanelBytes = m * ops::CeilAlign(inputParams_.kSize, X2Config::DATA_BLOCK);
    result.bPanelBytes = GetSizeWithDataType(weightN * ops::CeilAlign(inputParams_.kSize, c0), inputParams_.bDtype);
    return result;
}

bool X2PerGroupTiling::CanAFullLoad(uint64_t kAL1, uint64_t kBL1) const
{
    if (!CanUseResidentLayout() || inputParams_.batchC != 1UL || kAL1 < inputParams_.kSize || mCnt_ == 0UL ||
        aicoreParams_.aicNum < mCnt_ || aicoreParams_.aicNum % mCnt_ != 0UL || nCnt_ <= aicoreParams_.aicNum / mCnt_) {
        return false;
    }
    const auto layout = GetResidentLayout();
    if (layout.aPanelBytes > X2Config::FULL_LOAD_MAX_BYTES ||
        layout.aPanelBytes > aicoreParams_.l1Size / X2Config::BUFFER_COUNT) {
        return false;
    }
    const uint64_t maxKb = CalcAFullLoadStepK() * basicTiling_.baseK;
    return maxKb != 0UL &&
           (kBL1 == UINT64_MAX || (kBL1 >= basicTiling_.baseK && kBL1 <= maxKb && kBL1 % basicTiling_.baseK == 0UL &&
                                   inputParams_.groupSizeK % kBL1 == 0UL));
}

uint64_t X2PerGroupTiling::CalcAFullLoadStepK() const
{
    if (!CanUseResidentLayout() || inputParams_.groupSizeK % basicTiling_.baseK != 0UL) {
        return 0UL;
    }
    const uint64_t c0 = GetShapeWithDataType(X2Config::DATA_BLOCK, inputParams_.bDtype);
    const uint64_t weightN = ops::CeilAlign(static_cast<uint64_t>(basicTiling_.baseN),
                                            inputParams_.transB ? X2Config::CUBE_BLOCK : c0);
    const uint64_t baseBBytes = GetSizeWithDataType(weightN * basicTiling_.baseK, inputParams_.bDtype);
    return CalcFullLoadStepK(GetResidentLayout().aPanelBytes, baseBBytes);
}

uint64_t X2PerGroupTiling::CalcFullLoadStepK(uint64_t residentBytes, uint64_t baseWindowBytes) const
{
    const uint64_t scaleRowBytes = ops::CeilAlign(static_cast<uint64_t>(basicTiling_.baseN), X2Config::CUBE_BLOCK) *
                                   sizeof(uint64_t);
    // Reserve one row per scale buffer; the full-load layout batches more rows in the remaining L1.
    const uint64_t fixedBytes = residentBytes + scaleRowBytes * X2Config::BUFFER_COUNT;
    if (fixedBytes >= aicoreParams_.l1Size) {
        return 0UL;
    }
    const uint64_t maxSteps = (aicoreParams_.l1Size - fixedBytes) /
                              (baseWindowBytes * X2Config::FULL_LOAD_BUFFER_COUNT);
    const uint64_t groupSteps = inputParams_.groupSizeK / basicTiling_.baseK;
    // Match the per-channel 32 KiB copy target, retaining four windows within the L1 budget.
    uint64_t copySteps = ops::CeilDiv(X2Config::FULL_LOAD_MIN_COPY_BYTES, baseWindowBytes);
    if (copySteps == 1UL && groupSteps > 1UL && maxSteps > 1UL) {
        copySteps = 2UL;
    }
    uint64_t step = std::min({groupSteps, maxSteps, copySteps});
    // Each input window belongs to one quantization group; only the final K tail may be shorter.
    while (step > 1UL && groupSteps % step != 0UL) {
        --step;
    }
    return step;
}

uint64_t X2PerGroupTiling::CalcBFullLoadStepK() const
{
    if (!CanUseResidentLayout() || inputParams_.groupSizeK % basicTiling_.baseK != 0UL) {
        return 0UL;
    }
    const uint64_t alignedM = ops::CeilAlign(static_cast<uint64_t>(basicTiling_.baseM), X2Config::CUBE_BLOCK);
    return CalcFullLoadStepK(GetResidentLayout().bPanelBytes, alignedM * basicTiling_.baseK);
}

bool X2PerGroupTiling::CanBFullLoad(uint64_t kBL1, uint64_t kAL1) const
{
    if (!CanUseResidentLayout() || inputParams_.batchC != 1UL || kBL1 < inputParams_.kSize || nCnt_ == 0UL ||
        aicoreParams_.aicNum < nCnt_ || aicoreParams_.aicNum % nCnt_ != 0UL || mCnt_ <= aicoreParams_.aicNum / nCnt_) {
        return false;
    }
    const auto layout = GetResidentLayout();
    if (layout.bPanelBytes > X2Config::FULL_LOAD_MAX_BYTES ||
        layout.bPanelBytes > aicoreParams_.l1Size / X2Config::BUFFER_COUNT) {
        return false;
    }
    const uint64_t maxKa = CalcBFullLoadStepK() * basicTiling_.baseK;
    return maxKa != 0UL &&
           (kAL1 == UINT64_MAX || (kAL1 >= basicTiling_.baseK && kAL1 <= maxKa && kAL1 % basicTiling_.baseK == 0UL &&
                                   inputParams_.groupSizeK % kAL1 == 0UL));
}

void X2PerGroupTiling::SelectReuseMode(uint64_t kAL1, uint64_t kBL1)
{
    aFullLoad_ = CanAFullLoad(kAL1, kBL1);
    bFullLoad_ = !aFullLoad_ && CanBFullLoad(kBL1, kAL1);
}

// baseK divides groupSizeK, so one kL1 span can carry an integral number of complete groups.
// A/B use full baseM/baseN in L1; scale rows are fixed by the backend.
uint64_t X2PerGroupTiling::CalcL1StepK() const
{
    const uint64_t stepKPerGroup = inputParams_.groupSizeK / basicTiling_.baseK;
    const uint64_t baseK = basicTiling_.baseK;
    const uint64_t bC0 = GetShapeWithDataType(X2Config::DATA_BLOCK, inputParams_.bDtype);
    const uint64_t weightN = ops::CeilAlign(static_cast<uint64_t>(basicTiling_.baseN),
                                            inputParams_.transB ? 16UL : bC0);
    // Cost of one baseK step, matching depthA1/depthB1: stepM/stepN blocks along M/N, double buffered.
    const uint64_t stepABSize = qmmv3_tiling_const::DOUBLE_BUFFER_NUM *
                                (basicTiling_.stepM *
                                     optiling::GetSizeWithDataType(static_cast<uint64_t>(basicTiling_.baseM) * baseK,
                                                                   inputParams_.aDtype) +
                                 basicTiling_.stepN *
                                     optiling::GetSizeWithDataType(weightN * baseK, inputParams_.bDtype));
    // A8W4 keeps one scale row per L1 ping/pong slot. Legacy A8W8
    // reserves four rows. Neither layout grows scale storage with stepK.
    const bool isA8W4 = inputParams_.aDtype == ge::DT_INT8 && inputParams_.bDtype == ge::DT_INT4;
    const uint64_t scaleRows = isA8W4 ? X2Config::W4_STREAM_SCALE_ROWS : X2Config::W8_STREAM_SCALE_ROWS;
    const uint64_t scalePerTile = static_cast<uint64_t>(basicTiling_.baseN) * basicTiling_.stepN *
                                  static_cast<uint64_t>(ge::GetSizeByDataType(inputParams_.scaleDtype)) * scaleRows;
    if (stepKPerGroup == 0UL || stepABSize == 0UL) {
        return 0UL;
    }
    // Bias is consumed on AIV; neither weight type needs L1/BT bias storage.
    const uint64_t fixedL1Size = scalePerTile;
    const uint64_t budget = aicoreParams_.l1Size;

    // Whole-group branch: each group costs stepKPerGroup steps of A/B; the two scale rows are shared
    // by all groups of the tile through the slot pingpong.
    const uint64_t perGroupSize = stepKPerGroup * stepABSize;
    uint64_t groupCntInL1 = 0UL;
    if (budget > fixedL1Size) {
        groupCntInL1 = (budget - fixedL1Size) / perGroupSize;
    }
    if (groupCntInL1 > 0UL) {
        // Never reach past K: the tile would only hold groups that do not exist.
        const uint64_t groupCnt = ops::CeilDiv(inputParams_.kSize, inputParams_.groupSizeK);
        return std::min(groupCntInL1, groupCnt) * stepKPerGroup;
    }

    // Fraction branch: a single group does not fit, so take the largest factor of it that does. The
    // two scale rows still have to be there, since even a partial tile reads that group's scale.
    if (budget <= fixedL1Size) {
        return 0UL;
    }
    uint64_t stepK = std::min(stepKPerGroup, (budget - fixedL1Size) / stepABSize);
    while (stepK > 1UL && stepKPerGroup % stepK != 0UL) {
        --stepK;
    }
    return stepK;
}

void X2PerGroupTiling::SetQuantMode()
{
    // batch, biasDtype and groupSizeM/N/K are filled by the shared helper, do not set them by hand
    optiling::QuantBatchMatMulV3TilingUtil::SetCommonTilingData(inputParams_, tilingData_);
    // This mode has no x1Scale: the x1 side stays DEFAULT while the x2 side is per-group
    tilingData_.params.x1QuantMode = static_cast<uint32_t>(optiling::BasicQuantMode::DEFAULT);
    tilingData_.params.x2QuantMode = static_cast<uint32_t>(optiling::BasicQuantMode::PERGROUP_MODE);
    // The adapter alone decodes it; ASW tail/window fields keep their defaults.
}

// ===================== 5. LibApi tiling =====================

ge::graphStatus X2PerGroupTiling::DoLibApiTiling()
{
    tilingData_.matmulTiling.m = static_cast<uint32_t>(inputParams_.mSize);
    tilingData_.matmulTiling.n = static_cast<uint32_t>(inputParams_.nSize);
    tilingData_.matmulTiling.k = static_cast<uint32_t>(inputParams_.kSize);
    tilingData_.matmulTiling.baseM = basicTiling_.baseM;
    tilingData_.matmulTiling.baseN = basicTiling_.baseN;
    tilingData_.matmulTiling.baseK = basicTiling_.baseK;
    tilingData_.matmulTiling.kAL1 = basicTiling_.stepKa * basicTiling_.baseK;
    tilingData_.matmulTiling.kBL1 = basicTiling_.stepKb * basicTiling_.baseK;
    // Cube is bias-free. A8W4 checks the optional GM pointer on AIV (an absent
    // bias may still carry the INT32 dtype); A8W8 retains its FLOAT sentinel.
    tilingData_.matmulTiling.isBias = 0U;
    tilingData_.matmulTiling.dbL0C = static_cast<uint8_t>(basicTiling_.dbL0c);
    // These shared ABI fields have no consumer in X2 per-group. Keep them
    // neutral: the private load mode controls layout, and weights use default L2.
    tilingData_.matmulTiling.scaleKL1 = 0U;
    tilingData_.matmulTiling.nBufferNum = 0U;
    tilingData_.matmulTiling.weightMustHitL2 = 0U;
    SelectKernelType();

    return ge::GRAPH_SUCCESS;
}

void X2PerGroupTiling::SelectKernelType()
{
    const auto& mm = tilingData_.matmulTiling;
    SelectReuseMode(static_cast<uint64_t>(mm.kAL1), static_cast<uint64_t>(mm.kBL1));
    // This per-group path does not use the native MX L0 tile plan.
    tilingData_.adaptiveSlidingWin.mL0 = 0U;
    tilingData_.adaptiveSlidingWin.nL0 = 0U;
}

// ===================== 6. Workspace =====================

ge::graphStatus X2PerGroupTiling::GetWorkspaceSize()
{
    workspaceSize_ = inputParams_.libApiWorkSpaceSize;
    return ge::GRAPH_SUCCESS;
}

// ===================== 7. Write back =====================

ge::graphStatus X2PerGroupTiling::PostTiling()
{
    OP_TILING_CHECK(
        tilingDataSize_ % sizeof(uint64_t) != 0UL,
        CUBE_INNER_ERR_REPORT(inputParams_.opName, "Tiling data size[%zu] is not aligned to 8.", tilingDataSize_),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        context_->GetRawTilingData()->GetCapacity() < tilingDataSize_,
        CUBE_INNER_ERR_REPORT(inputParams_.opName, "context tiling data capacity %zu < actual tiling data size %zu.",
                              context_->GetRawTilingData()->GetCapacity(), tilingDataSize_),
        return ge::GRAPH_FAILED);
    errno_t ret = memcpy_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(),
                           static_cast<const void*>(&tilingData_), tilingDataSize_);
    if (ret != EOK) {
        CUBE_INNER_ERR_REPORT(context_->GetNodeName(), "Failed to copy memory with memcpy_s, ret=%d.", ret);
        return ge::GRAPH_FAILED;
    }
    context_->SetBlockDim(basicTiling_.usedCoreNum);
    // 1 = batchmode, which SetScheduleMode() requires for any cross-core-sync operator. Without it,
    // multi-stream runs may preempt part of the cores and the CrossCoreFlag wait deadlocks.
    context_->SetScheduleMode(1);
    context_->GetRawTilingData()->SetDataSize(tilingDataSize_);
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = workspaceSize_;
    return ge::GRAPH_SUCCESS;
}

// ===================== 8. Tiling key (invoked by SetTilingKey at the end of DoTiling) =====================

uint64_t X2PerGroupTiling::GetBatchMode() const
{
    // Always use WITH_BATCH; batchC == 1 executes one batch iteration.
    return TPL_WITH_BATCH;
}

uint64_t X2PerGroupTiling::GetKernelType() const
{
    if (bFullLoad_) {
        return TPL_VEC_EPILOGUE_X2_PERGROUP_BFULL;
    }
    if (aFullLoad_) {
        return TPL_VEC_EPILOGUE_X2_PERGROUP_AFULL;
    }
    return TPL_VEC_EPILOGUE_X2_PERGROUP;
}

uint64_t X2PerGroupTiling::GetApiLevel() const { return TPL_API_LEVEL_BASIC; }

uint64_t X2PerGroupTiling::GetTilingKey() const
{
    // Layout: ATRANS[0:1] BTRANS[2:3] BATCHMODE[4:7] KERNELTYPE[8:15] APILEVEL[16:17]
    return GET_TPL_TILING_KEY(static_cast<uint64_t>(inputParams_.transA), static_cast<uint64_t>(inputParams_.transB),
                              GetBatchMode(), GetKernelType(), GetApiLevel());
}

REGISTER_TILING_TEMPLATE_WITH_ARCH(QuantBatchMatmulV3, X2PerGroupTiling, supportedNpuArch, TILING_PRIORITY);
} // namespace optiling
