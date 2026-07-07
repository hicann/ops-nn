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
 * \file adaptive_sliding_window_tiling.cpp
 * \brief
 */
#include "common/op_host/op_tiling/tiling_type.h"
#include "log/log.h"
#include "error_util.h"
#include "adaptive_sliding_window_tiling.h"
#include "quant_batch_matmul_v3_tiling_util.h"
#include "../../../op_kernel/arch35/quant_batch_matmul_v3_apt_tiling_key.h"
#include "base_block_calculator.h"

using Ops::NN::MathUtil;
using namespace QuantBatchMatmulV3Arch35TilingKey;
namespace {

constexpr uint64_t BASIC_BLOCK_SIZE_16 = 16;
constexpr uint64_t BASIC_BLOCK_SIZE_512 = 512UL;
constexpr uint64_t BASIC_BLOCK_SIZE_1024 = 1024UL;

constexpr uint32_t UB_ALIGN_SIZE = 32;

constexpr uint32_t SCALER_FACTOR_DEFAULT = 1;
constexpr uint32_t SCALER_FACTOR_B_BIT = 8;
constexpr uint32_t SCALER_FACTOR_M_BIT = 16;
constexpr uint32_t SCALER_FACTOR_N_BIT = 24;

constexpr uint32_t VEC_CORE_GROUP_NUM = 2;

constexpr uint64_t LOAD_BALANCE_THRESHOLD = 1792; // Minimum M/N size to enable outer-axis load balancing.
} // namespace

namespace optiling {

AdaptiveSlidingWindowTiling::AdaptiveSlidingWindowTiling(gert::TilingContext* context)
    : QuantBatchMatmulV3TilingBase(context, false), tilingData_(tilingDataSelf_)
{}

AdaptiveSlidingWindowTiling::AdaptiveSlidingWindowTiling(gert::TilingContext* context,
                                                         DequantBmm::QuantBatchMatmulV3TilingDataParams* out)
    : QuantBatchMatmulV3TilingBase(context, true), tilingData_(out == nullptr ? tilingDataSelf_ : *out)
{}

void AdaptiveSlidingWindowTiling::Reset()
{
    isBf16Opt_ = false;

    if (!isTilingOut_) {
        tilingData_ = DequantBmm::QuantBatchMatmulV3TilingDataParams();
    }
}

ge::graphStatus AdaptiveSlidingWindowTiling::GetShapeAttrsInfo()
{
    if (!inputParams_.initFlag && !isTilingOut_) {
        auto tilingDataCapacity = context_->GetRawTilingData()->GetCapacity();
        OP_TILING_CHECK(
            memset_s(context_->GetRawTilingData()->GetData(), tilingDataCapacity, 0, tilingDataCapacity) != EOK,
            CUBE_INNER_ERR_REPORT(inputParams_.opName, "Failed to clear tiling data."), return ge::GRAPH_FAILED);
    }
    return QuantBatchMatmulV3TilingBase::GetShapeAttrsInfo();
}

ge::graphStatus AdaptiveSlidingWindowTiling::CheckContext()
{
    auto x1Shape = context_->GetInputShape(GetX1Idx());
    auto x1Desc = context_->GetInputDesc(GetX1Idx());
    auto x2Shape = context_->GetInputShape(GetX2Idx());
    auto x2Desc = context_->GetInputDesc(GetX2Idx());
    auto scaleShape = context_->GetInputShape(GetScaleIdx());
    auto scaleDesc = context_->GetInputDesc(GetScaleIdx());
    auto outputShape = context_->GetOutputShape(0);
    auto outputDesc = context_->GetOutputDesc(0);
    auto attrs = context_->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr,
                    CUBE_INNER_ERR_REPORT(inputParams_.opName, "Function context_->GetAttrs() failed!"),
                    return ge::GRAPH_FAILED);
    auto dtypeAttr = attrs->GetAttrPointer<int64_t>(0);

    OPS_CHECK_NULL_WITH_CONTEXT(context_, x1Shape);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, x1Desc);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, x2Shape);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, x2Desc);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, scaleShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, scaleDesc);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, outputShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, outputDesc);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, dtypeAttr);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData());
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData()->GetData());
    return ge::GRAPH_SUCCESS;
}

void AdaptiveSlidingWindowTiling::LoadBalanceDataReset()
{
    adaptiveWin_.mBaseTailSplitCnt = 1UL;
    adaptiveWin_.nBaseTailSplitCnt = 1UL;
    adaptiveWin_.mTailMain = 0UL;
    adaptiveWin_.nTailMain = 0UL;
}

bool AdaptiveSlidingWindowTiling::CheckDtype() const
{
    return QuantBatchMatMulV3TilingUtil::CheckDtype(context_, inputParams_, compileInfo_);
}

bool AdaptiveSlidingWindowTiling::CheckShape(const std::vector<gert::Shape*>& mandtoryShape,
                                             const gert::StorageShape* biasShape,
                                             const gert::StorageShape* pertokenShape,
                                             const std::vector<int64_t>& dimValueOfMKN) const
{
    return QuantBatchMatMulV3TilingUtil::CheckShape(context_, inputParams_, compileInfo_, mandtoryShape, biasShape,
                                                    pertokenShape, dimValueOfMKN);
}

bool AdaptiveSlidingWindowTiling::AnalyzeInputs() { return QuantBatchMatmulV3TilingBase::AnalyzeInputs(); }

ge::graphStatus AdaptiveSlidingWindowTiling::GetPlatformInfo()
{
    if (aicoreParams_.aicNum == 0UL || aicoreParams_.l1Size == 0UL || aicoreParams_.l0cSize == 0UL) {
        OP_LOGE(inputParams_.opName, "CoreNum/L1Size/L0cSize should not be 0. CoreNum: %lu, L1Size: %lu, L0cSize: %lu.",
                aicoreParams_.aicNum, aicoreParams_.l1Size, aicoreParams_.l0cSize);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

bool AdaptiveSlidingWindowTiling::CheckCoreNum() const { return true; }

bool AdaptiveSlidingWindowTiling::AnalyseSlidingWinInfo()
{
    if (!CalcBasicBlock()) {
        OP_LOGE(inputParams_.opName, "Invalid basic block.");
        return false;
    }
    CalcBlockWindowInfo();

    LoadBalanceDataReset();
    OptimizeEdgeBasicBlock();
    AnalyseFullLoadInfo();
    CalcTailRoundBasicBlockSplit();
    return true;
}

ge::graphStatus AdaptiveSlidingWindowTiling::DoOpTiling()
{
    SetBf16Compat();
    if (!CheckCoreNum()) {
        OP_LOGE(inputParams_.opName, "CheckCoreNum failed.");
        return ge::GRAPH_FAILED;
    }
    if (!AnalyseSlidingWinInfo()) {
        OP_LOGE(inputParams_.opName, "DoOpTiling failed.");
        return ge::GRAPH_FAILED;
    }
    if (!CalL1Tiling()) {
        OP_LOGE(inputParams_.opName, "CalL1Tiling failed.");
        return ge::GRAPH_FAILED;
    }
    if (inputParams_.isPertoken) {
        CalcUbTiling();
    }
    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AdaptiveSlidingWindowTiling::DoLibApiTiling()
{
    QuantBatchMatMulV3TilingUtil::SetBasicLibApiTiling(inputParams_, basicTiling_, tilingData_);
    if (inputParams_.isMxPerGroup) {
        tilingData_.matmulTiling.mxTypePara = (qmmv3_tiling_const::SCALER_FACTOR_MIN << SCALER_FACTOR_N_BIT) +
                                              (qmmv3_tiling_const::SCALER_FACTOR_MIN << SCALER_FACTOR_M_BIT);
        if (basicTiling_.scaleFactorA >= qmmv3_tiling_const::SCALER_FACTOR_MIN &&
            basicTiling_.scaleFactorA <= qmmv3_tiling_const::SCALER_FACTOR_MAX &&
            basicTiling_.scaleFactorB >= qmmv3_tiling_const::SCALER_FACTOR_MIN &&
            basicTiling_.scaleFactorB <= qmmv3_tiling_const::SCALER_FACTOR_MAX) {
            tilingData_.matmulTiling.mxTypePara += (basicTiling_.scaleFactorB << SCALER_FACTOR_B_BIT) +
                                                   basicTiling_.scaleFactorA;
        } else {
            tilingData_.matmulTiling.mxTypePara += (SCALER_FACTOR_DEFAULT << SCALER_FACTOR_B_BIT) +
                                                   SCALER_FACTOR_DEFAULT;
        }
    }
    return ge::GRAPH_SUCCESS;
}

uint64_t AdaptiveSlidingWindowTiling::GetBiasMode() const
{
    return QuantBatchMatMulV3TilingUtil::GetBiasMode(inputParams_);
}

uint64_t AdaptiveSlidingWindowTiling::GetKernelType() const
{
    return QuantBatchMatMulV3TilingUtil::GetKernelType(inputParams_, basicTiling_, isBf16Mix_, isAFullLoad_,
                                                       isBFullLoad_, isABFullLoad_);
}

uint64_t AdaptiveSlidingWindowTiling::GetApiLevel(NpuArch) const
{
    return static_cast<uint64_t>(QMMApiLevel::HIGH_LEVEL);
}

uint64_t AdaptiveSlidingWindowTiling::GetTilingKey() const
{
    uint64_t kernelType = GetKernelType();
    return GET_TPL_TILING_KEY(static_cast<uint64_t>(inputParams_.transA), static_cast<uint64_t>(inputParams_.transB),
                              GetBiasMode(), kernelType, GetApiLevel(compileInfo_.npuArch));
}

ge::graphStatus AdaptiveSlidingWindowTiling::GetWorkspaceSize()
{
    workspaceSize_ = inputParams_.libApiWorkSpaceSize;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AdaptiveSlidingWindowTiling::PostTiling()
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
                           GetTilingData(), tilingDataSize_);
    if (ret != EOK) {
        OP_LOGE(context_->GetNodeName(), "Failed to copy memory with memcpy_s, ret=%d.", ret);
        return ge::GRAPH_FAILED;
    }
    context_->SetBlockDim(basicTiling_.usedCoreNum);
    context_->GetRawTilingData()->SetDataSize(tilingDataSize_);
    size_t* workspaces = context_->GetWorkspaceSizes(1); // Set workspace size.
    OPS_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = workspaceSize_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AdaptiveSlidingWindowTiling::CalcUbTiling()
{
    uint64_t ubSize = static_cast<uint64_t>(aicoreParams_.ubSize);
    basicTiling_.ubCalcN = basicTiling_.baseN;
    // UB stores source int32, scale, per-token scale, and output, with double buffering for IO.
    // BF16 output is accounted for with int16-sized storage when estimating buffer usage.
    uint64_t ubCalc = static_cast<uint64_t>(qmmv3_tiling_const::DOUBLE_BUFFER_NUM *
                                            (sizeof(int32_t) + ge::GetSizeByDataType(inputParams_.cDtype))) *
                      static_cast<uint64_t>(basicTiling_.ubCalcN);
    // Reserve per-channel scale space when the quantization mode is not per-tensor.
    if (!inputParams_.isPerTensor) {
        ubSize -= ge::GetSizeByDataType(inputParams_.scaleDtype) * basicTiling_.ubCalcN;
    }
    // Reserve per-token scale space.
    ubCalc += qmmv3_tiling_const::DOUBLE_BUFFER_NUM * ge::GetSizeByDataType(inputParams_.perTokenScaleDtype);
    // Keep the UB allocation 32-byte aligned.
    ubSize -= qmmv3_tiling_const::DOUBLE_BUFFER_NUM *
              (UB_ALIGN_SIZE - ge::GetSizeByDataType(inputParams_.perTokenScaleDtype));
    basicTiling_.ubCalcM = static_cast<uint32_t>(ubSize / ubCalc);
    basicTiling_.ubCalcM = std::min(
        std::min(basicTiling_.ubCalcM, ops::CeilDiv(basicTiling_.baseM, VEC_CORE_GROUP_NUM)),
        ops::CeilDiv(static_cast<uint32_t>(inputParams_.mSize), VEC_CORE_GROUP_NUM));
    return ge::GRAPH_SUCCESS;
}

void AdaptiveSlidingWindowTiling::CalcBlockWindowInfo()
{
    adaptiveWin_.mBlockCnt = ops::CeilDiv(inputParams_.mSize, adaptiveWin_.baseM);
    adaptiveWin_.nBlockCnt = ops::CeilDiv(inputParams_.nSize, adaptiveWin_.baseN);
    adaptiveWin_.totalBlockCnt = GetBatchCoreCnt() * adaptiveWin_.mBlockCnt * adaptiveWin_.nBlockCnt;
    adaptiveWin_.mTail = inputParams_.mSize - (adaptiveWin_.mBlockCnt - 1UL) * adaptiveWin_.baseM;
    adaptiveWin_.nTail = inputParams_.nSize - (adaptiveWin_.nBlockCnt - 1UL) * adaptiveWin_.baseN;
    adaptiveWin_.totalWinCnt = ops::CeilDiv(adaptiveWin_.totalBlockCnt, aicoreParams_.aicNum);
    adaptiveWin_.tailWinBlockCnt = adaptiveWin_.totalBlockCnt % aicoreParams_.aicNum;
}

uint64_t AdaptiveSlidingWindowTiling::GetBatchCoreCnt() const { return 1UL; }

const void* AdaptiveSlidingWindowTiling::GetTilingData() const { return &tilingData_; }

void AdaptiveSlidingWindowTiling::SetBf16Compat()
{
    bool isMix = (inputParams_.scaleDtype != ge::DT_UINT64 && inputParams_.scaleDtype != ge::DT_INT64 &&
                  inputParams_.scaleDtype != ge::DT_FLOAT8_E8M0) &&
                 inputParams_.isPerChannel;
    bool isCompat = (inputParams_.scaleDtype != ge::DT_UINT64 && inputParams_.scaleDtype != ge::DT_INT64 &&
                     inputParams_.scaleDtype != ge::DT_FLOAT8_E8M0) &&
                    inputParams_.isPerTensor && inputParams_.aDtype == ge::DT_INT8 && inputParams_.hasBias &&
                    (inputParams_.biasDtype == ge::DT_FLOAT || inputParams_.biasDtype == ge::DT_BF16);
    bool isFp8OrHif8TTBiasMix = IsFp8OrHif8TTFloatBiasMix(inputParams_);
    isBf16Mix_ = (isMix || isCompat || isFp8OrHif8TTBiasMix) && (inputParams_.cDtype != ge::DT_INT32);
}

bool AdaptiveSlidingWindowTiling::IsMxKOdd() const
{
    return inputParams_.scaleDtype == ge::DT_FLOAT8_E8M0 &&
           ops::CeilDiv(inputParams_.kSize, qmmv3_tiling_const::MX_GROUP_SIZE) %
                   qmmv3_tiling_const::MXFP_MULTI_BASE_SIZE !=
               0;
}

bool AdaptiveSlidingWindowTiling::IsMxBackwardTrans() const
{
    return inputParams_.scaleDtype == ge::DT_FLOAT8_E8M0 && (inputParams_.transA || !inputParams_.transB);
}

bool AdaptiveSlidingWindowTiling::CheckBiasAndScale(uint64_t baseN, uint64_t dbL0c) const
{
    // INT32 bias consumes a 4 KB BT tile, so baseN must stay within 1024 without DB or 512 with DB.
    // UINT64/INT64 scale currently does not further limit baseN because libapi will tile a 512-wide scale.
    uint64_t maxBiasBaseN = dbL0c == 1UL ? BASIC_BLOCK_SIZE_1024 : BASIC_BLOCK_SIZE_512;
    // FB uses a 2 KB tile here, and the remaining split is handled consistently with libapi/TBE tiling.
    uint64_t maxScaleBaseN = dbL0c == 1UL ? BASIC_BLOCK_SIZE_512 : qmmv3_tiling_const::BASIC_BLOCK_SIZE_256;
    bool isBiasInvalid = inputParams_.hasBias && inputParams_.biasDtype != ge::DT_BF16 && baseN > maxBiasBaseN;
    bool isUbQuant = inputParams_.cDtype == ge::DT_BF16 || inputParams_.isPertoken;
    bool isScaleInvalid = !isUbQuant && baseN > maxScaleBaseN;
    return !(isBiasInvalid || isScaleInvalid);
}

void AdaptiveSlidingWindowTiling::SetTilingData()
{
    QuantBatchMatMulV3TilingUtil::SetBasicTilingData(inputParams_, basicTiling_, tilingData_);
    tilingData_.adaptiveSlidingWin.mTailTile = adaptiveWin_.mTailTile;
    tilingData_.adaptiveSlidingWin.nTailTile = adaptiveWin_.nTailTile;
    tilingData_.adaptiveSlidingWin.mBaseTailSplitCnt = static_cast<uint32_t>(adaptiveWin_.mBaseTailSplitCnt);
    tilingData_.adaptiveSlidingWin.nBaseTailSplitCnt = static_cast<uint32_t>(adaptiveWin_.nBaseTailSplitCnt);
    tilingData_.adaptiveSlidingWin.mTailMain = static_cast<uint32_t>(adaptiveWin_.mTailMain);
    tilingData_.adaptiveSlidingWin.nTailMain = static_cast<uint32_t>(adaptiveWin_.nTailMain);
}

uint32_t AdaptiveSlidingWindowTiling::CalUsedCoreNum()
{
    if (adaptiveWin_.totalWinCnt > 1UL || adaptiveWin_.tailWinBlockCnt == 0UL) {
        return aicoreParams_.aicNum;
    }

    return static_cast<uint32_t>(adaptiveWin_.tailWinBlockCnt * adaptiveWin_.mTailTile * adaptiveWin_.nTailTile);
}

uint32_t AdaptiveSlidingWindowTiling::CalUsedCoreNum(uint32_t mTile, uint32_t nTile)
{
    return mTile * nTile * static_cast<uint32_t>(adaptiveWin_.tailWinBlockCnt);
}

bool AdaptiveSlidingWindowTiling::IsInValidWeighNzTailSplit(uint64_t splitCnt, bool isPreSplit) const
{
    if (inputParams_.bFormat != ge::FORMAT_FRACTAL_NZ ||
        (!isAFullLoad_ && ((isPreSplit && adaptiveWin_.mTail >= adaptiveWin_.nTail) ||
                           (!isPreSplit && adaptiveWin_.mTail < adaptiveWin_.nTail)))) {
        return false;
    }

    uint64_t tailN = adaptiveWin_.baseN / splitCnt;
    return tailN % GetShapeWithDataType(qmmv3_tiling_const::L1_ALIGN_SIZE, inputParams_.bDtype) != 0UL;
}

uint64_t AdaptiveSlidingWindowTiling::GetTailBasicBlockSplitMax(bool isMSplit, uint64_t tileMax,
                                                                uint64_t splitSize) const
{
    const uint64_t baseMAlignSize = (inputParams_.isPerBlock ||
                                     (inputParams_.bFormat == ge::FORMAT_FRACTAL_NZ && inputParams_.transA &&
                                      adaptiveWin_.totalWinCnt == 1UL)) ?
                                        qmmv3_tiling_const::L2_ALIGN_SIZE :
                                        qmmv3_tiling_const::L1_ALIGN_SIZE;
    const uint64_t baseNAlignSize = inputParams_.isPerBlock ? qmmv3_tiling_const::L2_ALIGN_SIZE :
                                                              qmmv3_tiling_const::L1_ALIGN_SIZE;
    const uint64_t splitAlignNum = isMSplit ? (inputParams_.transA ?
                                                   GetShapeWithDataType(baseMAlignSize, inputParams_.aDtype) :
                                                   qmmv3_tiling_const::CUBE_BLOCK) :
                                              (!inputParams_.transB ?
                                                   GetShapeWithDataType(baseNAlignSize, inputParams_.bDtype) :
                                                   qmmv3_tiling_const::CUBE_BLOCK);
    return std::min(tileMax, MathUtil::CeilDivision(splitSize, splitAlignNum));
}

bool AdaptiveSlidingWindowTiling::CanIncreaseTailSplit(bool isPreSplitM, bool isPreSplit, uint64_t preSplit,
                                                       uint64_t secSplit, uint64_t splitMax)
{
    const uint64_t nextPreSplit = isPreSplit ? preSplit + 1UL : preSplit;
    const uint64_t nextSecSplit = isPreSplit ? secSplit : secSplit + 1UL;
    const uint64_t mTile = isPreSplitM ? nextPreSplit : nextSecSplit;
    const uint64_t nTile = isPreSplitM ? nextSecSplit : nextPreSplit;
    return (isPreSplit ? preSplit : secSplit) < splitMax &&
           CalUsedCoreNum(static_cast<uint32_t>(mTile), static_cast<uint32_t>(nTile)) <= aicoreParams_.aicNum;
}

uint64_t AdaptiveSlidingWindowTiling::GetTailSplitState(bool isPreSplitM, bool isPreSplit, uint64_t split,
                                                        uint64_t splitSize) const
{
    const bool isMSplit = isPreSplit == isPreSplitM;
    const bool needWeightNzCheck = !inputParams_.isPerBlock && inputParams_.bFormat == ge::FORMAT_FRACTAL_NZ &&
                                   (isAFullLoad_ || !isMSplit);
    // 0: invalid split, 1: valid split, 2: valid and preferred-aligned split.
    if (needWeightNzCheck && IsInValidWeighNzTailSplit(split, isPreSplit)) {
        return 0UL;
    }

    const bool isInnerAxis = isMSplit ? inputParams_.transA : !inputParams_.transB;
    if (!isInnerAxis || adaptiveWin_.totalWinCnt != 1UL) {
        return 1UL;
    }

    const uint64_t splitAlignNum = isMSplit ? GetShapeWithDataType(qmmv3_tiling_const::BASIC_BLOCK_SIZE_32,
                                                                   inputParams_.aDtype) :
                                              GetShapeWithDataType(qmmv3_tiling_const::BASIC_BLOCK_SIZE_32,
                                                                   inputParams_.bDtype);
    return MathUtil::CeilDivision(splitSize, split) % splitAlignNum == 0UL ? 2UL : 1UL;
}

void AdaptiveSlidingWindowTiling::CalcTailBasicBlockSplit(bool isPreSplitM, uint64_t preSplitMax, uint64_t secSplitMax,
                                                          uint64_t preSplitSize, uint64_t secSplitSize)
{
    uint64_t preSplitValid = 1UL;
    uint64_t secSplitValid = 1UL;
    uint64_t preSplitAlignValid = 1UL;
    uint64_t secSplitAlignValid = 1UL;

    uint64_t preSplit = 1UL;
    uint64_t secSplit = 1UL;
    while (CanIncreaseTailSplit(isPreSplitM, true, preSplit, secSplit, preSplitMax) ||
           CanIncreaseTailSplit(isPreSplitM, false, preSplit, secSplit, secSplitMax)) {
        if (CanIncreaseTailSplit(isPreSplitM, true, preSplit, secSplit, preSplitMax)) {
            ++preSplit;
            // 0: invalid split, 1: valid split, 2: valid and preferred-aligned split.
            const uint64_t splitState = GetTailSplitState(isPreSplitM, true, preSplit, preSplitSize);
            if (splitState > 0UL) {
                preSplitValid = preSplit;
            }
            if (splitState > 1UL) {
                preSplitAlignValid = preSplit;
            }
        }
        if (CanIncreaseTailSplit(isPreSplitM, false, preSplit, secSplit, secSplitMax)) {
            ++secSplit;
            // 0: invalid split, 1: valid split, 2: valid and preferred-aligned split.
            const uint64_t splitState = GetTailSplitState(isPreSplitM, false, secSplit, secSplitSize);
            if (splitState > 0UL) {
                secSplitValid = secSplit;
            }
            if (splitState > 1UL) {
                secSplitAlignValid = secSplit;
            }
        }
    }
    const uint64_t preTile = preSplitAlignValid != 1UL ? preSplitAlignValid : preSplitValid;
    const uint64_t secTile = secSplitAlignValid != 1UL ? secSplitAlignValid : secSplitValid;
    if (isPreSplitM) {
        adaptiveWin_.mTailTile = preTile;
        adaptiveWin_.nTailTile = secTile;
    } else {
        adaptiveWin_.mTailTile = secTile;
        adaptiveWin_.nTailTile = preTile;
    }
}

bool AdaptiveSlidingWindowTiling::CalcBasicBlock()
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

void AdaptiveSlidingWindowTiling::CalcTailBasicBlock()
{
    if (adaptiveWin_.tailWinBlockCnt == 0UL) {
        return;
    }
    const bool isPreSplitM = adaptiveWin_.mTail >= adaptiveWin_.nTail;
    const uint64_t tileMax = aicoreParams_.aicNum / adaptiveWin_.tailWinBlockCnt;
    const uint64_t tailBaseM = adaptiveWin_.mBaseTailSplitCnt != 1UL ? adaptiveWin_.mTailMain : adaptiveWin_.baseM;
    const uint64_t mTailSplitSize = std::min(inputParams_.mSize, tailBaseM);
    const uint64_t nTailSplitSize = std::min(inputParams_.nSize, adaptiveWin_.baseN);
    const uint64_t mTileMax = GetTailBasicBlockSplitMax(true, tileMax, mTailSplitSize);
    const uint64_t nTileMax = GetTailBasicBlockSplitMax(false, tileMax, nTailSplitSize);
    CalcTailBasicBlockSplit(isPreSplitM, isPreSplitM ? mTileMax : nTileMax, isPreSplitM ? nTileMax : mTileMax,
                            isPreSplitM ? mTailSplitSize : nTailSplitSize,
                            isPreSplitM ? nTailSplitSize : mTailSplitSize);
}

void AdaptiveSlidingWindowTiling::CalcTailBasicBlockAfullLoad()
{
    adaptiveWin_.mTailTile = 1UL;
    uint64_t nTile = 1UL;
    uint64_t nTileValid = 1UL;
    if (adaptiveWin_.tailWinBlockCnt != 0UL) {
        while (CalUsedCoreNum(adaptiveWin_.mTailTile, (nTile + 1UL)) <= aicoreParams_.aicNum &&
               adaptiveWin_.baseN / (nTile + 1UL) >= qmmv3_tiling_const::CUBE_BLOCK) {
            nTile += 1UL;
            if (IsInValidWeighNzTailSplit(nTile, true)) {
                continue;
            }
            nTileValid = nTile;
        }
    }
    adaptiveWin_.nTailTile = nTileValid;
}

void AdaptiveSlidingWindowTiling::CalcTailBasicBlockBfullLoad()
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

void AdaptiveSlidingWindowTiling::CalcTailBasicBlock4MmadS8S4()
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

void AdaptiveSlidingWindowTiling::CalcTailRoundBasicBlockSplit()
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

void AdaptiveSlidingWindowTiling::GetOuterMAxisTailCnt(uint64_t& baseTailSplitCnt, uint64_t& tailMain)
{
    uint64_t mTailSize = inputParams_.mSize % adaptiveWin_.baseM;
    uint64_t baseTailCntMax = std::min((adaptiveWin_.baseM - mTailSize) / BASIC_BLOCK_SIZE_16, adaptiveWin_.mBlockCnt);
    uint64_t windowSize = std::min(qmmv3_tiling_const::WINDOW_LEN, adaptiveWin_.mBlockCnt);
    uint64_t mainWindowNum = adaptiveWin_.mBlockCnt / windowSize - 1UL;
    uint64_t tailWindowSize = adaptiveWin_.mBlockCnt - mainWindowNum * windowSize;
    uint64_t perfRes = (mainWindowNum + 1UL) * adaptiveWin_.baseM;
    uint64_t mergeWindowNum = 1UL;

    for (uint64_t mergeLen = tailWindowSize - 1UL; mergeLen < baseTailCntMax;
         mergeLen += windowSize, ++mergeWindowNum) {
        uint64_t newTailMain = MathUtil::Align(
            MathUtil::CeilDivision((mergeLen * adaptiveWin_.baseM + mTailSize), mergeLen + 1UL), BASIC_BLOCK_SIZE_16);
        uint64_t curPerf = (mainWindowNum + 1UL - mergeWindowNum) * adaptiveWin_.baseM + mergeWindowNum * newTailMain;
        if (curPerf <= perfRes) {
            perfRes = curPerf;
            tailMain = newTailMain;
            baseTailSplitCnt = mergeLen + 1UL;
        }
    }
}

void AdaptiveSlidingWindowTiling::CalculateCurrentPerf(uint64_t mergeLen, uint64_t nTail, uint64_t& newTailMain,
                                                       uint64_t& curPerf)
{
    const uint64_t totalWindows = MathUtil::CeilDivision(adaptiveWin_.nBlockCnt * adaptiveWin_.mBlockCnt,
                                                         aicoreParams_.aicNum);
    newTailMain = MathUtil::Align(MathUtil::CeilDivision((mergeLen * adaptiveWin_.baseN + nTail), mergeLen + 1UL),
                                  BASIC_BLOCK_SIZE_16);
    uint64_t newTailLast = mergeLen * (adaptiveWin_.baseN - newTailMain) + nTail;
    uint64_t newMainRound = 0UL;
    uint64_t newTailRound = 0UL;

    if (mergeLen < adaptiveWin_.nBlockCnt - 1UL) {
        newMainRound = MathUtil::CeilDivision((adaptiveWin_.nBlockCnt - 1UL - mergeLen) * adaptiveWin_.mBlockCnt +
                                                  (mergeLen + 1UL) * adaptiveWin_.mBlockCnt % aicoreParams_.aicNum,
                                              aicoreParams_.aicNum);
    }
    if (mergeLen > 0UL) {
        newTailRound = std::min(
            MathUtil::CeilDivision(mergeLen * adaptiveWin_.mBlockCnt + adaptiveWin_.mBlockCnt % aicoreParams_.aicNum,
                                   aicoreParams_.aicNum),
            totalWindows - newMainRound);
    }

    curPerf = newMainRound * adaptiveWin_.baseN + newTailRound * newTailMain +
              (totalWindows - newMainRound - newTailRound) * newTailLast;
}

void AdaptiveSlidingWindowTiling::GetOuterNAxisTailCnt(uint64_t& baseTailSplitCnt, uint64_t& tailMain)
{
    uint64_t baseN = adaptiveWin_.baseN;
    uint64_t nTail = inputParams_.nSize % baseN;
    uint64_t blockCnt = adaptiveWin_.nBlockCnt * adaptiveWin_.mBlockCnt;
    uint64_t totalWindows = MathUtil::CeilDivision(blockCnt, aicoreParams_.aicNum);

    uint64_t mainWindows = MathUtil::CeilDivision(
        (adaptiveWin_.nBlockCnt - 1UL) * adaptiveWin_.mBlockCnt + adaptiveWin_.mBlockCnt % aicoreParams_.aicNum,
        aicoreParams_.aicNum);

    if (blockCnt <= aicoreParams_.aicNum || (adaptiveWin_.mBlockCnt % aicoreParams_.aicNum == 0UL &&
                                             (adaptiveWin_.nBlockCnt % qmmv3_tiling_const::WINDOW_LEN == 0UL ||
                                              qmmv3_tiling_const::WINDOW_LEN % adaptiveWin_.nBlockCnt == 0UL))) {
        mainWindows = totalWindows;
    }
    uint64_t tailWindows = totalWindows - mainWindows;
    uint64_t perfRes = mainWindows * baseN + tailWindows * nTail;

    uint64_t baseTailCntMax = std::min((baseN - nTail) / BASIC_BLOCK_SIZE_16, adaptiveWin_.nBlockCnt);
    for (uint64_t mergeLen = 1UL; mergeLen < baseTailCntMax; ++mergeLen) {
        uint64_t newTailMain = 0UL;
        uint64_t curPerf = 0UL;
        CalculateCurrentPerf(mergeLen, nTail, newTailMain, curPerf);
        if (curPerf < perfRes) {
            perfRes = curPerf;
            tailMain = newTailMain;
            baseTailSplitCnt = mergeLen + 1UL;
        }
    }
}

void AdaptiveSlidingWindowTiling::OptimizeEdgeBasicBlock()
{
    if (compileInfo_.supportMmadS8S4 || (inputParams_.transA && !inputParams_.transB) ||
        (inputParams_.isPerBlock && inputParams_.groupSizeM != 1UL)) {
        return;
    }

    if (adaptiveWin_.mBlockCnt == 1UL || adaptiveWin_.nBlockCnt == 1UL) {
        return;
    }

    uint64_t mBaseTail = static_cast<uint64_t>(inputParams_.mSize % adaptiveWin_.baseM);
    uint64_t nBaseTail = static_cast<uint64_t>(inputParams_.nSize % adaptiveWin_.baseN);
    bool isMxfp4 = inputParams_.isMxPerGroup && (inputParams_.aDtype == ge::DT_FLOAT4_E2M1);
    bool balanceAfterFixp = (inputParams_.kSize < BASIC_BLOCK_SIZE_1024 ||
                             (inputParams_.kSize == BASIC_BLOCK_SIZE_1024 &&
                              adaptiveWin_.nBlockCnt >= 8)); // With enough N tiles, imbalance impact is negligible.
    bool isInnerAxisAlign = GetSizeWithDataType(inputParams_.kSize, inputParams_.aDtype) %
                                qmmv3_tiling_const::MTE2_ADDRESS_ALIGN_SIZE ==
                            0UL;
    if (mBaseTail > 0UL && !inputParams_.transA &&
        (isInnerAxisAlign || (inputParams_.mSize >= LOAD_BALANCE_THRESHOLD && !isMxfp4))) {
        GetOuterMAxisTailCnt(adaptiveWin_.mBaseTailSplitCnt, adaptiveWin_.mTailMain);
    }
    if (nBaseTail > 0UL && inputParams_.transB && !balanceAfterFixp && !inputParams_.isPerBlock && !isMxfp4 &&
        (isInnerAxisAlign || (inputParams_.nSize >= LOAD_BALANCE_THRESHOLD))) {
        GetOuterNAxisTailCnt(adaptiveWin_.nBaseTailSplitCnt, adaptiveWin_.nTailMain);
    }
}

} // namespace optiling
