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
 * \file qbmm_streamk_tiling.cpp
 * \brief
 */
#include "qbmm_streamk_tiling.h"

#include <algorithm>
#include <limits>
#include <vector>

#include "base_block_calculator.h"
#include "common/op_host/op_tiling/tiling_type.h"
#include "error_util.h"
#include "l1_tiling_data_calculator.h"
#include "log/log.h"
#include "op_host/tiling_templates_registry.h"
#include "quant_batch_matmul_v3_tiling_strategy.h"

using namespace QuantBatchMatmulV3Arch35TilingKey;

namespace {
constexpr uint64_t STREAMK_RPC_WORKSPACE_MB = 16UL;
constexpr uint32_t SMALL_MN_EXPAND_RATIO = 3U;
constexpr uint64_t ASWT_A_FULL_LOAD_SINGLE_CORE_A_SCALER = 2UL;
constexpr uint64_t ASWT_A_FULL_LOAD_WINDOW_LEN = 4UL;
constexpr uint64_t UINT64_SATURATED = std::numeric_limits<uint64_t>::max();

using optiling::qmmv3_tiling_const::BASIC_BLOCK_SIZE_256;
using optiling::qmmv3_tiling_const::CORE_RATIO;
using optiling::qmmv3_tiling_const::CUBE_BLOCK;
using optiling::qmmv3_tiling_const::CUBE_REDUCE_BLOCK;
using optiling::qmmv3_tiling_const::DATA_SIZE_L0C;
using optiling::qmmv3_tiling_const::DOUBLE_BUFFER_NUM;
using optiling::qmmv3_tiling_const::L1_ALIGN_SIZE;
using optiling::qmmv3_tiling_const::L1_FOUR_BUFFER;
using optiling::qmmv3_tiling_const::L1_TWO_BUFFER;
using optiling::qmmv3_tiling_const::L2_ALIGN_SIZE;
using optiling::qmmv3_tiling_const::MXFP_DIVISOR_SIZE;
using optiling::qmmv3_tiling_const::MXFP_MULTI_BASE_SIZE;
using optiling::qmmv3_tiling_const::MX_GROUP_SIZE;
using optiling::qmmv3_tiling_const::NUM_HALF;
using optiling::qmmv3_tiling_const::SCALER_FACTOR_MAX;
using optiling::qmmv3_tiling_const::SCALER_FACTOR_MIN;
// Benefit thresholds are tuned for DAV_3510/Ascend950 QBMM MX StreamK. Sizes are bytes, time is ns, and permille
// values use 1000 as the denominator. The transfer model uses 4 TB/s for MTE2/AIV paths; the fixed overhead and
// safety margin come from ops-test-kit single-op profiling and cover StreamK scheduling/reduce overhead.
// The StreamK RPC workspace is sized in MiB and is reserved for the multi-block reduce synchronization path.
constexpr uint64_t STREAMK_RPC_WORKSPACE_SIZE = STREAMK_RPC_WORKSPACE_MB * 1024UL * 1024UL;
constexpr uint64_t BENEFIT_MIN_K_THRESHOLD = 4096UL;
constexpr uint64_t BENEFIT_TA0TB1_MIN_MN_THRESHOLD = 32UL;
constexpr uint64_t BENEFIT_TA0TB1_MIN_K_THRESHOLD = 8192UL;
constexpr uint64_t BENEFIT_MTE2_MIN_SAVED_BYTES = 2UL * 1024UL * 1024UL;
constexpr uint64_t BENEFIT_MTE2_MIN_SAVED_PERMILLE = 330UL;
constexpr uint64_t BENEFIT_MTE2_BW_BYTES_PER_NS = 4000UL;
constexpr uint64_t BENEFIT_AIV_BW_BYTES_PER_NS = 4000UL;
constexpr uint64_t BENEFIT_STREAMK_FIXED_OVERHEAD_NS = 500UL;
constexpr uint64_t BENEFIT_STREAMK_MIN_MARGIN_NS = 300UL;
constexpr uint64_t BENEFIT_TIME_SAFETY_NUM = 6UL;
constexpr uint64_t BENEFIT_TIME_SAFETY_DEN = 5UL;

const std::vector<int32_t> supportedNpuArch = {static_cast<int32_t>(NpuArch::DAV_3510)};
constexpr int32_t TILING_PRIORITY = optiling::strategy::MX_STREAMK;

uint64_t SafeCeilDiv(uint64_t value, uint64_t factor)
{
    return factor == 0UL ? 0UL : ops::CeilDiv(value, factor);
}

uint64_t CalcPermille(uint64_t numerator, uint64_t denominator)
{
    if (denominator == 0UL) {
        return 0UL;
    }
    auto permille = static_cast<unsigned __int128>(numerator) * 1000UL / denominator;
    return permille > UINT64_SATURATED ? UINT64_SATURATED : static_cast<uint64_t>(permille);
}

bool CheckedMul(uint64_t lhs, uint64_t rhs, uint64_t& result)
{
    if (lhs != 0UL && rhs > UINT64_SATURATED / lhs) {
        result = UINT64_SATURATED;
        return false;
    }
    result = lhs * rhs;
    return true;
}

bool CheckedAdd(uint64_t lhs, uint64_t rhs, uint64_t& result)
{
    if (rhs > UINT64_SATURATED - lhs) {
        result = UINT64_SATURATED;
        return false;
    }
    result = lhs + rhs;
    return true;
}

uint64_t SaturatingMul(uint64_t lhs, uint64_t rhs)
{
    uint64_t result = 0UL;
    (void)CheckedMul(lhs, rhs, result);
    return result;
}

uint64_t SaturatingAdd(uint64_t lhs, uint64_t rhs)
{
    uint64_t result = 0UL;
    (void)CheckedAdd(lhs, rhs, result);
    return result;
}

uint64_t SaturatingMulDiv(uint64_t value, uint64_t numerator, uint64_t denominator)
{
    if (denominator == 0UL) {
        return UINT64_SATURATED;
    }
    auto result = static_cast<unsigned __int128>(value) * numerator / denominator;
    return result > UINT64_SATURATED ? UINT64_SATURATED : static_cast<uint64_t>(result);
}

uint64_t CalcTransferTimeNs(uint64_t bytes, uint64_t bytesPerNs)
{
    return SafeCeilDiv(bytes, bytesPerNs);
}

bool CalcMnCnt(uint64_t batchC, uint64_t mCnt, uint64_t nCnt, uint64_t& mnCnt)
{
    uint64_t blockCnt = 0UL;
    return CheckedMul(mCnt, nCnt, blockCnt) && CheckedMul(batchC, blockCnt, mnCnt);
}

uint64_t GetStreamKTailMnCnt(uint64_t mnCnt, uint64_t coreNum)
{
    if (coreNum == 0UL || mnCnt == 0UL) {
        return 0UL;
    }
    return mnCnt < coreNum ? mnCnt : mnCnt % coreNum;
}

uint64_t GetDtypeBytes(ge::DataType dtype)
{
    if (dtype == ge::DT_INT4 || dtype == ge::DT_FLOAT4_E2M1 || dtype == ge::DT_FLOAT4_E1M2) {
        return 1UL;
    }
    return static_cast<uint64_t>(ge::GetSizeByDataType(dtype));
}

uint64_t GetShapeWithDataTypeLocal(uint64_t size, ge::DataType dtype)
{
    if (dtype == ge::DT_INT4 || dtype == ge::DT_FLOAT4_E2M1 || dtype == ge::DT_FLOAT4_E1M2) {
        return size + size;
    }
    uint64_t dtypeSize = GetDtypeBytes(dtype);
    return dtypeSize == 0UL ? 0UL : size / dtypeSize;
}

uint64_t GetSizeWithDataTypeLocal(uint64_t shape, ge::DataType dtype)
{
    if (dtype == ge::DT_INT4 || dtype == ge::DT_FLOAT4_E2M1 || dtype == ge::DT_FLOAT4_E1M2) {
        return shape / 2UL + shape % 2UL;
    }
    uint64_t dtypeSize = static_cast<uint64_t>(ge::GetSizeByDataType(dtype));
    return dtypeSize == 0UL ? 0UL : SaturatingMul(shape, dtypeSize);
}

uint64_t GetBenefitBaseMAlign(const optiling::QuantBatchMatmulInfo& inputParams)
{
    return inputParams.transA ? GetShapeWithDataTypeLocal(L1_ALIGN_SIZE, inputParams.aDtype) : CUBE_BLOCK;
}

uint64_t GetBenefitBaseNAlign(const optiling::QuantBatchMatmulInfo& inputParams)
{
    return inputParams.transB ? CUBE_BLOCK : GetShapeWithDataTypeLocal(L1_ALIGN_SIZE, inputParams.bDtype);
}

uint64_t GetBenefitBaseKAlign(const optiling::QuantBatchMatmulInfo& inputParams)
{
    return inputParams.isMxPerGroup ? MXFP_DIVISOR_SIZE : GetShapeWithDataTypeLocal(CUBE_REDUCE_BLOCK,
                                                                                    inputParams.aDtype);
}

std::vector<uint64_t> BuildBaseCandidates(uint64_t shape, uint64_t align, uint64_t coreNum)
{
    std::vector<uint64_t> bases;
    if (shape == 0UL || align == 0UL || coreNum == 0UL) {
        return bases;
    }
    uint64_t shapeAlign = ops::CeilAlign(shape, align);
    auto addBase = [&bases, shapeAlign](uint64_t base) {
        if (base != 0UL && base <= shapeAlign) {
            bases.push_back(base);
        }
    };
    addBase(ops::CeilAlign(std::min(shape, BASIC_BLOCK_SIZE_256), align));
    addBase(align);
    for (uint64_t tileCnt = 1UL; tileCnt <= coreNum; ++tileCnt) {
        addBase(ops::CeilAlign(SafeCeilDiv(shape, tileCnt), align));
    }
    std::sort(bases.begin(), bases.end());
    bases.erase(std::unique(bases.begin(), bases.end()), bases.end());
    return bases;
}

struct BenefitCandidate {
    uint64_t baseM = 0UL;
    uint64_t baseN = 0UL;
    uint64_t baseK = 0UL;
    uint64_t mCnt = 0UL;
    uint64_t nCnt = 0UL;
    uint64_t mnCnt = 0UL;
    uint64_t singleCoreK = 0UL;
    uint64_t streamKCnt = 1UL;
    uint64_t splitK = 1UL;
    uint64_t mte2Bytes = 0UL;
};

struct ActualStreamKShape {
    uint64_t coreNum = 0UL;
    uint64_t baseMAlign = 0UL;
    uint64_t baseNAlign = 0UL;
    uint64_t baseKAlign = 0UL;
    uint64_t baseM = 0UL;
    uint64_t baseN = 0UL;
    uint64_t mCnt = 0UL;
    uint64_t nCnt = 0UL;
    uint64_t mnCnt = 0UL;
    uint64_t streamKCnt = 1UL;
    uint64_t singleCoreK = 0UL;
};

struct AswtShadowInfo {
    uint64_t baseM = 0UL;
    uint64_t baseN = 0UL;
    uint64_t baseK = 0UL;
    uint64_t mCnt = 0UL;
    uint64_t nCnt = 0UL;
    uint64_t mnCnt = 0UL;
    uint64_t mte2Bytes = 0UL;
};

bool AddMte2TensorBytes(uint64_t repeatCnt, uint64_t outerDim, uint64_t innerDim, ge::DataType dtype, uint64_t& total)
{
    uint64_t elementCnt = 0UL;
    if (!CheckedMul(outerDim, innerDim, elementCnt)) {
        return false;
    }
    uint64_t bytes = GetSizeWithDataTypeLocal(elementCnt, dtype);
    if (bytes == UINT64_SATURATED || !CheckedMul(repeatCnt, bytes, bytes)) {
        return false;
    }
    return CheckedAdd(total, bytes, total);
}

uint64_t EstimateMte2Bytes(const optiling::QuantBatchMatmulInfo& inputParams, uint64_t mCnt, uint64_t nCnt)
{
    uint64_t scaleBytes = GetDtypeBytes(inputParams.scaleDtype);
    uint64_t pertokenScaleBytes = GetDtypeBytes(inputParams.perTokenScaleDtype);
    if (scaleBytes == 0UL || pertokenScaleBytes == 0UL) {
        return 0UL;
    }

    uint64_t scaleK = SaturatingMul(SafeCeilDiv(inputParams.kSize, MXFP_DIVISOR_SIZE), MXFP_MULTI_BASE_SIZE);
    if (scaleK == UINT64_SATURATED) {
        return UINT64_SATURATED;
    }

    uint64_t totalBytes = 0UL;
    bool ok = AddMte2TensorBytes(nCnt, inputParams.mSize, inputParams.kSize, inputParams.aDtype, totalBytes) &&
              AddMte2TensorBytes(mCnt, inputParams.nSize, inputParams.kSize, inputParams.bDtype, totalBytes) &&
              AddMte2TensorBytes(nCnt, inputParams.mSize, scaleK, inputParams.perTokenScaleDtype, totalBytes) &&
              AddMte2TensorBytes(mCnt, inputParams.nSize, scaleK, inputParams.scaleDtype, totalBytes);
    return ok ? totalBytes : UINT64_SATURATED;
}

bool CalcAswtShadowInfo(const optiling::QuantBatchMatmulInfo& inputParams,
                        const optiling::QuantBatchMatmulV3CompileInfo& compileInfo, AswtShadowInfo& info)
{
    if (compileInfo.aicNum == 0UL) {
        return false;
    }

    optiling::BaseBlockMode mode = compileInfo.supportMmadS8S4 ? optiling::BaseBlockMode::MMAD_S8S4 :
                                                                 optiling::BaseBlockMode::DEFAULT;
    optiling::BaseBlockCalculator calculator(inputParams, compileInfo, 1UL);
    if (!calculator.Compute(mode)) {
        return false;
    }

    const optiling::BaseBlockRes& baseBlockRes = calculator.GetOutput();
    if (baseBlockRes.baseM == 0UL || baseBlockRes.baseN == 0UL) {
        return false;
    }

    info.baseM = baseBlockRes.baseM;
    info.baseN = baseBlockRes.baseN;
    info.baseK = baseBlockRes.baseK;
    info.mCnt = SafeCeilDiv(inputParams.mSize, info.baseM);
    info.nCnt = SafeCeilDiv(inputParams.nSize, info.baseN);
    if (!CalcMnCnt(1UL, info.mCnt, info.nCnt, info.mnCnt)) {
        return false;
    }
    info.mte2Bytes = EstimateMte2Bytes(inputParams, info.mCnt, info.nCnt);
    return info.mnCnt != 0UL && info.mte2Bytes != 0UL && info.mte2Bytes != UINT64_SATURATED;
}

bool IsAswtAFullLoad(const optiling::QuantBatchMatmulInfo& inputParams,
                     const optiling::QuantBatchMatmulV3CompileInfo& compileInfo, const AswtShadowInfo& aswtInfo)
{
    if (inputParams.batchC != 1UL || compileInfo.aicNum == 0UL || aswtInfo.mCnt == 0UL ||
        aswtInfo.mnCnt == 0UL) {
        return false;
    }

    uint64_t singleCoreAElements = 0UL;
    if (!CheckedMul(aswtInfo.baseM, inputParams.kSize, singleCoreAElements)) {
        return false;
    }
    uint64_t singleCoreASize = GetSizeWithDataTypeLocal(singleCoreAElements, inputParams.aDtype);
    return singleCoreASize != UINT64_SATURATED &&
           singleCoreASize <= compileInfo.l1Size / ASWT_A_FULL_LOAD_SINGLE_CORE_A_SCALER &&
           aswtInfo.mCnt < ASWT_A_FULL_LOAD_WINDOW_LEN && compileInfo.aicNum % aswtInfo.mCnt == 0UL &&
           aswtInfo.mnCnt > compileInfo.aicNum;
}

bool BuildBenefitCandidate(const optiling::QuantBatchMatmulInfo& inputParams,
                           const optiling::QuantBatchMatmulV3CompileInfo& compileInfo, uint64_t baseM,
                           uint64_t baseN, BenefitCandidate& candidate)
{
    if (baseM == 0UL || baseN == 0UL || compileInfo.aicNum == 0UL) {
        return false;
    }
    uint64_t baseKAlign = GetBenefitBaseKAlign(inputParams);
    if (baseKAlign == 0UL) {
        return false;
    }

    uint64_t kValueMax = GetShapeWithDataTypeLocal(compileInfo.l0aSize / DOUBLE_BUFFER_NUM, inputParams.aDtype) /
                         std::max(baseM, baseN);
    kValueMax = ops::FloorAlign(kValueMax, baseKAlign);
    if (kValueMax < baseKAlign) {
        return false;
    }

    uint64_t kValueAlign = ops::CeilAlign(inputParams.kSize, baseKAlign);
    candidate.baseM = baseM;
    candidate.baseN = baseN;
    candidate.baseK = std::min(kValueAlign, kValueMax);
    candidate.mCnt = SafeCeilDiv(inputParams.mSize, baseM);
    candidate.nCnt = SafeCeilDiv(inputParams.nSize, baseN);
    if (!CalcMnCnt(inputParams.batchC, candidate.mCnt, candidate.nCnt, candidate.mnCnt)) {
        return false;
    }
    candidate.singleCoreK = inputParams.kSize;
    candidate.streamKCnt = 1UL;
    candidate.splitK = 1UL;
    candidate.mte2Bytes = EstimateMte2Bytes(inputParams, candidate.mCnt, candidate.nCnt);
    return candidate.mnCnt != 0UL && candidate.mte2Bytes != 0UL && candidate.mte2Bytes != UINT64_SATURATED;
}

bool InitActualStreamKShape(const optiling::QuantBatchMatmulInfo& inputParams,
                            const optiling::QuantBatchMatmulV3CompileInfo& compileInfo, ActualStreamKShape& shape)
{
    shape.coreNum = compileInfo.aicNum;
    shape.baseMAlign = GetBenefitBaseMAlign(inputParams);
    shape.baseNAlign = GetBenefitBaseNAlign(inputParams);
    shape.baseKAlign = GetBenefitBaseKAlign(inputParams);
    if (shape.coreNum == 0UL || shape.baseMAlign == 0UL || shape.baseNAlign == 0UL || shape.baseKAlign == 0UL) {
        return false;
    }

    shape.baseM = ops::CeilAlign(std::min(inputParams.mSize, BASIC_BLOCK_SIZE_256), shape.baseMAlign);
    shape.baseN = ops::CeilAlign(std::min(inputParams.nSize, BASIC_BLOCK_SIZE_256), shape.baseNAlign);
    shape.mCnt = SafeCeilDiv(inputParams.mSize, shape.baseM);
    shape.nCnt = SafeCeilDiv(inputParams.nSize, shape.baseN);
    if (!CalcMnCnt(inputParams.batchC, shape.mCnt, shape.nCnt, shape.mnCnt)) {
        return false;
    }
    shape.singleCoreK = inputParams.kSize;
    return shape.mnCnt != 0UL;
}

bool UpdateSmallMnActualStreamKShape(const optiling::QuantBatchMatmulInfo& inputParams, ActualStreamKShape& shape)
{
    if (shape.mCnt > shape.coreNum / SMALL_MN_EXPAND_RATIO && shape.mCnt < shape.coreNum / NUM_HALF) {
        shape.mCnt = shape.coreNum / NUM_HALF;
    }
    if (shape.nCnt > shape.coreNum / SMALL_MN_EXPAND_RATIO && shape.nCnt < shape.coreNum / NUM_HALF) {
        shape.nCnt = shape.coreNum / NUM_HALF;
    }
    shape.baseM = ops::CeilAlign(SafeCeilDiv(inputParams.mSize, shape.mCnt), shape.baseMAlign);
    shape.baseN = ops::CeilAlign(SafeCeilDiv(inputParams.nSize, shape.nCnt), shape.baseNAlign);
    if (!CalcMnCnt(inputParams.batchC, shape.mCnt, shape.nCnt, shape.mnCnt) || shape.mnCnt == 0UL) {
        return false;
    }
    shape.streamKCnt = std::max(shape.coreNum / shape.mnCnt, 1UL);
    shape.singleCoreK = SafeCeilDiv(inputParams.kSize, shape.streamKCnt);
    return true;
}

bool UpdateActualStreamKSchedule(const optiling::QuantBatchMatmulInfo& inputParams, ActualStreamKShape& shape)
{
    if (shape.mnCnt <= shape.coreNum / NUM_HALF) {
        if (!UpdateSmallMnActualStreamKShape(inputParams, shape)) {
            return false;
        }
    } else {
        uint64_t tailMnCnt = shape.mnCnt % shape.coreNum;
        if (tailMnCnt == 0UL) {
            shape.streamKCnt = 1UL;
            shape.singleCoreK = inputParams.kSize;
        } else {
            shape.streamKCnt = std::max(shape.coreNum / tailMnCnt, 1UL);
            shape.singleCoreK = SafeCeilDiv(inputParams.kSize, shape.streamKCnt);
            shape.streamKCnt = SafeCeilDiv(inputParams.kSize, shape.singleCoreK);
        }
    }
    shape.singleCoreK = ops::CeilAlign(shape.singleCoreK, shape.baseKAlign);
    return shape.singleCoreK != 0UL;
}

bool FillActualStreamKCandidate(const optiling::QuantBatchMatmulInfo& inputParams,
                                const optiling::QuantBatchMatmulV3CompileInfo& compileInfo,
                                const ActualStreamKShape& shape, BenefitCandidate& candidate)
{
    uint64_t kValueMax = GetShapeWithDataTypeLocal(compileInfo.l0aSize / DOUBLE_BUFFER_NUM, inputParams.aDtype) /
                         std::max(shape.baseM, shape.baseN);
    kValueMax = ops::FloorAlign(kValueMax, shape.baseKAlign);
    if (kValueMax < shape.baseKAlign) {
        return false;
    }

    if (!BuildBenefitCandidate(inputParams, compileInfo, shape.baseM, shape.baseN, candidate)) {
        return false;
    }

    candidate.mCnt = shape.mCnt;
    candidate.nCnt = shape.nCnt;
    if (!CalcMnCnt(inputParams.batchC, candidate.mCnt, candidate.nCnt, candidate.mnCnt)) {
        return false;
    }
    candidate.streamKCnt = SafeCeilDiv(inputParams.kSize, shape.singleCoreK);
    candidate.singleCoreK = shape.singleCoreK;
    candidate.splitK = std::max(SafeCeilDiv(inputParams.kSize, shape.singleCoreK), 1UL);
    candidate.baseK = std::min(std::min(shape.singleCoreK, kValueMax),
                               ops::CeilAlign(inputParams.kSize, shape.baseKAlign));
    candidate.mte2Bytes = EstimateMte2Bytes(inputParams, candidate.mCnt, candidate.nCnt);
    return candidate.mnCnt != 0UL && candidate.baseK != 0UL && candidate.mte2Bytes != 0UL &&
           candidate.mte2Bytes != UINT64_SATURATED;
}

bool BuildActualStreamKCandidate(const optiling::QuantBatchMatmulInfo& inputParams,
                                 const optiling::QuantBatchMatmulV3CompileInfo& compileInfo,
                                 BenefitCandidate& candidate)
{
    ActualStreamKShape shape;
    return InitActualStreamKShape(inputParams, compileInfo, shape) &&
           UpdateActualStreamKSchedule(inputParams, shape) &&
           FillActualStreamKCandidate(inputParams, compileInfo, shape, candidate);
}

struct BenefitExtraCost {
    uint64_t workspaceReadBytes = 0UL;
    uint64_t outputWriteBytes = 0UL;
    uint64_t fixedOverheadNs = BENEFIT_STREAMK_FIXED_OVERHEAD_NS;
    uint64_t totalBytes = 0UL;
    uint64_t totalTimeNs = 0UL;
};

BenefitExtraCost EstimateStreamKExtraBytes(const BenefitCandidate& candidate,
                                           const optiling::QuantBatchMatmulInfo& inputParams, uint64_t coreNum)
{
    BenefitExtraCost cost;
    uint64_t streamKTailMnCnt = GetStreamKTailMnCnt(candidate.mnCnt, coreNum);
    uint64_t tileElements = SaturatingMul(candidate.baseM, candidate.baseN);
    uint64_t workspaceTileBytes = SaturatingMul(tileElements, DATA_SIZE_L0C);
    uint64_t outputTileBytes = GetSizeWithDataTypeLocal(tileElements, inputParams.cDtype);
    cost.workspaceReadBytes = SaturatingMul(SaturatingMul(streamKTailMnCnt, candidate.splitK), workspaceTileBytes);
    cost.outputWriteBytes = SaturatingMul(streamKTailMnCnt, outputTileBytes);
    cost.totalBytes = SaturatingAdd(cost.workspaceReadBytes, cost.outputWriteBytes);
    cost.totalTimeNs = SaturatingAdd(CalcTransferTimeNs(cost.totalBytes, BENEFIT_AIV_BW_BYTES_PER_NS),
                                    cost.fixedOverheadNs);
    return cost;
}

struct BenefitGateEval {
    bool admit = false;
    const char* reason = "reject";
    uint64_t aswtBaseM = 0UL;
    uint64_t aswtBaseN = 0UL;
    uint64_t aswtMnCnt = 0UL;
    uint64_t aswtMte2Bytes = 0UL;
    uint64_t skBaseM = 0UL;
    uint64_t skBaseN = 0UL;
    uint64_t skBaseK = 0UL;
    uint64_t skMnCnt = 0UL;
    uint64_t skSingleCoreK = 0UL;
    uint64_t skSplitK = 1UL;
    uint64_t skMte2Bytes = 0UL;
    uint64_t savedMte2Bytes = 0UL;
    uint64_t extraBytes = 0UL;
    uint64_t workspaceExtraBytes = 0UL;
    uint64_t reduceExtraBytes = 0UL;
    uint64_t fixedOverheadNs = 0UL;
    uint64_t transposeExtraBytes = 0UL;
    uint64_t requiredSavedBytes = 0UL;
    uint64_t savedMte2Permille = 0UL;
    uint64_t extraCostPermille = 0UL;
    uint64_t requiredSavedPermille = 0UL;
    uint64_t savedMte2TimeNs = 0UL;
    uint64_t extraCostTimeNs = 0UL;
    uint64_t requiredSavedTimeNs = 0UL;
    uint64_t aswtUtilPermille = 0UL;
    uint64_t skUtilPermille = 0UL;
    uint64_t aswtMCnt = 0UL;
    uint64_t aswtNCnt = 0UL;
    bool aswtAFullLoad = false;
};


struct BenefitGateSearchResult {
    BenefitCandidate bestAswtUtil;
    BenefitCandidate bestAswtFullMte2;
    BenefitCandidate actualSk;
    AswtShadowInfo aswtShadow;
    bool hasAswtUtil = false;
    bool hasAswtFullMte2 = false;
    bool hasActualSk = false;
    bool hasAswtShadow = false;
    uint64_t bestAswtUtilSlots = 0UL;
};

bool CheckBenefitGateBasicParam(const optiling::QuantBatchMatmulInfo& inputParams,
                                const optiling::QuantBatchMatmulV3CompileInfo& compileInfo,
                                BenefitGateEval& eval)
{
    uint64_t coreNum = compileInfo.aicNum;
    uint64_t baseMAlign = GetBenefitBaseMAlign(inputParams);
    uint64_t baseNAlign = GetBenefitBaseNAlign(inputParams);
    if (coreNum == 0UL || baseMAlign == 0UL || baseNAlign == 0UL) {
        eval.reason = "invalid_basic_param";
        return false;
    }
    if (!inputParams.transA && inputParams.transB &&
        (inputParams.mSize <= BENEFIT_TA0TB1_MIN_MN_THRESHOLD ||
         inputParams.nSize <= BENEFIT_TA0TB1_MIN_MN_THRESHOLD ||
         inputParams.kSize <= BENEFIT_TA0TB1_MIN_K_THRESHOLD)) {
        eval.reason = "ta0tb1_small_shape";
        return false;
    }
    return true;
}

void UpdateBestAswtCandidate(const BenefitCandidate& candidate, uint64_t coreNum,
                             BenefitGateSearchResult& result)
{
    uint64_t utilSlots = std::min(candidate.mnCnt, coreNum);
    if (!result.hasAswtUtil || utilSlots > result.bestAswtUtilSlots ||
        (utilSlots == result.bestAswtUtilSlots && candidate.mte2Bytes < result.bestAswtUtil.mte2Bytes)) {
        result.bestAswtUtil = candidate;
        result.bestAswtUtilSlots = utilSlots;
        result.hasAswtUtil = true;
    }
}

void UpdateBestFullCoreMte2Candidate(const BenefitCandidate& candidate, uint64_t coreNum,
                                     BenefitGateSearchResult& result)
{
    if (candidate.mnCnt >= coreNum &&
        (!result.hasAswtFullMte2 || candidate.mte2Bytes < result.bestAswtFullMte2.mte2Bytes)) {
        result.bestAswtFullMte2 = candidate;
        result.hasAswtFullMte2 = true;
    }
}

void SearchAswtBenefitCandidates(const optiling::QuantBatchMatmulInfo& inputParams,
                                 const optiling::QuantBatchMatmulV3CompileInfo& compileInfo,
                                 BenefitGateSearchResult& result)
{
    uint64_t coreNum = compileInfo.aicNum;
    std::vector<uint64_t> baseMCandidates = BuildBaseCandidates(inputParams.mSize,
        GetBenefitBaseMAlign(inputParams), coreNum);
    std::vector<uint64_t> baseNCandidates = BuildBaseCandidates(inputParams.nSize,
        GetBenefitBaseNAlign(inputParams), coreNum);
    for (uint64_t baseM : baseMCandidates) {
        for (uint64_t baseN : baseNCandidates) {
            BenefitCandidate candidate;
            if (!BuildBenefitCandidate(inputParams, compileInfo, baseM, baseN, candidate)) {
                continue;
            }
            UpdateBestAswtCandidate(candidate, coreNum, result);
            UpdateBestFullCoreMte2Candidate(candidate, coreNum, result);
        }
    }
}

BenefitGateSearchResult BuildBenefitGateSearchResult(const optiling::QuantBatchMatmulInfo& inputParams,
                                                     const optiling::QuantBatchMatmulV3CompileInfo& compileInfo)
{
    BenefitGateSearchResult result;
    result.hasActualSk = BuildActualStreamKCandidate(inputParams, compileInfo, result.actualSk);
    result.hasAswtShadow = CalcAswtShadowInfo(inputParams, compileInfo, result.aswtShadow);
    SearchAswtBenefitCandidates(inputParams, compileInfo, result);
    return result;
}

void FillEvalWithAswtUtilCandidate(const optiling::QuantBatchMatmulInfo& inputParams,
                                   const optiling::QuantBatchMatmulV3CompileInfo& compileInfo,
                                   const BenefitGateSearchResult& result, BenefitGateEval& eval)
{
    uint64_t coreNum = compileInfo.aicNum;
    eval.aswtBaseM = result.bestAswtUtil.baseM;
    eval.aswtBaseN = result.bestAswtUtil.baseN;
    eval.aswtMnCnt = result.bestAswtUtil.mnCnt;
    eval.aswtMte2Bytes = result.bestAswtUtil.mte2Bytes;
    eval.aswtUtilPermille = CalcPermille(result.bestAswtUtilSlots, coreNum);
    if (!result.hasAswtShadow) {
        return;
    }
    eval.aswtBaseM = result.aswtShadow.baseM;
    eval.aswtBaseN = result.aswtShadow.baseN;
    eval.aswtMnCnt = result.aswtShadow.mnCnt;
    eval.aswtMte2Bytes = result.aswtShadow.mte2Bytes;
    eval.aswtUtilPermille = CalcPermille(std::min(result.aswtShadow.mnCnt, coreNum), coreNum);
    eval.aswtMCnt = result.aswtShadow.mCnt;
    eval.aswtNCnt = result.aswtShadow.nCnt;
    eval.aswtAFullLoad = IsAswtAFullLoad(inputParams, compileInfo, result.aswtShadow);
}

void FillEvalWithActualStreamK(const optiling::QuantBatchMatmulInfo& inputParams,
                               const BenefitCandidate& actualSk, uint64_t coreNum, BenefitGateEval& eval)
{
    BenefitExtraCost extraCost = EstimateStreamKExtraBytes(actualSk, inputParams, coreNum);
    eval.skBaseM = actualSk.baseM;
    eval.skBaseN = actualSk.baseN;
    eval.skBaseK = actualSk.baseK;
    eval.skMnCnt = actualSk.mnCnt;
    eval.skSingleCoreK = actualSk.singleCoreK;
    eval.skSplitK = actualSk.splitK;
    eval.skMte2Bytes = actualSk.mte2Bytes;
    eval.extraBytes = extraCost.totalBytes;
    eval.workspaceExtraBytes = extraCost.workspaceReadBytes;
    eval.reduceExtraBytes = extraCost.outputWriteBytes;
    eval.fixedOverheadNs = extraCost.fixedOverheadNs;
    eval.transposeExtraBytes = 0UL;
    eval.extraCostTimeNs = extraCost.totalTimeNs;
    eval.skUtilPermille = CalcPermille(std::min(SaturatingMul(actualSk.mnCnt, actualSk.splitK), coreNum), coreNum);
}

bool FinishSmallMnBenefitGate(const optiling::QuantBatchMatmulInfo& inputParams,
                              const BenefitGateSearchResult& result, uint64_t coreNum, BenefitGateEval& eval)
{
    if (result.bestAswtUtilSlots >= coreNum) {
        return false;
    }
    if (!result.hasActualSk) {
        eval.reason = "small_mn_no_sk_fill";
        return true;
    }
    if (eval.aswtMnCnt >= coreNum / NUM_HALF) {
        eval.reason = "small_mn_aswt_half_core";
        return true;
    }
    bool scheduleGainEnough = inputParams.kSize >= BENEFIT_MIN_K_THRESHOLD &&
                              SaturatingMul(result.actualSk.mnCnt, result.actualSk.splitK) >= coreNum;
    eval.admit = scheduleGainEnough;
    eval.reason = eval.admit ? "small_mn_core_fill" : "small_mn_extra_cost";
    return true;
}

void FillMte2SavingEval(const BenefitGateSearchResult& result, BenefitGateEval& eval)
{
    const BenefitCandidate& bestAswtFullMte2 = result.bestAswtFullMte2;
    const BenefitCandidate& actualSk = result.actualSk;
    eval.aswtBaseM = bestAswtFullMte2.baseM;
    eval.aswtBaseN = bestAswtFullMte2.baseN;
    eval.aswtMnCnt = bestAswtFullMte2.mnCnt;
    eval.aswtMte2Bytes = bestAswtFullMte2.mte2Bytes;
    eval.savedMte2Bytes = bestAswtFullMte2.mte2Bytes > actualSk.mte2Bytes ?
                              bestAswtFullMte2.mte2Bytes - actualSk.mte2Bytes :
                              0UL;
    eval.savedMte2Permille = CalcPermille(eval.savedMte2Bytes, eval.aswtMte2Bytes);
    eval.extraCostPermille = CalcPermille(eval.extraBytes, eval.aswtMte2Bytes);
    eval.savedMte2TimeNs = CalcTransferTimeNs(eval.savedMte2Bytes, BENEFIT_MTE2_BW_BYTES_PER_NS);
    eval.requiredSavedTimeNs = SaturatingAdd(
        SaturatingMulDiv(eval.extraCostTimeNs, BENEFIT_TIME_SAFETY_NUM, BENEFIT_TIME_SAFETY_DEN),
        BENEFIT_STREAMK_MIN_MARGIN_NS);
    eval.requiredSavedBytes = std::max(BENEFIT_MTE2_MIN_SAVED_BYTES,
                                       SaturatingMul(eval.requiredSavedTimeNs, BENEFIT_MTE2_BW_BYTES_PER_NS));
    eval.requiredSavedPermille = CalcPermille(eval.requiredSavedBytes, eval.aswtMte2Bytes);
}

void FinishMte2BenefitGate(const BenefitGateSearchResult& result, BenefitGateEval& eval)
{
    if (result.actualSk.splitK == 1UL) {
        eval.reason = "mte2_no_splitk";
        return;
    }
    eval.admit = eval.savedMte2Bytes >= BENEFIT_MTE2_MIN_SAVED_BYTES &&
                 eval.savedMte2Permille >= BENEFIT_MTE2_MIN_SAVED_PERMILLE &&
                 eval.savedMte2TimeNs >= eval.requiredSavedTimeNs;
    eval.reason = eval.admit ? "mte2_reuse_saving" : "mte2_extra_cost";
}

BenefitGateEval EvaluateStreamKBenefitGate(const optiling::QuantBatchMatmulInfo& inputParams,
                                           const optiling::QuantBatchMatmulV3CompileInfo& compileInfo)
{
    BenefitGateEval eval;
    if (!CheckBenefitGateBasicParam(inputParams, compileInfo, eval)) {
        return eval;
    }

    uint64_t coreNum = compileInfo.aicNum;
    BenefitGateSearchResult result = BuildBenefitGateSearchResult(inputParams, compileInfo);
    if (!result.hasAswtUtil) {
        eval.reason = "no_legal_aswt_candidate";
        return eval;
    }

    FillEvalWithAswtUtilCandidate(inputParams, compileInfo, result, eval);
    if (result.hasActualSk) {
        FillEvalWithActualStreamK(inputParams, result.actualSk, coreNum, eval);
    }

    if (result.hasActualSk && eval.skUtilPermille < eval.aswtUtilPermille) {
        eval.reason = "sk_util_lower_than_aswt";
        return eval;
    }
    if (FinishSmallMnBenefitGate(inputParams, result, coreNum, eval)) {
        return eval;
    }

    if (!result.hasAswtFullMte2 || !result.hasActualSk) {
        eval.reason = "mte2_no_full_core_pair";
        return eval;
    }
    if (eval.aswtAFullLoad) {
        eval.reason = "mte2_aswt_a_full_load";
        return eval;
    }

    FillMte2SavingEval(result, eval);
    FinishMte2BenefitGate(result, eval);
    return eval;
}

void LogBenefitGateEval(const char* opName, const optiling::QuantBatchMatmulInfo& inputParams,
                        const BenefitGateEval& benefitGate)
{
    OP_LOGD(opName,
            "QBMM_STREAMK_BENEFIT_GATE_%s reason=%s m=%lu n=%lu k=%lu transA=%d transB=%d "
            "aswtUtilPermille=%lu skUtilPermille=%lu aswtBaseM=%lu aswtBaseN=%lu aswtMnCnt=%lu "
            "aswtMCnt=%lu aswtNCnt=%lu aswtAFullLoad=%d "
            "skBaseM=%lu skBaseN=%lu skBaseK=%lu skMnCnt=%lu skSingleCoreK=%lu skSplitK=%lu "
            "aswtMte2Bytes=%lu skMte2Bytes=%lu savedMte2Bytes=%lu extraBytes=%lu "
            "workspaceExtraBytes=%lu reduceExtraBytes=%lu fixedOverheadNs=%lu transposeExtraBytes=%lu "
            "requiredSavedBytes=%lu savedMte2Permille=%lu extraCostPermille=%lu requiredSavedPermille=%lu "
            "savedMte2TimeNs=%lu extraCostTimeNs=%lu requiredSavedTimeNs=%lu.",
            benefitGate.admit ? "ADMIT" : "REJECT", benefitGate.reason, inputParams.mSize, inputParams.nSize,
            inputParams.kSize, inputParams.transA, inputParams.transB, benefitGate.aswtUtilPermille,
            benefitGate.skUtilPermille, benefitGate.aswtBaseM, benefitGate.aswtBaseN, benefitGate.aswtMnCnt,
            benefitGate.aswtMCnt, benefitGate.aswtNCnt, static_cast<int32_t>(benefitGate.aswtAFullLoad),
            benefitGate.skBaseM, benefitGate.skBaseN, benefitGate.skBaseK, benefitGate.skMnCnt,
            benefitGate.skSingleCoreK, benefitGate.skSplitK, benefitGate.aswtMte2Bytes, benefitGate.skMte2Bytes,
            benefitGate.savedMte2Bytes, benefitGate.extraBytes, benefitGate.workspaceExtraBytes,
            benefitGate.reduceExtraBytes, benefitGate.fixedOverheadNs, benefitGate.transposeExtraBytes,
            benefitGate.requiredSavedBytes, benefitGate.savedMte2Permille, benefitGate.extraCostPermille,
            benefitGate.requiredSavedPermille, benefitGate.savedMte2TimeNs, benefitGate.extraCostTimeNs,
            benefitGate.requiredSavedTimeNs);
}
} // namespace

namespace optiling {

QBMMV3StreamKTiling::QBMMV3StreamKTiling(gert::TilingContext* context)
    : QuantBatchMatmulV3TilingBase(context, false), tilingData_(tilingDataSelf_)
{
    Reset();
}

void QBMMV3StreamKTiling::Reset()
{
    isBf16Opt_ = false;
    isUbQuant_ = false;
    kL1_ = 1UL;
    scaleKL1_ = 1UL;
    if (!isTilingOut_) {
        tilingData_ = DequantBmm::QuantBatchMatmulV3StreamKBasicAPITilingData();
        if (context_ != nullptr && context_->GetRawTilingData() != nullptr &&
            context_->GetRawTilingData()->GetData() != nullptr) {
            errno_t ret = memset_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(),
                                   0, context_->GetRawTilingData()->GetCapacity());
            if (ret != EOK) {
                OP_LOGE("QuantBatchMatmulV3", "QBMM StreamK reset raw tiling data failed, ret=%d.", ret);
            }
        }
    }
}

ge::graphStatus QBMMV3StreamKTiling::GetPlatformInfo()
{
    OP_LOGE_IF(!SetPlatformInfoForTiling(), ge::GRAPH_FAILED, inputParams_.opName, "SetPlatformInfo failed.");
    OP_TILING_CHECK(
        aicoreParams_.aicNum == 0UL || aicoreParams_.l1Size == 0UL || aicoreParams_.l0cSize == 0UL,
        CUBE_INNER_ERR_REPORT(
            inputParams_.opName, "CoreNum/L1Size/L0cSize should not be 0. CoreNum: %lu, L1Size: %lu, L0cSize: %lu.",
            aicoreParams_.aicNum, aicoreParams_.l1Size, aicoreParams_.l0cSize),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QBMMV3StreamKTiling::GetShapeAttrsInfo()
{
    inputParams_.Reset();
    tilingDataSize_ = sizeof(DequantBmm::QuantBatchMatmulV3StreamKBasicAPITilingData);
    return QuantBatchMatmulV3TilingBase::GetShapeAttrsInfo();
}

bool QBMMV3StreamKTiling::CheckDtype() const
{
    return QuantBatchMatMulV3TilingUtil::CheckDtype(context_, inputParams_, compileInfo_);
}

bool QBMMV3StreamKTiling::CheckShape(const std::vector<gert::Shape*>& mandatoryShape,
                                    const gert::StorageShape* biasShape,
                                    const gert::StorageShape* pertokenShape,
                                    const std::vector<int64_t>& dimValueOfMKN) const
{
    return QuantBatchMatMulV3TilingUtil::CheckShape(
        context_, inputParams_, compileInfo_, mandatoryShape, biasShape, pertokenShape, dimValueOfMKN);
}

bool QBMMV3StreamKTiling::IsMxInput() const
{
    bool isMxfp8 = (inputParams_.aDtype == ge::DT_FLOAT8_E4M3FN || inputParams_.aDtype == ge::DT_FLOAT8_E5M2) &&
                   (inputParams_.bDtype == ge::DT_FLOAT8_E4M3FN || inputParams_.bDtype == ge::DT_FLOAT8_E5M2) &&
                   inputParams_.isMxPerGroup;
    bool isMxfp4 = inputParams_.isMxPerGroup && inputParams_.aDtype == ge::DT_FLOAT4_E2M1 &&
                   inputParams_.bDtype == ge::DT_FLOAT4_E2M1;
    return isMxfp8 || isMxfp4;
}

bool QBMMV3StreamKTiling::IsCapable()
{
    if (!IsMxInput()) {
        OP_LOGD(inputParams_.opName, "QBMM StreamK only supports MX per-group input.");
        return false;
    }
    if (inputParams_.batchC != 1UL) {
        OP_LOGD(inputParams_.opName, "QBMM StreamK only supports no-batch MX input, batchC=%lu.", inputParams_.batchC);
        return false;
    }
    if (compileInfo_.aivNum == 0UL) {
        OP_LOGD(inputParams_.opName, "QBMM StreamK requires AIV cores, aiv=%u aic=%u.", compileInfo_.aivNum,
                compileInfo_.aicNum);
        return false;
    }
    if (compileInfo_.aicNum == 0UL || compileInfo_.aivNum % CORE_RATIO != 0UL ||
        compileInfo_.aivNum / CORE_RATIO != compileInfo_.aicNum) {
        OP_LOGD(inputParams_.opName, "QBMM StreamK only supports aivNum == aicNum * 2, aiv=%u aic=%u.",
                compileInfo_.aivNum, compileInfo_.aicNum);
        return false;
    }
    if (context_ != nullptr && context_->GetDeterministic() == 1) {
        OP_LOGD(inputParams_.opName, "QBMM StreamK is disabled when deterministic mode is enabled.");
        return false;
    }
    if (!IsTensorapiCapable()) {
        OP_LOGD(inputParams_.opName, "QBMM StreamK requires Tensor API runtime support.");
        return false;
    }
    BenefitGateEval benefitGate = EvaluateStreamKBenefitGate(inputParams_, compileInfo_);
    LogBenefitGateEval(inputParams_.opName, inputParams_, benefitGate);
    if (!benefitGate.admit) {
        OP_LOGD(inputParams_.opName, "QBMM StreamK capability gate result: reject reason=%s.", benefitGate.reason);
    }
    return benefitGate.admit;
}

bool QBMMV3StreamKTiling::CalcBaseBlock()
{
    BaseBlockCalculator calculator(inputParams_, compileInfo_, inputParams_.batchC);
    if (!calculator.Compute(BaseBlockMode::STREAMK)) {
        return false;
    }
    const BaseBlockRes& baseBlockRes = calculator.GetOutput();
    basicTiling_.baseM = static_cast<uint32_t>(baseBlockRes.baseM);
    basicTiling_.baseN = static_cast<uint32_t>(baseBlockRes.baseN);
    basicTiling_.baseK = static_cast<uint32_t>(baseBlockRes.baseK);
    basicTiling_.singleCoreM = std::min(inputParams_.mSize, baseBlockRes.baseM);
    basicTiling_.singleCoreN = std::min(inputParams_.nSize, baseBlockRes.baseN);
    basicTiling_.singleCoreK = static_cast<uint32_t>(baseBlockRes.singleCoreK);
    mCnt_ = baseBlockRes.mCnt;
    nCnt_ = baseBlockRes.nCnt;
    streamKCnt_ = baseBlockRes.streamKCnt;
    basicTiling_.usedCoreNum = static_cast<uint32_t>(aicoreParams_.aicNum);
    basicTiling_.stepM = 1U;
    basicTiling_.stepN = 1U;
    basicTiling_.iterateOrder = 0U;
    basicTiling_.dbL0c =
        SaturatingMul(SaturatingMul(SaturatingMul(basicTiling_.baseM, basicTiling_.baseN), DATA_SIZE_L0C),
                      DOUBLE_BUFFER_NUM) <= aicoreParams_.l0cSize ? DOUBLE_BUFFER_NUM : 1U;
    return true;
}

bool QBMMV3StreamKTiling::CalcL1Tiling()
{
    L1TilingDataCalculator l1Calculator(inputParams_, compileInfo_, basicTiling_.baseM, basicTiling_.baseN,
                                        basicTiling_.baseK, basicTiling_.singleCoreK);
    if (!l1Calculator.Compute(L1TilingMode::STREAMK)) {
        return false;
    }
    const L1TilingData& l1TilingData = l1Calculator.GetOutput();
    basicTiling_.depthA1 = static_cast<uint32_t>(l1TilingData.depthKa_);
    basicTiling_.depthB1 = static_cast<uint32_t>(l1TilingData.depthKb_);
    basicTiling_.stepKa = static_cast<uint32_t>(l1TilingData.stepKa_);
    basicTiling_.stepKb = static_cast<uint32_t>(l1TilingData.stepKb_);
    basicTiling_.scaleFactorA = static_cast<uint32_t>(l1TilingData.scaleFactorA_);
    basicTiling_.scaleFactorB = static_cast<uint32_t>(l1TilingData.scaleFactorB_);
    kL1_ = std::min(static_cast<uint64_t>(basicTiling_.stepKa) * basicTiling_.baseK,
                    static_cast<uint64_t>(basicTiling_.singleCoreK));
    scaleKL1_ = std::min(l1TilingData.scaleFactorA_ * l1TilingData.stepKa_ * basicTiling_.baseK,
                         l1TilingData.scaleFactorB_ * l1TilingData.stepKb_ * basicTiling_.baseK);
    scaleKL1_ = std::max(scaleKL1_, kL1_);
    return true;
}

ge::graphStatus QBMMV3StreamKTiling::DoOpTiling()
{
    if (!CalcBaseBlock()) {
        OP_LOGE(inputParams_.opName, "QBMM StreamK CalcBaseBlock failed.");
        return ge::GRAPH_FAILED;
    }
    if (!CalcL1Tiling()) {
        OP_LOGE(inputParams_.opName, "QBMM StreamK CalcL1Tiling failed.");
        return ge::GRAPH_FAILED;
    }
    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

void QBMMV3StreamKTiling::CalculateNBufferNum4MX()
{
    uint64_t usedL1Size = GetSizeWithDataType(static_cast<uint64_t>(basicTiling_.baseN) * kL1_, inputParams_.bDtype) *
                          L1_FOUR_BUFFER;
    usedL1Size += GetSizeWithDataType(
                      static_cast<uint64_t>(basicTiling_.baseN) * ops::CeilDiv(scaleKL1_, MX_GROUP_SIZE),
                      inputParams_.scaleDtype) *
                  L1_TWO_BUFFER;
    if (inputParams_.hasBias) {
        usedL1Size += GetSizeWithDataType(static_cast<uint64_t>(basicTiling_.baseN), inputParams_.biasDtype) *
                      L1_TWO_BUFFER;
    }
    usedL1Size += GetSizeWithDataType(static_cast<uint64_t>(basicTiling_.baseM) * kL1_, inputParams_.aDtype) *
                  L1_FOUR_BUFFER;
    usedL1Size += GetSizeWithDataType(
                      static_cast<uint64_t>(basicTiling_.baseM) * ops::CeilDiv(scaleKL1_, MX_GROUP_SIZE),
                      inputParams_.perTokenScaleDtype) *
                  L1_TWO_BUFFER;
    tilingData_.matmulTiling.nBufferNum = usedL1Size < aicoreParams_.l1Size ? L1_FOUR_BUFFER : L1_TWO_BUFFER;
}

void QBMMV3StreamKTiling::SetTilingData()
{
    QuantBatchMatMulV3TilingUtil::SetCommonTilingData(inputParams_, tilingData_);
    tilingData_.params.x1QuantMode = static_cast<uint32_t>(BasicQuantMode::MX_PERGROUP_MODE);
    tilingData_.params.x2QuantMode = static_cast<uint32_t>(BasicQuantMode::MX_PERGROUP_MODE);
    tilingData_.matmulTiling.m = static_cast<uint32_t>(inputParams_.mSize);
    tilingData_.matmulTiling.n = static_cast<uint32_t>(inputParams_.nSize);
    tilingData_.matmulTiling.k = static_cast<uint32_t>(inputParams_.kSize);
    tilingData_.matmulTiling.baseM = basicTiling_.baseM;
    tilingData_.matmulTiling.baseN = basicTiling_.baseN;
    tilingData_.matmulTiling.baseK = basicTiling_.baseK;
    tilingData_.streamKTiling.singleCoreK = basicTiling_.singleCoreK;
    tilingData_.streamKTiling.kL1 = static_cast<uint32_t>(kL1_);
    tilingData_.matmulTiling.isBias = inputParams_.hasBias ? 1U : 0U;
    tilingData_.matmulTiling.dbL0C = static_cast<uint8_t>(basicTiling_.dbL0c);
    tilingData_.matmulTiling.stepKa = static_cast<uint16_t>(std::min(basicTiling_.stepKa, basicTiling_.stepKb));
    tilingData_.matmulTiling.stepKb = tilingData_.matmulTiling.stepKa;
    tilingData_.matmulTiling.scaleKL1 = static_cast<uint32_t>(scaleKL1_);
    tilingData_.matmulTiling.scaleFactorA =
        basicTiling_.scaleFactorA >= SCALER_FACTOR_MIN && basicTiling_.scaleFactorA <= SCALER_FACTOR_MAX ?
            static_cast<uint16_t>(basicTiling_.scaleFactorA) :
            static_cast<uint16_t>(SCALER_FACTOR_MIN);
    tilingData_.matmulTiling.scaleFactorB =
        basicTiling_.scaleFactorB >= SCALER_FACTOR_MIN && basicTiling_.scaleFactorB <= SCALER_FACTOR_MAX ?
            static_cast<uint16_t>(basicTiling_.scaleFactorB) :
            static_cast<uint16_t>(SCALER_FACTOR_MIN);
    // Current StreamK BlockMmad uses fixed double L1 buffers. nBufferNum is retained for BasicAPI tiling/log
    // compatibility and is not used by the StreamK kernel as a runtime tuning knob.
    CalculateNBufferNum4MX();
    // adaptiveSlidingWin is kept only to preserve the shared BasicAPI tiling data layout. StreamK scheduling uses
    // streamKTiling fields instead of ASW tail/window parameters, so fill neutral placeholders here.
    tilingData_.adaptiveSlidingWin.mTailTile = 1U;
    tilingData_.adaptiveSlidingWin.nTailTile = 1U;
    tilingData_.adaptiveSlidingWin.mBaseTailSplitCnt = 1U;
    tilingData_.adaptiveSlidingWin.nBaseTailSplitCnt = 1U;
    tilingData_.adaptiveSlidingWin.mTailMain = 0U;
    tilingData_.adaptiveSlidingWin.nTailMain = 0U;
    OP_LOGD(inputParams_.opName,
            "QBMM StreamK SetTilingData: M=%u N=%u K=%u baseM=%u baseN=%u baseK=%u singleCoreK=%u "
            "kL1=%u scaleKL1=%u usedCore=%u stepK=%u scaleFactorA=%u scaleFactorB=%u nBuffer=%u.",
            tilingData_.matmulTiling.m, tilingData_.matmulTiling.n, tilingData_.matmulTiling.k,
            tilingData_.matmulTiling.baseM, tilingData_.matmulTiling.baseN, tilingData_.matmulTiling.baseK,
            tilingData_.streamKTiling.singleCoreK, tilingData_.streamKTiling.kL1,
            tilingData_.matmulTiling.scaleKL1, basicTiling_.usedCoreNum, tilingData_.matmulTiling.stepKa,
            tilingData_.matmulTiling.scaleFactorA, tilingData_.matmulTiling.scaleFactorB,
            tilingData_.matmulTiling.nBufferNum);
}

ge::graphStatus QBMMV3StreamKTiling::DoLibApiTiling()
{
    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

uint64_t QBMMV3StreamKTiling::GetBiasMode() const
{
    return TPL_EXCLUDE_FROM_TEMPLATE;
}

uint64_t QBMMV3StreamKTiling::GetKernelType() const
{
    return TPL_VEC_EPILOGUE_STREAMK_WITH_MMAPI;
}

uint64_t QBMMV3StreamKTiling::GetTilingKey() const
{
    return GET_TPL_TILING_KEY(static_cast<uint64_t>(inputParams_.transA), static_cast<uint64_t>(inputParams_.transB),
                              GetBiasMode(), GetKernelType(), TPL_API_LEVEL_BLAZE);
}

ge::graphStatus QBMMV3StreamKTiling::GetWorkspaceSize()
{
    workspaceSize_ = inputParams_.libApiWorkSpaceSize +
                     aicoreParams_.aicNum * BASIC_BLOCK_SIZE_256 * BASIC_BLOCK_SIZE_256 * DATA_SIZE_L0C +
                     STREAMK_RPC_WORKSPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QBMMV3StreamKTiling::PostTiling()
{
    OP_TILING_CHECK(tilingDataSize_ % sizeof(uint64_t) != 0UL,
                    CUBE_INNER_ERR_REPORT(inputParams_.opName, "Tiling data size[%zu] is not aligned to 8.",
                                          tilingDataSize_),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(context_->GetRawTilingData()->GetCapacity() < tilingDataSize_,
                    CUBE_INNER_ERR_REPORT(inputParams_.opName,
                                          "context tiling data capacity %zu < actual tiling data size %zu.",
                                          context_->GetRawTilingData()->GetCapacity(), tilingDataSize_),
                    return ge::GRAPH_FAILED);
    errno_t ret = memcpy_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(),
                           reinterpret_cast<void*>(&tilingData_), tilingDataSize_);
    if (ret != EOK) {
        OP_LOGE(context_->GetNodeName(), "Failed to copy memory with memcpy_s, ret=%d.", ret);
        return ge::GRAPH_FAILED;
    }
    context_->SetBlockDim(basicTiling_.usedCoreNum);
    context_->SetScheduleMode(1); // StreamK kernel uses cross-core synchronization and requires batch mode.
    context_->GetRawTilingData()->SetDataSize(tilingDataSize_);
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = workspaceSize_;
    return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_TEMPLATE_WITH_ARCH(QuantBatchMatmulV3, QBMMV3StreamKTiling, supportedNpuArch, TILING_PRIORITY);
} // namespace optiling
