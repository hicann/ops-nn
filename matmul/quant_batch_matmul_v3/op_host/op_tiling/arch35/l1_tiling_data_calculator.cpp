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
 * \file l1_tiling_data_calculator.cpp
 * \brief
 */
#include "l1_tiling_data_calculator.h"

#include <algorithm>
#include "error_util.h"

namespace {
constexpr uint64_t L1_ALIGN_SIZE = 32UL;
constexpr uint64_t BASIC_BLOCK_SIZE_256 = 256UL;
constexpr uint64_t BASIC_BLOCK_SIZE_512 = 512UL;
constexpr uint64_t MTE2_MIN_LOAD_SIZE = 32768UL;
constexpr uint64_t L2_ALIGN_SIZE = 128UL;
constexpr uint64_t MX_GROUP_SIZE = 32UL;
constexpr uint64_t MXFP_MULTI_BASE_SIZE = 2UL;
constexpr uint32_t DB_SIZE = 2U;
constexpr uint32_t NUM_HALF = 2U;
constexpr uint32_t SCALER_FACTOR_MAX = 127U;
constexpr uint32_t SCALER_FACTOR_MIN = 1U;

uint64_t GetSizeWithDataType(uint64_t shape, ge::DataType dtype)
{
    if (dtype == ge::DT_FLOAT4_E2M1 || dtype == ge::DT_FLOAT4_E1M2 || dtype == ge::DT_INT4) {
        return (shape + 1UL) >> 1UL;
    }
    return shape * static_cast<uint64_t>(ge::GetSizeByDataType(dtype));
}
} // namespace

namespace optiling {

L1TilingDataCalculator::L1TilingDataCalculator(
    const QuantBatchMatmulInfo& inputParams, const QuantBatchMatmulV3CompileInfo& compileInfo, uint64_t baseM,
    uint64_t baseN, uint64_t baseK)
    : inputParams_(inputParams), compileInfo_(compileInfo), baseM_(baseM), baseN_(baseN), baseK_(baseK)
{}

bool L1TilingDataCalculator::Compute(L1TilingMode mode)
{
    l1TilingData_ = L1TilingData();
    if (!ValidateInput() || !InitLeftL1Size()) {
        return false;
    }
    switch (mode) {
        case L1TilingMode::A_L1_FULL_LOAD:
            return ComputeL1TilingAL1FullLoad();
        case L1TilingMode::DEFAULT:
        default:
            return ComputeL1TilingDefault();
    }
}

const L1TilingData& L1TilingDataCalculator::GetOutput() const
{
    return l1TilingData_;
}

bool L1TilingDataCalculator::ValidateInput() const
{
    OP_TILING_CHECK(
        baseM_ == 0UL || baseN_ == 0UL || baseK_ == 0UL,
        CUBE_INNER_ERR_REPORT(
            inputParams_.opName,
            "BaseM, baseN and baseK should be greater than 0, but baseM: %lu, baseN: %lu, baseK: %lu.", baseM_, baseN_,
            baseK_),
        return false);
    return true;
}

bool L1TilingDataCalculator::InitLeftL1Size()
{
    uint64_t biasDtypeSize = ge::GetSizeByDataType(inputParams_.biasDtype);
    uint64_t scaleDtypeSize = ge::GetSizeByDataType(inputParams_.scaleDtype);
    if (inputParams_.cDtype == ge::DT_INT32) {
        scaleDtypeSize = 0UL;
    }
    uint64_t singleCoreBiasSize = inputParams_.hasBias ? baseN_ * biasDtypeSize : 0UL;
    uint64_t singleCoreScaleSize = inputParams_.isPerChannel ? baseN_ * scaleDtypeSize : 0UL;
    OP_TILING_CHECK(
        compileInfo_.l1Size < singleCoreBiasSize,
        CUBE_INNER_ERR_REPORT(
            inputParams_.opName, "Subtraction underflow: l1Size(%lu) - singleCoreBiasSize(%lu).", compileInfo_.l1Size,
            singleCoreBiasSize),
        return false);
    uint64_t leftL1SizeAfterBias = compileInfo_.l1Size - singleCoreBiasSize;
    OP_TILING_CHECK(
        leftL1SizeAfterBias < singleCoreScaleSize,
        CUBE_INNER_ERR_REPORT(
            inputParams_.opName, "Subtraction underflow: leftL1SizeAfterBias(%lu) - singleCoreScaleSize(%lu).",
            leftL1SizeAfterBias, singleCoreScaleSize),
        return false);
    leftL1Size_ = leftL1SizeAfterBias - singleCoreScaleSize;
    return true;
}

bool L1TilingDataCalculator::ComputeL1TilingDefault()
{
    if (inputParams_.isMxPerGroup) {
        uint64_t baseASize = GetSizeWithDataType(baseM_ * baseK_, inputParams_.aDtype);
        uint64_t baseBSize = GetSizeWithDataType(baseN_ * baseK_, inputParams_.bDtype);
        uint64_t scaleBaseK = ops::CeilAlign(ops::CeilDiv(baseK_, MX_GROUP_SIZE), MXFP_MULTI_BASE_SIZE);
        uint64_t baseScaleASize = GetSizeWithDataType(scaleBaseK * baseM_, inputParams_.perTokenScaleDtype);
        uint64_t baseScaleBSize = GetSizeWithDataType(scaleBaseK * baseN_, inputParams_.scaleDtype);
        uint64_t baseL1Size = baseASize + baseBSize + baseScaleASize + baseScaleBSize;
        OP_TILING_CHECK(
            baseL1Size == 0UL,
            CUBE_INNER_ERR_REPORT(
                inputParams_.opName, "Invalid MX L1 divisor: baseL1Size(%lu) should be greater than 0.", baseL1Size),
            return false);
        uint64_t depthInit = GetDepthA1B1(leftL1Size_, baseL1Size, 1UL);
        OP_TILING_CHECK(
            baseL1Size > leftL1Size_ / depthInit,
            CUBE_INNER_ERR_REPORT(
                inputParams_.opName, "Subtraction underflow: leftL1Size(%lu) - depthInit(%lu) * baseL1Size(%lu).",
                leftL1Size_, depthInit, baseL1Size),
            return false);
        uint64_t usedL1SizeByDepthInit = depthInit * baseL1Size;
        uint64_t leftL1SizeByDepthInit = leftL1Size_ - usedL1SizeByDepthInit;
        uint64_t depthASec = GetDepthA1B1(leftL1SizeByDepthInit, (baseASize + baseScaleASize) * depthInit, depthInit);
        uint64_t depthBSec = GetDepthA1B1(leftL1SizeByDepthInit, (baseBSize + baseScaleBSize) * depthInit, depthInit);
        l1TilingData_.depthKa_ = std::max(depthASec, depthBSec);
        l1TilingData_.depthKb_ = l1TilingData_.depthKa_;
        if (l1TilingData_.depthKa_ * baseL1Size > leftL1Size_) {
            l1TilingData_.depthKa_ = depthASec >= depthBSec ? depthASec : depthInit;
            l1TilingData_.depthKb_ = depthASec < depthBSec ? depthBSec : depthInit;
        }
        if (!CalStepKs() || !CalScaleFactors(baseASize, baseBSize, baseScaleASize, baseScaleBSize)) {
            return false;
        }
    } else {
        // With baseM/baseN/baseK <= 256 and 1B elements, each A/B base block is at most 16KB.
        // Split remaining L1 evenly between A and B to estimate depth.
        uint64_t baseASize = GetSizeWithDataType(baseM_ * baseK_, inputParams_.aDtype);
        uint64_t baseBSize = GetSizeWithDataType(baseN_ * baseK_, inputParams_.bDtype);
        OP_TILING_CHECK(
            baseASize == 0UL || baseBSize == 0UL,
            CUBE_INNER_ERR_REPORT(
                inputParams_.opName,
                "Invalid L1 base size: baseASize(%lu) and baseBSize(%lu) should be greater than 0.", baseASize,
                baseBSize),
            return false);
        l1TilingData_.depthKa_ = leftL1Size_ / NUM_HALF / baseASize;
        l1TilingData_.depthKb_ = leftL1Size_ / NUM_HALF / baseBSize;
        if (!CalStepKs()) {
            return false;
        }
    }
    return true;
}

bool L1TilingDataCalculator::ComputeL1TilingAL1FullLoad()
{
    l1TilingData_.stepKa_ = ops::CeilDiv(inputParams_.kSize, baseK_);
    l1TilingData_.depthKa_ = l1TilingData_.stepKa_;

    uint64_t singleCoreASize = GetSizeWithDataType(baseM_ * inputParams_.kSize, inputParams_.aDtype);
    uint64_t alignedSingleCoreASize = ops::CeilAlign(singleCoreASize, L1_ALIGN_SIZE);
    OP_TILING_CHECK(
        leftL1Size_ < alignedSingleCoreASize,
        CUBE_INNER_ERR_REPORT(
            inputParams_.opName, "Subtraction underflow: leftL1Size(%lu) - alignedSingleCoreASize(%lu).", leftL1Size_,
            alignedSingleCoreASize),
        return false);
    uint64_t leftL1SizeAfterFullA = leftL1Size_ - alignedSingleCoreASize;
    uint64_t baseBSize = GetSizeWithDataType(baseN_ * baseK_, inputParams_.bDtype);
    if (inputParams_.isMxPerGroup) {
        uint64_t singleCoreScaleASize = GetSizeWithDataType(
            std::min(inputParams_.mSize, baseM_) *
                ops::CeilAlign(ops::CeilDiv(inputParams_.kSize, MX_GROUP_SIZE), MXFP_MULTI_BASE_SIZE),
            inputParams_.perTokenScaleDtype);
        uint64_t alignedSingleCoreScaleASize = ops::CeilAlign(singleCoreScaleASize, L1_ALIGN_SIZE);
        l1TilingData_.scaleFactorA_ = 1UL;
        OP_TILING_CHECK(
            leftL1SizeAfterFullA < alignedSingleCoreScaleASize,
            CUBE_INNER_ERR_REPORT(
                inputParams_.opName,
                "Subtraction underflow: leftL1SizeAfterFullA(%lu) - alignedSingleCoreScaleASize(%lu).",
                leftL1SizeAfterFullA, alignedSingleCoreScaleASize),
            return false);
        leftL1SizeAfterFullA -= alignedSingleCoreScaleASize;
        l1TilingData_.depthKb_ = GetDepthB1AfullLoad(leftL1SizeAfterFullA);
        l1TilingData_.stepKb_ =
            l1TilingData_.depthKb_ == 1UL ? l1TilingData_.depthKb_ : l1TilingData_.depthKb_ / DB_SIZE;
        uint64_t alignedBSize = ops::CeilAlign(l1TilingData_.depthKb_ * baseBSize, L1_ALIGN_SIZE);
        OP_TILING_CHECK(
            leftL1SizeAfterFullA < alignedBSize,
            CUBE_INNER_ERR_REPORT(
                inputParams_.opName, "Subtraction underflow: leftL1SizeAfterFullA(%lu) - alignedBSize(%lu).",
                leftL1SizeAfterFullA, alignedBSize),
            return false);
        l1TilingData_.scaleFactorB_ = GetScaleFactorBAfullLoad(leftL1SizeAfterFullA - alignedBSize);
    } else {
        l1TilingData_.depthKb_ = GetDepthB1AfullLoad(leftL1SizeAfterFullA);
        l1TilingData_.stepKb_ =
            l1TilingData_.depthKb_ == 1UL ? l1TilingData_.depthKb_ : l1TilingData_.depthKb_ / DB_SIZE;
    }
    return true;
}

bool L1TilingDataCalculator::CalStepKs()
{
    l1TilingData_.stepKa_ = l1TilingData_.depthKa_ / DB_SIZE;
    l1TilingData_.stepKb_ = l1TilingData_.depthKb_ / DB_SIZE;

    if (l1TilingData_.stepKa_ * baseK_ > inputParams_.kSize) {
        l1TilingData_.stepKa_ = ops::CeilDiv(inputParams_.kSize, baseK_);
    }
    if (l1TilingData_.stepKb_ * baseK_ > inputParams_.kSize) {
        l1TilingData_.stepKb_ = ops::CeilDiv(inputParams_.kSize, baseK_);
    }
    OP_TILING_CHECK(
        l1TilingData_.stepKa_ == 0UL || l1TilingData_.stepKb_ == 0UL,
        CUBE_INNER_ERR_REPORT(
            inputParams_.opName,
            "Invalid stepK before alignment: stepKa(%lu), stepKb(%lu), depthKa(%lu), depthKb(%lu).",
            l1TilingData_.stepKa_, l1TilingData_.stepKb_, l1TilingData_.depthKa_, l1TilingData_.depthKb_),
        return false);
    if (l1TilingData_.stepKa_ > l1TilingData_.stepKb_) {
        l1TilingData_.stepKa_ = l1TilingData_.stepKa_ / l1TilingData_.stepKb_ * l1TilingData_.stepKb_;
    }
    if (l1TilingData_.stepKb_ > l1TilingData_.stepKa_) {
        l1TilingData_.stepKb_ = l1TilingData_.stepKb_ / l1TilingData_.stepKa_ * l1TilingData_.stepKa_;
    }
    if (inputParams_.isPerBlock || inputParams_.isMxPerGroup) {
        // Limit max stepK to 4 to avoid issue queue stalls.
        l1TilingData_.stepKa_ = std::min(l1TilingData_.stepKa_, 4UL);
        l1TilingData_.stepKb_ = std::min(l1TilingData_.stepKb_, 4UL);
    }
    if (inputParams_.isPerBlock && inputParams_.hasBias) {
        l1TilingData_.stepKa_ = (l1TilingData_.stepKa_ + l1TilingData_.stepKb_) / NUM_HALF;
        l1TilingData_.stepKb_ = l1TilingData_.stepKa_;
    }
    l1TilingData_.depthKa_ = l1TilingData_.stepKa_ * DB_SIZE;
    l1TilingData_.depthKb_ = l1TilingData_.stepKb_ * DB_SIZE;
    return true;
}

bool L1TilingDataCalculator::CalScaleFactors(
    uint64_t baseASize, uint64_t baseBSize, uint64_t baseScaleASize, uint64_t baseScaleBSize)
{
    uint64_t biasDtypeSize = ge::GetSizeByDataType(inputParams_.biasDtype);
    uint64_t baseBiasSize = inputParams_.hasBias ? baseN_ * biasDtypeSize : 0UL;
    OP_TILING_CHECK(
        baseScaleASize == 0UL || baseScaleBSize == 0UL,
        CUBE_INNER_ERR_REPORT(
            inputParams_.opName,
            "Invalid scale factor divisor: baseScaleASize(%lu) and baseScaleBSize(%lu) should be greater than 0.",
            baseScaleASize, baseScaleBSize),
        return false);

    // Compute scaleFactorA/scaleFactorB from K-axis constraints first.
    uint64_t scaleFactorAMax = std::min(MTE2_MIN_LOAD_SIZE / baseScaleASize, static_cast<uint64_t>(SCALER_FACTOR_MAX));
    uint64_t scaleFactorBMax = std::min(MTE2_MIN_LOAD_SIZE / baseScaleBSize, static_cast<uint64_t>(SCALER_FACTOR_MAX));
    uint64_t scaleFactorA = inputParams_.kSize / (l1TilingData_.stepKa_ * baseK_);
    uint64_t scaleFactorB = inputParams_.kSize / (l1TilingData_.stepKb_ * baseK_);
    l1TilingData_.scaleFactorA_ = std::max(static_cast<uint64_t>(SCALER_FACTOR_MIN), scaleFactorA);
    l1TilingData_.scaleFactorB_ = std::max(static_cast<uint64_t>(SCALER_FACTOR_MIN), scaleFactorB);
    l1TilingData_.scaleFactorA_ = std::min(scaleFactorAMax, l1TilingData_.scaleFactorA_);
    l1TilingData_.scaleFactorB_ = std::min(scaleFactorBMax, l1TilingData_.scaleFactorB_);

    // Then apply the L1 size constraint.
    uint64_t usedASize = l1TilingData_.depthKa_ * baseASize;
    uint64_t usedBSize = l1TilingData_.depthKb_ * baseBSize;
    OP_TILING_CHECK(
        compileInfo_.l1Size < usedASize,
        CUBE_INNER_ERR_REPORT(
            inputParams_.opName, "Subtraction underflow: l1Size(%lu) - usedASize(%lu).", compileInfo_.l1Size,
            usedASize),
        return false);
    uint64_t leftL1Size = compileInfo_.l1Size - usedASize;
    OP_TILING_CHECK(
        leftL1Size < usedBSize,
        CUBE_INNER_ERR_REPORT(
            inputParams_.opName, "Subtraction underflow: leftL1Size(%lu) - usedBSize(%lu).", leftL1Size, usedBSize),
        return false);
    leftL1Size -= usedBSize;
    OP_TILING_CHECK(
        leftL1Size < baseBiasSize,
        CUBE_INNER_ERR_REPORT(
            inputParams_.opName, "Subtraction underflow: leftL1Size(%lu) - baseBiasSize(%lu).", leftL1Size,
            baseBiasSize),
        return false);
    leftL1Size -= baseBiasSize;
    uint64_t scaleADivisor = l1TilingData_.depthKa_ * baseScaleASize;
    uint64_t scaleBDivisor = l1TilingData_.depthKb_ * baseScaleBSize;
    uint64_t scaleInitDivisor = scaleADivisor + scaleBDivisor;
    uint64_t scaleInit = leftL1Size / scaleInitDivisor;
    if (l1TilingData_.scaleFactorA_ <= scaleInit && l1TilingData_.scaleFactorB_ > scaleInit) {
        uint64_t usedScaleASize = l1TilingData_.scaleFactorA_ * l1TilingData_.depthKa_ * baseScaleASize;
        leftL1Size -= usedScaleASize;
        l1TilingData_.scaleFactorB_ = std::min(leftL1Size / scaleBDivisor, l1TilingData_.scaleFactorB_);
    } else if (l1TilingData_.scaleFactorB_ <= scaleInit && l1TilingData_.scaleFactorA_ > scaleInit) {
        uint64_t usedScaleBSize = l1TilingData_.scaleFactorB_ * l1TilingData_.depthKb_ * baseScaleBSize;
        leftL1Size -= usedScaleBSize;
        l1TilingData_.scaleFactorA_ = std::min(leftL1Size / scaleADivisor, l1TilingData_.scaleFactorA_);
    } else if (l1TilingData_.scaleFactorA_ > scaleInit && l1TilingData_.scaleFactorB_ > scaleInit) {
        uint64_t usedScaleInitSize =
            scaleInit * l1TilingData_.depthKb_ * baseScaleBSize + scaleInit * l1TilingData_.depthKa_ * baseScaleASize;
        leftL1Size -= usedScaleInitSize;
        uint64_t scaleASec = std::min(leftL1Size / scaleADivisor, l1TilingData_.scaleFactorA_ - scaleInit);
        uint64_t scaleBSec = std::min(leftL1Size / scaleBDivisor, l1TilingData_.scaleFactorB_ - scaleInit);
        l1TilingData_.scaleFactorA_ = scaleASec >= scaleBSec ? scaleASec + scaleInit : scaleInit;
        l1TilingData_.scaleFactorB_ = scaleASec < scaleBSec ? scaleBSec + scaleInit : scaleInit;
    }
    return true;
}

uint64_t L1TilingDataCalculator::GetDepthA1B1(uint64_t leftSize, uint64_t perDepthSize, uint64_t depthInit) const
{
    if (perDepthSize == 0UL) {
        return depthInit;
    }
    if (depthInit > 1UL && perDepthSize > DB_SIZE * MTE2_MIN_LOAD_SIZE) {
        return depthInit;
    }
    uint64_t depthScale = leftSize / perDepthSize;
    if (depthInit > 1UL) {
        uint64_t baseKSize = GetSizeWithDataType(baseK_, inputParams_.aDtype);
        if (baseKSize == 0UL) {
            return depthInit;
        }
        while ((depthScale * baseKSize) % BASIC_BLOCK_SIZE_512 != 0UL &&
               (depthScale * baseKSize) > BASIC_BLOCK_SIZE_512) {
            depthScale -= 1UL;
        }
        if ((depthScale * baseKSize) % BASIC_BLOCK_SIZE_512 != 0UL &&
            (depthScale * baseKSize) >= BASIC_BLOCK_SIZE_256) {
            depthScale = BASIC_BLOCK_SIZE_256 / baseKSize;
        }
        depthScale = std::max(depthScale, 1UL);
    } else {
        // Depth grows by powers of two.
        constexpr uint64_t INDEX = 2UL;
        depthScale = 1UL;
        while (depthScale * perDepthSize < leftSize) {
            depthScale *= INDEX;
        }
        depthScale = depthScale == 1UL ? depthScale : depthScale / INDEX;
    }
    return depthInit * depthScale;
}

uint64_t L1TilingDataCalculator::GetDepthB1AfullLoad(uint64_t leftSize) const
{
    // Align inner axis to 128B.
    uint64_t stepKbBase = 1UL;
    if (inputParams_.transB) {
        uint64_t singleBaseKSize = GetSizeWithDataType(baseK_, inputParams_.bDtype);
        if (singleBaseKSize != 0UL && singleBaseKSize < L2_ALIGN_SIZE) {
            stepKbBase = ops::CeilDiv(L2_ALIGN_SIZE, singleBaseKSize);
        }
    }

    // Keep single-copy size at least 32KB without exceeding L1.
    uint64_t scaleBBaseK =
        inputParams_.isMxPerGroup ? ops::CeilAlign(ops::CeilDiv(baseK_, MX_GROUP_SIZE), MXFP_MULTI_BASE_SIZE) : 0UL;
    uint64_t baseBSize = GetSizeWithDataType(baseN_ * (baseK_ + scaleBBaseK) * stepKbBase, inputParams_.bDtype);
    if (baseBSize == 0UL) {
        return DB_SIZE;
    }
    uint64_t stepKbBaseScale = 1UL;
    if (leftSize >= MTE2_MIN_LOAD_SIZE * DB_SIZE) {
        stepKbBaseScale = ops::CeilDiv(MTE2_MIN_LOAD_SIZE, baseBSize);
    } else {
        stepKbBaseScale = ops::CeilDiv(leftSize / DB_SIZE, baseBSize);
    }
    stepKbBase *= stepKbBaseScale;

    // stepKb 1 can block MTE1 parallelism; use at least 2 when L1 allows.
    uint64_t refinedStepKb = 2UL;
    if (stepKbBase == 1UL && inputParams_.kSize > baseK_ && leftSize > baseBSize * refinedStepKb) {
        stepKbBase = refinedStepKb;
    }
    return stepKbBase * DB_SIZE;
}

uint64_t L1TilingDataCalculator::GetScaleFactorBAfullLoad(uint64_t leftSize) const
{
    uint64_t scaleBaseK = ops::CeilAlign(ops::CeilDiv(baseK_, MX_GROUP_SIZE), MXFP_MULTI_BASE_SIZE);
    uint64_t baseScaleBSize = GetSizeWithDataType(baseN_ * scaleBaseK, inputParams_.scaleDtype);
    if (baseScaleBSize == 0UL || l1TilingData_.stepKb_ == 0UL || l1TilingData_.depthKb_ == 0UL) {
        return 1UL;
    }

    uint64_t scaleFactorBBase = 1UL;
    if (inputParams_.transB) {
        // K is the inner axis.
        uint64_t singleScaleBBaseKSize = GetSizeWithDataType(scaleBaseK, inputParams_.scaleDtype);
        if (singleScaleBBaseKSize != 0UL && singleScaleBBaseKSize < L2_ALIGN_SIZE) {
            scaleFactorBBase = ops::CeilDiv(L2_ALIGN_SIZE, singleScaleBBaseKSize);
        }
    }

    uint64_t scaleFactorBMaxFromK = inputParams_.kSize / (l1TilingData_.stepKb_ * baseK_);
    scaleFactorBMaxFromK = std::min(static_cast<uint64_t>(SCALER_FACTOR_MAX), scaleFactorBMaxFromK);
    scaleFactorBMaxFromK = std::max(static_cast<uint64_t>(SCALER_FACTOR_MIN), scaleFactorBMaxFromK);

    uint64_t scaleFactorB = 1UL;
    uint64_t scaleFactorBMax =
        std::min(MTE2_MIN_LOAD_SIZE * DB_SIZE, leftSize) / (baseScaleBSize * l1TilingData_.depthKb_);
    if (scaleFactorBMax != 0UL) {
        if (scaleFactorBBase <= scaleFactorBMaxFromK && scaleFactorBMax >= scaleFactorBBase) {
            // Keep 128B inner-axis alignment while meeting 32KB copy size and L1 limits.
            scaleFactorB = std::min(scaleFactorBMax / scaleFactorBBase * scaleFactorBBase, scaleFactorBMaxFromK);
        } else {
            // If all scaleB data along K cannot satisfy 128B alignment, use copy size to compute scaleFactorB.
            scaleFactorB = std::min(scaleFactorBMax, scaleFactorBMaxFromK);
        }
    }
    return scaleFactorB;
}

} // namespace optiling
