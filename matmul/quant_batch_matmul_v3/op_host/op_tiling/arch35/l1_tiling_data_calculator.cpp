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
#include "quant_batch_matmul_v3_tiling_util.h"

namespace {

constexpr uint64_t MTE2_MIN_LOAD_SIZE = 32768UL;
constexpr uint64_t MIN_CARRY_DATA_SIZE_32K = 32UL * 1024UL;
constexpr uint64_t FULL_LOAD_DATA_SIZE_64K = 64UL * 1024UL;
constexpr uint64_t CACHE_LINE_512B = 512UL;
constexpr uint32_t MAX_STEPK_With_BL1_FULL = 8U;

uint64_t GetSizeWithDataType(uint64_t shape, ge::DataType dtype)
{
    if (dtype == ge::DT_FLOAT4_E2M1 || dtype == ge::DT_FLOAT4_E1M2 || dtype == ge::DT_INT4) {
        return (shape + 1UL) >> 1UL;
    }
    return shape * static_cast<uint64_t>(ge::GetSizeByDataType(dtype));
}

uint64_t GetSizeWithDataTypeLut(uint64_t shape, ge::DataType dtype)
{
    // lut查表逻辑: 原始数据DT_INT2和DT_UINT1，查表后转DT_INT4; 原始数据DT_INT4, 查表后转DT_INT8
    bool is4BitInput = (dtype == ge::DT_INT2 || dtype == ge::DT_UINT1);
    bool is8BitInput = dtype == ge::DT_INT4;

    if (is4BitInput) {
        return (shape + 1) >> 1;
    } else if (is8BitInput) {
        return shape;
    } else {
        return shape * static_cast<uint64_t>(ge::GetSizeByDataType(dtype));
    }
}
} // namespace

namespace optiling {

L1TilingDataCalculator::L1TilingDataCalculator(const QuantBatchMatmulInfo& inputParams,
                                               const QuantBatchMatmulV3CompileInfo& compileInfo, uint64_t baseM,
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
            isAFullLoad_ = true;
            return ComputeL1TilingAL1FullLoad();
        case L1TilingMode::B_L1_FULL_LOAD:
            isBFullLoad_ = true;
            return ComputeL1TilingBL1FullLoad();
        case L1TilingMode::PASS_OPTIMIZED:
            return ComputeL1TilingMmadS8S4();
        case L1TilingMode::PASS_OPTIMIZED_A_FULL_LOAD:
            isAFullLoad_ = true;
            return ComputeL1TilingMmadS8S4();
        case L1TilingMode::PASS_OPTIMIZED_B_FULL_LOAD:
            isBFullLoad_ = true;
            return ComputeL1TilingMmadS8S4();
        case L1TilingMode::PASS_OPTIMIZED_AB_FULL_LOAD:
            isAFullLoad_ = true;
            isBFullLoad_ = true;
            isABFullLoad_ = true;
            return ComputeL1TilingMmadS8S4();
        case L1TilingMode::DEPTH_MAXIMIZED:
            return ComputeL1TilingMmadS8S4LUT();
        case L1TilingMode::DEPTH_MAXIMIZED_A_FULL_LOAD:
            isAFullLoad_ = true;
            return ComputeL1TilingMmadS8S4LUT();
        case L1TilingMode::DEFAULT:
        default:
            return ComputeL1TilingDefault();
    }
}

const L1TilingData& L1TilingDataCalculator::GetOutput() const { return l1TilingData_; }

bool L1TilingDataCalculator::ValidateInput() const
{
    OP_TILING_CHECK(baseM_ == 0UL || baseN_ == 0UL || baseK_ == 0UL,
                    CUBE_INNER_ERR_REPORT(
                        inputParams_.opName,
                        "BaseM, baseN and baseK should be greater than 0, but baseM: %lu, baseN: %lu, baseK: %lu.",
                        baseM_, baseN_, baseK_),
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
    uint64_t biasL1Size = inputParams_.hasBias ? baseN_ * biasDtypeSize * qmmv3_tiling_const::DOUBLE_BUFFER_NUM : 0UL;
    uint64_t scaleL1Size = inputParams_.isPerChannel ? baseN_ * scaleDtypeSize * qmmv3_tiling_const::DOUBLE_BUFFER_NUM :
                                                       0UL;
    OP_TILING_CHECK(compileInfo_.l1Size < biasL1Size,
                    CUBE_INNER_ERR_REPORT(inputParams_.opName, "Subtraction underflow: l1Size(%lu) - biasL1Size(%lu).",
                                          compileInfo_.l1Size, biasL1Size),
                    return false);
    uint64_t leftL1SizeAfterBias = compileInfo_.l1Size - biasL1Size;
    OP_TILING_CHECK(leftL1SizeAfterBias < scaleL1Size,
                    CUBE_INNER_ERR_REPORT(inputParams_.opName,
                                          "Subtraction underflow: leftL1SizeAfterBias(%lu) - scaleL1Size(%lu).",
                                          leftL1SizeAfterBias, scaleL1Size),
                    return false);
    leftL1Size_ = leftL1SizeAfterBias - scaleL1Size;
    return true;
}

bool L1TilingDataCalculator::ComputeL1TilingDefault()
{
    if (inputParams_.isMxPerGroup) {
        uint64_t baseASize = GetSizeWithDataType(baseM_ * baseK_, inputParams_.aDtype);
        uint64_t baseBSize = GetSizeWithDataType(baseN_ * baseK_, inputParams_.bDtype);
        uint64_t scaleBaseK = ops::CeilAlign(ops::CeilDiv(baseK_, qmmv3_tiling_const::MX_GROUP_SIZE),
                                             qmmv3_tiling_const::MXFP_MULTI_BASE_SIZE);
        uint64_t baseScaleASize = GetSizeWithDataType(scaleBaseK * baseM_, inputParams_.perTokenScaleDtype);
        uint64_t baseScaleBSize = GetSizeWithDataType(scaleBaseK * baseN_, inputParams_.scaleDtype);
        uint64_t baseL1Size = baseASize + baseBSize + baseScaleASize + baseScaleBSize;
        OP_TILING_CHECK(
            baseL1Size == 0UL,
            CUBE_INNER_ERR_REPORT(inputParams_.opName,
                                  "Invalid MX L1 divisor: baseL1Size(%lu) should be greater than 0.", baseL1Size),
            return false);
        uint64_t depthInit = GetDepthA1B1();
        l1TilingData_.depthKa_ = depthInit;
        l1TilingData_.depthKb_ = depthInit;
        if (!CalStepKs() || !CalScaleFactors(baseASize, baseBSize)) {
            return false;
        }
    } else {
        // With baseM/baseN/baseK <= 256 and 1B elements, each A/B base block is at most 16KB.
        // Split remaining L1 evenly between A and B to estimate depth.
        uint64_t baseASize = GetSizeWithDataType(baseM_ * baseK_, inputParams_.aDtype);
        uint64_t baseBSize = GetSizeWithDataType(baseN_ * baseK_, inputParams_.bDtype);
        OP_TILING_CHECK(
            baseASize == 0UL || baseBSize == 0UL,
            CUBE_INNER_ERR_REPORT(inputParams_.opName,
                                  "Invalid L1 base size: baseASize(%lu) and baseBSize(%lu) should be greater than 0.",
                                  baseASize, baseBSize),
            return false);
        l1TilingData_.depthKa_ = leftL1Size_ / qmmv3_tiling_const::NUM_HALF / baseASize;
        l1TilingData_.depthKb_ = leftL1Size_ / qmmv3_tiling_const::NUM_HALF / baseBSize;
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

    uint64_t alignedSingleCoreASize = GetSingleCoreAFullLoadSize();
    OP_TILING_CHECK(leftL1Size_ < alignedSingleCoreASize,
                    CUBE_INNER_ERR_REPORT(inputParams_.opName,
                                          "Subtraction underflow: leftL1Size(%lu) - alignedSingleCoreASize(%lu).",
                                          leftL1Size_, alignedSingleCoreASize),
                    return false);
    uint64_t leftL1SizeAfterFullA = leftL1Size_ - alignedSingleCoreASize;
    uint64_t baseBSize = GetSizeWithDataType(baseN_ * baseK_, inputParams_.bDtype);
    if (inputParams_.isMxPerGroup) {
        uint64_t singleCoreScaleASize = GetSizeWithDataType(
            std::min(inputParams_.mSize, baseM_) *
                ops::CeilAlign(ops::CeilDiv(inputParams_.kSize, qmmv3_tiling_const::MX_GROUP_SIZE),
                               qmmv3_tiling_const::MXFP_MULTI_BASE_SIZE),
            inputParams_.perTokenScaleDtype);
        uint64_t alignedSingleCoreScaleASize = ops::CeilAlign(singleCoreScaleASize, qmmv3_tiling_const::L1_ALIGN_SIZE);
        l1TilingData_.scaleFactorA_ = 1UL;
        OP_TILING_CHECK(leftL1SizeAfterFullA < alignedSingleCoreScaleASize,
                        CUBE_INNER_ERR_REPORT(
                            inputParams_.opName,
                            "Subtraction underflow: leftL1SizeAfterFullA(%lu) - alignedSingleCoreScaleASize(%lu).",
                            leftL1SizeAfterFullA, alignedSingleCoreScaleASize),
                        return false);
        leftL1SizeAfterFullA -= alignedSingleCoreScaleASize;
        l1TilingData_.depthKb_ = GetDepthB1AfullLoad(leftL1SizeAfterFullA);
        l1TilingData_.stepKb_ = l1TilingData_.depthKb_ == 1UL ?
                                    l1TilingData_.depthKb_ :
                                    l1TilingData_.depthKb_ / qmmv3_tiling_const::DOUBLE_BUFFER_NUM;
        uint64_t alignedBSize = ops::CeilAlign(l1TilingData_.depthKb_ * baseBSize, qmmv3_tiling_const::L1_ALIGN_SIZE);
        OP_TILING_CHECK(leftL1SizeAfterFullA < alignedBSize,
                        CUBE_INNER_ERR_REPORT(inputParams_.opName,
                                              "Subtraction underflow: leftL1SizeAfterFullA(%lu) - alignedBSize(%lu).",
                                              leftL1SizeAfterFullA, alignedBSize),
                        return false);
        l1TilingData_.scaleFactorB_ = GetScaleFactorBAfullLoad(leftL1SizeAfterFullA - alignedBSize);
    } else {
        l1TilingData_.depthKb_ = GetDepthB1AfullLoad(leftL1SizeAfterFullA);
        l1TilingData_.stepKb_ = l1TilingData_.depthKb_ == 1UL ?
                                    l1TilingData_.depthKb_ :
                                    l1TilingData_.depthKb_ / qmmv3_tiling_const::DOUBLE_BUFFER_NUM;
    }
    return true;
}

bool L1TilingDataCalculator::ComputeL1TilingBL1FullLoad()
{
    l1TilingData_.stepKb_ = ops::CeilDiv(inputParams_.kSize, baseK_);
    l1TilingData_.depthKb_ = l1TilingData_.stepKb_;

    uint64_t alignedSingleCoreBSize = GetSingleCoreBFullLoadSize();
    OP_TILING_CHECK(leftL1Size_ < alignedSingleCoreBSize,
                    CUBE_INNER_ERR_REPORT(inputParams_.opName,
                                          "Subtraction underflow: leftL1Size(%lu) - alignedSingleCoreBSize(%lu).",
                                          leftL1Size_, alignedSingleCoreBSize),
                    return false);
    uint64_t leftL1SizeAfterFullB = leftL1Size_ - alignedSingleCoreBSize;
    l1TilingData_.depthKa_ = GetDepthA1BfullLoad(leftL1SizeAfterFullB);
    l1TilingData_.stepKa_ = l1TilingData_.depthKa_ == 1UL ?
                                l1TilingData_.depthKa_ :
                                l1TilingData_.depthKa_ / qmmv3_tiling_const::DOUBLE_BUFFER_NUM;
    return true;
}

bool L1TilingDataCalculator::CalStepKs()
{
    l1TilingData_.stepKa_ = l1TilingData_.depthKa_ / qmmv3_tiling_const::DOUBLE_BUFFER_NUM;
    l1TilingData_.stepKb_ = l1TilingData_.depthKb_ / qmmv3_tiling_const::DOUBLE_BUFFER_NUM;

    if (l1TilingData_.stepKa_ * baseK_ > inputParams_.kSize) {
        l1TilingData_.stepKa_ = ops::CeilDiv(inputParams_.kSize, baseK_);
    }
    if (l1TilingData_.stepKb_ * baseK_ > inputParams_.kSize) {
        l1TilingData_.stepKb_ = ops::CeilDiv(inputParams_.kSize, baseK_);
    }
    OP_TILING_CHECK(l1TilingData_.stepKa_ == 0UL || l1TilingData_.stepKb_ == 0UL,
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
        // The current pertile/perblock tiling constraints keep the averaged stepK within L1 capacity.
        l1TilingData_.stepKa_ = (l1TilingData_.stepKa_ + l1TilingData_.stepKb_) / qmmv3_tiling_const::NUM_HALF;
        l1TilingData_.stepKb_ = l1TilingData_.stepKa_;
    }
    l1TilingData_.depthKa_ = l1TilingData_.stepKa_ * qmmv3_tiling_const::DOUBLE_BUFFER_NUM;
    l1TilingData_.depthKb_ = l1TilingData_.stepKb_ * qmmv3_tiling_const::DOUBLE_BUFFER_NUM;
    return true;
}

bool L1TilingDataCalculator::CalScaleFactors(uint64_t baseASize, uint64_t baseBSize)
{
    uint64_t usedABSize = l1TilingData_.depthKa_ * baseASize + l1TilingData_.depthKb_ * baseBSize;
    OP_TILING_CHECK(
        leftL1Size_ < usedABSize,
        CUBE_INNER_ERR_REPORT(inputParams_.opName, "Subtraction underflow: leftL1Size(%lu) - usedABSize(%lu).",
                              leftL1Size_, usedABSize),
        return false);
    uint64_t leftL1Size = leftL1Size_ - usedABSize;

    uint64_t stepK = std::min(l1TilingData_.stepKa_, l1TilingData_.stepKb_);
    uint64_t kL1 = stepK * baseK_;

    // One MX scale group covers 64 K elements and stores two scale values. A/B scale buffers are double-buffered,
    // so this is the L1 cost for extending scaleKL1 by one 64-element group.
    uint64_t scaleGroupL1Size = (GetSizeWithDataType(baseM_, inputParams_.perTokenScaleDtype) +
                                 GetSizeWithDataType(baseN_, inputParams_.scaleDtype)) *
                                qmmv3_tiling_const::MXFP_MULTI_BASE_SIZE * qmmv3_tiling_const::DOUBLE_BUFFER_NUM;
    uint64_t maxScaleKL1ByL1 = leftL1Size / scaleGroupL1Size * qmmv3_tiling_const::MXFP_DIVISOR_SIZE;
    uint64_t maxScaleKL1 = ops::CeilAlign(inputParams_.kSize, kL1);
    uint64_t scaleKL1 = std::min(maxScaleKL1ByL1, maxScaleKL1);
    // Libapi scaleFactor is scaleKL1 / kL1, so scaleKL1 must be rounded down to a kL1 multiple.
    scaleKL1 = scaleKL1 / kL1 * kL1;

    OP_TILING_CHECK(
        scaleKL1 == 0UL,
        CUBE_INNER_ERR_REPORT(inputParams_.opName, "Insufficient L1 for MX scale buffer. leftL1Size(%lu), kL1(%lu).",
                              leftL1Size, kL1),
        return false);

    uint64_t scaleFactor = scaleKL1 / kL1;
    l1TilingData_.scaleFactorA_ = scaleFactor;
    l1TilingData_.scaleFactorB_ = scaleFactor;
    return true;
}

uint64_t L1TilingDataCalculator::GetDepthA1B1() const
{
    constexpr uint64_t INDEX = 2UL;
    uint64_t depth = 1UL;
    uint64_t scaleKL1 = std::min(inputParams_.kSize, qmmv3_tiling_const::ESTIMATED_SCALE_K);
    uint64_t baseABSize = GetSizeWithDataType(baseM_ * baseK_, inputParams_.aDtype) +
                          GetSizeWithDataType(baseN_ * baseK_, inputParams_.bDtype);
    uint64_t baseScaleSize = GetSizeWithDataType(baseM_, inputParams_.perTokenScaleDtype) +
                             GetSizeWithDataType(baseN_, inputParams_.scaleDtype);
    // Reserve a bounded scale window while searching depth. If depth later covers more K than the estimate,
    // scaleL1Size is raised accordingly before accepting that depth.
    uint64_t scaleL1Size = baseScaleSize * ops::CeilDiv(scaleKL1, qmmv3_tiling_const::MXFP_DIVISOR_SIZE) *
                           qmmv3_tiling_const::MXFP_MULTI_BASE_SIZE * qmmv3_tiling_const::DOUBLE_BUFFER_NUM;
    while (depth * baseABSize + scaleL1Size <= leftL1Size_) {
        depth *= INDEX;
        uint64_t kL1 = depth / qmmv3_tiling_const::DOUBLE_BUFFER_NUM * baseK_;
        if (kL1 > scaleKL1) {
            scaleKL1 = kL1;
            scaleL1Size = baseScaleSize * ops::CeilDiv(scaleKL1, qmmv3_tiling_const::MXFP_DIVISOR_SIZE) *
                          qmmv3_tiling_const::MXFP_MULTI_BASE_SIZE * qmmv3_tiling_const::DOUBLE_BUFFER_NUM;
        }
    }
    return depth == 1UL ? depth : depth / INDEX;
}

uint64_t L1TilingDataCalculator::GetDepthB1AfullLoad(uint64_t leftSize) const
{
    // Align inner axis to 128B.
    uint64_t stepKbBase = 1UL;
    if (inputParams_.transB) {
        uint64_t singleBaseKSize = GetSizeWithDataType(baseK_, inputParams_.bDtype);
        if (singleBaseKSize != 0UL && singleBaseKSize < qmmv3_tiling_const::L2_ALIGN_SIZE) {
            stepKbBase = ops::CeilDiv(qmmv3_tiling_const::L2_ALIGN_SIZE, singleBaseKSize);
        }
    }

    // Keep single-copy size at least 32KB without exceeding L1.
    uint64_t scaleBBaseK = inputParams_.isMxPerGroup ?
                               ops::CeilAlign(ops::CeilDiv(baseK_, qmmv3_tiling_const::MX_GROUP_SIZE),
                                              qmmv3_tiling_const::MXFP_MULTI_BASE_SIZE) :
                               0UL;
    uint64_t baseBSize = GetSizeWithDataType(baseN_ * (baseK_ + scaleBBaseK) * stepKbBase, inputParams_.bDtype);
    if (baseBSize == 0UL) {
        return qmmv3_tiling_const::DOUBLE_BUFFER_NUM;
    }
    uint64_t stepKbBaseScale = 1UL;
    if (leftSize >= MTE2_MIN_LOAD_SIZE * qmmv3_tiling_const::DOUBLE_BUFFER_NUM) {
        stepKbBaseScale = ops::CeilDiv(MTE2_MIN_LOAD_SIZE, baseBSize);
    } else {
        stepKbBaseScale = ops::CeilDiv(leftSize / qmmv3_tiling_const::DOUBLE_BUFFER_NUM, baseBSize);
    }
    stepKbBase *= stepKbBaseScale;

    // stepKb 1 can block MTE1 parallelism; use at least 2 when L1 allows.
    uint64_t refinedStepKb = 2UL;
    if (stepKbBase == 1UL && inputParams_.kSize > baseK_ && leftSize > baseBSize * refinedStepKb) {
        stepKbBase = refinedStepKb;
    }
    return stepKbBase * qmmv3_tiling_const::DOUBLE_BUFFER_NUM;
}

uint64_t L1TilingDataCalculator::GetDepthA1BfullLoad(uint64_t leftSize) const
{
    // Align inner axis to 128B.
    uint64_t stepKaBase = 1UL;
    if (!inputParams_.transA) {
        uint64_t singleBaseKSize = GetSizeWithDataType(baseK_, inputParams_.aDtype);
        if (singleBaseKSize != 0UL && singleBaseKSize < qmmv3_tiling_const::L2_ALIGN_SIZE) {
            stepKaBase = ops::CeilDiv(qmmv3_tiling_const::L2_ALIGN_SIZE, singleBaseKSize);
        }
    }

    uint64_t baseASize = GetSizeWithDataType(baseM_ * baseK_ * stepKaBase, inputParams_.aDtype);
    if (baseASize == 0UL) {
        return qmmv3_tiling_const::DOUBLE_BUFFER_NUM;
    }
    uint64_t stepKaBaseScale = 1UL;
    if (leftSize >= MTE2_MIN_LOAD_SIZE * qmmv3_tiling_const::DOUBLE_BUFFER_NUM) {
        stepKaBaseScale = ops::CeilDiv(MTE2_MIN_LOAD_SIZE, baseASize);
    } else {
        stepKaBaseScale = ops::CeilDiv(leftSize / qmmv3_tiling_const::DOUBLE_BUFFER_NUM, baseASize);
    }
    stepKaBase *= stepKaBaseScale;

    uint64_t refinedStepKa = 2UL;
    if (stepKaBase == 1UL && inputParams_.kSize > baseK_ && leftSize > baseASize * refinedStepKa) {
        stepKaBase = refinedStepKa;
    }
    return stepKaBase * qmmv3_tiling_const::DOUBLE_BUFFER_NUM;
}

uint64_t L1TilingDataCalculator::GetScaleFactorBAfullLoad(uint64_t leftSize) const
{
    uint64_t scaleBaseK = ops::CeilAlign(ops::CeilDiv(baseK_, qmmv3_tiling_const::MX_GROUP_SIZE),
                                         qmmv3_tiling_const::MXFP_MULTI_BASE_SIZE);
    uint64_t baseScaleBSize = GetSizeWithDataType(baseN_ * scaleBaseK, inputParams_.scaleDtype);
    if (baseScaleBSize == 0UL || l1TilingData_.stepKb_ == 0UL || l1TilingData_.depthKb_ == 0UL) {
        return 1UL;
    }

    uint64_t scaleFactorBBase = 1UL;
    if (inputParams_.transB) {
        // K is the inner axis.
        uint64_t singleScaleBBaseKSize = GetSizeWithDataType(scaleBaseK, inputParams_.scaleDtype);
        if (singleScaleBBaseKSize != 0UL && singleScaleBBaseKSize < qmmv3_tiling_const::L2_ALIGN_SIZE) {
            scaleFactorBBase = ops::CeilDiv(qmmv3_tiling_const::L2_ALIGN_SIZE, singleScaleBBaseKSize);
        }
    }

    uint64_t scaleFactorBMaxFromK = inputParams_.kSize / (l1TilingData_.stepKb_ * baseK_);
    scaleFactorBMaxFromK = std::min(static_cast<uint64_t>(qmmv3_tiling_const::SCALER_FACTOR_MAX), scaleFactorBMaxFromK);
    scaleFactorBMaxFromK = std::max(static_cast<uint64_t>(qmmv3_tiling_const::SCALER_FACTOR_MIN), scaleFactorBMaxFromK);

    uint64_t scaleFactorB = 1UL;
    uint64_t scaleFactorBMax = std::min(MTE2_MIN_LOAD_SIZE * qmmv3_tiling_const::DOUBLE_BUFFER_NUM, leftSize) /
                               (baseScaleBSize * l1TilingData_.depthKb_);
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

bool L1TilingDataCalculator::ComputeL1TilingMmadS8S4()
{
    l1TilingData_.stepKa_ = 1UL;
    l1TilingData_.stepKb_ = 1UL;
    l1TilingData_.depthKa_ = 1UL;
    l1TilingData_.depthKb_ = 1UL;
    if (isABFullLoad_) {
        return true;
    }

    uint64_t maxStepK = ops::CeilDiv(inputParams_.kSize, baseK_);
    CarryDataSizePass(leftL1Size_, maxStepK);
    BalanceStepKPass(leftL1Size_);
    PostCacheLinePass(leftL1Size_, maxStepK);

    l1TilingData_.stepKa_ = isAFullLoad_ ? 1UL : l1TilingData_.stepKa_;
    l1TilingData_.depthKa_ = isAFullLoad_ ? l1TilingData_.stepKa_ :
                                            l1TilingData_.stepKa_ * qmmv3_tiling_const::DOUBLE_BUFFER_NUM;
    l1TilingData_.stepKb_ = isBFullLoad_ ? 1UL : l1TilingData_.stepKb_;
    l1TilingData_.depthKb_ = isBFullLoad_ ? l1TilingData_.stepKb_ :
                                            l1TilingData_.stepKb_ * qmmv3_tiling_const::DOUBLE_BUFFER_NUM;
    return true;
}

bool L1TilingDataCalculator::ComputeL1TilingMmadS8S4LUT()
{
    // LUT 场景采用mm api Norm模板，stepK无意义，默认1
    l1TilingData_.stepKa_ = 1UL;
    l1TilingData_.stepKb_ = 1UL;

    // LUT 场景采用mm api Norm模板，depthA1 depthB1尽可能用满L1空间。分asw和al1full两种情况讨论
    l1TilingData_.depthKa_ = 1UL;
    l1TilingData_.depthKb_ = 1UL;

    uint64_t maxDepth = ops::CeilDiv(inputParams_.kSize, baseK_);

    uint64_t oneBaseADataSize = GetSizeWithDataType(baseM_ * baseK_, inputParams_.aDtype);
    uint64_t oneBaseBDataSize = GetSizeWithDataTypeLut(baseN_ * baseK_, inputParams_.bDtype);

    if (isAFullLoad_) {
        OP_TILING_CHECK(leftL1Size_ < GetSingleCoreAFullLoadSize(),
                        CUBE_INNER_ERR_REPORT(inputParams_.opName,
                                              "Subtraction underflow: leftL1Size(%lu) - alignedSingleCoreASize(%lu).",
                                              leftL1Size_, GetSingleCoreAFullLoadSize()),
                        return false);
        l1TilingData_.depthKb_ = std::min(ops::FloorDiv(leftL1Size_ - GetSingleCoreAFullLoadSize(), oneBaseBDataSize),
                                          maxDepth);
    } else {
        l1TilingData_.depthKa_ = std::min(ops::FloorDiv(leftL1Size_, oneBaseADataSize + oneBaseBDataSize), maxDepth);
        l1TilingData_.depthKb_ = l1TilingData_.depthKa_;
    }
    return true;
}

uint64_t L1TilingDataCalculator::GetSingleCoreAFullLoadSize() const
{
    if (inputParams_.isMxPerGroup) {
        // MX kernels consume K in 64-element groups; scale bytes are reserved separately.
        return GetSizeWithDataType(baseM_ * ops::CeilAlign(inputParams_.kSize, qmmv3_tiling_const::MXFP_DIVISOR_SIZE),
                                   inputParams_.aDtype);
    }
    return baseM_ * (inputParams_.transA ?
                         GetSizeWithDataType(ops::CeilAlign(inputParams_.kSize, qmmv3_tiling_const::CUBE_BLOCK),
                                             inputParams_.aDtype) :
                         ops::CeilAlign(GetSizeWithDataType(inputParams_.kSize, inputParams_.aDtype),
                                        qmmv3_tiling_const::CUBE_REDUCE_BLOCK));
}

uint64_t L1TilingDataCalculator::GetSingleCoreBFullLoadSize() const
{
    if (inputParams_.isMxPerGroup) {
        // MX kernels consume K in 64-element groups; scale bytes are reserved separately.
        return GetSizeWithDataType(baseN_ * ops::CeilAlign(inputParams_.kSize, qmmv3_tiling_const::MXFP_DIVISOR_SIZE),
                                   inputParams_.bDtype);
    }
    return baseN_ * (inputParams_.transB ?
                         ops::CeilAlign(GetSizeWithDataType(inputParams_.kSize, inputParams_.bDtype),
                                        qmmv3_tiling_const::CUBE_REDUCE_BLOCK) :
                         GetSizeWithDataType(ops::CeilAlign(inputParams_.kSize, qmmv3_tiling_const::CUBE_BLOCK),
                                             inputParams_.bDtype));
}

bool L1TilingDataCalculator::CheckL1Size(uint64_t leftL1Size, uint64_t tempStepKa, uint64_t tempStepKb) const
{
    uint64_t singleCoreASize = tempStepKa * GetSizeWithDataType(baseM_ * baseK_, inputParams_.aDtype) *
                               qmmv3_tiling_const::DOUBLE_BUFFER_NUM;
    uint64_t singleCoreBSize = tempStepKb * GetSizeWithDataType(baseN_ * baseK_, inputParams_.bDtype) *
                               qmmv3_tiling_const::DOUBLE_BUFFER_NUM;

    if (isAFullLoad_) {
        singleCoreASize = GetSingleCoreAFullLoadSize();
    } else if (isBFullLoad_) {
        singleCoreBSize = GetSingleCoreBFullLoadSize();
    }
    return leftL1Size >= singleCoreASize + singleCoreBSize;
}

void L1TilingDataCalculator::AdjustStepK(uint64_t leftL1Size, uint64_t& tempStepKa, uint64_t& tempStepKb,
                                         bool isStepKa) const
{
    uint64_t oneBaseADataSize = GetSizeWithDataType(baseM_ * baseK_, inputParams_.aDtype) *
                                qmmv3_tiling_const::DOUBLE_BUFFER_NUM;
    uint64_t oneBaseBDataSize = GetSizeWithDataType(baseN_ * baseK_, inputParams_.bDtype) *
                                qmmv3_tiling_const::DOUBLE_BUFFER_NUM;
    if (isStepKa) {
        uint64_t singleCoreBSize = tempStepKb * oneBaseBDataSize;
        if (isBFullLoad_) {
            singleCoreBSize = GetSingleCoreBFullLoadSize();
        }
        if (leftL1Size < singleCoreBSize + oneBaseADataSize) {
            return;
        }
        tempStepKa = (leftL1Size - singleCoreBSize) / oneBaseADataSize;
    } else {
        uint64_t singleCoreASize = tempStepKa * oneBaseADataSize;
        if (isAFullLoad_) {
            singleCoreASize = GetSingleCoreAFullLoadSize();
        }
        if (leftL1Size < singleCoreASize + oneBaseBDataSize) {
            return;
        }
        tempStepKb = (leftL1Size - singleCoreASize) / oneBaseBDataSize;
    }
}

void L1TilingDataCalculator::CarryDataSizePass(uint64_t leftL1Size, uint64_t maxStepK)
{
    uint64_t tempStepKa = l1TilingData_.stepKa_;
    uint64_t tempStepKb = l1TilingData_.stepKb_;
    if (!isAFullLoad_) {
        tempStepKa = ops::CeilDiv(MIN_CARRY_DATA_SIZE_32K, GetSizeWithDataType(baseM_ * baseK_, inputParams_.aDtype));
        tempStepKa = std::min(tempStepKa, maxStepK);
        if (!CheckL1Size(leftL1Size, tempStepKa, tempStepKb)) {
            AdjustStepK(leftL1Size, tempStepKa, tempStepKb, true);
        }
    }
    if (!isBFullLoad_) {
        tempStepKb = ops::CeilDiv(MIN_CARRY_DATA_SIZE_32K, GetSizeWithDataType(baseN_ * baseK_, inputParams_.bDtype));
        tempStepKb = std::min(tempStepKb, maxStepK);
        if (!CheckL1Size(leftL1Size, tempStepKa, tempStepKb)) {
            AdjustStepK(leftL1Size, tempStepKa, tempStepKb, false);
        }
    }

    if (tempStepKa < l1TilingData_.stepKa_ || tempStepKb < l1TilingData_.stepKb_) {
        return;
    }
    l1TilingData_.stepKa_ = tempStepKa;
    l1TilingData_.stepKb_ = tempStepKb;
}

void L1TilingDataCalculator::BalanceStepKPass(uint64_t leftL1Size)
{
    if (isAFullLoad_ || isBFullLoad_ || l1TilingData_.stepKa_ == l1TilingData_.stepKb_) {
        return;
    }
    uint64_t biggerStepK = std::max(l1TilingData_.stepKa_, l1TilingData_.stepKb_);
    if (CheckL1Size(leftL1Size, biggerStepK, biggerStepK)) {
        l1TilingData_.stepKa_ = biggerStepK;
        l1TilingData_.stepKb_ = biggerStepK;
        return;
    }

    uint64_t tempStepKa = l1TilingData_.stepKa_;
    uint64_t tempStepKb = l1TilingData_.stepKb_;
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
    l1TilingData_.stepKa_ = tempStepKa;
    l1TilingData_.stepKb_ = tempStepKb;
}

void L1TilingDataCalculator::L1FullLoadCacheLinePass(uint64_t& tempStepKa, uint64_t& tempStepKb, uint64_t aCacheLine,
                                                     uint64_t bCacheLine)
{
    if (aCacheLine == 0 || bCacheLine == 0) {
        OP_LOGE(inputParams_.opName, "Invalid aCacheLine or bCacheLine.");
        return;
    }
    uint32_t maxStepKWithSmallCase = 4U * qmmv3_tiling_const::DOUBLE_BUFFER_NUM;
    if (ops::CeilDiv(inputParams_.kSize, baseK_) < maxStepKWithSmallCase) {
        return;
    }
    bool isEnableA = CACHE_LINE_512B % aCacheLine == 0UL;
    bool isEnableB = CACHE_LINE_512B % bCacheLine == 0UL;
    if (isAFullLoad_) {
        if (inputParams_.transB && isEnableB && (baseN_ * bCacheLine <= FULL_LOAD_DATA_SIZE_64K)) {
            tempStepKb *= (CACHE_LINE_512B / bCacheLine);
        }
    } else {
        if (ops::CeilDiv(inputParams_.kSize, baseK_) < MAX_STEPK_With_BL1_FULL) {
            return;
        }
        if (!inputParams_.transA && isEnableA && (baseM_ * aCacheLine <= FULL_LOAD_DATA_SIZE_64K)) {
            tempStepKa *= (CACHE_LINE_512B / aCacheLine);
        }
    }
}

void L1TilingDataCalculator::NONL1FullLoadCacheLinePass(uint64_t& tempStepKa, uint64_t& tempStepKb, uint64_t aCacheLine,
                                                        uint64_t bCacheLine)
{
    if (aCacheLine == 0 || bCacheLine == 0) {
        OP_LOGE(inputParams_.opName, "Invalid aCacheLine or bCacheLine.");
        return;
    }
    bool isEnableA = CACHE_LINE_512B % aCacheLine == 0UL;
    bool isEnableB = CACHE_LINE_512B % bCacheLine == 0UL;
    uint64_t aDataSize = GetSizeWithDataType(baseM_ * baseK_, inputParams_.aDtype) * tempStepKa;
    uint64_t bDataSize = GetSizeWithDataType(baseN_ * baseK_, inputParams_.bDtype) * tempStepKb;
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

void L1TilingDataCalculator::PostCacheLinePass(uint64_t leftL1Size, uint64_t maxStepK)
{
    if (inputParams_.transA && !inputParams_.transB) {
        return;
    }
    uint64_t tempStepKa = l1TilingData_.stepKa_;
    uint64_t tempStepKb = l1TilingData_.stepKb_;
    uint64_t aCacheLine = GetSizeWithDataType(baseK_, inputParams_.aDtype) * tempStepKa;
    uint64_t bCacheLine = GetSizeWithDataType(baseK_, inputParams_.bDtype) * tempStepKb;

    if (isAFullLoad_ || isBFullLoad_) {
        L1FullLoadCacheLinePass(tempStepKa, tempStepKb, aCacheLine, bCacheLine);
    } else {
        NONL1FullLoadCacheLinePass(tempStepKa, tempStepKb, aCacheLine, bCacheLine);
    }

    if (!CheckL1Size(leftL1Size, tempStepKa, tempStepKb) || tempStepKa > maxStepK || tempStepKb > maxStepK) {
        return;
    }
    l1TilingData_.stepKa_ = tempStepKa;
    l1TilingData_.stepKb_ = tempStepKb;
}

} // namespace optiling
