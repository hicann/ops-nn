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
 * \file base_block_calculator.cpp
 * \brief
 */
#include "base_block_calculator.h"

#include <algorithm>
#include "error_util.h"
#include "util/math_util.h"

namespace {
constexpr uint64_t CUBE_BLOCK = 16UL;
constexpr uint64_t L1_ALIGN_SIZE = 32UL;
constexpr uint64_t L2_ALIGN_SIZE = 128UL;
constexpr uint64_t CUBE_REDUCE_BLOCK = 32UL;
constexpr uint64_t MXFP_DIVISOR_SIZE = 64UL;
constexpr uint64_t PER_BLOCK_SIZE = 128UL;
constexpr uint64_t BASIC_BLOCK_SIZE_128 = 128UL;
constexpr uint64_t BASIC_BLOCK_SIZE_256 = 256UL;
constexpr uint64_t BASIC_BLOCK_SIZE_32 = 32UL;
constexpr uint64_t PER_BLOCK_BASE_SIZE_256 = 256UL;
constexpr uint64_t BASEM_BASEN_RATIO = 2UL;
constexpr uint64_t BASEK_LIMIT = 4095UL;
constexpr uint32_t DB_SIZE = 2U;
constexpr uint32_t NUM_HALF = 2U;
constexpr uint32_t DOUBLE_CORE_NUM = 2U;

uint64_t GetShapeWithDataType(uint64_t size, ge::DataType dtype)
{
    if (dtype == ge::DT_INT4 || dtype == ge::DT_FLOAT4_E2M1 || dtype == ge::DT_FLOAT4_E1M2) {
        return size + size;
    }
    uint64_t dtypeSize = static_cast<uint64_t>(ge::GetSizeByDataType(dtype));
    return dtypeSize == 0UL ? 0UL : size / dtypeSize;
}
} // namespace

namespace optiling {

using Ops::NN::MathUtil;

BaseBlockCalculator::BaseBlockCalculator(
    const QuantBatchMatmulInfo& inputParams, const QuantBatchMatmulV3CompileInfo& compileInfo, uint64_t batchCoreCnt)
    : inputParams_(inputParams), compileInfo_(compileInfo), batchCoreCnt_(batchCoreCnt)
{}

bool BaseBlockCalculator::Compute(BaseBlockMode mode)
{
    baseBlockRes_ = BaseBlockRes();
    if (!ValidateInput()) {
        return false;
    }
    switch (mode) {
        case BaseBlockMode::PERBLOCK:
            ComputeBaseBlockPerblock();
            break;
        case BaseBlockMode::MMAD_S8S4:
            ComputeBaseBlockMmadS8S4();
            break;
        case BaseBlockMode::DEFAULT:
        default:
            ComputeBaseBlockDefault();
            break;
    }
    if (!AdjustBaseBlock(mode)) {
        OP_LOGE(inputParams_.opName, "Failed to adjust base block.");
        return false;
    }
    return ValidateBaseBlock();
}

const BaseBlockRes& BaseBlockCalculator::GetOutput() const
{
    return baseBlockRes_;
}

bool BaseBlockCalculator::ValidateInput() const
{
    OP_TILING_CHECK(
        compileInfo_.aicNum == 0UL || batchCoreCnt_ == 0UL,
        CUBE_INNER_ERR_REPORT(
            inputParams_.opName,
            "Invalid BaseBlockCalculator input: aicNum(%lu) and batchCoreCnt(%lu) should be greater than 0.",
            static_cast<uint64_t>(compileInfo_.aicNum), batchCoreCnt_),
        return false);
    uint64_t baseMAlignSize = GetBaseMAlignSize();
    uint64_t baseNAlignSize = GetBaseNAlignSize();
    uint64_t baseKAlignSize = GetBaseKAlignSize();
    OP_TILING_CHECK(
        baseMAlignSize == 0UL || baseNAlignSize == 0UL || baseKAlignSize == 0UL,
        CUBE_INNER_ERR_REPORT(
            inputParams_.opName,
            "Invalid BaseBlockCalculator divisor: baseMAlignSize(%lu), baseNAlignSize(%lu) and baseKAlignSize(%lu) "
            "should be greater than 0.",
            baseMAlignSize, baseNAlignSize, baseKAlignSize),
        return false);
    return true;
}

bool BaseBlockCalculator::ValidateBaseBlock() const
{
    OP_TILING_CHECK(
        baseBlockRes_.baseM == 0UL || baseBlockRes_.baseN == 0UL || baseBlockRes_.baseK == 0UL,
        CUBE_INNER_ERR_REPORT(
            inputParams_.opName,
            "BaseM, baseN and baseK should be greater than 0, but baseM: %lu, baseN: %lu, baseK: %lu.",
            baseBlockRes_.baseM, baseBlockRes_.baseN, baseBlockRes_.baseK),
        return false);
    return true;
}

void BaseBlockCalculator::ComputeBaseBlockDefault()
{
    baseBlockRes_.baseM = ops::CeilAlign(std::min(inputParams_.mSize, BASIC_BLOCK_SIZE_256), GetBaseMAlignSize());
    baseBlockRes_.baseN = ops::CeilAlign(std::min(inputParams_.nSize, BASIC_BLOCK_SIZE_256), GetBaseNAlignSize());

    uint64_t baseKDefaultSize = GetShapeWithDataType(BASIC_BLOCK_SIZE_128, inputParams_.aDtype);
    baseBlockRes_.baseK = ops::CeilAlign(std::min(baseKDefaultSize, inputParams_.kSize), GetBaseKAlignSize());
}

void BaseBlockCalculator::ComputeBaseBlockPerblock()
{
    if (inputParams_.mSize <= PER_BLOCK_SIZE || inputParams_.mSize % PER_BLOCK_SIZE != 0UL) {
        baseBlockRes_.baseM = inputParams_.groupSizeM == 1UL && inputParams_.mSize < PER_BLOCK_SIZE ?
                                  ops::CeilAlign(inputParams_.mSize, GetBaseMAlignSize()) :
                                  PER_BLOCK_SIZE;
        baseBlockRes_.baseN = inputParams_.nSize % PER_BLOCK_SIZE == 0UL ? PER_BLOCK_BASE_SIZE_256 : PER_BLOCK_SIZE;
    } else {
        baseBlockRes_.baseM = PER_BLOCK_BASE_SIZE_256;
        baseBlockRes_.baseN = PER_BLOCK_SIZE;
    }
    baseBlockRes_.baseK = PER_BLOCK_SIZE;
}

void BaseBlockCalculator::ComputeBaseBlockMmadS8S4()
{
    baseBlockRes_.baseM = ops::CeilAlign(std::min(inputParams_.mSize, BASIC_BLOCK_SIZE_256), GetBaseMAlignSize());
    baseBlockRes_.baseN = ops::CeilAlign(std::min(inputParams_.nSize, BASIC_BLOCK_SIZE_256), GetBaseNAlignSize());

    uint64_t basicBlockSizeA = BASIC_BLOCK_SIZE_128;
    uint64_t basicBlockSizeB = BASIC_BLOCK_SIZE_128;
    if (baseBlockRes_.baseM != 0UL && compileInfo_.l0aSize / DB_SIZE / baseBlockRes_.baseM >= BASIC_BLOCK_SIZE_256) {
        basicBlockSizeA = BASIC_BLOCK_SIZE_256;
    }
    if (baseBlockRes_.baseN != 0UL && compileInfo_.l0bSize / DB_SIZE / baseBlockRes_.baseN >= BASIC_BLOCK_SIZE_256) {
        basicBlockSizeB = BASIC_BLOCK_SIZE_256;
    }
    uint64_t minBaseK = std::min(
        std::min(
            GetShapeWithDataType(basicBlockSizeA, inputParams_.aDtype),
            GetShapeWithDataType(basicBlockSizeB, inputParams_.bDtype)),
        inputParams_.kSize);
    uint64_t maxAlignSize = std::max(GetBaseKAlignSize(), GetShapeWithDataType(CUBE_REDUCE_BLOCK, inputParams_.bDtype));
    baseBlockRes_.baseK = ops::CeilAlign(minBaseK, maxAlignSize);
}

uint64_t BaseBlockCalculator::GetBaseMAlignSize() const
{
    return inputParams_.transA ? GetShapeWithDataType(L1_ALIGN_SIZE, inputParams_.aDtype) : CUBE_BLOCK;
}

uint64_t BaseBlockCalculator::GetBaseNAlignSize() const
{
    return inputParams_.transB ? CUBE_BLOCK : GetShapeWithDataType(L1_ALIGN_SIZE, inputParams_.bDtype);
}

uint64_t BaseBlockCalculator::GetBaseKAlignSize() const
{
    return inputParams_.isMxPerGroup ? MXFP_DIVISOR_SIZE : GetShapeWithDataType(CUBE_REDUCE_BLOCK, inputParams_.aDtype);
}

bool BaseBlockCalculator::AdjustBaseBlock(BaseBlockMode mode)
{
    uint64_t oriBlock = batchCoreCnt_ * ops::CeilDiv(inputParams_.mSize, baseBlockRes_.baseM) *
                        ops::CeilDiv(inputParams_.nSize, baseBlockRes_.baseN);
    if (oriBlock >= compileInfo_.aicNum) {
        return true;
    }
    switch (compileInfo_.npuArch) {
        case NpuArch::DAV_3510:
            return mode == BaseBlockMode::PERBLOCK ? AdjustBaseBlockPerblock() : AdjustBaseBlockDefault();
        case NpuArch::DAV_RESV:
            return AdjustBaseBlockMmadS8S4(oriBlock);
        default:
            OP_LOGE(inputParams_.opName, "Failed to find the AdjustBaseBlock function for the current NPU architecture.");
            return false;
    }
}

bool BaseBlockCalculator::AdjustBaseBlockDefault()
{
    uint64_t baseMAlignNum =
        inputParams_.transA ? GetShapeWithDataType(L2_ALIGN_SIZE, inputParams_.aDtype) : CUBE_BLOCK;
    uint64_t baseNAlignNum =
        inputParams_.transB ? CUBE_BLOCK : GetShapeWithDataType(L2_ALIGN_SIZE, inputParams_.bDtype);
    uint64_t baseKAlignNum = (inputParams_.transA && !inputParams_.transB) ?
                                 GetShapeWithDataType(BASIC_BLOCK_SIZE_32, inputParams_.aDtype) :
                                 GetShapeWithDataType(L2_ALIGN_SIZE, inputParams_.aDtype);
    if (IsMxBackwardTrans()) {
        baseKAlignNum = GetShapeWithDataType(MXFP_DIVISOR_SIZE, inputParams_.aDtype);
    }
    OP_TILING_CHECK(
        baseMAlignNum == 0UL || baseNAlignNum == 0UL || baseKAlignNum == 0UL,
        CUBE_INNER_ERR_REPORT(
            inputParams_.opName,
            "Invalid base block alignment: baseMAlignNum(%lu), baseNAlignNum(%lu), baseKAlignNum(%lu) should be "
            "greater than 0.",
            baseMAlignNum, baseNAlignNum, baseKAlignNum),
        return false);
    uint64_t mMaxtile = MathUtil::CeilDivision(inputParams_.mSize, baseMAlignNum);
    uint64_t nMaxtile = MathUtil::CeilDivision(inputParams_.nSize, baseNAlignNum);
    uint64_t tempBaseM = baseBlockRes_.baseM;
    uint64_t tempBaseN = baseBlockRes_.baseN;
    uint64_t coreNumMN = compileInfo_.aicNum / batchCoreCnt_;
    if (mMaxtile * nMaxtile < coreNumMN && (inputParams_.transA || !inputParams_.transB)) {
        return true;
    }

    uint64_t mCore = MathUtil::CeilDivision(inputParams_.mSize, baseBlockRes_.baseM);
    uint64_t nCore = MathUtil::CeilDivision(inputParams_.nSize, baseBlockRes_.baseN);
    if (mMaxtile < nMaxtile || (mMaxtile == nMaxtile && baseNAlignNum == CUBE_BLOCK)) {
        tempBaseM = ops::CeilAlign(MathUtil::CeilDivision(inputParams_.mSize, mCore), baseMAlignNum);
        mCore = MathUtil::CeilDivision(inputParams_.mSize, tempBaseM);
        nCore = coreNumMN / mCore;
        tempBaseN = ops::CeilAlign(MathUtil::CeilDivision(inputParams_.nSize, nCore), baseNAlignNum);
    } else {
        tempBaseN = ops::CeilAlign(MathUtil::CeilDivision(inputParams_.nSize, nCore), baseNAlignNum);
        nCore = MathUtil::CeilDivision(inputParams_.nSize, tempBaseN);
        mCore = coreNumMN / nCore;
        tempBaseM = ops::CeilAlign(MathUtil::CeilDivision(inputParams_.mSize, mCore), baseMAlignNum);
    }

    while (tempBaseN >= tempBaseM * BASEM_BASEN_RATIO && nCore < coreNumMN / NUM_HALF && tempBaseN != baseNAlignNum) {
        nCore = nCore * DOUBLE_CORE_NUM;
        mCore = coreNumMN / nCore;
        tempBaseM = ops::CeilAlign(MathUtil::CeilDivision(inputParams_.mSize, mCore), baseMAlignNum);
        tempBaseN = ops::CeilAlign(MathUtil::CeilDivision(inputParams_.nSize, nCore), baseNAlignNum);
        mCore = MathUtil::CeilDivision(inputParams_.mSize, tempBaseM);
        nCore = MathUtil::CeilDivision(inputParams_.nSize, tempBaseN);
    }
    while (tempBaseM >= tempBaseN * BASEM_BASEN_RATIO && mCore < coreNumMN / NUM_HALF && tempBaseM != baseMAlignNum) {
        mCore = mCore * DOUBLE_CORE_NUM;
        nCore = coreNumMN / mCore;
        tempBaseM = ops::CeilAlign(MathUtil::CeilDivision(inputParams_.mSize, mCore), baseMAlignNum);
        tempBaseN = ops::CeilAlign(MathUtil::CeilDivision(inputParams_.nSize, nCore), baseNAlignNum);
        mCore = MathUtil::CeilDivision(inputParams_.mSize, tempBaseM);
        nCore = MathUtil::CeilDivision(inputParams_.nSize, tempBaseN);
    }
    uint64_t kValueAlign = ops::CeilAlign(inputParams_.kSize, baseKAlignNum);
    uint64_t kValueMax =
        GetShapeWithDataType(compileInfo_.l0aSize / DB_SIZE, inputParams_.aDtype) / std::max(tempBaseM, tempBaseN);
    if (kValueMax >= baseKAlignNum) {
        baseBlockRes_.baseM = tempBaseM;
        baseBlockRes_.baseN = tempBaseN;
        kValueMax = ops::FloorAlign(kValueMax, baseKAlignNum);
        baseBlockRes_.baseK = std::min(kValueAlign, kValueMax);
        baseBlockRes_.baseK = baseBlockRes_.baseK > BASEK_LIMIT ?
                                  ops::CeilAlign(baseBlockRes_.baseK / NUM_HALF, baseKAlignNum) :
                                  baseBlockRes_.baseK;
        baseBlockRes_.useTailWinLogic = false;
    }
    return true;
}

bool BaseBlockCalculator::AdjustBaseBlockPerblock()
{
    uint64_t coreNumMN = compileInfo_.aicNum / batchCoreCnt_;
    if (inputParams_.groupSizeM == 1UL) {
        return AdjustBaseBlockPertile(coreNumMN);
    } else if (
        baseBlockRes_.baseM == PER_BLOCK_BASE_SIZE_256 &&
        ops::CeilDiv(inputParams_.mSize, PER_BLOCK_SIZE) * ops::CeilDiv(inputParams_.nSize, baseBlockRes_.baseN) <=
            coreNumMN) {
        baseBlockRes_.baseM = PER_BLOCK_SIZE;
    } else if (
        baseBlockRes_.baseN == PER_BLOCK_BASE_SIZE_256 &&
        ops::CeilDiv(inputParams_.mSize, baseBlockRes_.baseM) * ops::CeilDiv(inputParams_.nSize, PER_BLOCK_SIZE) <=
            coreNumMN) {
        baseBlockRes_.baseN = PER_BLOCK_SIZE;
    }
    return true;
}

bool BaseBlockCalculator::AdjustBaseBlockPertile(uint64_t coreNumMN)
{
    uint64_t baseMAlignNum =
        inputParams_.transA ? GetShapeWithDataType(L2_ALIGN_SIZE, inputParams_.aDtype) : CUBE_BLOCK;
    uint64_t baseNAlignNum =
        !inputParams_.transB ? GetShapeWithDataType(L2_ALIGN_SIZE, inputParams_.bDtype) : CUBE_BLOCK;
    uint64_t adjustBaseM = baseBlockRes_.baseM;
    uint64_t adjustBaseN = baseBlockRes_.baseN;
    uint64_t adjustMCore = MathUtil::CeilDivision(inputParams_.mSize, adjustBaseM);
    uint64_t adjustNCore = MathUtil::CeilDivision(inputParams_.nSize, adjustBaseN);

    adjustMCore = coreNumMN / adjustNCore;
    adjustBaseM = ops::CeilAlign(MathUtil::CeilDivision(inputParams_.mSize, adjustMCore), baseMAlignNum);
    adjustMCore = MathUtil::CeilDivision(inputParams_.mSize, adjustBaseM);

    while (adjustBaseN / NUM_HALF >= baseNAlignNum &&
           adjustMCore * MathUtil::CeilDivision(inputParams_.nSize, adjustBaseN / NUM_HALF) <= coreNumMN) {
        adjustBaseN = adjustBaseN / NUM_HALF;
        adjustNCore = MathUtil::CeilDivision(inputParams_.nSize, adjustBaseN);
    }

    while (adjustBaseN > adjustBaseM * BASEM_BASEN_RATIO && adjustBaseN / NUM_HALF >= baseNAlignNum) {
        uint64_t tempBaseN = adjustBaseN / NUM_HALF;
        uint64_t tempNCore = MathUtil::CeilDivision(inputParams_.nSize, tempBaseN);
        uint64_t tempMCore = coreNumMN / tempNCore;
        uint64_t tempBaseM = ops::CeilAlign(MathUtil::CeilDivision(inputParams_.mSize, tempMCore), baseMAlignNum);
        tempMCore = MathUtil::CeilDivision(inputParams_.mSize, tempBaseM);
        uint64_t tempUsedCoreNum = tempMCore * tempNCore;

        if (tempUsedCoreNum > coreNumMN) {
            break;
        }
        adjustBaseM = tempBaseM;
        adjustBaseN = tempBaseN;
        adjustMCore = tempMCore;
        adjustNCore = tempNCore;
    }

    baseBlockRes_.baseM = adjustBaseM;
    baseBlockRes_.baseN = adjustBaseN;
    return true;
}

bool BaseBlockCalculator::AdjustBaseBlockMmadS8S4(uint64_t oriBlock)
{
    uint64_t baseMAlignNum =
        inputParams_.transA ? GetShapeWithDataType(L2_ALIGN_SIZE, inputParams_.aDtype) : CUBE_BLOCK;
    uint64_t baseNAlignNum = L2_ALIGN_SIZE;
    uint64_t baseKAlignNum = (inputParams_.transA && !inputParams_.transB) ?
                                 GetShapeWithDataType(BASIC_BLOCK_SIZE_32, inputParams_.aDtype) :
                                 GetShapeWithDataType(L2_ALIGN_SIZE, inputParams_.aDtype);
    baseKAlignNum = std::min(baseKAlignNum, inputParams_.kSize);
    OP_TILING_CHECK(
        baseMAlignNum == 0UL || baseNAlignNum == 0UL || baseKAlignNum == 0UL,
        CUBE_INNER_ERR_REPORT(
            inputParams_.opName,
            "Invalid MMAD S8S4 alignment: baseMAlignNum(%lu), baseNAlignNum(%lu), baseKAlignNum(%lu) should be "
            "greater than 0.",
            baseMAlignNum, baseNAlignNum, baseKAlignNum),
        return false);
    uint64_t mMaxtile = MathUtil::CeilDivision(inputParams_.mSize, baseMAlignNum);
    uint64_t nMaxtile = MathUtil::CeilDivision(inputParams_.nSize, baseNAlignNum);
    uint64_t tempBaseM = baseMAlignNum;
    uint64_t tempBaseN = baseNAlignNum;
    uint64_t aicNum = static_cast<uint64_t>(compileInfo_.aicNum);
    bool optimalFound = false;
    if (mMaxtile * nMaxtile < oriBlock && mMaxtile != 1UL && nMaxtile != 1UL) {
        return true;
    }
    if (mMaxtile == 1UL || nMaxtile == 1UL) {
        tempBaseM =
            mMaxtile == 1UL ?
                baseMAlignNum :
                std::max(
                    baseMAlignNum, ops::CeilAlign(MathUtil::CeilDivision(inputParams_.mSize, aicNum), baseMAlignNum));
        tempBaseN =
            nMaxtile == 1UL ?
                baseNAlignNum :
                std::max(
                    baseNAlignNum, ops::CeilAlign(MathUtil::CeilDivision(inputParams_.nSize, aicNum), baseNAlignNum));
        optimalFound = true;
    } else {
        optimalFound = CalculateOptimalSplit(tempBaseM, tempBaseN, baseMAlignNum, baseNAlignNum, baseKAlignNum);
    }
    uint64_t kValueAlign = ops::CeilAlign(inputParams_.kSize, baseKAlignNum);
    uint64_t kValueMax =
        GetShapeWithDataType(compileInfo_.l0aSize / DB_SIZE, inputParams_.aDtype) / std::max(tempBaseM, tempBaseN);
    if (kValueMax >= baseKAlignNum && optimalFound) {
        baseBlockRes_.baseM = tempBaseM;
        baseBlockRes_.baseN = tempBaseN;
        kValueMax = ops::FloorAlign(kValueMax, baseKAlignNum);
        baseBlockRes_.baseK = std::min(kValueAlign, kValueMax);
        baseBlockRes_.baseK = baseBlockRes_.baseK > BASEK_LIMIT ? baseBlockRes_.baseK / NUM_HALF : baseBlockRes_.baseK;
        uint64_t maxAlignSize = std::max(
            GetShapeWithDataType(CUBE_REDUCE_BLOCK, inputParams_.aDtype),
            GetShapeWithDataType(CUBE_REDUCE_BLOCK, inputParams_.bDtype));
        OP_TILING_CHECK(
            maxAlignSize == 0UL,
            CUBE_INNER_ERR_REPORT(
                inputParams_.opName, "Invalid MMAD S8S4 K alignment: maxAlignSize(%lu) should be greater than 0.",
                maxAlignSize),
            return false);
        baseBlockRes_.baseK = ops::CeilAlign(baseBlockRes_.baseK, maxAlignSize);
        baseBlockRes_.useTailWinLogic = false;
    }
    return true;
}

bool BaseBlockCalculator::CalculateOptimalSplit(
    uint64_t& baseM, uint64_t& baseN, uint64_t baseMAlignNum, uint64_t baseNAlignNum, uint64_t baseKAlignNum) const
{
    uint64_t mMaxtile = MathUtil::CeilDivision(inputParams_.mSize, baseMAlignNum);
    uint64_t nMaxtile = MathUtil::CeilDivision(inputParams_.nSize, baseNAlignNum);
    uint64_t maxUsedCore = MathUtil::CeilDivision(inputParams_.mSize, baseBlockRes_.baseM) *
                           MathUtil::CeilDivision(inputParams_.nSize, baseBlockRes_.baseN);
    uint64_t maxDiff = UINT64_MAX;
    uint64_t iterMSplite = std::min(mMaxtile, static_cast<uint64_t>(compileInfo_.aicNum));
    uint64_t iterNSplite = std::min(nMaxtile, static_cast<uint64_t>(compileInfo_.aicNum));
    uint64_t l0aHalfShape = GetShapeWithDataType(compileInfo_.l0aSize / DB_SIZE, inputParams_.aDtype);
    bool optimalFound = false;
    for (uint64_t mFactor = 1UL; mFactor <= iterMSplite; ++mFactor) {
        for (uint64_t nFactor = 1UL; nFactor <= iterNSplite; ++nFactor) {
            uint64_t tempMBase = mFactor * baseMAlignNum;
            uint64_t mCore = MathUtil::CeilDivision(inputParams_.mSize, tempMBase);
            uint64_t tempNBase = nFactor * baseNAlignNum;
            uint64_t nCore = MathUtil::CeilDivision(inputParams_.nSize, tempNBase);
            uint64_t usedCore = mCore * nCore;
            uint64_t diff = (tempMBase >= tempNBase) ? tempMBase - tempNBase : tempNBase - tempMBase;
            uint64_t kValueMax = l0aHalfShape / std::max(tempMBase, tempNBase);
            if (usedCore > compileInfo_.aicNum) {
                continue;
            }
            if ((usedCore > maxUsedCore || (usedCore == maxUsedCore && diff < maxDiff)) && kValueMax >= baseKAlignNum) {
                maxUsedCore = usedCore;
                maxDiff = diff;
                baseM = tempMBase;
                baseN = tempNBase;
                optimalFound = true;
            }
        }
    }
    return optimalFound;
}

bool BaseBlockCalculator::IsMxBackwardTrans() const
{
    return inputParams_.scaleDtype == ge::DT_FLOAT8_E8M0 && (inputParams_.transA || !inputParams_.transB);
}

} // namespace optiling
