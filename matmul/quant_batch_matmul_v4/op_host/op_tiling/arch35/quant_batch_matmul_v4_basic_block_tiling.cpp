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
 * \file quant_batch_matmul_v4_basic_block_tiling.cpp
 * \brief
 */

#include "quant_batch_matmul_v4_basic_block_tiling.h"
#include "../../../../common/op_host/math_util.h"

namespace optiling {
using namespace matmul_v4;
void QuantBatchMatmulV4BasicBlockTiling::Init()
{
    opName_ = nullptr;
    aByteSize_ = 0.0;
    bByteSize_ = 0.0;
    biasByteSize_ = 0.0;
    yScaleByteSize_ = 0.0;
    aDtypeBits_ = 0;
    bDtypeBits_ = 0;
    biasDtypeBits_ = 0;
    vFRate_ = VF_RATE_FPA8W4;

    transA_ = false;
    transB_ = true;
    hasBias_ = false;
    weightNzFlag_ = false;
    isMxType_ = false;

    optionalResults_.clear();

    basicBlockParam_.mSize = 1;
    basicBlockParam_.nSize = 1;
    basicBlockParam_.kSize = 1;
    basicBlockParam_.kDim = 1;
    basicBlockParam_.singleK = 1;
    basicBlockParam_.groupSize = 0;

    InitBasicBlockParam();
    InitPlarformParam();
}

void QuantBatchMatmulV4BasicBlockTiling::InitPlarformParam()
{
    platformParam_.blockNum = 0;
    platformParam_.aicNum = 0;
    platformParam_.ubSize = 0;
    platformParam_.l1Size = 0;
    platformParam_.l0aSize = 0;
    platformParam_.l0bSize = 0;
    platformParam_.l0cSize = 0;
    platformParam_.cacheLine = 0;
    platformParam_.minCacheLine = 0;

    platformParam_.frequency = 0;
    platformParam_.hbmBW = 0;
    platformParam_.l2BW = 0;
}

void QuantBatchMatmulV4BasicBlockTiling::InitL1TilingParam()
{
    basicBlockParam_.l1Param.iterateOrder = 1;
    basicBlockParam_.l1Param.stepM = 1;
    basicBlockParam_.l1Param.stepN = 1;
    basicBlockParam_.l1Param.stepKa = 1;
    basicBlockParam_.l1Param.stepKb = 1;
    basicBlockParam_.l1Param.A1BufferNum = 1;
    basicBlockParam_.l1Param.B1BufferNum = 1;
}

void QuantBatchMatmulV4BasicBlockTiling::InitBasicBlockParam()
{
    // mSize, nSize, kSize, singleK 采用SetShape传入的结果
    basicBlockParam_.singleM = 1;
    basicBlockParam_.singleN = 1;
    basicBlockParam_.mDim = 1;
    basicBlockParam_.nDim = 1;
    basicBlockParam_.mte2DataSize = 0;
    basicBlockParam_.fixpDataSize = 0;
    basicBlockParam_.mxType = static_cast<int64_t>(isMxType_);

    basicBlockParam_.basicBlock.baseM = 1;
    basicBlockParam_.basicBlock.baseN = 1;
    basicBlockParam_.basicBlock.baseK = 1;
    basicBlockParam_.basicBlock.mte2BW = 0;
    basicBlockParam_.basicBlock.mte2MinBW = 0;
    basicBlockParam_.basicBlock.mte2BWRatio = 0;
    basicBlockParam_.basicBlock.mte2TailBWRatio = 0;

    InitL1TilingParam();
}

void QuantBatchMatmulV4BasicBlockTiling::Reset()
{
    InitBasicBlockParam();
    optionalResults_.clear();
}

void QuantBatchMatmulV4BasicBlockTiling::SetPlatformParam(const PlatformParam &param)
{
    platformParam_ = param;
    OP_LOGI(opName_,
            "Platform info: blockNum: %ld, aicNum: %ld, ubSize: %ld, l1Size: %ld, l0aSize: %ld, l0bSize: %ld, l0cSize: "
            "%ld, cacheLine: %ld, minCacheLine: %ld, frequency: %lf, hbmBW: %lf, l2BW: %lf",
            platformParam_.blockNum, platformParam_.aicNum, platformParam_.ubSize, platformParam_.l1Size,
            platformParam_.l0aSize, platformParam_.l0bSize, platformParam_.l0cSize, platformParam_.cacheLine,
            platformParam_.minCacheLine, platformParam_.frequency, platformParam_.hbmBW, platformParam_.l2BW);
}

void QuantBatchMatmulV4BasicBlockTiling::SetShape(int64_t mSize, int64_t nSize, int64_t kSize, int64_t groupSize)
{
    basicBlockParam_.mSize = mSize;
    basicBlockParam_.nSize = nSize;
    basicBlockParam_.kSize = kSize;
    basicBlockParam_.singleK = kSize;
    basicBlockParam_.groupSize = groupSize;
    OP_LOGI(opName_, "Init shape param, mSize: %ld, nSize: %ld, kSize: %ld, groupSize: %ld", basicBlockParam_.mSize,
            basicBlockParam_.nSize, basicBlockParam_.kSize, basicBlockParam_.groupSize);
}

void QuantBatchMatmulV4BasicBlockTiling::SetAttr(const char *opName, bool transA, bool transB, bool hasBias,
                                                       bool weightNzFlag)
{
    opName_ = opName;
    transA_ = transA;
    transB_ = transB;
    hasBias_ = hasBias;
    weightNzFlag_ = weightNzFlag;
    auto transA_str = transA_ ? "true" : "false";
    auto transB_str = transB_ ? "true" : "false";
    auto hasBias_str = hasBias_ ? "true" : "false";
    auto weightNzFlag_str = weightNzFlag_ ? "true" : "false";
    OP_LOGI(opName_, "Init attr param, transA_: %s, transB_: %s, hasBias_: %s, weightNzFlag: %s",
            transA_str, transB_str, hasBias_str, weightNzFlag_str);
}

void QuantBatchMatmulV4BasicBlockTiling::SetQuantType(bool isMxType)
{
    isMxType_ = isMxType;
    vFRate_ = isMxType_ ? VF_RATE_MXA8W4 : VF_RATE_FPA8W4;
    basicBlockParam_.mxType = static_cast<int64_t>(isMxType);
    OP_LOGI(opName_, "isMxType_: %d, vFRate_: %ld", isMxType_, vFRate_);
}

void QuantBatchMatmulV4BasicBlockTiling::SetDtypeBits(const int64_t aDtypeBits, const int64_t bDtypeBits,
    const int64_t biasDtypeBits, const int64_t yScaleDtypeBits)
{
    aDtypeBits_ = aDtypeBits;
    bDtypeBits_ = bDtypeBits;
    biasDtypeBits_ = biasDtypeBits;
    aByteSize_ = aDtypeBits / BYTE_BITS;
    bByteSize_ = bDtypeBits / BYTE_BITS;
    biasByteSize_ = biasDtypeBits / BYTE_BITS;
    yScaleByteSize_ = yScaleDtypeBits / BYTE_BITS;
    OP_LOGI(opName_, "Init byteSize, aByteSize_: %lf, bByteSize_: %lf, biasByteSize_: %lf , yScaleByteSize_: %lf",
        aByteSize_, bByteSize_, biasByteSize_, yScaleByteSize_);
}

bool QuantBatchMatmulV4BasicBlockTiling::ValidateInputParam() const
{
    OP_TILING_CHECK(basicBlockParam_.mSize <= 0 || basicBlockParam_.nSize <= 0 || basicBlockParam_.kSize <= 0,
                    VECTOR_INNER_ERR_REPORT_TILIING(
                        opName_, "Invalid param, shape size must gt 0, mSize: %ld, nSize: %ld, kSize: %ld",
                        basicBlockParam_.mSize, basicBlockParam_.nSize, basicBlockParam_.kSize),
                    return false);

    OP_TILING_CHECK(
        aDtypeBits_ <= 0 || bDtypeBits_ <= 0 || (hasBias_ && biasDtypeBits_ <= 0),
        VECTOR_INNER_ERR_REPORT_TILIING(
            opName_,
            "Invalid param, dtypeBits must be greater than 0, aDtypeBits_: %ld, bDtypeBits_: %ld, biasDtypeBits_: %ld",
            aDtypeBits_, bDtypeBits_, biasDtypeBits_),
        return false);

    OP_TILING_CHECK(basicBlockParam_.groupSize < 0,
                    VECTOR_INNER_ERR_REPORT_TILIING(
                        opName_, "Invalid param, groupSize must be greater than or equalt to 0, groupSize: %ld",
                        basicBlockParam_.groupSize),
                    return false);

    return true;
}

int64_t QuantBatchMatmulV4BasicBlockTiling::GetCycleVF(const BasicBlockParam &basicBlockParam) const
{
    const int64_t singleMLoop = basicBlockParam.basicBlock.baseK * basicBlockParam.l1Param.stepKb >=
        basicBlockParam.singleK ? 1 : CeilDiv(basicBlockParam.singleM, basicBlockParam.basicBlock.baseM);
    const int64_t vfProcessLen = basicBlockParam.basicBlock.baseK * basicBlockParam.l1Param.stepKb;
    const int64_t vfAlignLen = ops::CeilAlign(vfProcessLen, VF_PROCESS_ELEM << 1);
    const int64_t realVfRate = vFRate_ * vfProcessLen / vfAlignLen;
    const int64_t aivParallel = min(BUFF_NUM_2, basicBlockParam.l1Param.B1BufferNum);
    return basicBlockParam.singleN * basicBlockParam.singleK / aivParallel * singleMLoop / realVfRate;
}

int64_t QuantBatchMatmulV4BasicBlockTiling::GetCycleBMte2(const BasicBlockParam &basicBlockParam) const
{
    const int64_t singleMLoop = basicBlockParam.basicBlock.baseK * basicBlockParam.l1Param.stepKb >=
        basicBlockParam.singleK ? 1 : CeilDiv(basicBlockParam.singleM, basicBlockParam.basicBlock.baseM);
    const int64_t aivParallel = min(BUFF_NUM_2, basicBlockParam.l1Param.B1BufferNum);
    return basicBlockParam.singleN * basicBlockParam.singleK * static_cast<int64_t>(bByteSize_) /
        aivParallel * singleMLoop / MTE2_AIV_RATE;
}

int64_t QuantBatchMatmulV4BasicBlockTiling::GetCycleAMte2(const BasicBlockParam &basicBlockParam) const
{
    const int64_t singleNLoop = basicBlockParam.basicBlock.baseK * basicBlockParam.l1Param.stepKa >=
        basicBlockParam.singleK ? 1 : CeilDiv(basicBlockParam.singleN, basicBlockParam.basicBlock.baseN);
    return basicBlockParam.singleM * basicBlockParam.singleN * static_cast<int64_t>(bByteSize_) *
        singleNLoop / MTE2_AIC_RATE;
}

int64_t QuantBatchMatmulV4BasicBlockTiling::GetCycleCube(const BasicBlockParam &basicBlockParam) const
{
    return basicBlockParam.singleM * basicBlockParam.singleN * basicBlockParam.singleK / CUBE_ELEM_FP8;
}

double QuantBatchMatmulV4BasicBlockTiling::GetMinMte2BW(int64_t baseM, int64_t baseN, int64_t mDim,
                                                              int64_t nDim) const
{
    if (mDim * nDim * baseM * baseN == 0) {
        return 0;
    }
    // 求V100平台在给定基本块条件下达到cube bound要求的最低MTE2带宽
    // 0.001含义：将计算结果转换为TB/s
    return CUBE_ELEM_FP8 * platformParam_.frequency * 0.001 * static_cast<double>(mDim) *
           static_cast<double>(nDim) *
           (aByteSize_ / static_cast<double>(baseN) + bByteSize_ / static_cast<double>(baseM));
}

double QuantBatchMatmulV4BasicBlockTiling::GetMte2BW(int64_t baseM, int64_t baseN, int64_t mDim,
                                                           int64_t nDim) const
{
    // 估算V100平台当前分核条件下的综合MTE2带宽
    double res =
        static_cast<double>(mDim * nDim * (baseM * aByteSize_ + bByteSize_ * baseN)) /
        (static_cast<double>(mDim * baseM * aByteSize_ + nDim * baseN * bByteSize_) / platformParam_.hbmBW +
         static_cast<double>((mDim * nDim - mDim) * baseM * aByteSize_ + baseN * (mDim * nDim - nDim) * bByteSize_) /
             platformParam_.l2BW) *
        (static_cast<double>(mDim * nDim) / static_cast<double>(platformParam_.blockNum));

    return res;
}

double QuantBatchMatmulV4BasicBlockTiling::GetMte2BWRatio(int64_t baseM, int64_t baseN, int64_t mDim,
                                                                int64_t nDim) const
{
    if (GetMinMte2BW(baseM, baseN, mDim, nDim) > 0) {
        return GetMte2BW(baseM, baseN, mDim, nDim) / GetMinMte2BW(baseM, baseN, mDim, nDim);
    }
    return 0;
}

void QuantBatchMatmulV4BasicBlockTiling::GetMte2DataSize(BasicBlockParam &basicBlockParam,
                                                                int64_t &mte2DataSize, int64_t &iterateOrder) const
{
    if (isMxType_) {
        GetMte2DataSizeMx(basicBlockParam, mte2DataSize);
    } else {
        GetMte2DataSizeGroup(basicBlockParam, mte2DataSize, iterateOrder);
    }
}

void QuantBatchMatmulV4BasicBlockTiling::GetMte2DataSizeGroup(BasicBlockParam &basicBlockParam,
                                                               int64_t &mte2DataSize, int64_t &iterateOrder) const
{
    // 给定单核L1、L0切分条件，计算MTE2总搬运量
    const int64_t singleMLoop =
        CeilDiv(basicBlockParam.singleM, basicBlockParam.l1Param.stepM * basicBlockParam.basicBlock.baseM);
    const int64_t singleNLoop =
        CeilDiv(basicBlockParam.singleN, basicBlockParam.l1Param.stepN * basicBlockParam.basicBlock.baseN);
    const bool A1KFullLoadFlag = basicBlockParam.basicBlock.baseK * basicBlockParam.l1Param.stepKa >=
        basicBlockParam.singleK;
    const bool B1KFullLoadFlag = basicBlockParam.basicBlock.baseK * basicBlockParam.l1Param.stepKb >=
        basicBlockParam.singleK;

    if (A1KFullLoadFlag && B1KFullLoadFlag) {
        const int64_t mte2DataSizeOrderM = static_cast<int64_t>(basicBlockParam.singleM * basicBlockParam.singleK *
            aDtypeBits_ + singleMLoop * basicBlockParam.singleN * basicBlockParam.singleK * bDtypeBits_);
        const int64_t mte2DataSizeOrderN = static_cast<int64_t>(basicBlockParam.singleN * basicBlockParam.singleK *
            bDtypeBits_ + singleNLoop * basicBlockParam.singleM * basicBlockParam.singleK * aDtypeBits_);
        mte2DataSize = min(mte2DataSizeOrderN, mte2DataSizeOrderM);
        iterateOrder = mte2DataSizeOrderM < mte2DataSizeOrderN ? 0 : 1;
    } else if (A1KFullLoadFlag) {
        mte2DataSize = static_cast<int64_t>(basicBlockParam.singleM * basicBlockParam.singleK * aDtypeBits_ +
            singleMLoop * basicBlockParam.singleN * basicBlockParam.singleK * bDtypeBits_);
    } else if (B1KFullLoadFlag) {
        mte2DataSize = static_cast<int64_t>(basicBlockParam.singleN * basicBlockParam.singleK * bDtypeBits_ +
            singleNLoop * basicBlockParam.singleM * basicBlockParam.singleK * aDtypeBits_);
    } else {
        mte2DataSize = static_cast<int64_t>(singleMLoop * basicBlockParam.singleN * basicBlockParam.singleK *
            bDtypeBits_ + singleNLoop * basicBlockParam.singleM * basicBlockParam.singleK * aDtypeBits_);
    }
    mte2DataSize *= basicBlockParam.mDim * basicBlockParam.nDim / BYTE_SIZE;
}

void QuantBatchMatmulV4BasicBlockTiling::GetMte2DataSizeMx(BasicBlockParam& basicBlockParam,
    int64_t& mte2DataSize) const
{
    const int64_t singleMLoop =
        CeilDiv(basicBlockParam.singleM, basicBlockParam.l1Param.stepM * basicBlockParam.basicBlock.baseM);
    const int64_t singleNLoop =
        CeilDiv(basicBlockParam.singleN, basicBlockParam.l1Param.stepN * basicBlockParam.basicBlock.baseN);
    int64_t scaleKloopNumTotal =
        CeilDiv(basicBlockParam.singleK, basicBlockParam.basicBlock.baseK * basicBlockParam.l1Param.stepKa *
                                             basicBlockParam.l1Param.scaleFactor);
    // 每行搬运量都小于cacheline, 按照cacheline大小计算
    int64_t scaleSingleK = basicBlockParam.singleK / basicBlockParam.groupSize;
    int64_t scaleKLoopNumHbm = min(CeilDiv(scaleSingleK, platformParam_.cacheLine), scaleKloopNumTotal);
    int64_t scaleKLoopNumL2 = scaleKloopNumTotal - scaleKLoopNumHbm;
    int64_t scaleASizeHbm = basicBlockParam.singleM * platformParam_.cacheLine * scaleKLoopNumHbm;
    int64_t scaleASizeL2 = basicBlockParam.singleM * platformParam_.cacheLine * scaleKLoopNumL2 /
                           CeilDiv(platformParam_.cacheLine, scaleSingleK);
    int64_t scaleBSizeHbm = basicBlockParam.singleN * platformParam_.cacheLine * scaleKLoopNumHbm;
    int64_t scaleBSizeL2 = basicBlockParam.singleN * platformParam_.cacheLine * scaleKLoopNumL2 /
                           CeilDiv(platformParam_.cacheLine, scaleSingleK);
    int64_t sizeX1 = basicBlockParam.singleM * basicBlockParam.singleK * aDtypeBits_ / BYTE_SIZE;
    int64_t sizeX2 = basicBlockParam.singleN * basicBlockParam.singleK * bDtypeBits_ / BYTE_SIZE;

    int64_t mte2DataSizeHbm =
        (sizeX1 + scaleASizeHbm) * basicBlockParam.mDim + (sizeX2 + scaleBSizeHbm) * basicBlockParam.nDim;
    int64_t mte2DataSizeL2 = (scaleASizeHbm + sizeX1) * (platformParam_.blockNum - basicBlockParam.mDim) +
                             (scaleBSizeHbm + sizeX2) * (platformParam_.blockNum - basicBlockParam.nDim) +
                             (scaleASizeL2 + scaleBSizeL2 + (scaleASizeHbm + sizeX1) * (singleNLoop - 1) +
                              (scaleBSizeHbm + sizeX2) * (singleMLoop - 1)) *
                                 basicBlockParam.nDim * basicBlockParam.mDim;
    mte2DataSize = mte2DataSizeHbm + mte2DataSizeL2;
    basicBlockParam.basicBlock.mte2BW =
        mte2DataSize / (mte2DataSizeHbm / platformParam_.hbmBW + mte2DataSizeL2 / platformParam_.l2BW);
}

void QuantBatchMatmulV4BasicBlockTiling::UpdateL1Param(L1TilingParam &l1TilingParam, int64_t &mte2DataSize,
                                                             double &mte2Cost, bool &updateFlag)
{
    int64_t tmpMte2DataSize = 0;
    int64_t tmpIterateOrder = 1;
    GetMte2DataSize(basicBlockParam_, tmpMte2DataSize, tmpIterateOrder);
    const double tmpMte2Cost = GetMte2Cost(tmpMte2DataSize, basicBlockParam_.basicBlock.mte2BW);
    // GetMte2DataSize 在非 mx 场景会忽略stepK，导致仅保留stepK = 1
    bool lessCost = isMxType_ ? tmpMte2Cost < mte2Cost : tmpMte2Cost <= mte2Cost;
    if (lessCost) {
        l1TilingParam = basicBlockParam_.l1Param;
        l1TilingParam.iterateOrder = tmpIterateOrder;
        mte2Cost = tmpMte2Cost;
        mte2DataSize = tmpMte2DataSize;
        updateFlag = true;
    }
}

bool QuantBatchMatmulV4BasicBlockTiling::CheckL1TilingInvalid(int64_t stepKa, int64_t stepKb, int64_t stepKMax)
{
    bool invalidFlag = (stepKb * basicBlockParam_.basicBlock.baseK) % basicBlockParam_.groupSize > 0 &&
                       basicBlockParam_.groupSize % (stepKb * basicBlockParam_.basicBlock.baseK) > 0;
    invalidFlag = invalidFlag || (weightNzFlag_ && basicBlockParam_.l1Param.B1BufferNum != BUFF_NUM_4 &&
                                  basicBlockParam_.l1Param.B1BufferNum != BUFF_NUM_1);
    // Mx场景，要求stepKa=stepKb, 当L1 K方向不全载时，kL1大于等于cacheline
    invalidFlag = invalidFlag ||
                  (isMxType_ && ((stepKa != stepKb) ||
                                 ((stepKa * basicBlockParam_.basicBlock.baseK * basicBlockParam_.l1Param.B1BufferNum <
                                   basicBlockParam_.kSize) &&
                                  basicBlockParam_.kSize > platformParam_.cacheLine &&
                                  stepKa * basicBlockParam_.basicBlock.baseK < platformParam_.cacheLine)));
    invalidFlag = invalidFlag || (ops::CeilAlign(basicBlockParam_.basicBlock.baseN, BLOCK_CUBE * BUFF_NUM_2) *
                                      static_cast<int64_t>(stepKb * basicBlockParam_.basicBlock.baseK * aByteSize_) >
                                  MAX_BL1_SIZE);
    invalidFlag = invalidFlag || stepKa == 0 || stepKb == 0 ||
                  (stepKa < stepKMax && stepKb < stepKMax && stepKa % stepKb > 0 && stepKb % stepKa > 0);
    invalidFlag =
        invalidFlag || GetL1LoadSize(basicBlockParam_.basicBlock, basicBlockParam_.l1Param) > platformParam_.l1Size;
    if (basicBlockParam_.l1Param.B1BufferNum == BUFF_NUM_3) {
        basicBlockParam_.l1Param.B1BufferNum = BUFF_NUM_2;
    }
    if (basicBlockParam_.l1Param.A1BufferNum == BUFF_NUM_3) {
        basicBlockParam_.l1Param.A1BufferNum = BUFF_NUM_2;
    }
    if (weightNzFlag_) {
        int64_t kBl1Size =
            std::min(basicBlockParam_.singleK, basicBlockParam_.l1Param.stepKb * basicBlockParam_.basicBlock.baseK);
        int64_t nBl1Size =
            std::min(basicBlockParam_.singleN, basicBlockParam_.l1Param.stepN * basicBlockParam_.basicBlock.baseN);
        invalidFlag =
            invalidFlag ||
            (!transB_ && (nBl1Size % NZ_BASIC_BLOCK_ALIGN_SIZE != 0 && kBl1Size % NZ_BASIC_BLOCK_ALIGN_SIZE != 0)) ||
            (transB_ && kBl1Size % NZ_BASIC_BLOCK_ALIGN_SIZE != 0);
    }
    return invalidFlag;
}

void QuantBatchMatmulV4BasicBlockTiling::UpdateL1ParamWithScaleFactor(L1TilingParam &l1TilingParam,
                                                                      int64_t &mte2DataSize, double &mte2Cost,
                                                                      bool &ret, const StepKParam &stepKParams)
{
    int64_t scaleFactorMax =
    isMxType_ ? CeilDiv(basicBlockParam_.singleK, basicBlockParam_.basicBlock.baseK * stepKParams.stepKaTmp) : 1;
    scaleFactorMax = max(scaleFactorMax, SCALE_FACTOR_MAX);  // 确保能取到1
    for (int64_t scaleFactorTmp = 1; scaleFactorTmp < scaleFactorMax; scaleFactorTmp++) {
        basicBlockParam_.l1Param = {1, 1, 1, stepKParams.stepKaTmp, stepKParams.stepKbTmp,
                                min(CeilDiv(stepKParams.stepKMax, stepKParams.stepKaTmp), BUFF_NUM_4),
                                min(CeilDiv(stepKParams.stepKMax, stepKParams.stepKbTmp), BUFF_NUM_4), scaleFactorTmp};
        if (CheckL1TilingInvalid(stepKParams.stepKaTmp, stepKParams.stepKbTmp, stepKParams.stepKMax)) {
            continue;
        }
        UpdateL1Param(l1TilingParam, mte2DataSize, mte2Cost, ret);
    }
}

bool QuantBatchMatmulV4BasicBlockTiling::GetL1TilingResult(L1TilingParam &l1TilingParam,
                                                                          int64_t &mte2DataSize, double &mte2Cost)
{
    if (basicBlockParam_.basicBlock.baseK > basicBlockParam_.singleK) {
        return false;
    }
    const int64_t stepKaMin = 1;
    const int64_t stepKbMin = 1;

    const int64_t stepKMax = CeilDiv(basicBlockParam_.singleK, basicBlockParam_.basicBlock.baseK);
    int64_t stepKaMax = min(CeilDiv(stepKMax, BUFF_NUM_2), STEPK_CUSTOM);
    int64_t stepKbMax = min(CeilDiv(stepKMax, BUFF_NUM_2), STEPK_CUSTOM);
    bool ret = false;
    for (int64_t stepKaTmp = stepKaMin; stepKaTmp <= stepKaMax; stepKaTmp += stepKaMin) {
        for (int64_t stepKbTmp = stepKbMin; stepKbTmp <= stepKbMax; stepKbTmp += stepKbMin) {
            StepKParam stepKParam = {stepKMax, stepKaTmp, stepKbTmp};
            UpdateL1ParamWithScaleFactor(l1TilingParam, mte2DataSize, mte2Cost, ret, stepKParam);
        }
    }
    return ret;
}

void QuantBatchMatmulV4BasicBlockTiling::AddOptionalSolution()
{
    L1TilingParam finalL1TilingParam;
    int64_t minMte2DataSize = numeric_limits<int64_t>::max();
    double minMte2Cost = numeric_limits<double>::max();
    // 选取其中mte2Cost最小的一组tiling，放入可选解集
    if (GetL1TilingResult(finalL1TilingParam, minMte2DataSize, minMte2Cost)) {
        basicBlockParam_.cycleVF = GetCycleVF(basicBlockParam_);
        basicBlockParam_.cycleBMte2 = GetCycleBMte2(basicBlockParam_);
        basicBlockParam_.cycleAMte2 = GetCycleAMte2(basicBlockParam_);
        basicBlockParam_.cycleCube = GetCycleCube(basicBlockParam_);

        basicBlockParam_.mte2DataSize = minMte2DataSize;
        basicBlockParam_.l1Param = finalL1TilingParam;
        optionalResults_.push_back(basicBlockParam_);
    }
}

void QuantBatchMatmulV4BasicBlockTiling::GetNdBasicBlockBaseKSolution(const int64_t baseM, const int64_t baseN)
{
    // baseK选取能开启L0 DB的最大值，另外满足32对齐，方便MTE2 cache line对齐
    int64_t baseK = max((BASE_MK_LIMIT / max(baseM, baseN)) / BLOCK_CUBE * BLOCK_CUBE, BLOCK_CUBE);
    baseK = min(ops::CeilAlign(basicBlockParam_.kSize, BLOCK_CUBE), ops::CeilAlign(baseK, VF_PROCESS_ELEM));
    if ((baseM * baseK * BUFF_NUM_2 > platformParam_.l0aSize) ||
        (baseN * baseK * BUFF_NUM_2 > platformParam_.l0bSize)) {
        return;
    }
    double minBW = GetMinMte2BW(baseM, baseN, basicBlockParam_.mDim, basicBlockParam_.nDim);
    double curBW = GetMte2BW(baseM, baseN, basicBlockParam_.mDim, basicBlockParam_.nDim);
    double mteBWRatio = GetMte2BWRatio(baseM, baseN, basicBlockParam_.mDim, basicBlockParam_.nDim);
    basicBlockParam_.basicBlock = {baseM, baseN, baseK, curBW, minBW, mteBWRatio, 0.0};

    AddOptionalSolution();
}

void QuantBatchMatmulV4BasicBlockTiling::GetNzBasicBlockBaseKSolution(const int64_t baseM, const int64_t baseN)
{
    int64_t baseK = max((BASE_MK_LIMIT / max(baseM, baseN)) / BLOCK_CUBE * BLOCK_CUBE, BLOCK_CUBE);
    constexpr int64_t baseKAlignThreshold = 128L;
    if (baseM >= K_MTE2_ALIGN_BASE_M_THRESHOLD && baseK > baseKAlignThreshold &&
        basicBlockParam_.kSize > K_MTE2_ALIGN_K_THRESHOLD) {
        baseK = ops::FloorAlign(baseK, baseKAlignThreshold);
    }
    baseK = min(ops::CeilAlign(basicBlockParam_.kSize, BLOCK_CUBE), baseK);
    if ((baseM * baseK * BUFF_NUM_2 > platformParam_.l0aSize) ||
        (baseN * baseK * BUFF_NUM_2 > platformParam_.l0bSize)) {
        return;
    }
    if (transB_) {
        if (baseK % NZ_BASIC_BLOCK_ALIGN_SIZE != 0) {
            baseK = ops::FloorAlign(baseK, NZ_BASIC_BLOCK_ALIGN_SIZE);
            baseK = max(baseK, NZ_BASIC_BLOCK_ALIGN_SIZE);
        }
    } else {
        if (baseK % NZ_BASIC_BLOCK_ALIGN_SIZE != 0 || baseN % NZ_BASIC_BLOCK_ALIGN_SIZE != 0) {
            return;
        }
    }
    double minBW = GetMinMte2BW(baseM, baseN, basicBlockParam_.mDim, basicBlockParam_.nDim);
    double curBW = GetMte2BW(baseM, baseN, basicBlockParam_.mDim, basicBlockParam_.nDim);
    double mteBWRatio = GetMte2BWRatio(baseM, baseN, basicBlockParam_.mDim, basicBlockParam_.nDim);
    basicBlockParam_.basicBlock = {baseM, baseN, baseK, curBW, minBW, mteBWRatio, 0.0};
    AddOptionalSolution();
}

void QuantBatchMatmulV4BasicBlockTiling::GetBasicBlockSolution()
{
    // 给定singleM、singleN、mDim、nDim，获取可行基本块集
    int64_t maxMNSize = isMxType_ ? BASE_MN_LIMIT_BUFF_2 : BASE_MN_LIMIT_BUFF_1;
    // 当nSize大于核数*BASE_N_THRESHOLD时，baseN取小于BASE_N_THRESHOLD的值会让计算轮次增加，无性能收益
    int64_t baseNMin =
        basicBlockParam_.nSize >= BASE_N_THRESHOLD * platformParam_.blockNum ? BASE_N_THRESHOLD : BLOCK_CUBE;
    int64_t baseMMin =
        basicBlockParam_.mSize >= BASE_M_THRESHOLD * platformParam_.blockNum ? BASE_M_THRESHOLD : BLOCK_CUBE;
    for (int64_t baseM = baseMMin; baseM <= basicBlockParam_.singleM; baseM += BLOCK_CUBE) {
        int64_t baseNMax = min(BASE_BLOCK_MAX, (maxMNSize / baseM) / BLOCK_CUBE * BLOCK_CUBE);
        baseNMax = min(basicBlockParam_.singleN, baseNMax);
        for (int64_t baseN = baseNMax; baseN >= baseNMin; baseN -= BLOCK_CUBE) {
            if (baseM * baseN > maxMNSize || baseM > BASE_BLOCK_MAX || baseN > BASE_BLOCK_MAX) {
                continue;
            }
            if (weightNzFlag_) {
                GetNzBasicBlockBaseKSolution(baseM, baseN);
            } else {
                GetNdBasicBlockBaseKSolution(baseM, baseN);
            }
        }
    }
}

int64_t QuantBatchMatmulV4BasicBlockTiling::GetL1LoadSize(const BasicBlock &basicBlock,
                                                          const L1TilingParam &l1Param) const
{
    int64_t l1LoadSize =
        static_cast<int64_t>(basicBlock.baseM * basicBlock.baseK * l1Param.stepKa * l1Param.A1BufferNum * aDtypeBits_ +
                             basicBlock.baseN * basicBlock.baseK * l1Param.stepKb * l1Param.B1BufferNum * aDtypeBits_ +
                             basicBlock.baseN * biasDtypeBits_) / BYTE_SIZE;
    if (isMxType_) {
        l1LoadSize += basicBlock.baseK * (basicBlock.baseM * l1Param.stepKa + basicBlock.baseN * l1Param.stepKb) *
                      l1Param.scaleFactor * BUFF_NUM_2 / basicBlockParam_.groupSize;
        if (hasBias_) {
            l1LoadSize += (l1Param.B1BufferNum - 1) * basicBlock.baseN * biasDtypeBits_ / BYTE_SIZE;
        }
    }
    return l1LoadSize;
}

void QuantBatchMatmulV4BasicBlockTiling::PrintFinalResult(const BasicBlockParam &param, bool enable) const
{
    if (enable) {
        OP_LOGD(opName_,
                "Tiling result: mSize: %ld, nSize: %ld, kSize: %ld, groupSize: %ld, singleM: %ld, singleN: %ld, "
                "singleK: %ld, mDim: %ld, nDim: %ld, kDim: %ld, mte2DataSize: %ld, fixpDataSize: %ld, iterateOrder: "
                "%ld, stepM: %ld, stepN: %ld, stepKa: %ld, stepKb: %ld, A1BufferNum: %ld, B1BufferNum: %ld, baseM: "
                "%ld, baseN: %ld, baseK: %ld, mte2BW: %lf, mte2MinBW: %lf, mte2BWRatio: %lf, mte2TailBWRatio: %lf, "
                "cycleVF: %ld, cycleBMte2: %ld, cycleAMte2: %ld, cycleCube: %ld",
                param.mSize, param.nSize, param.kSize, basicBlockParam_.groupSize, param.singleM, param.singleN,
                param.singleK, param.mDim, param.nDim, param.kDim, param.mte2DataSize, param.fixpDataSize,
                param.l1Param.iterateOrder, param.l1Param.stepM, param.l1Param.stepN, param.l1Param.stepKa,
                param.l1Param.stepKb, param.l1Param.A1BufferNum, param.l1Param.B1BufferNum, param.basicBlock.baseM,
                param.basicBlock.baseN, param.basicBlock.baseK, param.basicBlock.mte2BW, param.basicBlock.mte2MinBW,
                param.basicBlock.mte2BWRatio, param.basicBlock.mte2TailBWRatio,
                param.cycleVF, param.cycleBMte2, param.cycleAMte2, param.cycleCube);
    }
}

bool QuantBatchMatmulV4BasicBlockTiling::ValidateTilingResult() const
{
    OP_TILING_CHECK(basicBlockParam_.mDim * basicBlockParam_.nDim * basicBlockParam_.kDim > platformParam_.blockNum,
                    VECTOR_INNER_ERR_REPORT_TILIING(
                        opName_, "Invalid block dim, mDim: %ld, nDim: %ld, kDim: %ld, maxDimNum: %ld",
                        basicBlockParam_.mDim, basicBlockParam_.nDim, basicBlockParam_.kDim, platformParam_.blockNum),
                    return false);

    OP_TILING_CHECK(GetL1LoadSize(basicBlockParam_.basicBlock, basicBlockParam_.l1Param) > platformParam_.l1Size,
                    VECTOR_INNER_ERR_REPORT_TILIING(
                        opName_, "The load size exceeds L1 buffer limit, load size: %ld, L1 buffer size: %ld",
                        GetL1LoadSize(basicBlockParam_.basicBlock, basicBlockParam_.l1Param), platformParam_.l1Size),
                    return false);

    const int64_t a2Size = static_cast<int64_t>(basicBlockParam_.basicBlock.baseM * basicBlockParam_.basicBlock.baseK *
        aByteSize_) * BUFF_NUM_2;
    const int64_t b2Size = static_cast<int64_t>(basicBlockParam_.basicBlock.baseN * basicBlockParam_.basicBlock.baseK *
        aByteSize_) * BUFF_NUM_2;

    OP_TILING_CHECK(
        a2Size > platformParam_.l0aSize || b2Size > platformParam_.l0bSize || a2Size == 0 || b2Size == 0,
        VECTOR_INNER_ERR_REPORT_TILIING(
            opName_,
            "The load size may exceed L0 buffer limit, L0A load size: %ld, L0B load size: %ld, L0 buffer size: %ld",
            a2Size, b2Size, platformParam_.l0aSize),
        return false);

    int64_t stepKMax = CeilDiv(basicBlockParam_.singleK, basicBlockParam_.basicBlock.baseK);

    OP_TILING_CHECK((basicBlockParam_.l1Param.stepKa < stepKMax && basicBlockParam_.l1Param.stepKb < stepKMax) &&
                        (basicBlockParam_.l1Param.stepKa % basicBlockParam_.l1Param.stepKb > 0 &&
                         basicBlockParam_.l1Param.stepKb % basicBlockParam_.l1Param.stepKa > 0),
                    VECTOR_INNER_ERR_REPORT_TILIING(
                        opName_, "Invalid stepK, stepKa (%ld) should be divisible by stepKb (%ld) or otherwise",
                        basicBlockParam_.l1Param.stepKa, basicBlockParam_.l1Param.stepKb),
                    return false);

    return true;
}

bool QuantBatchMatmulV4BasicBlockTiling::GetFinalResult()
{
    bool ret = true;
    if (!optionalResults_.empty()) {
        sort(optionalResults_.begin(), optionalResults_.end(), CompareOptionalResult);
        basicBlockParam_ = optionalResults_.front();
    }

    PrintFinalResult(basicBlockParam_, ret);
    return ret && ValidateTilingResult();
}

/*
    该函数根据给定shape计算遍历能够满足cube bound且绑满核的解，再找出cycle最小的解。后续还需考虑：
    1）核间偏移非128B对齐引入的MTE2拆包、fixpipe写出效率下降；
    2）避免fixpipe bound需要考虑kL1；
    3）尾块分析；
*/
bool QuantBatchMatmulV4BasicBlockTiling::GetBasicBlockTiling()
{
    OP_TILING_CHECK(!ValidateInputParam(), VECTOR_INNER_ERR_REPORT_TILIING(opName_, "Invalid input param"),
                    return false);

    Reset();
    int64_t mDimMax = min(CeilDiv(basicBlockParam_.mSize, BLOCK_CUBE), platformParam_.blockNum);
    for (int64_t mDim = mDimMax; mDim >= 1; mDim--) {
        int64_t nDimMax = min(CeilDiv(basicBlockParam_.nSize, BLOCK_CUBE), platformParam_.blockNum / mDim);
        for (int64_t nDim = nDimMax; nDim >= 1; nDim--) {
            if ((!optionalResults_.empty()) &&
                // 0.8含义：当已有可选解时剪枝分核情况小于0.8倍总核数的解
                mDim * nDim < 0.8 * platformParam_.blockNum) {
                continue;
            }
            basicBlockParam_.mDim = mDim;
            basicBlockParam_.nDim = nDim;
            basicBlockParam_.singleM =
                ops::CeilAlign(CeilDiv(basicBlockParam_.mSize, basicBlockParam_.mDim), BLOCK_CUBE);
            int64_t alignSize;
            if (isMxType_) {
                alignSize = weightNzFlag_ ? BLOCK_CUBE : NZ_SINGLE_N_ALIGN_SIZE;
            } else {
                // 原逻辑为 alignSize = weightNzFlag_ ? BLOCK_CUBE : NZ_SINGLE_N_ALIGN_SIZE，
                // 这里仅修改 pergroup nz下的对齐
                alignSize = NZ_SINGLE_N_ALIGN_SIZE;
            }
            basicBlockParam_.singleN = ops::CeilAlign(CeilDiv(basicBlockParam_.nSize, basicBlockParam_.nDim),
                                                      alignSize);
            int64_t mDimFixed = CeilDiv(basicBlockParam_.mSize, basicBlockParam_.singleM);
            int64_t nDimFixed = CeilDiv(basicBlockParam_.nSize, basicBlockParam_.singleN);
            if (mDimFixed != basicBlockParam_.mDim || nDimFixed != basicBlockParam_.nDim) {
                continue;
            }
            GetBasicBlockSolution();
        }
    }

    return GetFinalResult();
}
}  // namespace optiling
