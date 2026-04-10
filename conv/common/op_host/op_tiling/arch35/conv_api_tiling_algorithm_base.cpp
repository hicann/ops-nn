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
 * \file conv_api_tiling_algorithm_base.cpp
 * \brief
 */

#include <cstdint>
#include "conv_api_tiling_algorithm_base.h"

using namespace std;

namespace conv_tiling {
ConvTilingAlgorithmBase::ConvTilingAlgorithmBase(ConvTilingBase *tilingIns)
{
    tilingIns_ = tilingIns;
    this->fMapDTypeSize = DTYPE_SIZE_TAB.at(tilingIns_->descInfo.fMapType.dtype);
    this->weightDTypeSize = DTYPE_SIZE_TAB.at(tilingIns_->descInfo.weightType.dtype);
    if (tilingIns_->hasBias) {
        this->biasDTypeSize = DTYPE_SIZE_TAB.at(tilingIns_->descInfo.biasType.dtype);
    }
    if (tilingIns_->hasScale) {
        this->scaleDtypeSize = DTYPE_SIZE_TAB.at(tilingIns_->descInfo.scaleType.dtype);
    }
}

uint64_t ConvTilingAlgorithmBase::CalcAL0Size(uint64_t mL0, uint64_t kL0) const
{
    return AlignB(mL0, tilingIns_->cubeInfo.m0) * AlignB(kL0, tilingIns_->cubeInfo.k0) * this->dbValue.pbAL0 *
        this->fMapDTypeSize * tilingIns_->innerBatch;
}

uint64_t ConvTilingAlgorithmBase::CalcBL0Size(uint64_t kL0, uint64_t nL0) const
{
    return AlignB(kL0, tilingIns_->cubeInfo.k0) * AlignB(nL0, tilingIns_->cubeInfo.n0) * this->dbValue.pbBL0 *
        this->weightDTypeSize;
}

uint64_t ConvTilingAlgorithmBase::CalcCL0Size(uint64_t mL0, uint64_t nL0) const
{
    // use mmad dtype size
    return AlignB(mL0, tilingIns_->cubeInfo.m0) * AlignB(nL0, tilingIns_->cubeInfo.n0) * this->dbValue.pbCL0 *
        DTYPE_SIZE_TAB.at(tilingIns_->cubeInfo.madType) * tilingIns_->innerBatch;
}

bool ConvTilingAlgorithmBase::CheckL0Buffer(uint64_t currmL0, uint64_t currkL0, uint64_t currnL0) const
{
    if (CalcAL0Size(currmL0, currkL0) > tilingIns_->platformInfo.l0ASize ||
        CalcBL0Size(currkL0, currnL0) > tilingIns_->platformInfo.l0BSize ||
        CalcCL0Size(currmL0, currnL0) > tilingIns_->platformInfo.l0CSize) {
            return false;
    } else {
        return true;
    }
}

uint64_t ConvTilingAlgorithmBase::CalcBTSize(uint64_t nL0) const
{
    // use biasBT dtype size
    return tilingIns_->hasBias ? AlignB(nL0, tilingIns_->cubeInfo.n0) * biasDTypeSize : 0;
}

uint64_t ConvTilingAlgorithmBase::CalcFBSize(uint64_t nL0) const
{
    return AlignB(nL0, tilingIns_->cubeInfo.n0) * tilingIns_->shapeInfo.channelWiseCoeff * FP16_DTYPE_SIZE;
}

uint64_t ConvTilingAlgorithmBase::InferHiL1(uint64_t hoL1, int64_t hi) const
{
    int64_t khDilated = (tilingIns_->shapeInfo.singlekH - 1) * tilingIns_->attrInfo.dilationH + 1;
    int64_t tmpHiL1 = (hoL1 - 1) * tilingIns_->attrInfo.strideH + khDilated;
    if (tmpHiL1 > hi) {
        tmpHiL1 = hi;
    }

    return tmpHiL1;
}

uint64_t ConvTilingAlgorithmBase::InferWiL1(uint64_t woL1, int64_t wi) const
{
    if (woL1 >= static_cast<uint64_t>(tilingIns_->shapeInfo.singleWo) && tilingIns_->isC04Flag) {
        return tilingIns_->shapeInfo.orgWi;
    }

    int64_t kwDilated = (tilingIns_->shapeInfo.singlekW - 1) * tilingIns_->attrInfo.dilationW + 1;
    int64_t tmpWiL1 = (woL1 - 1) * tilingIns_->attrInfo.strideW + kwDilated;
    if (tmpWiL1 > wi) {
        tmpWiL1 = wi;
    }

    return tmpWiL1;
}

void ConvTilingAlgorithmBase::PrintRanges(std::vector<uint64_t> inputRanges, std::string rangeName) const
{
    std::string res = "";
    res += rangeName;
    res += ": ";
    for (auto v: inputRanges) {
        res += to_string(v);
        res += " ";
    }
    TILING_LOG_DEBUG("%s", res.c_str());
}

void ConvTilingAlgorithmBase::ResetOptGroupDoubleBuffer(bool resetFlag)
{
    if (resetFlag) {
        uint64_t curAL1Size = 0;
        if (tilingIns_->outputOrder == static_cast<int8_t>(OutputOrder::M)) {
            uint64_t curHoAL1 = min(static_cast<uint64_t>(tilingIns_->shapeInfo.orgHo),
                static_cast<uint64_t>(tilingIns_->l1TilingInfo.mAL1 / tilingIns_->shapeInfo.orgWo + CONST_VALUE_2));
            uint64_t curHiAL1 = InferHiL1(curHoAL1, tilingIns_->shapeInfo.orgHi);
            curAL1Size = AlignB(curHiAL1 * tilingIns_->shapeInfo.orgWi * tilingIns_->l1TilingInfo.kAL1 /
                tilingIns_->shapeInfo.orgkH / tilingIns_->shapeInfo.orgkW * this->fMapDTypeSize, C0_SIZE);
        } else {
            uint64_t curHiAL1 = InferHiL1(tilingIns_->l1TilingInfo.hoAL1, tilingIns_->shapeInfo.orgHi);
            uint64_t curWiAL1 = InferWiL1(tilingIns_->l1TilingInfo.woAL1, tilingIns_->shapeInfo.orgWi);
            curAL1Size = AlignB(curHiAL1 * curWiAL1 * tilingIns_->l1TilingInfo.kAL1 /
                tilingIns_->shapeInfo.orgkH / tilingIns_->shapeInfo.orgkW * this->fMapDTypeSize, C0_SIZE);
        }

        uint64_t curBL1Size =
            AlignB(tilingIns_->l1TilingInfo.kBL1 * tilingIns_->l1TilingInfo.nBL1 * this->weightDTypeSize, C0_SIZE);

        uint64_t curBiasSize = tilingIns_->hasBias ?
            AlignB(tilingIns_->shapeInfo.orgCo * this->biasDTypeSize, C0_SIZE) : 0;
        uint64_t needL1Size = (curAL1Size + curBL1Size) * CONST_VALUE_2 + curBiasSize;

        if (needL1Size <= tilingIns_->platformInfo.l1Size) {
            dbValue.pbBL1 = DOUBLE_BUFFER_NUM;
            dbValue.pbAL1 = DOUBLE_BUFFER_NUM;
        }
    }
}

bool ConvTilingAlgorithmBase::CheckOptGroupPreload() const
{
    bool sceneFlag = tilingIns_->optGroupFlag && tilingIns_->innerBatch == 1 &&
        !tilingIns_->isC04Flag && !tilingIns_->isDmaFlag;

    uint64_t kSize = tilingIns_->shapeInfo.singleCi1 * tilingIns_->shapeInfo.singlekH *
        tilingIns_->shapeInfo.singlekW * tilingIns_->cubeInfo.k0;
    bool kAL1FullloadFlag = tilingIns_->l1TilingInfo.kAL1 == kSize;
    bool kBL1FullloadFlag = tilingIns_->l1TilingInfo.kBL1 == kSize;
    bool nBL1FullloadFlag = tilingIns_->l1TilingInfo.nBL1 == static_cast<uint64_t>(tilingIns_->shapeInfo.singleCo);
    bool fullLoadFlag = kAL1FullloadFlag && kBL1FullloadFlag && nBL1FullloadFlag;

    bool otherFlag = tilingIns_->l1TilingInfo.iterateMNOrder == IterateMNOrder::ITER_M_FST &&
                     tilingIns_->ubTilingInfo.mUb == 0 && tilingIns_->ubTilingInfo.nUb == 0;

    bool multiMNFlag = tilingIns_->l1TilingInfo.mAL1 == tilingIns_->l0TilingInfo.mL0;
    if (tilingIns_->outputOrder == static_cast<int8_t>(OutputOrder::HW)) {
        multiMNFlag = tilingIns_->l1TilingInfo.hoAL1 == tilingIns_->l0TilingInfo.hoL0 &&
                     tilingIns_->l1TilingInfo.woAL1 == tilingIns_->l0TilingInfo.woL0;
    }
    multiMNFlag = multiMNFlag && (tilingIns_->l1TilingInfo.nBL1 == tilingIns_->l0TilingInfo.nL0);

    return (sceneFlag && fullLoadFlag && otherFlag && multiMNFlag);
}

void ConvTilingAlgorithmBase::SetPBufferRes()
{
    bool resetOptGroupFlag = CheckOptGroupPreload();
    ResetOptGroupDoubleBuffer(resetOptGroupFlag);

    tilingIns_->dbValue.pBufferFlag = 0;
    tilingIns_->dbValue.pbBL1 = dbValue.pbBL1;
    tilingIns_->dbValue.pbAL1 = dbValue.pbAL1;
    tilingIns_->dbValue.pbCL0 = dbValue.pbCL0;
    tilingIns_->dbValue.pbBL0 = dbValue.pbBL0;
    tilingIns_->dbValue.pbAL0 = dbValue.pbAL0;
    tilingIns_->dbValue.pBufferFlag = tilingIns_->dbValue.pBufferFlag |
                                      (tilingIns_->dbValue.pbBL1 == DOUBLE_BUFFER_NUM ? 1 : 0);
    tilingIns_->dbValue.pBufferFlag = (tilingIns_->dbValue.pBufferFlag << 1) |
                                      (tilingIns_->dbValue.pbAL1 == DOUBLE_BUFFER_NUM ? 1 : 0);
    tilingIns_->dbValue.pBufferFlag = (tilingIns_->dbValue.pBufferFlag << 1) |
                                      (tilingIns_->dbValue.pbCL0 == DOUBLE_BUFFER_NUM ? 1 : 0);
    tilingIns_->dbValue.pBufferFlag = (tilingIns_->dbValue.pBufferFlag << 1) |
                                      (tilingIns_->dbValue.pbBL0 == DOUBLE_BUFFER_NUM ? 1 : 0);
    tilingIns_->dbValue.pBufferFlag = (tilingIns_->dbValue.pBufferFlag << 1) |
                                      (tilingIns_->dbValue.pbAL0 == DOUBLE_BUFFER_NUM ? 1 : 0);
    TILING_LOG_DEBUG("pBufferFlag: %ld, pbAL0: %d, pbBL0: %d, pbCL0: %d, pbAL1: %d, pbBL1: %d, ",
                     tilingIns_->dbValue.pBufferFlag, tilingIns_->dbValue.pbAL0, tilingIns_->dbValue.pbBL0,
                     tilingIns_->dbValue.pbCL0, tilingIns_->dbValue.pbAL1, tilingIns_->dbValue.pbBL1);
}
}