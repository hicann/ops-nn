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
 * \file conv3d_tiling_algorithm_pointwise.cpp
 * \brief
 */

#include <cstdint>
#include "conv3d_api_tiling_algorithm.h"
#include "conv3d_api_tiling_algorithm_pointwise.h"

using namespace std;

namespace Conv3dApiTiling {

bool Conv3dTilingAlgorithmPointWise::CheckL1Buffer() const
{
    // MaL1 * KaL1 * dtype_size
    uint64_t currentFmL1Size =
        this->l1TilingRange.mAL1ValueRange.at(this->l1TilingIdx.mAL1Idx) * this->doubleBufferValue.pbAL1 *
        this->l1TilingRange.kAL1Range.at(this->l1TilingIdx.kAL1Idx) *
        this->fMapDTypeSize;
    uint64_t currentWeightL1Size =
        this->l1TilingRange.kBL1Range.at(this->l1TilingIdx.kBL1Idx) * this->doubleBufferValue.pbBL1 *
        this->l1TilingRange.nBL1ValueRange.at(this->l1TilingIdx.nBL1Idx) * this->weightDTypeSize;
    uint64_t biasSize = this->l1TilingFlag.isBiasFullLoad ?
        tilingIns_->shapeCalc.singleCo1 * tilingIns_->cubeInfo.n0 * this->biasDTypeSize :
        this->l0TilingParams.nL0 * this->biasDTypeSize;
    uint64_t currentBiasL1Size = tilingIns_->hasBias ?
        biasSize * POINT_WISE_BIAS_CUBE_UNIT : INITIAL_SIZE;
    // cal current fm in L1
    if (this->l1TilingFlag.abL1Mode == L1TilingMode::ALL_FULL_LOAD) {
        currentFmL1Size = this->l1TilingCalc.fmapFullLoadL1Size;
        currentWeightL1Size = this->l1TilingCalc.weightFullLoadL1Size;
    } else if (this->l1TilingFlag.abL1Mode == L1TilingMode::FULL_LOAD_AL1) {
        currentFmL1Size = this->l1TilingCalc.fmapFullLoadL1Size;
    } else if (this->l1TilingFlag.abL1Mode == L1TilingMode::FULL_LOAD_BL1) {
        currentWeightL1Size = this->l1TilingCalc.weightFullLoadL1Size;
    }

    uint64_t l1SizeCur = currentFmL1Size + currentWeightL1Size + currentBiasL1Size;

    return l1SizeCur <= tilingIns_->platformInfo.l1Size;
}

uint64_t Conv3dTilingAlgorithmPointWise::CalcBTSize([[maybe_unused]] uint64_t currnL0) const
{
    return INITIAL_SIZE;
}

void Conv3dTilingAlgorithmPointWise::WeightBypassDecision()
{
    return;
}

int64_t Conv3dTilingAlgorithmPointWise::InitCalcL1Params()
{
    this->l1TilingCalc.ci0HkWk = static_cast<uint64_t>(tilingIns_->shapeInfo.singlekH *
                                 tilingIns_->shapeInfo.singlekW * static_cast<int64_t>(tilingIns_->cubeInfo.k0));
    this->l1TilingCalc.alignCinKhKwKd = AlignB(static_cast<uint64_t>(tilingIns_->shapeInfo.singleCi),
                                               static_cast<uint64_t>(tilingIns_->cubeInfo.k0)) *
                                               static_cast<uint64_t>(tilingIns_->shapeInfo.orgkH *
                                               tilingIns_->shapeInfo.orgkW * tilingIns_->shapeInfo.orgkD);
    // cal fmap weight full load in L1 size
    uint64_t fampSizeInL1 = static_cast<uint64_t>(tilingIns_->shapeInfo.singlekD) * tilingIns_->shapeCalc.singleCi1 *
        tilingIns_->shapeCalc.singleM1 * static_cast<uint64_t>(tilingIns_->cubeInfo.m0 * tilingIns_->cubeInfo.k0);
    if ((fampSizeInL1 * this->fMapDTypeSize) / this->fMapDTypeSize != fampSizeInL1) {
        TILING_ERROR_LOG("fmap size in l1 is overflow uint64, initcalc l1 params failed!");
        return INVALID_VALUE;
    }
    this->l1TilingCalc.fmapFullLoadL1Size = static_cast<uint64_t>(tilingIns_->shapeInfo.singlekD) *
        AlignB(tilingIns_->shapeCalc.singleCi1 * static_cast<uint64_t>(tilingIns_->cubeInfo.k0), POINT_WISE_ALIGN_UNIT) *
        tilingIns_->shapeCalc.singleM1 * static_cast<uint64_t>(tilingIns_->cubeInfo.m0) * this->fMapDTypeSize;
    uint64_t weightSizeInL1 = static_cast<uint64_t>(tilingIns_->shapeInfo.singlekD) * tilingIns_->shapeCalc.singleCi1 *
        this->l1TilingCalc.ci0HkWk * tilingIns_->shapeCalc.singleCo1 * static_cast<uint64_t>(tilingIns_->cubeInfo.n0);
    if ((weightSizeInL1 * this->weightDTypeSize) / weightDTypeSize != weightSizeInL1) {
        TILING_ERROR_LOG("weight size in l1 is overflow uint64, initcalc l1 params failed!");
        return INVALID_VALUE;
    }
    this->l1TilingCalc.weightFullLoadL1Size = static_cast<uint64_t>(tilingIns_->shapeInfo.singlekD) *
        tilingIns_->shapeCalc.singleCi1 * this->l1TilingCalc.ci0HkWk * tilingIns_->shapeCalc.singleCo1 *
        static_cast<uint64_t>(tilingIns_->cubeInfo.n0) * this->weightDTypeSize;
    // cal min/kfullload fmap size in L1(mL0)
    uint64_t multiK0 = tilingIns_->descInfo.fMapType.dtype == ConvDtype::FLOAT32
        ? static_cast<uint64_t>(CONST_VALUE_2)
        : static_cast<uint64_t>(1);
    this->l1TilingCalc.fmapMinLoadL1Size = tilingIns_->cubeInfo.k0 * this->l0TilingParams.mL0 *
        this->fMapDTypeSize * this->doubleBufferValue.pbAL1 * multiK0;

    this->l1TilingCalc.fmapKL1FullLoadSize = static_cast<uint64_t>(tilingIns_->shapeInfo.singlekD) * tilingIns_->shapeCalc.singleCi1 *
        static_cast<uint64_t>(tilingIns_->cubeInfo.k0) * this->l0TilingParams.mL0 * static_cast<uint64_t>(this->doubleBufferValue.pbAL1) *
        this->fMapDTypeSize;
    // cal min/kfullload weiht size in L1
    this->l1TilingCalc.weightMinLoadL1Size = this->l1TilingCalc.ci0HkWk * this->l0TilingParams.nL0 *
        this->doubleBufferValue.pbBL1 * this->weightDTypeSize * multiK0;
    this->l1TilingCalc.weightKL1FullLoadSize = static_cast<uint64_t>(tilingIns_->shapeInfo.singlekD) * tilingIns_->shapeCalc.singleCi1 *
        this->l1TilingCalc.ci0HkWk * this->l0TilingParams.nL0 * static_cast<uint64_t>(this->doubleBufferValue.pbBL1) * this->weightDTypeSize;
    // cal bias size in L1
    this->l1TilingCalc.biasMinLoadL1Size = tilingIns_->hasBias ? this->l0TilingParams.nL0 * POINT_WISE_BIAS_CUBE_UNIT *
                                            this->biasDTypeSize : INITIAL_SIZE;

    return VALID_VALUE;
}

void Conv3dTilingAlgorithmPointWise::InitABL1TilingMode()
{
    // init L1 fmap weight full load case and init mode
    if (this->l1TilingCalc.fmapFullLoadL1Size + this->l1TilingCalc.weightFullLoadL1Size +
        this->l1TilingCalc.biasMinLoadL1Size <= tilingIns_->platformInfo.l1Size) {
        this->l1TilingFlag.abL1Mode = L1TilingMode::ALL_FULL_LOAD;
        this->doubleBufferValue.pbAL1 = ENABLE_DOUBLE_BUFFER;
        this->doubleBufferValue.pbBL1 = ENABLE_DOUBLE_BUFFER;
        return;
    }
    if (this->l1TilingCalc.fmapFullLoadL1Size <= this->l1TilingCalc.weightFullLoadL1Size) {
        if (this->l1TilingCalc.weightFullLoadL1Size + this->l1TilingCalc.fmapMinLoadL1Size +
            this->l1TilingCalc.biasMinLoadL1Size <= tilingIns_->platformInfo.l1Size) {
            this->l1TilingFlag.abL1Mode = L1TilingMode::FULL_LOAD_BL1;
            this->doubleBufferValue.pbBL1 = ENABLE_DOUBLE_BUFFER;
            return;
        } else if (this->l1TilingCalc.fmapFullLoadL1Size + this->l1TilingCalc.weightMinLoadL1Size +
            this->l1TilingCalc.biasMinLoadL1Size <= tilingIns_->platformInfo.l1Size) {
            this->l1TilingFlag.abL1Mode = L1TilingMode::FULL_LOAD_AL1;
            this->doubleBufferValue.pbAL1 = ENABLE_DOUBLE_BUFFER;
            return;
        }
    } else {
        if (this->l1TilingCalc.fmapFullLoadL1Size + this->l1TilingCalc.weightMinLoadL1Size +
            this->l1TilingCalc.biasMinLoadL1Size <= tilingIns_->platformInfo.l1Size) {
            this->l1TilingFlag.abL1Mode = L1TilingMode::FULL_LOAD_AL1;
            this->doubleBufferValue.pbAL1 = ENABLE_DOUBLE_BUFFER;
            return;
        } else if (this->l1TilingCalc.weightFullLoadL1Size + this->l1TilingCalc.fmapMinLoadL1Size +
            this->l1TilingCalc.biasMinLoadL1Size <= tilingIns_->platformInfo.l1Size) {
            this->l1TilingFlag.abL1Mode = L1TilingMode::FULL_LOAD_BL1;
            this->doubleBufferValue.pbBL1 = ENABLE_DOUBLE_BUFFER;
            return;
        }
    }
    // other case None can full load in L1 case
    this->l1TilingFlag.abL1Mode = L1TilingMode::NONE_FULL_LOAD;
    return;
}


void Conv3dTilingAlgorithmPointWise::GetL1TilingRange()
{
    if (tilingIns_->descInfo.fMapType.dtype == ConvDtype::FLOAT32) {
        CalcFactorPointWise(tilingIns_->shapeCalc.singleCi1, this->l1TilingRange.kAL1Range);
        CalcFactorPointWise(tilingIns_->shapeCalc.singleCi1, this->l1TilingRange.kBL1Range);
    } else {
        CalcCommFactor(tilingIns_->shapeCalc.singleCi1, tilingIns_->shapeCalc.singleCi1, this->l1TilingRange.kAL1Range);
        CalcCommFactor(tilingIns_->shapeCalc.singleCi1, tilingIns_->shapeCalc.singleCi1, this->l1TilingRange.kBL1Range);
    }

    VectorElementMultip(this->l1TilingRange.kAL1Range, this->l1TilingCalc.ci0HkWk);
    VectorElementMultip(this->l1TilingRange.kBL1Range, this->l1TilingCalc.ci0HkWk);

    // cal mAL1Value and nBL1Value
    uint64_t multiNBL1Max = CeilDiv(tilingIns_->shapeCalc.singleCo1 * tilingIns_->cubeInfo.n0,
                                    this->l0TilingParams.nL0);
    CalcCommFactor(multiNBL1Max, multiNBL1Max, this->l1TilingRange.nBL1ValueRange);
    VectorElementMultip(this->l1TilingRange.nBL1ValueRange, l0TilingParams.nL0);
    uint64_t multiMAL1Max = CeilDiv(CeilDiv(static_cast<uint64_t>(tilingIns_->shapeInfo.singleM),
                                            static_cast<uint64_t>(tilingIns_->cubeInfo.m0)) * static_cast<uint64_t>(tilingIns_->cubeInfo.m0),
                                    this->l0TilingParams.mL0);
    CalcCommFactor(multiMAL1Max, multiMAL1Max, this->l1TilingRange.mAL1ValueRange);
    VectorElementMultip(this->l1TilingRange.mAL1ValueRange, l0TilingParams.mL0);
}

void Conv3dTilingAlgorithmPointWise::GetKL0TilingRangeCommon(uint64_t k0)
{
    uint64_t maxKAL12L0Loop = CeilDiv(this->l1TilingRange.kAL1Range.at(this->l1TilingIdx.kAL1Idx),
                                    k0);
    uint64_t maxKBL12L0Loop = CeilDiv(this->l1TilingRange.kBL1Range.at(this->l1TilingIdx.kBL1Idx),
                                    k0);
    uint64_t factorK = Gcd(maxKAL12L0Loop, maxKBL12L0Loop);
    CalcCommFactor(factorK, factorK, this->l0TilingRange.kL0Range);
    VectorElementMultip(this->l0TilingRange.kL0Range, k0);
}

void Conv3dTilingAlgorithmPointWise::GetKL0TilingDecision()
{
     // get k0 range according to kal1 and kbl1
    if (tilingIns_->descInfo.fMapType.dtype == ConvDtype::FLOAT32) {
        GetKL0TilingRangeCommon(POINT_WISE_ALIGN_UNIT);
    } else {
        GetKL0TilingRangeCommon(tilingIns_->cubeInfo.k0);
    }

    if (FixL0PingpongDecision()) {
        // when fix l0 pingpong res, kl0 decision is full load
        return;
    }

    // kL0 decision
    while (this->l0TilingIdx.kL0Idx < this->l0TilingRange.kL0Range.size() &&
           CheckL0Buffer(this->l0TilingParams.mL0, this->l0TilingRange.kL0Range.at(this->l0TilingIdx.kL0Idx),
                         this->l0TilingParams.nL0)) {
        this->l0TilingIdx.kL0Idx++;
    }
    this->l0TilingIdx.kL0Idx = this->l0TilingIdx.kL0Idx == INITIAL_INDEX ?
        INITIAL_INDEX : static_cast<uint64_t>(this->l0TilingIdx.kL0Idx - 1);
    this->l0TilingParams.kL0 = this->l0TilingRange.kL0Range.at(this->l0TilingIdx.kL0Idx);
    tilingIns_->l0TilingInfo.kL0 = this->l0TilingParams.kL0;
    tilingIns_->l0TilingInfo.kL0xorgCoAlignN0 = this->l0TilingParams.kL0 * this->l0TilingParams.orgCoAlignN0;
    return;
}

uint64_t Conv3dTilingAlgorithmPointWise::CalcL1SizeForL0Tiling(uint64_t currmL0, uint64_t currnL0) const
{
    uint64_t usedL1Size = currmL0 * AlignB(tilingIns_->cubeInfo.k0, POINT_WISE_ALIGN_UNIT) *
                          fMapDTypeSize * this->doubleBufferValue.pbAL1;
    usedL1Size += currnL0 * AlignB(tilingIns_->cubeInfo.k0, POINT_WISE_ALIGN_UNIT) *
                    weightDTypeSize * this->doubleBufferValue.pbBL1;
    if (tilingIns_->hasBias) {
        uint64_t biasSize = currnL0 * biasDTypeSize * POINT_WISE_BIAS_CUBE_UNIT;
        usedL1Size += biasSize;
    }
    return usedL1Size;
}

uint64_t Conv3dTilingAlgorithmPointWise::L1NoFullLoadFmapSize() const
{
    uint64_t fmapSingleCoreL1Load =  AlignB(static_cast<uint64_t>(tilingIns_->shapeInfo.singleM), static_cast<uint64_t>(tilingIns_->cubeInfo.m0)) *
        AlignB(tilingIns_->shapeCalc.singleCi1 * static_cast<uint64_t>(tilingIns_->cubeInfo.k0), POINT_WISE_ALIGN_UNIT) *
        static_cast<uint64_t>(tilingIns_->shapeInfo.singlekD) * this->fMapDTypeSize;
    return fmapSingleCoreL1Load;
}

bool Conv3dTilingAlgorithmPointWise::CoreL1TilingMinWeightBypass() const
{
    return false;
}

bool Conv3dTilingAlgorithmPointWise::NoneKABL1FullLoadWeightBypass() const
{
    return false;
}

void Conv3dTilingAlgorithmPointWise::SetKAL1KBL1TailRes()
{
    this->l1TilingParams.kAL1 = this->l1TilingRange.kAL1Range.at(this->l1TilingIdx.kAL1Idx);
    uint64_t kAL1TailCheck = static_cast<uint64_t>(this->tilingIns_->shapeInfo.singleCi) % this->l1TilingParams.kAL1;
    this->l1TilingParams.kAL1Tail = kAL1TailCheck == 0 ? this->l1TilingParams.kAL1 : kAL1TailCheck;

    this->l1TilingParams.kBL1 = this->l1TilingRange.kBL1Range.at(this->l1TilingIdx.kBL1Idx);
    uint64_t kBL1TailCheck = static_cast<uint64_t>(this->tilingIns_->shapeInfo.singleCi) % this->l1TilingParams.kBL1;
    this->l1TilingParams.kBL1Tail = kBL1TailCheck == 0 ? this->l1TilingParams.kBL1 : kBL1TailCheck;
}

} // namespace Conv3dApiTiling