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
 * \file conv_api_tiling_algorithm_Mmode.cpp
 * \brief
 */

#include <cstdint>
#include "conv_api_tiling_algorithm_Mmode.h"

using namespace std;

namespace conv_tiling_algo_m {
// decide tiling
int64_t ConvTilingAlgorithmMmode::Process()
{
    InitPingPong();
    GetL0Tiling();
    if (GetL1Tiling() == -1) {
        return -1;
    }
    GetUbTiling();
    // set pb res
    SetPBufferRes();
    return 0;
}

bool ConvTilingAlgorithmMmode::CheckMinL1Tiling()
{
    if (tilingIns_->shapeInfo.orgWo == 0) {
        TILING_LOG_ERROR("zero div in CheckMinL1Tiling.");
        return false;
    }
    uint64_t minHoAL1 = min(tilingIns_->cubeInfo.m0 / tilingIns_->shapeInfo.orgWo + CONST_VALUE_2,
                            tilingIns_->shapeInfo.orgHo);
    uint64_t minHiAL1 = InferHiL1(minHoAL1, tilingIns_->shapeInfo.orgHi); // 2
    uint64_t minKAL1 = tilingIns_->isC04Flag ? C04_CIN_SIZE : tilingIns_->cubeInfo.k0;
    uint64_t minAL1Size = AlignB(tilingIns_->innerBatch * minHiAL1 * tilingIns_->shapeInfo.orgWi * minKAL1 * this->fMapDTypeSize, C0_SIZE);
    uint64_t minKBL1 = tilingIns_->isC04Flag ? AlignB(C04_CIN_SIZE * tilingIns_->shapeInfo.singlekH *
        tilingIns_->shapeInfo.singlekW, tilingIns_->cubeInfo.k0) :
        tilingIns_->shapeInfo.singlekH * tilingIns_->shapeInfo.singlekW * tilingIns_->cubeInfo.k0;
 
    uint64_t minBL1Size = AlignB(minKBL1 * tilingIns_->cubeInfo.n0 * this->weightDTypeSize, C0_SIZE);
    uint64_t minBiasSize = tilingIns_->hasBias ? AlignB(tilingIns_->cubeInfo.n0 * this->biasDTypeSize, C0_SIZE) : 0;
    uint64_t curUsedL1Size = minAL1Size + minBiasSize + minBL1Size;
 
    if (tilingIns_->hasScale) {
        // cal channelwise fix params according to channelWiseCoefficient
        float minFixpipeParamsL1Size = tilingIns_->shapeInfo.channelWiseCoeff *
            static_cast<float>(tilingIns_->cubeInfo.n0 * FP16_DTYPE_SIZE);
        curUsedL1Size += static_cast<uint64_t>(minFixpipeParamsL1Size);
    }

    if (curUsedL1Size > tilingIns_->platformInfo.l1Size) {
        TILING_LOG_ERROR("when w fullyload, minL1LoadSize > L1size.");
        return false;
    }
    return true;
}

int64_t ConvTilingAlgorithmMmode::GetL1Tiling()
{
    if (!CheckMinL1Tiling()) {
        return -1;
    }
    if (PreProcessingL1Tiling() == -1) {
        return -1;
    }
    // tiling decision
    CoreL1TilingDecision();
    // bias load check
    BiasL1TilingDecision();
    // get kl0 tiling decision
    GetKL0TilingDecision();
    // update l1 pingpong decision
    UpdateL1DoubelBuffer();

    SetL1TilingRes();
    CheckL0CDoubleBuffer();
    SetL0TilingRes();
    return 0;
}

int64_t ConvTilingAlgorithmMmode::PreProcessingL1Tiling()
{
    // cal some common variables
    if (InitCalcL1Params() == -1) {
        return -1;
    }
    // get kL1，mL1 and nL1 range value
    GetL1TilingRange();
    // cal full load condition and init range start idx
    InitL1TiLing();
    return 0;
}

bool ConvTilingAlgorithmMmode::CheckL1Buffer()
{
    // check if l1 buffer is overflow in current tiling decision
    uint64_t hoL1Load = min((this->l1TilingRange.mAL1ValueRange.at(this->l1TilingIdx.mAL1Idx) /
        tilingIns_->shapeInfo.orgWo) + 2, static_cast<uint64_t>(tilingIns_->shapeInfo.orgHo));
    uint64_t hiL1Load = InferHiL1(hoL1Load, tilingIns_->shapeInfo.orgHi);
    uint64_t tmpKAL1 = this->l1TilingRange.kAL1Range.at(this->l1TilingIdx.kAL1Idx) /
        (tilingIns_->shapeInfo.singlekH * tilingIns_->shapeInfo.singlekW);
    uint64_t currentFmL1Size = tmpKAL1 * hiL1Load * tilingIns_->shapeInfo.orgWi * this->fMapDTypeSize *
        this->dbValue.pbAL1 * tilingIns_->innerBatch;
    uint64_t currentWeightL1Size =
        this->l1TilingRange.kBL1Range.at(this->l1TilingIdx.kBL1Idx) * this->dbValue.pbBL1 *
        this->l1TilingRange.nBL1ValueRange.at(this->l1TilingIdx.nBL1Idx) * this->weightDTypeSize;
    uint64_t currentBiasL1Size = tilingIns_->hasBias ?
        (this->l1TilingFlag.isBiasFullLoad ? tilingIns_->shapeInfo.singleCo1 * tilingIns_->cubeInfo.n0 *
        this->biasDTypeSize : this->l0TilingParams.nL0 * this->biasDTypeSize) : 0;
    uint64_t currentFixpParamsSize = this->l1TilingFlag.isFixpFullLoad ? tilingIns_->shapeInfo.singleCo1 *
        tilingIns_->cubeInfo.n0 * tilingIns_->shapeInfo.channelWiseCoeff * FP16_DTYPE_SIZE :
        this->l0TilingParams.nL0 * tilingIns_->shapeInfo.channelWiseCoeff * FP16_DTYPE_SIZE;

    // cal current fm in L1
    if (this->l1TilingFlag.abL1Mode == L1TilingMode::ALL_FULL_LOAD) {
        currentFmL1Size = this->l1TilingCalc.fmapFullLoadL1Size;
        currentWeightL1Size = this->l1TilingCalc.weightFullLoadL1Size;
    } else if (this->l1TilingFlag.abL1Mode == L1TilingMode::FULL_LOAD_AL1) {
        currentFmL1Size = this->l1TilingCalc.fmapFullLoadL1Size;
    } else if (this->l1TilingFlag.abL1Mode == L1TilingMode::FULL_LOAD_BL1) {
        currentWeightL1Size = this->l1TilingCalc.weightFullLoadL1Size;
    }

    uint64_t l1SizeCur = currentFmL1Size + currentWeightL1Size + currentBiasL1Size + currentFixpParamsSize;

    return l1SizeCur <= tilingIns_->platformInfo.l1Size;
}

int64_t ConvTilingAlgorithmMmode::InitCalcL1Params()
{
    this->l1TilingCalc.ci0HkWk = tilingIns_->shapeInfo.singlekH * tilingIns_->shapeInfo.singlekW *
        tilingIns_->cubeInfo.k0;
    // in c04 mode, ci0 = 4, ci1 = 1, kAL1 and kBL1 all fullLoad, no matter in minL1Size or maxL1Size calculation
    this->l1TilingCalc.c04KSizeAlign = AlignB(C04_CIN_SIZE * tilingIns_->shapeInfo.singlekH *
                                              tilingIns_->shapeInfo.singlekW, tilingIns_->cubeInfo.k0);
    this->l1TilingCalc.kBL1FullLoadSize = tilingIns_->isC04Flag ? this->l1TilingCalc.c04KSizeAlign :
        tilingIns_->shapeInfo.singleCi1 * this->l1TilingCalc.ci0HkWk;
    this->l1TilingCalc.kAL1FullLoadSize = tilingIns_->isC04Flag ? C04_CIN_SIZE :
        tilingIns_->shapeInfo.singleCi1 * tilingIns_->cubeInfo.k0;
    // cal fmap weight full load in L1 size
    uint64_t hoL1FullLoad = min((tilingIns_->shapeInfo.singleM / tilingIns_->shapeInfo.orgWo) + 2,
        tilingIns_->shapeInfo.orgHo);
    uint64_t hiL1FullLoad = InferHiL1(hoL1FullLoad, tilingIns_->shapeInfo.orgHi);
    if ((tilingIns_->shapeInfo.singlekD * tilingIns_->shapeInfo.singleCi1 * hiL1FullLoad *
        tilingIns_->shapeInfo.orgWi * tilingIns_->cubeInfo.k0 * this->fMapDTypeSize) / this->fMapDTypeSize !=
        (tilingIns_->shapeInfo.singlekD * tilingIns_->shapeInfo.singleCi1 * hiL1FullLoad *
        tilingIns_->shapeInfo.orgWi * tilingIns_->cubeInfo.k0)) {
        TILING_LOG_ERROR("fmap size in l1 is overflow uint64, initcalc l1 params failed!");
        return -1;
    }
    this->l1TilingCalc.fmapFullLoadL1Size = tilingIns_->shapeInfo.singlekD * this->l1TilingCalc.kAL1FullLoadSize *
        hiL1FullLoad * tilingIns_->shapeInfo.orgWi * this->fMapDTypeSize * tilingIns_->innerBatch;

    if ((tilingIns_->shapeInfo.singlekD * tilingIns_->shapeInfo.singleCi1 * this->l1TilingCalc.ci0HkWk *
        tilingIns_->shapeInfo.singleCo1 * tilingIns_->cubeInfo.n0 * this->weightDTypeSize) / weightDTypeSize !=
        (tilingIns_->shapeInfo.singlekD * tilingIns_->shapeInfo.singleCi1 * this->l1TilingCalc.ci0HkWk *
        tilingIns_->shapeInfo.singleCo1 * tilingIns_->cubeInfo.n0)) {
        TILING_LOG_ERROR("weight size in l1 is overflow uint64, initcalc l1 params failed!");
        return -1;
    }
    this->l1TilingCalc.weightFullLoadL1Size = tilingIns_->shapeInfo.singlekD * this->l1TilingCalc.kBL1FullLoadSize *
        tilingIns_->shapeInfo.singleCo1 * tilingIns_->cubeInfo.n0 * this->weightDTypeSize;

    // cal min/kfullload fmap size in L1(mL0)
    uint64_t hoL1MinLoad = min((this->l0TilingParams.mL0 / tilingIns_->shapeInfo.orgWo) + 2,
        static_cast<uint64_t>(tilingIns_->shapeInfo.orgHo));
    uint64_t hiL1MinLoad = InferHiL1(hoL1MinLoad, tilingIns_->shapeInfo.orgHi);
    uint64_t kAL1MinLoad = tilingIns_->isC04Flag ? this->l1TilingCalc.kAL1FullLoadSize : tilingIns_->cubeInfo.k0;
    uint64_t kBL1MinLoad = tilingIns_->isC04Flag ? this->l1TilingCalc.kBL1FullLoadSize : this->l1TilingCalc.ci0HkWk;
    this->l1TilingCalc.fmapMinLoadL1Size = kAL1MinLoad * hiL1MinLoad * tilingIns_->shapeInfo.orgWi *
        this->fMapDTypeSize * this->dbValue.pbAL1 * tilingIns_->innerBatch;
    this->l1TilingCalc.fmapKL1FullLoadSize = static_cast<uint64_t>(tilingIns_->shapeInfo.singlekD) *
        this->l1TilingCalc.kAL1FullLoadSize * hiL1MinLoad * static_cast<uint64_t>(tilingIns_->shapeInfo.orgWi) *
        static_cast<uint64_t>(this->dbValue.pbAL1) * this->fMapDTypeSize * static_cast<uint64_t>(tilingIns_->innerBatch);
    // cal min/kfullload weiht size in L1
    this->l1TilingCalc.weightMinLoadL1Size = max(kBL1MinLoad * this->l0TilingParams.nL0 *
        this->dbValue.pbBL1 * this->weightDTypeSize, tilingIns_->platformInfo.l0BSize);
    this->l1TilingCalc.weightKL1FullLoadSize = tilingIns_->shapeInfo.singlekD * this->l1TilingCalc.kBL1FullLoadSize *
        this->l0TilingParams.nL0 * this->dbValue.pbBL1 * this->weightDTypeSize;
    // cal bias size in L1
    this->l1TilingCalc.biasMinLoadL1Size = tilingIns_->hasBias ? this->l0TilingParams.nL0 * this->biasDTypeSize : 0;
    this->l1TilingCalc.fixpMinLoadL1Size =
        this->l0TilingParams.nL0 * tilingIns_->shapeInfo.channelWiseCoeff * FP16_DTYPE_SIZE;
    return 0;
}

void ConvTilingAlgorithmMmode::GetL1TilingRange()
{
    // cal kAL1 and kBL1 range, cin1 factor or (1~dk) cin1
    CalcCommFactor(tilingIns_->shapeInfo.singleCi1, tilingIns_->shapeInfo.singleCi1, this->l1TilingRange.kAL1Range);
    CalcCommFactor(tilingIns_->shapeInfo.singleCi1, tilingIns_->shapeInfo.singleCi1, this->l1TilingRange.kBL1Range);
    // singlekD > 0 is ensured in base.cpp
    for (uint64_t dk = 2; dk <= static_cast<uint64_t>(tilingIns_->shapeInfo.singlekD); ++dk) {
        this->l1TilingRange.kAL1Range.emplace_back(dk * tilingIns_->shapeInfo.singleCi1);
        this->l1TilingRange.kBL1Range.emplace_back(dk * tilingIns_->shapeInfo.singleCi1);
    }

    // kAL1 has limit of postk due to load3d instr
    const uint64_t limitKAL1 = (POSTK_LIMIT + tilingIns_->cubeInfo.k0) / this->l1TilingCalc.ci0HkWk;
    std::vector<uint64_t>::iterator up = std::upper_bound(this->l1TilingRange.kAL1Range.begin(),
                                                          this->l1TilingRange.kAL1Range.end(), limitKAL1);
    this->l1TilingRange.kAL1Range.erase(up, this->l1TilingRange.kAL1Range.end());
    this->l1TilingRange.kAL1Range.shrink_to_fit();

    VectorElementMultip(this->l1TilingRange.kAL1Range, this->l1TilingCalc.ci0HkWk);
    VectorElementMultip(this->l1TilingRange.kBL1Range, this->l1TilingCalc.ci0HkWk);

    if (tilingIns_->isC04Flag) {
        // c04 mode require kAL1 = kBL1 all fullLoad
        this->l1TilingRange.kAL1Range.assign(1, this->l1TilingCalc.c04KSizeAlign);
        this->l1TilingRange.kBL1Range.assign(1, this->l1TilingCalc.c04KSizeAlign);
    }

    auto effectiveCount = count_if(this->l1TilingRange.kAL1Range.begin(), this->l1TilingRange.kAL1Range.end(),
                                   [](uint64_t x) {
                                       return x <= MAX_16_BIT_NUM;
                                   });
    this->l1TilingRange.kAL1Range.resize(effectiveCount);

    // cal mAL1Value and nBL1Value
    uint64_t multiNBL1Max = CeilDiv(tilingIns_->shapeInfo.singleCo1 * tilingIns_->cubeInfo.n0,
                                    this->l0TilingParams.nL0);
    CalcCommFactor(multiNBL1Max, multiNBL1Max, this->l1TilingRange.nBL1ValueRange);
    VectorElementMultip(this->l1TilingRange.nBL1ValueRange, l0TilingParams.nL0);
    uint64_t multiMAL1Max = CeilDiv(CeilDiv(tilingIns_->shapeInfo.singleM,
                                            tilingIns_->cubeInfo.m0) * tilingIns_->cubeInfo.m0,
                                    this->l0TilingParams.mL0);
    CalcCommFactor(multiMAL1Max, multiMAL1Max, this->l1TilingRange.mAL1ValueRange);
    VectorElementMultip(this->l1TilingRange.mAL1ValueRange, l0TilingParams.mL0);
    // load3d m start position <= LOAD3D_M_START_POS_LIMIT
    if (tilingIns_->platformInfo.socVersion == platform_ascendc::SocVersion::ASCEND950 ||
        tilingIns_->platformInfo.socVersion == platform_ascendc::SocVersion::MC62CM12A) {
        auto mL1EffectiveCount = count_if(this->l1TilingRange.mAL1ValueRange.begin(),
            this->l1TilingRange.mAL1ValueRange.end(), [](uint64_t x) { return x <= LOAD3D_M_START_POS_LIMIT; });
        this->l1TilingRange.mAL1ValueRange.resize(mL1EffectiveCount);
    }
}

void ConvTilingAlgorithmMmode::InitL1TiLing()
{
    this->l1TilingInitMap[L1TilingMode::NONE_FULL_LOAD].SetIdx(0, 0, 0, 0);
    this->l1TilingInitMap[L1TilingMode::FULL_LOAD_AL1].SetIdx(this->l1TilingRange.kAL1Range.size() - 1, 0,
                                                              this->l1TilingRange.mAL1ValueRange.size() - 1, 0);
    this->l1TilingInitMap[L1TilingMode::FULL_LOAD_BL1].SetIdx(0, this->l1TilingRange.kBL1Range.size() - 1,
                                                              0, this->l1TilingRange.nBL1ValueRange.size() - 1);
    this->l1TilingInitMap[L1TilingMode::ALL_FULL_LOAD].SetIdx(this->l1TilingRange.kAL1Range.size() - 1,
                                                              this->l1TilingRange.kBL1Range.size() - 1,
                                                              this->l1TilingRange.mAL1ValueRange.size() - 1,
                                                              this->l1TilingRange.nBL1ValueRange.size() - 1);

    InitABL1TilingMode();
    this->l1TilingIdx = this->l1TilingInitMap.at(this->l1TilingFlag.abL1Mode);
}

void ConvTilingAlgorithmMmode::InitABL1TilingMode()
{
    auto fullLoadML1 = tilingIns_->cubeInfo.m0 * tilingIns_->shapeInfo.singleM1;
    bool mL1ExceedInstrLimit = fullLoadML1 > LOAD3D_M_START_POS_LIMIT;

    // Check if all full load is possible
    if (IsAllFullLoadPossible(mL1ExceedInstrLimit)) {
        this->l1TilingFlag.abL1Mode = L1TilingMode::ALL_FULL_LOAD;
        this->dbValue.pbAL1 = 1;
        if (tilingIns_->enableInnerBatch && CeilDiv(tilingIns_->shapeInfo.singleBatch, tilingIns_->innerBatch) > 1) {
            this->l1TilingFlag.abL1Mode = L1TilingMode::FULL_LOAD_BL1;
            this->dbValue.pbAL1 = CONST_VALUE_2;
        }
        return;
    }

    // weightFullLoadL1Size is bigger than fmapFullLoadL1Size
    if (IsWeightFullLoadDominant() && IsWeightFullLoadPossible()) {
        this->l1TilingFlag.abL1Mode = L1TilingMode::FULL_LOAD_BL1;
        return;
    } else if (IsWeightFullLoadDominant() && IsFmapFullLoadPossible(mL1ExceedInstrLimit)) {
        this->l1TilingFlag.abL1Mode = L1TilingMode::FULL_LOAD_AL1;
        this->dbValue.pbAL1 = 1;
        if(tilingIns_->enableInnerBatch && CeilDiv(tilingIns_->shapeInfo.singleBatch, tilingIns_->innerBatch) > 1) {
            this->l1TilingFlag.abL1Mode = L1TilingMode::NONE_FULL_LOAD;
            this->dbValue.pbAL1 = CONST_VALUE_2;
        }
        return;
    }

    // weightFullLoadL1Size is smaller than fmapFullLoadL1Size
    if (IsFmapFullLoadPossible(mL1ExceedInstrLimit)) {
        this->l1TilingFlag.abL1Mode = L1TilingMode::FULL_LOAD_AL1;
        this->dbValue.pbAL1 = 1;
        if (tilingIns_->enableInnerBatch && CeilDiv(tilingIns_->shapeInfo.singleBatch, tilingIns_->innerBatch) > 1) {
            if (IsWeightFullLoadPossible()) {
                this->l1TilingFlag.abL1Mode = L1TilingMode::FULL_LOAD_BL1;
                this->dbValue.pbAL1 = CONST_VALUE_2;
                return;
            }
            this->l1TilingFlag.abL1Mode = L1TilingMode::NONE_FULL_LOAD;
            this->dbValue.pbAL1 = CONST_VALUE_2;
        }
        return;
    } else if (IsWeightFullLoadPossible()) {
        this->l1TilingFlag.abL1Mode = L1TilingMode::FULL_LOAD_BL1;
        return;
    }
    // If none of the above conditions are met, set to NONE_FULL_LOAD
    this->l1TilingFlag.abL1Mode = L1TilingMode::NONE_FULL_LOAD;
    return;
}

bool ConvTilingAlgorithmMmode::IsAllFullLoadPossible(bool mL1ExceedInstrLimit)
{
    return (this->l1TilingCalc.fmapFullLoadL1Size + this->l1TilingCalc.weightFullLoadL1Size +
            this->l1TilingCalc.biasMinLoadL1Size + this->l1TilingCalc.fixpMinLoadL1Size <=
            tilingIns_->platformInfo.l1Size) && !mL1ExceedInstrLimit;
}

bool ConvTilingAlgorithmMmode::IsWeightFullLoadDominant()
{
    return (this->l1TilingCalc.fmapFullLoadL1Size <= this->l1TilingCalc.weightFullLoadL1Size *
           tilingIns_->platformInfo.abL1mte2BandWidthCof);
}

bool ConvTilingAlgorithmMmode::IsWeightFullLoadPossible()
{
    return (this->l1TilingCalc.weightFullLoadL1Size + this->l1TilingCalc.fmapMinLoadL1Size +
            this->l1TilingCalc.biasMinLoadL1Size + this->l1TilingCalc.fixpMinLoadL1Size <=
            tilingIns_->platformInfo.l1Size);
}

bool ConvTilingAlgorithmMmode::IsFmapFullLoadPossible(bool mL1ExceedInstrLimit)
{
    return (this->l1TilingCalc.weightMinLoadL1Size + this->l1TilingCalc.fmapFullLoadL1Size +
            this->l1TilingCalc.biasMinLoadL1Size + this->l1TilingCalc.fixpMinLoadL1Size <=
            tilingIns_->platformInfo.l1Size) && !mL1ExceedInstrLimit;
}

void ConvTilingAlgorithmMmode::FmapL1FullLoadIter()
{
    // iter kBL1
    while (this->l1TilingIdx.kBL1Idx < this->l1TilingRange.kBL1Range.size() && CheckL1Buffer()) {
        this->l1TilingIdx.kBL1Idx++;
    }
    this->l1TilingIdx.kBL1Idx = this->l1TilingIdx.kBL1Idx == 0 ? 0 : this->l1TilingIdx.kBL1Idx - 1;
    if (this->l1TilingIdx.kBL1Idx != this->l1TilingRange.kBL1Range.size() - 1) {
        return;
    }
    // iter nBL1
    while (this->l1TilingIdx.nBL1Idx < this->l1TilingRange.nBL1ValueRange.size() && CheckL1Buffer()) {
        this->l1TilingIdx.nBL1Idx++;
    }
    this->l1TilingIdx.nBL1Idx = this->l1TilingIdx.nBL1Idx == 0 ? 0 : this->l1TilingIdx.nBL1Idx - 1;
}

void ConvTilingAlgorithmMmode::CoreL1TilingDecision()
{
    // when fmap and weight can both full laod, it can set res and return directly
    uint64_t tmpKAL1max = tilingIns_->isC04Flag ? this->l1TilingCalc.c04KSizeAlign :
        tilingIns_->shapeInfo.singleCi1 * this->l1TilingCalc.ci0HkWk;
    if (this->l1TilingFlag.abL1Mode == L1TilingMode::ALL_FULL_LOAD) {
        this->l1TilingParams.kAL1 = tilingIns_->shapeInfo.singlekD * tmpKAL1max;
        this->l1TilingParams.kBL1 = tilingIns_->shapeInfo.singlekD * this->l1TilingCalc.kBL1FullLoadSize;
        this->l1TilingParams.mAL1Value = tilingIns_->cubeInfo.m0 * tilingIns_->shapeInfo.singleM1;
        this->l1TilingParams.nBL1Value = tilingIns_->cubeInfo.n0 * tilingIns_->shapeInfo.singleCo1;
        this->l1TilingFlag.iterateMNOrder = IterateMNOrder::ITER_M_FST;
        return;
    }
    // when only fmap full load in L1, nfirset and iter kbl1 then nbl1
    if (this->l1TilingFlag.abL1Mode == L1TilingMode::FULL_LOAD_AL1) {
        this->l1TilingParams.kAL1 = tilingIns_->shapeInfo.singlekD * tmpKAL1max;
        this->l1TilingParams.mAL1Value = tilingIns_->cubeInfo.m0 * tilingIns_->shapeInfo.singleM1;
        this->l1TilingFlag.iterateMNOrder = IterateMNOrder::ITER_N_FST;
        // normal case
        FmapL1FullLoadIter();
        return;
    }
    // when only weight full load in L1, mfirset and iter kal1 then mal1
    if (this->l1TilingFlag.abL1Mode == L1TilingMode::FULL_LOAD_BL1) {
        this->l1TilingParams.kBL1 = tilingIns_->shapeInfo.singlekD * this->l1TilingCalc.kBL1FullLoadSize;
        this->l1TilingParams.nBL1Value = tilingIns_->cubeInfo.n0 * tilingIns_->shapeInfo.singleCo1;
        this->l1TilingFlag.iterateMNOrder = IterateMNOrder::ITER_M_FST;
        // normal case
        WeightL1FullLoadIter();
        return;
    }
    // both cannot full load, cal if K can full load in L1 and decision
    InitKL1LoadFlag();
    L1NoFullLoadIter();
    return;
}

void ConvTilingAlgorithmMmode::WeightL1FullLoadIter()
{
    // iter kAL1
    while (this->l1TilingIdx.kAL1Idx < this->l1TilingRange.kAL1Range.size() && CheckL1Buffer()) {
        this->l1TilingIdx.kAL1Idx++;
    }
    this->l1TilingIdx.kAL1Idx = this->l1TilingIdx.kAL1Idx == 0 ? 0 : this->l1TilingIdx.kAL1Idx - 1;
    if (this->l1TilingIdx.kAL1Idx != this->l1TilingRange.kAL1Range.size() - 1) {
        return;
    }
    // iter mAL1
    while (this->l1TilingIdx.mAL1Idx < this->l1TilingRange.mAL1ValueRange.size() && CheckL1Buffer()) {
        this->l1TilingIdx.mAL1Idx++;
    }
    this->l1TilingIdx.mAL1Idx = this->l1TilingIdx.mAL1Idx == 0 ? 0 : this->l1TilingIdx.mAL1Idx - 1;
}

void ConvTilingAlgorithmMmode::InitKL1LoadFlag()
{
    // check if KAL1, KBL1 can both/either/none can full load in L1
    if (this->l1TilingCalc.fmapKL1FullLoadSize + this->l1TilingCalc.weightKL1FullLoadSize +
        this->l1TilingCalc.biasMinLoadL1Size + this->l1TilingCalc.fixpMinLoadL1Size <=
        tilingIns_->platformInfo.l1Size) {
        this->l1TilingFlag.kAL1CanFullLoadFlag = true;
        this->l1TilingFlag.kBL1CanFullLoadFlag = true;
        this->l1TilingFlag.kABL1CanFullLoadFlag = true;
        return;
    }
    if (this->l1TilingCalc.fmapKL1FullLoadSize + this->l1TilingCalc.weightMinLoadL1Size +
        this->l1TilingCalc.biasMinLoadL1Size + this->l1TilingCalc.fixpMinLoadL1Size <=
        tilingIns_->platformInfo.l1Size) {
        this->l1TilingFlag.kAL1CanFullLoadFlag = true;
    }
    if (this->l1TilingCalc.fmapMinLoadL1Size + this->l1TilingCalc.weightKL1FullLoadSize +
        this->l1TilingCalc.biasMinLoadL1Size + this->l1TilingCalc.fixpMinLoadL1Size <=
        tilingIns_->platformInfo.l1Size) {
        this->l1TilingFlag.kBL1CanFullLoadFlag = true;
    }
}

void ConvTilingAlgorithmMmode::KAL1FullLoadIter()
{
    this->l1TilingFlag.iterateMNOrder = IterateMNOrder::ITER_N_FST;
    // iter kBL1
    while (this->l1TilingIdx.kBL1Idx < this->l1TilingRange.kBL1Range.size() && CheckL1Buffer()) {
        this->l1TilingIdx.kBL1Idx++;
    }
    this->l1TilingIdx.kBL1Idx = this->l1TilingIdx.kBL1Idx == 0 ? 0 : this->l1TilingIdx.kBL1Idx - 1;
    // iter mAL1
    while (this->l1TilingIdx.mAL1Idx < this->l1TilingRange.mAL1ValueRange.size() && CheckL1Buffer()) {
        this->l1TilingIdx.mAL1Idx++;
    }
    this->l1TilingIdx.mAL1Idx = this->l1TilingIdx.mAL1Idx == 0 ? 0 : this->l1TilingIdx.mAL1Idx - 1;
}

void ConvTilingAlgorithmMmode::KBL1FullLoadIter()
{
    this->l1TilingFlag.iterateMNOrder = IterateMNOrder::ITER_M_FST;
    // iter kAL1
    while (this->l1TilingIdx.kAL1Idx < this->l1TilingRange.kAL1Range.size() && CheckL1Buffer()) {
        this->l1TilingIdx.kAL1Idx++;
    }
    this->l1TilingIdx.kAL1Idx = this->l1TilingIdx.kAL1Idx == 0 ? 0 : this->l1TilingIdx.kAL1Idx - 1;
    // iter nBL1
    while (this->l1TilingIdx.nBL1Idx < this->l1TilingRange.nBL1ValueRange.size() && CheckL1Buffer()) {
        this->l1TilingIdx.nBL1Idx++;
    }
    this->l1TilingIdx.nBL1Idx = this->l1TilingIdx.nBL1Idx == 0 ? 0 : this->l1TilingIdx.nBL1Idx - 1;
}

uint64_t ConvTilingAlgorithmMmode::KABL1FullLoadIterN()
{
    // iter mAL1
    uint64_t startIdxBk = this->l1TilingIdx.mAL1Idx;
    uint64_t tmpMAL1Idx = 0;
    while (this->l1TilingIdx.mAL1Idx < this->l1TilingRange.mAL1ValueRange.size() && CheckL1Buffer()) {
        this->l1TilingIdx.mAL1Idx++;
    }
    this->l1TilingIdx.mAL1Idx = this->l1TilingIdx.mAL1Idx == 0 ? 0 : this->l1TilingIdx.mAL1Idx - 1;
    tmpMAL1Idx = this->l1TilingIdx.mAL1Idx;
    this->l1TilingIdx.mAL1Idx = startIdxBk;
    return tmpMAL1Idx;
}

uint64_t ConvTilingAlgorithmMmode::KABL1FullLoadIterM()
{
    // iter nBL1
    uint64_t startIdxBk = this->l1TilingIdx.nBL1Idx;
    uint64_t tmpNBL1Idx = 0;
    while (this->l1TilingIdx.nBL1Idx < this->l1TilingRange.nBL1ValueRange.size() && CheckL1Buffer()) {
        this->l1TilingIdx.nBL1Idx++;
    }
    this->l1TilingIdx.nBL1Idx = this->l1TilingIdx.nBL1Idx == 0 ? 0 : this->l1TilingIdx.nBL1Idx - 1;
    tmpNBL1Idx = this->l1TilingIdx.nBL1Idx;
    this->l1TilingIdx.nBL1Idx = startIdxBk;
    return tmpNBL1Idx;
}

void ConvTilingAlgorithmMmode::NoneKABL1FullLoadIter()
{
    // iter kAL1 and kBL1 together
    while (this->l1TilingIdx.kAL1Idx < this->l1TilingRange.kAL1Range.size() && CheckL1Buffer()) {
        this->l1TilingIdx.kAL1Idx++;
        this->l1TilingIdx.kBL1Idx++;
    }
    this->l1TilingFlag.iterateMNOrder = IterateMNOrder::ITER_M_FST;
    this->l1TilingIdx.kAL1Idx = this->l1TilingIdx.kAL1Idx == 0 ? 0 : this->l1TilingIdx.kAL1Idx - 1;
    this->l1TilingIdx.kBL1Idx = this->l1TilingIdx.kBL1Idx == 0 ? 0 : this->l1TilingIdx.kBL1Idx - 1;
}

bool ConvTilingAlgorithmMmode::IsKAL1orKBL1FullLoad()
{
    uint64_t hoL1SingleCore = min((tilingIns_->shapeInfo.singleM / tilingIns_->shapeInfo.orgWo) + 2,
        tilingIns_->shapeInfo.orgHo);
    uint64_t hiL1SingleCore = InferHiL1(hoL1SingleCore, tilingIns_->shapeInfo.orgHi);
    uint64_t fmapSingleCoreL1Load = this->l1TilingCalc.kAL1FullLoadSize * hiL1SingleCore *
        tilingIns_->shapeInfo.orgWi * tilingIns_->shapeInfo.singlekD;
    uint64_t weightSingleCoreL1Load = this->l1TilingCalc.weightKL1FullLoadSize;

    if ((this->l1TilingFlag.kBL1CanFullLoadFlag || this->l1TilingFlag.kAL1CanFullLoadFlag) &&
        !this->l1TilingFlag.kABL1CanFullLoadFlag) {
        // kAL1 full load
        // fmap_mte2_size: dout*fmapSingleCoreL1Load
        // weight_mte2_size: dout*weightSingleCoreL1Load*(loop m) * bwCof
        uint64_t onlyKAL1FullloadMte2Size = fmapSingleCoreL1Load * tilingIns_->shapeInfo.singleDo +
            weightSingleCoreL1Load * tilingIns_->shapeInfo.singleDo * tilingIns_->platformInfo.abL1mte2BandWidthCof *
            CeilDiv(tilingIns_->shapeInfo.singleM, this->l0TilingParams.mL0);
        // kBL1 full load
        // fmap_mte2_size: dout*fmapSingleCoreL1Load*(loop n)
        // weight_mte2_size: weightSingleCoreL1Load * bwCof
        uint64_t onlyKBL1FullloadMte2Size = fmapSingleCoreL1Load * tilingIns_->shapeInfo.singleDo *
            CeilDiv(tilingIns_->shapeInfo.singleCo, this->l0TilingParams.nL0) + weightSingleCoreL1Load *
            tilingIns_->platformInfo.abL1mte2BandWidthCof;
        if (onlyKAL1FullloadMte2Size < onlyKBL1FullloadMte2Size) {
            this->l1TilingIdx.kAL1Idx = this->l1TilingRange.kAL1Range.size() - 1;
            KAL1FullLoadIter();
        } else {
            this->l1TilingIdx.kBL1Idx = this->l1TilingRange.kBL1Range.size() - 1;
            KBL1FullLoadIter();
        }
        return true;
    }
    return false;
}

bool ConvTilingAlgorithmMmode::IsKAL1andKBL1FullLoad()
{
    uint64_t hoL1SingleCore = min((tilingIns_->shapeInfo.singleM / tilingIns_->shapeInfo.orgWo) + 2,
        tilingIns_->shapeInfo.orgHo);
    uint64_t hiL1SingleCore = InferHiL1(hoL1SingleCore, tilingIns_->shapeInfo.orgHi);
    uint64_t fmapSingleCoreL1Load = this->l1TilingCalc.kAL1FullLoadSize * hiL1SingleCore *
        tilingIns_->shapeInfo.orgWi * tilingIns_->shapeInfo.singlekD;
    uint64_t weightSingleCoreL1Load = this->l1TilingCalc.weightKL1FullLoadSize;

    if (this->l1TilingFlag.kABL1CanFullLoadFlag) {
        this->l1TilingIdx.kAL1Idx = this->l1TilingRange.kAL1Range.size() - 1;
        this->l1TilingIdx.kBL1Idx = this->l1TilingRange.kBL1Range.size() - 1;
        uint64_t tmpMAL1Idx = KABL1FullLoadIterN();
        uint64_t tmpNBL1Idx = KABL1FullLoadIterM();
        // iterN
        // fmap_mte2_size: dout*fmapSingleCoreL1Load
        // weight_mte2_size: dout*weightSingleCoreL1Load*(loop m) * bwCof
        uint64_t bothFullloadIterNMte2Size = fmapSingleCoreL1Load * tilingIns_->shapeInfo.singleDo +
            weightSingleCoreL1Load * tilingIns_->shapeInfo.singleDo * tilingIns_->platformInfo.abL1mte2BandWidthCof *
            CeilDiv(tilingIns_->shapeInfo.singleM, this->l1TilingRange.mAL1ValueRange.at(tmpMAL1Idx));
        // iterM
        // fmap_mte2_size: dout*fmapSingleCoreL1Load*(loop n)
        // weight_mte2_size: weightSingleCoreL1Load * bwCof
        uint64_t bothFullloadIterMMte2Size = fmapSingleCoreL1Load * tilingIns_->shapeInfo.singleDo *
            CeilDiv(tilingIns_->shapeInfo.singleCo, this->l1TilingRange.nBL1ValueRange.at(tmpNBL1Idx)) +
            weightSingleCoreL1Load * tilingIns_->platformInfo.abL1mte2BandWidthCof;
        if (bothFullloadIterNMte2Size < bothFullloadIterMMte2Size) {
            this->l1TilingFlag.iterateMNOrder = IterateMNOrder::ITER_N_FST;
            this->l1TilingIdx.mAL1Idx = tmpMAL1Idx;
        } else {
            this->l1TilingFlag.iterateMNOrder = IterateMNOrder::ITER_M_FST;
            this->l1TilingIdx.nBL1Idx = tmpNBL1Idx;
        }
        return true;
    }
    return false;
}

void ConvTilingAlgorithmMmode::L1NoFullLoadIter()
{
    // only kAL1 full load L1
    if (this->l1TilingFlag.kAL1CanFullLoadFlag && !this->l1TilingFlag.kBL1CanFullLoadFlag) {
        this->l1TilingIdx.kAL1Idx = this->l1TilingRange.kAL1Range.size() - 1;
        KAL1FullLoadIter();
        return;
    }
    // only kBL1 full load L1
    if (this->l1TilingFlag.kBL1CanFullLoadFlag && !this->l1TilingFlag.kAL1CanFullLoadFlag) {
        this->l1TilingIdx.kBL1Idx = this->l1TilingRange.kBL1Range.size() - 1;
        KBL1FullLoadIter();
        return;
    }
    // either kAL1 or kBL1 can full load
    if (IsKAL1orKBL1FullLoad()) {
        return;
    }

    // both kAL1 and kBL1 can full load
    if (IsKAL1andKBL1FullLoad()) {
        return;
    }
    // None of kAL1 and kBL1 can full load in L1
    NoneKABL1FullLoadIter();
    return;
}

void ConvTilingAlgorithmMmode::BiasL1TilingDecision()
{
    // decide bias and fixpParams in L1
    if (!tilingIns_->hasBias && !tilingIns_->hasScale) {
        return;
    }

    if (this->l1TilingFlag.isBiasFullLoad && this->l1TilingFlag.isFixpFullLoad) {
        return;
    }

    // decide if bias can fullload in L1, than decide if fixpipe Params can full load in L1
    bool isSupportSoc = tilingIns_->platformInfo.socVersion == platform_ascendc::SocVersion::ASCEND950 ||
                        tilingIns_->platformInfo.socVersion == platform_ascendc::SocVersion::ASCEND910_55 ||
                        tilingIns_->platformInfo.socVersion == platform_ascendc::SocVersion::MC62CM12A;
    if (!this->l1TilingFlag.isBiasFullLoad) {
        this->l1TilingFlag.isBiasFullLoad = true;
        bool exceedDataCopyLimits = isSupportSoc && tilingIns_->shapeInfo.singleCo1 * tilingIns_->cubeInfo.n0 *
            this->biasDTypeSize > DATACOPYPARAMS_BURSTLEN_MAX;
        if (!CheckL1Buffer() || exceedDataCopyLimits) {
            this->l1TilingFlag.isBiasFullLoad = false;
        }
    }
    if (!this->l1TilingFlag.isFixpFullLoad) {
        this->l1TilingFlag.isFixpFullLoad = true;
        bool exceedDataCopyLimits = isSupportSoc && tilingIns_->shapeInfo.singleCo1 * tilingIns_->cubeInfo.n0 *
            this->scaleDtypeSize > DATACOPYPARAMS_BURSTLEN_MAX;
        if (!CheckL1Buffer() || exceedDataCopyLimits) {
            this->l1TilingFlag.isFixpFullLoad = false;
        }
    }

    return;
}

void ConvTilingAlgorithmMmode::SetL0TilingRes()
{
    tilingIns_->l0TilingInfo.kL0 = this->l0TilingParams.kL0;
    tilingIns_->l0TilingInfo.mL0 = l0TilingParams.mL0;
    tilingIns_->l0TilingInfo.nL0 = l0TilingParams.nL0;
}

void ConvTilingAlgorithmMmode::GetKL0TilingDecision()
{
    // get k0 range according kal1 and kbl1
    uint64_t maxKAL12L0Loop = CeilDiv(this->l1TilingRange.kAL1Range.at(this->l1TilingIdx.kAL1Idx),
                                      tilingIns_->cubeInfo.k0);
    uint64_t maxKBL12L0Loop = CeilDiv(this->l1TilingRange.kBL1Range.at(this->l1TilingIdx.kBL1Idx),
                                      tilingIns_->cubeInfo.k0);
    uint64_t factorK = Gcd(maxKAL12L0Loop, maxKBL12L0Loop);
    CalcCommFactor(factorK, factorK, this->l0TilingRange.kL0Range);
    VectorElementMultip(this->l0TilingRange.kL0Range, tilingIns_->cubeInfo.k0);
    // kL0 decision
    while (this->l0TilingIdx.kL0Idx < this->l0TilingRange.kL0Range.size() &&
            CheckL0Buffer(this->l0TilingParams.mL0, this->l0TilingRange.kL0Range.at(this->l0TilingIdx.kL0Idx),
                          this->l0TilingParams.nL0)) {
        this->l0TilingIdx.kL0Idx++;
    }
    this->l0TilingIdx.kL0Idx = this->l0TilingIdx.kL0Idx == 0 ? 0 : this->l0TilingIdx.kL0Idx - 1;
    this->l0TilingParams.kL0 = this->l0TilingRange.kL0Range.at(this->l0TilingIdx.kL0Idx);

    return;
}

void ConvTilingAlgorithmMmode::UpdateL1DoubelBuffer()
{
    if (this->l1TilingFlag.abL1Mode == L1TilingMode::ALL_FULL_LOAD ||
        this->l1TilingFlag.abL1Mode == L1TilingMode::FULL_LOAD_AL1) {
        this->dbValue.pbAL1 = 1;
    }
    if (this->l1TilingFlag.abL1Mode != L1TilingMode::FULL_LOAD_BL1 &&
        (this->l1TilingFlag.abL1Mode != L1TilingMode::ALL_FULL_LOAD)) {
        this->dbValue.pbBL1 = DOUBLE_BUFFER_NUM;
        if (!CheckL1Buffer()) {
            this->dbValue.pbBL1 = 1;
        }
    }
    if (this->l1TilingFlag.abL1Mode == L1TilingMode::ALL_FULL_LOAD ||
        this->l1TilingFlag.abL1Mode == L1TilingMode::FULL_LOAD_BL1) {
        this->dbValue.pbBL1 = 1;
    }
    return;
}

void ConvTilingAlgorithmMmode::SetL1TilingRes()
{
    this->l1TilingParams.kAL1 = this->l1TilingRange.kAL1Range.at(this->l1TilingIdx.kAL1Idx);
    this->l1TilingParams.kBL1 = this->l1TilingRange.kBL1Range.at(this->l1TilingIdx.kBL1Idx);
    this->l1TilingParams.nBL1Value = this->l1TilingRange.nBL1ValueRange.at(this->l1TilingIdx.nBL1Idx);
    this->l1TilingParams.mAL1Value = this->l1TilingRange.mAL1ValueRange.at(this->l1TilingIdx.mAL1Idx);
    tilingIns_->l1TilingInfo.al1FullLoad = false;
    tilingIns_->l1TilingInfo.bl1FullLoad = false;
    if (this->l1TilingFlag.abL1Mode == L1TilingMode::FULL_LOAD_BL1) {
        this->l1TilingParams.nBL1Value = tilingIns_->shapeInfo.singleCo1 * tilingIns_->cubeInfo.n0;
        tilingIns_->l1TilingInfo.bl1FullLoad = true;
    } else if (this->l1TilingFlag.abL1Mode == L1TilingMode::FULL_LOAD_AL1) {
        this->l1TilingParams.mAL1Value = CeilDiv(tilingIns_->shapeInfo.singleM,
                                                 tilingIns_->cubeInfo.m0) * tilingIns_->cubeInfo.m0;
        tilingIns_->l1TilingInfo.al1FullLoad = true;
    } else if (this->l1TilingFlag.abL1Mode == L1TilingMode::ALL_FULL_LOAD) {
        this->l1TilingParams.nBL1Value = tilingIns_->shapeInfo.singleCo1 * tilingIns_->cubeInfo.n0;
        this->l1TilingParams.mAL1Value = CeilDiv(tilingIns_->shapeInfo.singleM,
                                                 tilingIns_->cubeInfo.m0) * tilingIns_->cubeInfo.m0;
        tilingIns_->l1TilingInfo.al1FullLoad = true;
        tilingIns_->l1TilingInfo.bl1FullLoad = true;
    }

    tilingIns_->l1TilingInfo.kAL1 = this->l1TilingParams.kAL1;
    tilingIns_->l1TilingInfo.kBL1 = this->l1TilingParams.kBL1;
    tilingIns_->l1TilingInfo.mAL1 = this->l1TilingParams.mAL1Value;
    tilingIns_->l1TilingInfo.nBL1 = this->l1TilingParams.nBL1Value;
    tilingIns_->l1TilingInfo.iterateMNOrder = this->l1TilingFlag.iterateMNOrder;
    tilingIns_->l1TilingInfo.biasFullLoadFlag = this->l1TilingFlag.isBiasFullLoad;
    tilingIns_->l1TilingInfo.fixpParamsFullLoadFlag = this->l1TilingFlag.isFixpFullLoad;
}

uint64_t ConvTilingAlgorithmMmode::CalcL1SizeForL0Tiling(uint64_t currmL0, uint64_t currnL0, uint64_t reserveWeightSize)
{
    // Calculate AL1 size
    uint64_t kAL1Min = tilingIns_->isC04Flag ? C04_CIN_SIZE : tilingIns_->cubeInfo.k0;
    uint64_t kBL1Min = tilingIns_->isC04Flag ? AlignB(C04_CIN_SIZE * tilingIns_->shapeInfo.singlekH *
        tilingIns_->shapeInfo.singlekW, tilingIns_->cubeInfo.k0) :
        tilingIns_->shapeInfo.singlekH * tilingIns_->shapeInfo.singlekW * tilingIns_->cubeInfo.k0;
    uint64_t mAL1Min = currmL0;
    uint64_t hoAL1Min = mAL1Min / tilingIns_->shapeInfo.orgWo + CONST_VALUE_2;
    uint64_t hiAL1Min = InferHiL1(hoAL1Min, tilingIns_->shapeInfo.orgHi);
    uint64_t usedL1Size = hiAL1Min * tilingIns_->shapeInfo.orgWi * kAL1Min * fMapDTypeSize * this->dbValue.pbAL1;
    uint64_t weightSize = currnL0 * kBL1Min * weightDTypeSize * this->dbValue.pbBL1;
    usedL1Size += max(weightSize, reserveWeightSize);
    if (tilingIns_->hasBias) {
        uint64_t biasSize = currnL0 * biasDTypeSize;
        usedL1Size += biasSize;
    }
    if (tilingIns_->hasScale) {
        uint64_t scaleSize = static_cast<uint64_t>(tilingIns_->shapeInfo.channelWiseCoeff *
                                                   static_cast<float>(currnL0 * FP16_DTYPE_SIZE));
        usedL1Size += scaleSize;
    }

    return usedL1Size;
}

int64_t ConvTilingAlgorithmMmode::GetL0Tiling()
{
    GetL0TilingRange();
    L0TilingDecision();
    return 0;
}

void ConvTilingAlgorithmMmode::InitPingPong()
{
    this->dbValue.pbAL1 = DOUBLE_BUFFER_NUM;
    this->dbValue.pbBL1 = DOUBLE_BUFFER_NUM;
    uint64_t l0BSize = tilingIns_->platformInfo.l0BSize;
    if (CalcL1SizeForL0Tiling(tilingIns_->cubeInfo.m0, tilingIns_->cubeInfo.n0,
        l0BSize) <= tilingIns_->platformInfo.l1Size) {
            this->dbValue.pbAL1 = DOUBLE_BUFFER_NUM;
            this->dbValue.pbBL1 = DOUBLE_BUFFER_NUM;
    } else {
        this->dbValue.pbAL1 = (CalcL1SizeForL0Tiling(tilingIns_->cubeInfo.m0, tilingIns_->cubeInfo.n0) <=
                               tilingIns_->platformInfo.l1Size) ? DOUBLE_BUFFER_NUM : 1;
        this->dbValue.pbBL1 = 1;
    }
    this->dbValue.pbAL0 = DOUBLE_BUFFER_NUM;
    this->dbValue.pbBL0 = DOUBLE_BUFFER_NUM;
    this->dbValue.pbCL0 = 1;
}

void ConvTilingAlgorithmMmode::GetL0TilingRange()
{
    // Get nL0 range
    uint64_t nL0Max = min(tilingIns_->platformInfo.l0BSize / (tilingIns_->cubeInfo.k0 * this->dbValue.pbBL0 *
                                                                weightDTypeSize),
                          tilingIns_->platformInfo.l0CSize / (tilingIns_->cubeInfo.m0 * this->dbValue.pbCL0 *
                                                                DTYPE_SIZE_TAB.at(tilingIns_->cubeInfo.madType)));
    CalcCommFactorWithPowerOfTwo(tilingIns_->shapeInfo.singleCo1, nL0Max / tilingIns_->cubeInfo.n0,
                                 l0TilingRange.nL0Range);
    VectorElementMultip(l0TilingRange.nL0Range, tilingIns_->cubeInfo.n0);

    // Get mL0 range
    uint64_t mL0Max = min(tilingIns_->platformInfo.l0ASize / (tilingIns_->cubeInfo.k0 * this->dbValue.pbAL0 *
                          fMapDTypeSize),
                          tilingIns_->platformInfo.l0CSize / (tilingIns_->cubeInfo.n0 * this->dbValue.pbCL0 *
                          DTYPE_SIZE_TAB.at(tilingIns_->cubeInfo.madType)));
    CalcCommFactorWithPowerOfTwo(CeilDiv(tilingIns_->shapeInfo.singleM, tilingIns_->cubeInfo.m0),
                                 mL0Max / tilingIns_->cubeInfo.m0, l0TilingRange.mL0Range);
    VectorElementMultip(l0TilingRange.mL0Range, tilingIns_->cubeInfo.m0);
}

void ConvTilingAlgorithmMmode::L0TilingDecision()
{
    l0TilingIdx.mL0Idx = 0;
    l0TilingIdx.nL0Idx = 0;
    l0TilingParams.kL0 = tilingIns_->cubeInfo.k0;
    l0TilingParams.mL0 = l0TilingRange.mL0Range[l0TilingIdx.mL0Idx];
    l0TilingParams.nL0 = l0TilingRange.nL0Range[l0TilingIdx.nL0Idx];

    bool updateML0Index = false;
    bool l0BufferNotOverflowFlag = CheckL0Buffer(l0TilingParams.mL0, l0TilingParams.kL0, l0TilingParams.nL0);
    uint64_t usedL1Size = CalcL1SizeForL0Tiling(l0TilingParams.mL0, l0TilingParams.nL0);
    uint64_t usedBTSize = CalcBTSize(l0TilingParams.nL0);
    uint64_t usedFBSize = CalcFBSize(l0TilingParams.nL0);
    uint64_t mL0RangeLen = l0TilingRange.mL0Range.size();
    uint64_t nL0RangeLen = l0TilingRange.nL0Range.size();

    bool overFlow = true;
    while (l0BufferNotOverflowFlag && usedL1Size <= tilingIns_->platformInfo.l1Size &&
           usedBTSize <= tilingIns_->platformInfo.btSize && usedFBSize <= tilingIns_->platformInfo.fbSize) {
        if ((l0TilingIdx.mL0Idx == mL0RangeLen - 1) && (l0TilingIdx.nL0Idx == nL0RangeLen - 1)) {
            overFlow = false;
            break;
        }
        if (l0TilingParams.mL0 <= l0TilingParams.nL0) {
            updateML0Index = (l0TilingIdx.mL0Idx == (mL0RangeLen - 1)) ? false : true;
        } else {
            updateML0Index = (l0TilingIdx.nL0Idx == (nL0RangeLen - 1)) ? true : false;
        }

        if (updateML0Index && (l0TilingIdx.mL0Idx < mL0RangeLen - 1)) {
            ++l0TilingIdx.mL0Idx;
        } else if (!updateML0Index && (l0TilingIdx.nL0Idx < nL0RangeLen - 1)) {
            ++l0TilingIdx.nL0Idx;
        }

        l0TilingParams.mL0 = l0TilingRange.mL0Range[l0TilingIdx.mL0Idx];
        l0TilingParams.nL0 = l0TilingRange.nL0Range[l0TilingIdx.nL0Idx];

        l0BufferNotOverflowFlag = CheckL0Buffer(l0TilingParams.mL0, l0TilingParams.kL0, l0TilingParams.nL0);
        usedL1Size = CalcL1SizeForL0Tiling(l0TilingParams.mL0, l0TilingParams.nL0);
        usedBTSize = CalcBTSize(l0TilingParams.nL0);
        usedFBSize = CalcFBSize(l0TilingParams.nL0);
    }

    if (overFlow) {
        if (updateML0Index) {
            l0TilingIdx.mL0Idx = l0TilingIdx.mL0Idx == 0 ? 0 : l0TilingIdx.mL0Idx - 1;
        } else {
            l0TilingIdx.nL0Idx = l0TilingIdx.nL0Idx == 0 ? 0 : l0TilingIdx.nL0Idx - 1;
        }
    }

    l0TilingParams.mL0 = l0TilingRange.mL0Range[l0TilingIdx.mL0Idx];
    l0TilingParams.nL0 = l0TilingRange.nL0Range[l0TilingIdx.nL0Idx];

    CalFormulaicInnerBatch();
}

void ConvTilingAlgorithmMmode::CalFormulaicInnerBatch()
{
    if (tilingIns_->isC04Flag || tilingIns_->attrInfo.groups > 1 || tilingIns_->isDmaFlag ||
        tilingIns_->descInfo.fMapType.format == ConvFormat::NCDHW ||
        tilingIns_->descInfo.fMapType.format == ConvFormat::NDHWC) {
        return;
    }
    uint64_t mDim = CeilDiv(static_cast<uint64_t>(tilingIns_->shapeInfo.orgWo * tilingIns_->shapeInfo.orgHo),
        static_cast<uint64_t>(tilingIns_->shapeInfo.singleM));
    if (tilingIns_->shapeInfo.singleBatch == static_cast<int64_t>(1) || mDim != static_cast<uint64_t>(1)) {
        return;
    }
    tilingIns_->enableInnerBatch = true;
 
    uint64_t innerBatchLimit1 = tilingIns_->platformInfo.l0ASize / (CONST_VALUE_2 * this->fMapDTypeSize *
                                                                    l0TilingParams.mL0 * l0TilingParams.kL0);
    uint64_t innerBatchLimit2 = tilingIns_->platformInfo.l0CSize / (l0TilingParams.mL0 * l0TilingParams.nL0 *
                                DTYPE_SIZE_TAB.at(tilingIns_->cubeInfo.madType) * CONST_VALUE_2);
    uint64_t currentBiasL1Size = tilingIns_->hasBias ?
        (this->l1TilingFlag.isBiasFullLoad ? tilingIns_->shapeInfo.singleCo1 * tilingIns_->cubeInfo.n0 *
        this->biasDTypeSize : this->l0TilingParams.nL0 * this->biasDTypeSize) : 0;
    uint64_t innerBatchLimit3 = (tilingIns_->platformInfo.l1Size - currentBiasL1Size -
                                l0TilingParams.nL0 * static_cast<uint64_t>(tilingIns_->shapeInfo.orgkH *
                                tilingIns_->shapeInfo.orgkW) * l0TilingParams.kL0 * this->weightDTypeSize *
                                CONST_VALUE_2) / (static_cast<uint64_t>(tilingIns_->shapeInfo.orgWi *
                                tilingIns_->shapeInfo.orgHi) * l0TilingParams.kL0 *
                                this->fMapDTypeSize * CONST_VALUE_2);
    uint64_t innerBatchLimit4 = tilingIns_->shapeInfo.singleBatch;
    if (innerBatchLimit1 == 0 || innerBatchLimit2 == 0 || innerBatchLimit3 == 0 || innerBatchLimit4 == 0) {
        tilingIns_->enableInnerBatch = false;
        return;
    }
    tilingIns_->innerBatch = static_cast<uint32_t>(min(min(innerBatchLimit1, innerBatchLimit2),
        min(innerBatchLimit3, innerBatchLimit4)));
    tilingIns_->innerBatch = min(tilingIns_->innerBatch,
        static_cast<uint32_t>(CeilDiv(static_cast<uint64_t>(tilingIns_->shapeInfo.singleBatch),
        CeilDiv(static_cast<uint64_t>(tilingIns_->shapeInfo.singleBatch), tilingIns_->innerBatch))));
    if (tilingIns_->innerBatch == 1) {
        tilingIns_->enableInnerBatch = false;
    }
    return;
}

void ConvTilingAlgorithmMmode::CheckL0CDoubleBuffer()
{
    if (CalcCL0Size(l0TilingParams.mL0, l0TilingParams.nL0) <= (tilingIns_->platformInfo.l0CSize / DOUBLE_BUFFER_NUM)) {
        this->dbValue.pbCL0 = DOUBLE_BUFFER_NUM;
    }
    bool kFullLoadFlag = l0TilingParams.kL0 == tilingIns_->shapeInfo.singlekD * tilingIns_->shapeInfo.singleCi1 *
                                               this->l1TilingCalc.ci0HkWk;
    if (kFullLoadFlag && static_cast<uint64_t>(tilingIns_->shapeInfo.singleM) <= l0TilingParams.mL0) {
        this->dbValue.pbAL0 = 1;
    }

    if (kFullLoadFlag && static_cast<uint64_t>(tilingIns_->shapeInfo.singleCo) <= l0TilingParams.nL0) {
        this->dbValue.pbBL0 = 1;
    }

    if (this->dbValue.pbAL0 == DOUBLE_BUFFER_NUM && this->dbValue.pbBL0 == DOUBLE_BUFFER_NUM && kFullLoadFlag) {
        if (tilingIns_->l1TilingInfo.iterateMNOrder == IterateMNOrder::ITER_M_FST) {
            this->dbValue.pbBL0 = 1;
        } else {
            this->dbValue.pbAL0 = 1;
        }
    }
    if (this->dbValue.pbAL0 == DOUBLE_BUFFER_NUM && this->dbValue.pbBL0 == 1) {
        tilingIns_->l1TilingInfo.iterateMNOrder = IterateMNOrder::ITER_M_FST;
    } else if (this->dbValue.pbAL0 == 1 && this->dbValue.pbBL0 == DOUBLE_BUFFER_NUM) {
        tilingIns_->l1TilingInfo.iterateMNOrder = IterateMNOrder::ITER_N_FST;
    }
}

void ConvTilingAlgorithmMmode::ScaleBiasUbTilingDecision()
{
    if (l0TilingParams.mL0 == 0 || l0TilingParams.nL0 == 0) {
        return;
    }
    if (!tilingIns_->isScaleBiasInUb) {
        return;
    }

    uint64_t mUb = 0;
    std::vector<uint64_t> mUbRange;
    CalcCommFactor(l0TilingParams.mL0, l0TilingParams.mL0, mUbRange);

    // output + scale + bias(fp16/bf16) + bias(fp32) <= UB_SIZE
    // output + scale + bias(fp32) <= UB_SIZE
    uint64_t totalScaleBiasTypeSize = tilingIns_->descInfo.biasType.dtype == ConvDtype::FLOAT32 ?
        TOTAL_SCALE_BIAS_32_TYPE_SIZE : TOTAL_SCALE_BIAS_16_TYPE_SIZE;
    for (int32_t i = mUbRange.size() - 1; i >= 0; i--) {
        if (l0TilingParams.nL0 * (mUbRange[i] * MAX_OUT_TYPE_SIZE + totalScaleBiasTypeSize) <=
            tilingIns_->platformInfo.ubSize) {
            mUb = mUbRange[i];
            break;
        }
    }

    // if mUb fullload mL0, split mL0 to use more vector
    if (mUb == l0TilingParams.mL0) {
        mUb = CeilDiv(l0TilingParams.mL0, VEC_NUM_PER_CUBE_910D);
    }
    tilingIns_->ubTilingInfo.mUb = mUb;
    tilingIns_->ubTilingInfo.nUb = l0TilingParams.nL0; // nUb fullload nL0 by default;
}

void ConvTilingAlgorithmMmode::GetUbTiling()
{
    ScaleBiasUbTilingDecision();
}
} // namespace conv_tiling_algo_m