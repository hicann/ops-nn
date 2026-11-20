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
 * \file conv_api_tiling_algorithm_HWmode.cpp
 * \brief
 */

#include <iostream>
#include <cstdint>
#include "conv_api_tiling_algorithm_BBmode.h"

using namespace std;
namespace conv_tiling_algo_bb {
// [API Tiling] First segment interface
void ConvTilingAlgorithmBBmode::AdjustM()
{
    uint64_t hwOut = tilingIns_->shapeInfo.orgHo * tilingIns_->shapeInfo.orgWo;
    uint64_t fCut = basicBlockInfoPtr->batch * basicBlockInfoPtr->doCut * CeilDiv(hwOut, basicBlockInfoPtr->mTile);
    if (tilingIns_->enableInnerBatch) {
        fCut = CeilDiv(
            CeilDiv(basicBlockInfoPtr->batch * CeilDiv(hwOut, tilingIns_->cubeInfo.m0) * tilingIns_->cubeInfo.m0,
                    basicBlockInfoPtr->mTile),
            tilingIns_->innerBatch);
    }
    uint64_t fMapped = CeilDiv(fCut, basicBlockInfoPtr->fDim); // max fmap cuts in one singleCore
    fActive = CeilDiv(fCut, fMapped);                          // active cores
    // max m cuts in one singleCore
    uint64_t mMappedHoWo = !tilingIns_->enableInnerBatch ?
                               basicBlockInfoPtr->fDim * fMapped / basicBlockInfoPtr->batch :
                               static_cast<uint64_t>(1);
    uint64_t mTileAdjusted = CeilDiv(hwOut, tilingIns_->cubeInfo.m0 * mMappedHoWo) * tilingIns_->cubeInfo.m0;
    uint64_t mCutAdjusted = CeilDiv(hwOut, mTileAdjusted);
    uint64_t fCutAdjusted = basicBlockInfoPtr->batch * basicBlockInfoPtr->doCut * mCutAdjusted;
    if (tilingIns_->enableInnerBatch) {
        fCutAdjusted = CeilDiv(
            CeilDiv(basicBlockInfoPtr->batch * CeilDiv(hwOut, tilingIns_->cubeInfo.m0) * tilingIns_->cubeInfo.m0,
                    mTileAdjusted),
            tilingIns_->innerBatch);
    }
    uint64_t fActiveAdjusted = CeilDiv(fCutAdjusted, CeilDiv(fCutAdjusted, basicBlockInfoPtr->fDim));
    if ((basicBlockInfoPtr->useStrictAdjustM && fActiveAdjusted > fActive) ||
        (!basicBlockInfoPtr->useStrictAdjustM && fActiveAdjusted >= fActive)) {
        basicBlockInfoPtr->mTile = static_cast<uint32_t>(mTileAdjusted);
        basicBlockInfoPtr->mCut = static_cast<uint32_t>(mCutAdjusted);
        // in basic block multiM=multiN=1, wiAl1=oriWi
        basicBlockInfoPtr->mIn = InferHiL1(basicBlockInfoPtr->mTile);
        fActive = fActiveAdjusted;
    }
}

void ConvTilingAlgorithmBBmode::AdjustN()
{
    // active cores
    nActive = CeilDiv(basicBlockInfoPtr->nCut, CeilDiv(basicBlockInfoPtr->nCut, basicBlockInfoPtr->nDim));
    uint64_t nTileAdjusted = CeilDiv(tilingIns_->shapeInfo.orgCo,
                                     CeilDiv(basicBlockInfoPtr->nCut, basicBlockInfoPtr->nDim) *
                                         basicBlockInfoPtr->nDim * tilingIns_->cubeInfo.n0) *
                             tilingIns_->cubeInfo.n0;
    // fixpipe buffer limit
    nTileAdjusted = tilingIns_->shapeInfo.channelWiseCoeff > 0 ?
                        min((tilingIns_->platformInfo.fbSize /
                             static_cast<uint64_t>(tilingIns_->shapeInfo.channelWiseCoeff * FP16_DTYPE_SIZE)) /
                                tilingIns_->cubeInfo.n0 * tilingIns_->cubeInfo.n0,
                            nTileAdjusted) :
                        nTileAdjusted;
    nTileAdjusted = tilingIns_->hasBias ? min((tilingIns_->platformInfo.btSize / this->biasDTypeSize), nTileAdjusted) :
                                          nTileAdjusted; // bias table limit
    uint64_t nCutAdjusted = CeilDiv(tilingIns_->shapeInfo.orgCo, nTileAdjusted);
    // adjusted active cores
    uint64_t nActiveAdjusted = CeilDiv(nCutAdjusted, CeilDiv(nCutAdjusted, basicBlockInfoPtr->nDim));
    if ((basicBlockInfoPtr->useStrictAdjustM && nActiveAdjusted > nActive) ||
        (!basicBlockInfoPtr->useStrictAdjustM && nActiveAdjusted >= nActive)) {
        basicBlockInfoPtr->nTile = static_cast<uint32_t>(nTileAdjusted);
        basicBlockInfoPtr->nCut = static_cast<uint32_t>(nCutAdjusted);
        nActive = nActiveAdjusted;
    } else {
        nTileAdjusted = static_cast<uint64_t>(basicBlockInfoPtr->nTile);
        nTileAdjusted = tilingIns_->shapeInfo.channelWiseCoeff > 0 ?
                            min((tilingIns_->platformInfo.fbSize /
                                 static_cast<uint64_t>(tilingIns_->shapeInfo.channelWiseCoeff * FP16_DTYPE_SIZE)),
                                nTileAdjusted) :
                            nTileAdjusted;
        nTileAdjusted = tilingIns_->hasBias ? min((tilingIns_->platformInfo.btSize / this->biasDTypeSize),
                                                  nTileAdjusted) :
                                              nTileAdjusted; // bias table limit
        basicBlockInfoPtr->nTile = static_cast<uint32_t>(nTileAdjusted);
        basicBlockInfoPtr->nCut = static_cast<uint32_t>(nCutAdjusted);
    }
}

uint64_t ConvTilingAlgorithmBBmode::InferHiL1(uint32_t multiMTileSize) const
{
    uint64_t inputHoL1 = static_cast<uint64_t>(multiMTileSize) / tilingIns_->shapeInfo.orgWo + CONST_VALUE_2;
    uint64_t khDilated = CalcKhDilated(tilingIns_->shapeInfo.singlekH, tilingIns_->attrInfo.dilationH);
    uint64_t inputHiL1 = Conv2DInferHiL1(inputHoL1, khDilated, tilingIns_->shapeInfo.orgHi,
                                         tilingIns_->attrInfo.strideH);
    return inputHiL1 * tilingIns_->shapeInfo.orgWi;
}

void ConvTilingAlgorithmBBmode::CalcCoreUtilization() const
{
    basicBlockInfoPtr->coreUtilization = static_cast<float>(fActive) * static_cast<float>(nActive) /
                                         (static_cast<float>(basicBlockInfoPtr->aicoreNum) /
                                          static_cast<float>(basicBlockInfoPtr->groupDim));
}

void ConvTilingAlgorithmBBmode::TryBiasAndScaleFullLoad(const int64_t& usedL1Size)
{
    preSetFullLoadFlag.biasFullLoad = false;
    preSetFullLoadFlag.fixpFullLoad = false;
    int64_t biasL1FullLoadExtraSize = tilingIns_->hasBias ? (tilingIns_->shapeInfo.singleCo -
                                                             static_cast<int64_t>(basicBlockInfoPtr->nTile)) *
                                                                this->biasDTypeSize :
                                                            0;
    int64_t scaleL1FullLoadExtraSize = (tilingIns_->shapeInfo.singleCo -
                                        static_cast<int64_t>(basicBlockInfoPtr->nTile)) *
                                       tilingIns_->shapeInfo.channelWiseCoeff * FP16_DTYPE_SIZE;
    if (usedL1Size - biasL1FullLoadExtraSize - scaleL1FullLoadExtraSize >= 0) {
        preSetFullLoadFlag.biasFullLoad = true;
        preSetFullLoadFlag.fixpFullLoad = true;
    }
    // bias has larger dtypeSize, limitation is more strict
    if (tilingIns_->hasBias && (tilingIns_->shapeInfo.singleCo * this->biasDTypeSize > DATACOPYPARAMS_BURSTLEN_MAX)) {
        preSetFullLoadFlag.biasFullLoad = false;
        preSetFullLoadFlag.fixpFullLoad = false;
    }
    return;
}

bool ConvTilingAlgorithmBBmode::CheckL1SpaceEnough(uint64_t usedAL1Size, uint64_t usedBL1Size)
{
    int64_t usedL1Size = availableL1Size - static_cast<int64_t>(usedAL1Size) - static_cast<int64_t>(usedBL1Size);
    if (usedL1Size < 0) {
        return false;
    }
    TryBiasAndScaleFullLoad(usedL1Size);
    return true;
}

void ConvTilingAlgorithmBBmode::CalcMNFullLoadFlag() const
{
    basicBlockInfoPtr->fCut = basicBlockInfoPtr->mCut * basicBlockInfoPtr->batch * basicBlockInfoPtr->doCut;
    if (tilingIns_->enableInnerBatch) {
        basicBlockInfoPtr->fCut = CeilDiv(CeilDiv(basicBlockInfoPtr->batch *
                                                      CeilDiv(tilingIns_->shapeInfo.orgHo * tilingIns_->shapeInfo.orgWo,
                                                              tilingIns_->cubeInfo.m0) *
                                                      tilingIns_->cubeInfo.m0,
                                                  basicBlockInfoPtr->mTile),
                                          tilingIns_->innerBatch) *
                                  basicBlockInfoPtr->doCut;
    }
    basicBlockInfoPtr->mAl1FullLoad = (basicBlockInfoPtr->fCut > basicBlockInfoPtr->fDim) ? false : true;
    basicBlockInfoPtr->nBl1FullLoad = (basicBlockInfoPtr->nCut > basicBlockInfoPtr->nDim) ? false : true;
}

void ConvTilingAlgorithmBBmode::CalcAvalibleL1Size()
{
    uint64_t biasL1Size = tilingIns_->hasBias ?
                              (basicBlockInfoPtr->biasFullLoad ? tilingIns_->shapeInfo.singleCo * this->biasDTypeSize :
                                                                 basicBlockInfoPtr->nTile * this->biasDTypeSize) :
                              0;
    uint64_t scaleL1Size = (basicBlockInfoPtr->fixpFullLoad ?
                                tilingIns_->shapeInfo.singleCo * tilingIns_->shapeInfo.channelWiseCoeff *
                                    FP16_DTYPE_SIZE :
                                basicBlockInfoPtr->nTile * tilingIns_->shapeInfo.channelWiseCoeff * FP16_DTYPE_SIZE);
    availableL1Size = tilingIns_->platformInfo.l1Size - biasL1Size - scaleL1Size;

    fullLoadFactor = tilingIns_->shapeInfo.orgkD > 0 ? static_cast<uint64_t>(tilingIns_->shapeInfo.orgkD) : 1;

    fmapL1FullLoadSize = tilingIns_->shapeInfo.orgHi * tilingIns_->shapeInfo.orgWi * tilingIns_->shapeInfo.orgCi *
                         basicBlockInfoPtr->batch / basicBlockInfoPtr->fDim * fMapDTypeSize * fullLoadFactor;
    weightL1FullLoadSize = tilingIns_->shapeInfo.orgkH * tilingIns_->shapeInfo.orgkW * tilingIns_->shapeInfo.orgCi *
                           tilingIns_->shapeInfo.orgCo / basicBlockInfoPtr->nDim * weightDTypeSize * fullLoadFactor;
}

void ConvTilingAlgorithmBBmode::CalcWeightCoeff()
{
    if (tilingIns_->descInfo.weightType.format == ConvFormat::FRACTAL_Z ||
        tilingIns_->descInfo.weightType.format == ConvFormat::FRACTAL_Z_C04) {
        weightCoeff = FMAP_WEIGHT_BANDWIDTH_COEFF_1V1;
        return;
    }

    if (!tilingIns_->optGroupFlag) {
        if (tilingIns_->shapeInfo.orgkH * tilingIns_->shapeInfo.orgkW == KERNEL_1X1 ||
            tilingIns_->descInfo.weightType.format == ConvFormat::NHWC ||
            tilingIns_->descInfo.weightType.format == ConvFormat::NDHWC) {
            weightCoeff = FMAP_WEIGHT_BANDWIDTH_COEFF_1V1; // 1: hyperparameter due to ND2NZ
        } else if (tilingIns_->shapeInfo.orgkH * tilingIns_->shapeInfo.orgkW > KERNEL_3X3) {
            weightCoeff = FMAP_WEIGHT_BANDWIDTH_COEFF_6V1; // 6: hyperparameter
        } else {
            weightCoeff = FMAP_WEIGHT_BANDWIDTH_COEFF_4V1; // 4: DN2NZ hardware shifter limits
        }
    } else {
        // optGroupFlag is true, use NDDMA, weight coeff equal to 1, only increase GM->UB Head cost
        weightCoeff = FMAP_WEIGHT_BANDWIDTH_COEFF_1V1;
    }
}

double ConvTilingAlgorithmBBmode::CalcL1LoadScore() const
{
    uint64_t aEffectiveSize = tilingIns_->shapeInfo.orgHi * tilingIns_->shapeInfo.orgWi * fullLoadFactor *
                              basicBlockInfoPtr->doCut * basicBlockInfoPtr->batch;
    uint64_t bEffectiveSize = tilingIns_->shapeInfo.orgCo * tilingIns_->shapeInfo.orgkH * fullLoadFactor *
                              tilingIns_->shapeInfo.orgkW * weightCoeff;
    double l1LoadScoreTemp = l1ScoreBase +
                             static_cast<double>(1.0) /
                                 (aEffectiveSize / (tilingIns_->shapeInfo.orgkH * tilingIns_->shapeInfo.orgkW) *
                                      mRepeats +
                                  bEffectiveSize * nRepeats);
    return l1LoadScoreTemp;
}

bool ConvTilingAlgorithmBBmode::UpdateL1LoadStrategy(ConvTilingAlgorithmBBmode* bbPtr)
{
    auto l1LoadStrategy = l1LoadStrategies[l1LoadStrategyType];
    bool ret = l1LoadStrategy->GetL1LoadStrategy(bbPtr);
    if (!ret) {
        return false;
    }
    double l1LoadScoreTemp = CalcL1LoadScore();
    if (l1LoadScoreTemp > basicBlockInfoPtr->l1LoadScore) {
        basicBlockInfoPtr->l1LoadScore = l1LoadScoreTemp;
        basicBlockInfoPtr->kAl1FullLoad = preSetFullLoadFlag.kAl1FullLoad;
        basicBlockInfoPtr->kBl1FullLoad = preSetFullLoadFlag.kBl1FullLoad;
        basicBlockInfoPtr->biasFullLoad = preSetFullLoadFlag.biasFullLoad;
        basicBlockInfoPtr->fixpFullLoad = preSetFullLoadFlag.fixpFullLoad;
        return true;
    }
    return false;
}

void ConvTilingAlgorithmBBmode::TryKABFullLoadL1Stratgy()
{
    bool updateRet = false;
    // 1> kAl1FullLoad && kBl1FullLoad
    preSetFullLoadFlag.kAl1FullLoad = true;
    preSetFullLoadFlag.kBl1FullLoad = true;
    if (basicBlockInfoPtr->mAl1FullLoad || basicBlockInfoPtr->nBl1FullLoad) {
        // K_AND_MAL1_FULL_LOAD is same as K_AND_NBL1_FULL_LOAD, because none of them have repeated loading
        l1LoadStrategyType = L1LoadStrategyType::K_AND_MAL1_FULL_LOAD;
        updateRet = UpdateL1LoadStrategy(this);
    } else {
        // KAL1 and KBL1 FullLoad, but Fmap and weight can not fullLoad
        l1LoadStrategyType = L1LoadStrategyType::K_AND_NONE_FULL_LOAD;
        updateRet = UpdateL1LoadStrategy(this);
    }
    if (updateRet) {
        if (weightL1FullLoadSize > fmapL1FullLoadSize) {
            basicBlockInfoPtr->iterateMNOrder = IterateMNOrder::ITER_M_FST;
        } else {
            basicBlockInfoPtr->iterateMNOrder = IterateMNOrder::ITER_N_FST;
        }
        l1TilingParams = tmpL1TilingParams;
    }
}

void ConvTilingAlgorithmBBmode::TryNFstLoadL1Stratgy()
{
    bool updateRet = false;
    // 2> iterateMNOrder == IterateMNOrder::ITER_N_FST
    preSetFullLoadFlag.kAl1FullLoad = true;
    preSetFullLoadFlag.kBl1FullLoad = false;
    if (basicBlockInfoPtr->mAl1FullLoad) {
        l1LoadStrategyType = L1LoadStrategyType::FMAP_FULL_LOAD;
        updateRet = UpdateL1LoadStrategy(this);
    } else {
        l1LoadStrategyType = L1LoadStrategyType::FMAP_K_FULL_LOAD;
        updateRet = UpdateL1LoadStrategy(this);
    }
    if (updateRet) {
        basicBlockInfoPtr->iterateMNOrder = IterateMNOrder::ITER_N_FST;
        l1TilingParams = tmpL1TilingParams;
    }
}

void ConvTilingAlgorithmBBmode::TryMFstLoadL1Stratgy()
{
    bool updateRet = false;
    // 3> iterateMNOrder == IterateMNOrder::ITER_M_FST
    preSetFullLoadFlag.kAl1FullLoad = false;
    preSetFullLoadFlag.kBl1FullLoad = true;
    if (basicBlockInfoPtr->nBl1FullLoad) {
        l1LoadStrategyType = L1LoadStrategyType::WEIGHT_FULL_LOAD;
        updateRet = UpdateL1LoadStrategy(this);
    } else if (basicBlockInfoPtr->kBl1FullLoad) {
        l1LoadStrategyType = L1LoadStrategyType::WEIGHT_K_FULL_LOAD;
        updateRet = UpdateL1LoadStrategy(this);
    }
    if (updateRet) {
        basicBlockInfoPtr->iterateMNOrder = IterateMNOrder::ITER_M_FST;
        l1TilingParams = tmpL1TilingParams;
    }
}

void ConvTilingAlgorithmBBmode::TryKAllSplitLoadL1Stratgy()
{
    bool updateRet = false;
    // 4> KAllSplit
    preSetFullLoadFlag.kAl1FullLoad = false;
    preSetFullLoadFlag.kBl1FullLoad = false;
    l1LoadStrategyType = L1LoadStrategyType::K_ALL_SPLIT;
    updateRet = UpdateL1LoadStrategy(this);
    if (updateRet) {
        if (weightL1FullLoadSize > fmapL1FullLoadSize) {
            basicBlockInfoPtr->iterateMNOrder = IterateMNOrder::ITER_M_FST;
        } else {
            basicBlockInfoPtr->iterateMNOrder = IterateMNOrder::ITER_N_FST;
        }
        l1TilingParams = tmpL1TilingParams;
    }
}

void ConvTilingAlgorithmBBmode::CalcBestL1LoadStratgy()
{
    // firstly, m and n can full load or not can be judged by amount of Cut and Dim.
    CalcMNFullLoadFlag();
    CalcAvalibleL1Size();
    CalcWeightCoeff();

    // secondly, try different K fullLoad sence. (not excceed L1Size and record best load strategy)
    if (singleCi1xC0 * tilingIns_->shapeInfo.singlekH * tilingIns_->shapeInfo.singlekW * fullLoadFactor <=
        MAX_16_BIT_NUM) { // PostK Limit
        // 1> kAl1FullLoad && kBl1FullLoad
        TryKABFullLoadL1Stratgy();

        // 2> iterateMNOrder == IterateMNOrder::ITER_N_FST
        TryNFstLoadL1Stratgy();
    }
    // 3> iterateMNOrder == IterateMNOrder::ITER_M_FST
    TryMFstLoadL1Stratgy();

    // 4> KAllSplit
    TryKAllSplitLoadL1Stratgy();
}

// ========= K all split L1 Load Strategy =========
bool ConvTilingAlgorithmBBmode::KAllSplit::GetL1LoadStrategy(ConvTilingAlgorithmBBmode* bbPtr)
{
    bbPtr->mRepeats = bbPtr->basicBlockInfoPtr->nCut;
    bbPtr->nRepeats = !bbPtr->tilingIns_->enableInnerBatch ?
                          bbPtr->basicBlockInfoPtr->doCut * bbPtr->basicBlockInfoPtr->batch *
                              bbPtr->basicBlockInfoPtr->mCut :
                          CeilDiv(CeilDiv(bbPtr->basicBlockInfoPtr->batch *
                                              CeilDiv(static_cast<uint32_t>(bbPtr->tilingIns_->shapeInfo.orgHo *
                                                                            bbPtr->tilingIns_->shapeInfo.orgWo),
                                                      bbPtr->tilingIns_->cubeInfo.m0) *
                                              bbPtr->tilingIns_->cubeInfo.m0,
                                          bbPtr->basicBlockInfoPtr->mTile),
                                  bbPtr->tilingIns_->innerBatch);
    bbPtr->l1ScoreBase = L1_SCORE_BASE_0_POINTS;
    return true;
}

// ========= K all Full Load L1 Load Strategy =========
bool ConvTilingAlgorithmBBmode::GetL1LoadStrategyForKFullLoadCommon()
{
    dbA = basicBlockInfoPtr->mAl1FullLoad && preSetFullLoadFlag.kAl1FullLoad ? 1 : DOUBLE_BUFFER_NUM;
    dbB = basicBlockInfoPtr->nBl1FullLoad && preSetFullLoadFlag.kBl1FullLoad ? 1 : DOUBLE_BUFFER_NUM;

    tmpL1TilingParams.kAL1 = singleCi1xC0 * fullLoadFactor;
    tmpL1TilingParams.kBL1 = singleCi1xC0 * tilingIns_->shapeInfo.orgkH * tilingIns_->shapeInfo.orgkW * fullLoadFactor;
    uint64_t usedAL1Size = dbA * fMapDTypeSize * basicBlockInfoPtr->mIn * tilingIns_->innerBatch *
                           tmpL1TilingParams.kAL1;
    uint64_t usedBL1Size = dbB * weightDTypeSize * basicBlockInfoPtr->nTile * tmpL1TilingParams.kBL1;
    if (!CheckL1SpaceEnough(usedAL1Size, usedBL1Size)) {
        return false;
    }
    return true;
}

bool ConvTilingAlgorithmBBmode::KFullLoadAndMOrNFullLoad::GetL1LoadStrategy(ConvTilingAlgorithmBBmode* bbPtr)
{
    if (!bbPtr->GetL1LoadStrategyForKFullLoadCommon()) {
        return false;
    }

    bbPtr->mRepeats = bbPtr->basicBlockInfoPtr->nDim;
    bbPtr->nRepeats = bbPtr->basicBlockInfoPtr->fDim;
    bbPtr->l1ScoreBase = L1_SCORE_BASE_5_POINTS;
    return true;
}

bool ConvTilingAlgorithmBBmode::KAndNoneFullLoad::GetL1LoadStrategy(ConvTilingAlgorithmBBmode* bbPtr)
{
    if (!bbPtr->GetL1LoadStrategyForKFullLoadCommon()) {
        return false;
    }

    bbPtr->mRepeats = bbPtr->basicBlockInfoPtr->nDim;
    bbPtr->nRepeats = !bbPtr->tilingIns_->enableInnerBatch ?
                          bbPtr->basicBlockInfoPtr->doCut * bbPtr->basicBlockInfoPtr->batch *
                              bbPtr->basicBlockInfoPtr->mCut :
                          CeilDiv(CeilDiv(bbPtr->basicBlockInfoPtr->batch *
                                              CeilDiv(static_cast<uint32_t>(bbPtr->tilingIns_->shapeInfo.orgHo *
                                                                            bbPtr->tilingIns_->shapeInfo.orgWo),
                                                      bbPtr->tilingIns_->cubeInfo.m0) *
                                              bbPtr->tilingIns_->cubeInfo.m0,
                                          bbPtr->basicBlockInfoPtr->mTile),
                                  bbPtr->tilingIns_->innerBatch);
    bbPtr->l1ScoreBase = L1_SCORE_BASE_5_POINTS;
    return true;
}

void ConvTilingAlgorithmBBmode::GetL1LoadStrategyOneInputFullLoad()
{
    this->mRepeats = basicBlockInfoPtr->nDim;
    this->nRepeats = basicBlockInfoPtr->fDim;
    this->l1ScoreBase = L1_SCORE_BASE_4_POINTS;
}
// ========= Nfirst L1 Load Strategy =========
bool ConvTilingAlgorithmBBmode::GetL1LoadStrategyForNFirstCommon()
{
    dbA = basicBlockInfoPtr->mAl1FullLoad && preSetFullLoadFlag.kAl1FullLoad ? 1 : DOUBLE_BUFFER_NUM;
    dbB = basicBlockInfoPtr->nBl1FullLoad && preSetFullLoadFlag.kBl1FullLoad ? 1 : DOUBLE_BUFFER_NUM;

    tmpL1TilingParams.kAL1 = singleCi1xC0 * fullLoadFactor;
    uint64_t usedAL1Size = dbA * fMapDTypeSize * tmpL1TilingParams.mAL1Value * tmpL1TilingParams.kAL1 *
                           tilingIns_->innerBatch;
    // weight not full load, resever space size
    uint64_t usedBL1Size = max(
        tilingIns_->platformInfo.l0BSize,
        static_cast<uint64_t>(basicBlockInfoPtr->nTile * tilingIns_->shapeInfo.orgkH * tilingIns_->shapeInfo.orgkW *
                              tilingIns_->cubeInfo.k0 * weightDTypeSize));
    if (!CheckL1SpaceEnough(usedAL1Size, usedBL1Size)) {
        return false;
    }
    return true;
}

bool ConvTilingAlgorithmBBmode::FmapFullLoad::GetL1LoadStrategy(ConvTilingAlgorithmBBmode* bbPtr)
{
    bbPtr->tmpL1TilingParams.mAL1Value = bbPtr->basicBlockInfoPtr->mIn;
    if (!bbPtr->GetL1LoadStrategyForNFirstCommon()) {
        return false;
    }
    bbPtr->GetL1LoadStrategyOneInputFullLoad();
    return true;
}

bool ConvTilingAlgorithmBBmode::FmapKFullLoad::GetL1LoadStrategy(ConvTilingAlgorithmBBmode* bbPtr)
{
    bbPtr->tmpL1TilingParams.mAL1Value = bbPtr->basicBlockInfoPtr->mIn;
    if (!bbPtr->GetL1LoadStrategyForNFirstCommon()) {
        return false;
    }

    bbPtr->mRepeats = bbPtr->basicBlockInfoPtr->nDim;
    bbPtr->nRepeats = !bbPtr->tilingIns_->enableInnerBatch ?
                          bbPtr->basicBlockInfoPtr->doCut * bbPtr->basicBlockInfoPtr->batch *
                              bbPtr->basicBlockInfoPtr->mCut :
                          CeilDiv(CeilDiv(bbPtr->basicBlockInfoPtr->batch *
                                              CeilDiv(static_cast<uint32_t>(bbPtr->tilingIns_->shapeInfo.orgHo *
                                                                            bbPtr->tilingIns_->shapeInfo.orgWo),
                                                      bbPtr->tilingIns_->cubeInfo.m0) *
                                              bbPtr->tilingIns_->cubeInfo.m0,
                                          bbPtr->basicBlockInfoPtr->mTile),
                                  bbPtr->tilingIns_->innerBatch);
    bbPtr->l1ScoreBase = L1_SCORE_BASE_3_POINTS;
    return true;
}
// ========= Mfirst L1 Load Strategy =========
bool ConvTilingAlgorithmBBmode::GetL1LoadStrategyForMFirstCommon()
{
    dbA = basicBlockInfoPtr->mAl1FullLoad && preSetFullLoadFlag.kAl1FullLoad ? 1 : DOUBLE_BUFFER_NUM;
    dbB = basicBlockInfoPtr->nBl1FullLoad && preSetFullLoadFlag.kBl1FullLoad ? 1 : DOUBLE_BUFFER_NUM;

    tmpL1TilingParams.kBL1 = singleCi1xC0 * tilingIns_->shapeInfo.orgkH * tilingIns_->shapeInfo.orgkW * fullLoadFactor;
    uint64_t usedAL1Size = max(
        tilingIns_->platformInfo.l0ASize,
        static_cast<uint64_t>(basicBlockInfoPtr->mIn * tilingIns_->innerBatch * tilingIns_->cubeInfo.k0 *
                              fMapDTypeSize)); // fmap not full load, resever space size
    uint64_t usedBL1Size = dbB * weightDTypeSize * tmpL1TilingParams.nBL1Value * tmpL1TilingParams.kBL1;
    if (!CheckL1SpaceEnough(usedAL1Size, usedBL1Size)) {
        return false;
    }
    return true;
}

bool ConvTilingAlgorithmBBmode::WeightFullLoad::GetL1LoadStrategy(ConvTilingAlgorithmBBmode* bbPtr)
{
    bbPtr->tmpL1TilingParams.nBL1Value = bbPtr->basicBlockInfoPtr->nTile;
    if (!bbPtr->GetL1LoadStrategyForMFirstCommon()) {
        return false;
    }
    bbPtr->GetL1LoadStrategyOneInputFullLoad();
    return true;
}

bool ConvTilingAlgorithmBBmode::WeightKFullLoad::GetL1LoadStrategy(ConvTilingAlgorithmBBmode* bbPtr)
{
    bbPtr->tmpL1TilingParams.nBL1Value = bbPtr->basicBlockInfoPtr->nTile;
    if (!bbPtr->GetL1LoadStrategyForMFirstCommon()) {
        return false;
    }
    bbPtr->mRepeats = bbPtr->basicBlockInfoPtr->nCut;
    bbPtr->nRepeats = bbPtr->basicBlockInfoPtr->fDim;
    bbPtr->l1ScoreBase = L1_SCORE_BASE_3_POINTS;
    return true;
}

// [API Tiling] Second segment interface
void ConvTilingAlgorithmBBmode::InitPingPong()
{
    this->dbValue.pbAL1 = DOUBLE_BUFFER_NUM;
    this->dbValue.pbBL1 = DOUBLE_BUFFER_NUM;
    this->dbValue.pbAL0 = DOUBLE_BUFFER_NUM;
    this->dbValue.pbBL0 = DOUBLE_BUFFER_NUM;
    this->dbValue.pbCL0 = SINGLE_BUFFER_NUM;
}

int64_t ConvTilingAlgorithmBBmode::Process()
{
    InitPingPong();
    if (GetL1Tiling(this) == -1) {
        return -1;
    }
    GetL0Tiling();
    if (optiling::conv_ops_tiling::IsDAV920Arch(tilingIns_->platformInfo.npuArch)) {
        GetUbTiling();
    }
    SetPBufferRes();
    return 0;
}

int64_t ConvTilingAlgorithmBBmode::GetL1Tiling(ConvTilingAlgorithmBBmode* bbPtr)
{
    CalcAvalibleL1Size();
    fmapL1FullLoadSize = bbPtr->basicBlockInfoPtr->mIn * singleCi1xC0 * fMapDTypeSize * tilingIns_->innerBatch *
                         fullLoadFactor; // MultiM=1时Fmap全载大小
    weightL1FullLoadSize = tilingIns_->shapeInfo.orgkH * tilingIns_->shapeInfo.orgkW * singleCi1xC0 *
                           bbPtr->basicBlockInfoPtr->nTile * weightDTypeSize *
                           fullLoadFactor; //  MultiN=1时Weight全载大小

    // set default mAL1Value and nBL1Value
    bbPtr->l1TilingParams.mAL1Value = bbPtr->basicBlockInfoPtr->mIn;
    bbPtr->l1TilingParams.nBL1Value = bbPtr->basicBlockInfoPtr->nTile;
    if (bbPtr->basicBlockInfoPtr->mTile >= tilingIns_->shapeInfo.singleM) {
        basicBlockInfoPtr->mAl1FullLoad = true;
    }

    // KABL1 FullLoad
    if (basicBlockInfoPtr->kAl1FullLoad && basicBlockInfoPtr->kBl1FullLoad) {
        GetKABFullLoadL1TilingParams();
        // Fmap resides in L1, iterateMNOrder == ITER_N_FST
    } else if (basicBlockInfoPtr->iterateMNOrder == IterateMNOrder::ITER_N_FST) {
        if (basicBlockInfoPtr->kAl1FullLoad && basicBlockInfoPtr->mAl1FullLoad) {
            l1LoadStrategyType = L1LoadStrategyType::FMAP_FULL_LOAD;
        } else if (basicBlockInfoPtr->kAl1FullLoad) {
            l1LoadStrategyType = L1LoadStrategyType::FMAP_K_FULL_LOAD;
        } else {
            l1LoadStrategyType = L1LoadStrategyType::N_FIRST_K_SPLIT;
        }
        // Weight resides in L1, iterateMNOrder == ITER_M_FST
    } else {
        if (basicBlockInfoPtr->kBl1FullLoad && basicBlockInfoPtr->nBl1FullLoad) {
            l1LoadStrategyType = L1LoadStrategyType::WEIGHT_FULL_LOAD;
        } else if (basicBlockInfoPtr->kBl1FullLoad) {
            l1LoadStrategyType = L1LoadStrategyType::WEIGHT_K_FULL_LOAD;
        } else {
            l1LoadStrategyType = L1LoadStrategyType::M_FIRST_K_SPLIT;
        }
    }

    OP_LOGD(tilingIns_->nodeType, "l1LoadStrategyType is: %u", static_cast<uint32_t>(l1LoadStrategyType));
    if (!GetL1TilingParams(this, l1LoadStrategyType)) {
        return -1;
    };
    UpdateFinalFullLoadFlag();

    SetL1TilingRes();
    return 0;
}

void ConvTilingAlgorithmBBmode::GetL0Tiling()
{
    GetKL0();
    CheckL0CDoubleBuffer();
    if (optiling::conv_ops_tiling::IsDAV920Arch(tilingIns_->platformInfo.npuArch)) {
        TryOptimizeL0MNTile();
    }
    SetL0TilingRes();
}

void ConvTilingAlgorithmBBmode::TryOptimizeL0MNTile()
{
    uint64_t singleM = static_cast<uint64_t>(tilingIns_->shapeInfo.singleM);
    uint64_t singleCo = static_cast<uint64_t>(tilingIns_->shapeInfo.singleCo);
    uint64_t m0 = tilingIns_->cubeInfo.m0;
    uint64_t n0 = tilingIns_->cubeInfo.n0;
    uint64_t kL0 = basicBlockInfoPtr->kTile;
    uint64_t maxMTileByL0A = tilingIns_->platformInfo.l0ASize / fMapDTypeSize / this->dbValue.pbAL0 / kL0 /
                             tilingIns_->innerBatch;
    uint64_t maxNTileByL0B = tilingIns_->platformInfo.l0BSize / weightDTypeSize / this->dbValue.pbBL0 / kL0;
    uint64_t maxMTile = min(AlignB(min(maxMTileByL0A, singleM), m0), tilingIns_->l1TilingInfo.mAL1);
    uint64_t maxNTile = min(AlignB(min(maxNTileByL0B, singleCo), n0), tilingIns_->l1TilingInfo.nBL1);

    for (uint64_t tM = maxMTile; tM >= m0; tM -= m0) {
        for (uint64_t tN = maxNTile; tN >= n0; tN -= n0) {
            L0MNTileEvalResult result;
            if (!EvalL0MNTileCandidate(tM, tN, result)) {
                continue;
            }
            basicBlockInfoPtr->mTile = static_cast<uint32_t>(tM);
            basicBlockInfoPtr->nTile = static_cast<uint32_t>(tN);
            basicBlockInfoPtr->mCut = static_cast<uint32_t>(CeilDiv(singleM, tM));
            basicBlockInfoPtr->nCut = static_cast<uint32_t>(CeilDiv(singleCo, tN));
            this->dbValue.pbAL0 = result.pbAL0;
            this->dbValue.pbBL0 = result.pbBL0;
            return;
        }
    }
}

bool ConvTilingAlgorithmBBmode::EvalL0MNTileCandidate(uint64_t tryMTile, uint64_t tryNTile,
                                                      L0MNTileEvalResult& result) const
{
    uint64_t kL0 = basicBlockInfoPtr->kTile;
    uint64_t m0 = tilingIns_->cubeInfo.m0;
    uint64_t n0 = tilingIns_->cubeInfo.n0;
    bool kFullLoadFlag = kL0 == static_cast<uint64_t>(tilingIns_->shapeInfo.singlekD) *
                                    tilingIns_->shapeInfo.singleCi1 * tilingIns_->cubeInfo.k0 *
                                    static_cast<uint64_t>(tilingIns_->shapeInfo.singlekH) *
                                    static_cast<uint64_t>(tilingIns_->shapeInfo.singlekW);
    uint64_t singleM = static_cast<uint64_t>(tilingIns_->shapeInfo.singleM);
    uint64_t singleCo = static_cast<uint64_t>(tilingIns_->shapeInfo.singleCo);
    bool al0SingleBufferOk = !tilingIns_->enableInnerBatch ||
                             tilingIns_->innerBatch >= static_cast<uint64_t>(tilingIns_->shapeInfo.singleBatch);
    result.pbAL0 = (kFullLoadFlag && singleM <= tryMTile && al0SingleBufferOk) ? SINGLE_BUFFER_NUM :
                                                                                 this->dbValue.pbAL0;
    result.pbBL0 = (kFullLoadFlag && singleCo <= tryNTile) ? SINGLE_BUFFER_NUM : this->dbValue.pbBL0;
    uint64_t mL0Align = AlignB(tryMTile, m0);
    uint64_t nL0Align = AlignB(tryNTile, n0);
    uint64_t madDTypeSize = DTYPE_SIZE_TAB.at(tilingIns_->cubeInfo.madType);
    uint8_t pbCL0 = this->dbValue.pbCL0;
    uint64_t aL0Size = mL0Align * kL0 * result.pbAL0 * fMapDTypeSize * tilingIns_->innerBatch;
    uint64_t bL0Size = kL0 * nL0Align * result.pbBL0 * weightDTypeSize;
    uint64_t cL0Size = mL0Align * nL0Align * pbCL0 * madDTypeSize * tilingIns_->innerBatch;
    return aL0Size <= tilingIns_->platformInfo.l0ASize && bL0Size <= tilingIns_->platformInfo.l0BSize &&
           cL0Size <= tilingIns_->platformInfo.l0CSize;
}

void ConvTilingAlgorithmBBmode::CheckL0CDoubleBuffer()
{
    uint32_t mL0 = basicBlockInfoPtr->mTile;
    uint32_t kL0 = basicBlockInfoPtr->kTile;
    uint32_t nL0 = basicBlockInfoPtr->nTile;
    if (CalcCL0Size(mL0, nL0) <= (tilingIns_->platformInfo.l0CSize / DOUBLE_BUFFER_NUM)) {
        this->dbValue.pbCL0 = DOUBLE_BUFFER_NUM;
    }
    // todo fix shapeInfo.singlekD = -1
    bool kFullLoadFlag = optiling::conv_ops_tiling::IsDAV920Arch(tilingIns_->platformInfo.npuArch) &&
                         kL0 == tilingIns_->shapeInfo.singlekD * tilingIns_->shapeInfo.singleCi1 *
                                    tilingIns_->cubeInfo.k0 * tilingIns_->shapeInfo.singlekH *
                                    tilingIns_->shapeInfo.singlekW;
    if (kFullLoadFlag && static_cast<uint64_t>(tilingIns_->shapeInfo.singleM) <= mL0 &&
        tilingIns_->innerBatch >= static_cast<uint64_t>(tilingIns_->shapeInfo.singleBatch)) {
        this->dbValue.pbAL0 = SINGLE_BUFFER_NUM;
    }

    if (kFullLoadFlag && static_cast<uint64_t>(tilingIns_->shapeInfo.singleCo) <= nL0) {
        this->dbValue.pbBL0 = SINGLE_BUFFER_NUM;
    }

    if (this->dbValue.pbAL0 == DOUBLE_BUFFER_NUM && this->dbValue.pbBL0 == DOUBLE_BUFFER_NUM && kFullLoadFlag) {
        if (tilingIns_->l1TilingInfo.iterateMNOrder == IterateMNOrder::ITER_M_FST) {
            this->dbValue.pbBL0 = SINGLE_BUFFER_NUM;
        } else {
            this->dbValue.pbAL0 = SINGLE_BUFFER_NUM;
        }
    }
    if (this->dbValue.pbAL0 == DOUBLE_BUFFER_NUM && this->dbValue.pbBL0 == SINGLE_BUFFER_NUM) {
        tilingIns_->l1TilingInfo.iterateMNOrder = IterateMNOrder::ITER_M_FST;
    } else if (this->dbValue.pbAL0 == SINGLE_BUFFER_NUM && this->dbValue.pbBL0 == DOUBLE_BUFFER_NUM) {
        tilingIns_->l1TilingInfo.iterateMNOrder = IterateMNOrder::ITER_N_FST;
    }
}

void ConvTilingAlgorithmBBmode::GetKL0() const
{
    uint64_t kAL1 = tilingIns_->l1TilingInfo.kAL1;
    uint64_t kBL1 = tilingIns_->l1TilingInfo.kBL1;
    uint64_t kL1 = min(kAL1, kBL1) / tilingIns_->cubeInfo.k0;
    uint64_t l0ASize = tilingIns_->platformInfo.l0ASize;
    uint64_t l0BSize = tilingIns_->platformInfo.l0BSize;
    uint64_t kL0Max = min(l0ASize / fMapDTypeSize / DOUBLE_BUFFER_NUM / basicBlockInfoPtr->mTile /
                              tilingIns_->innerBatch,
                          l0BSize / weightDTypeSize / DOUBLE_BUFFER_NUM / basicBlockInfoPtr->nTile) /
                      tilingIns_->cubeInfo.k0;
    vector<uint64_t> factors;
    CalcCommFactor(kL1, kL0Max, factors);
    auto maxFactorPtr = max_element(factors.begin(), factors.end());
    if (maxFactorPtr == factors.end()) {
        OP_LOGE(tilingIns_->nodeType, "Input data for GetKL0 is illegal in BasicBlockMode.");
        return;
    }
    basicBlockInfoPtr->kTile = static_cast<uint32_t>((*maxFactorPtr) * tilingIns_->cubeInfo.k0);
}

void ConvTilingAlgorithmBBmode::SetL1TilingRes()
{
    tilingIns_->l1TilingInfo.kAL1 = this->l1TilingParams.kAL1 * tilingIns_->shapeInfo.orgkH *
                                    tilingIns_->shapeInfo.orgkW;
    tilingIns_->l1TilingInfo.kBL1 = this->l1TilingParams.kBL1;
    tilingIns_->l1TilingInfo.mAL1 = basicBlockInfoPtr->mAl1FullLoad ?
                                        tilingIns_->shapeInfo.singleM1 * tilingIns_->cubeInfo.m0 :
                                        this->multiM * basicBlockInfoPtr->mTile;
    tilingIns_->l1TilingInfo.nBL1 = basicBlockInfoPtr->nBl1FullLoad ?
                                        tilingIns_->shapeInfo.singleCo1 * tilingIns_->cubeInfo.n0 :
                                        this->multiN * basicBlockInfoPtr->nTile;
    tilingIns_->l1TilingInfo.iterateMNOrder = basicBlockInfoPtr->iterateMNOrder;
    tilingIns_->l1TilingInfo.biasFullLoadFlag = basicBlockInfoPtr->biasFullLoad;
    tilingIns_->l1TilingInfo.fixpParamsFullLoadFlag = basicBlockInfoPtr->fixpFullLoad;
}

void ConvTilingAlgorithmBBmode::SetL0TilingRes()
{
    tilingIns_->l0TilingInfo.kL0 = basicBlockInfoPtr->kTile;
    tilingIns_->l0TilingInfo.mL0 = basicBlockInfoPtr->mTile;
    tilingIns_->l0TilingInfo.nL0 = basicBlockInfoPtr->nTile;
}

void ConvTilingAlgorithmBBmode::L1LoadStrategyKFullLoad::GetL1LoadTilingBothFullLoad(ConvTilingAlgorithmBBmode* bbPtr,
                                                                                     uint64_t maxMAL1Iter,
                                                                                     uint64_t maxNBL1Iter) const
{
    bbPtr->basicBlockInfoPtr->iterateMNOrder = IterateMNOrder::ITER_M_FST;
    bbPtr->multiM = maxMAL1Iter;
    bbPtr->multiN = maxNBL1Iter;
    bbPtr->l1TilingParams.mAL1Value = min(
        static_cast<uint64_t>(bbPtr->multiM) * bbPtr->basicBlockInfoPtr->mIn,
        static_cast<uint64_t>(bbPtr->tilingIns_->shapeInfo.orgHi * bbPtr->tilingIns_->shapeInfo.orgWi));
    bbPtr->l1TilingParams.nBL1Value = min(static_cast<uint64_t>(bbPtr->multiN) * bbPtr->basicBlockInfoPtr->nTile,
                                          static_cast<uint64_t>(bbPtr->tilingIns_->shapeInfo.singleCo));
}

void ConvTilingAlgorithmBBmode::L1LoadStrategyKFullLoad::GetL1LoadTilingFmapFullLoad(ConvTilingAlgorithmBBmode* bbPtr,
                                                                                     uint64_t maxMAL1Iter) const
{
    bbPtr->basicBlockInfoPtr->iterateMNOrder = IterateMNOrder::ITER_N_FST;
    bbPtr->multiM = maxMAL1Iter;
    bbPtr->multiN = 1;
    bbPtr->l1TilingParams.mAL1Value = min(
        static_cast<uint64_t>(bbPtr->multiM) * bbPtr->basicBlockInfoPtr->mIn,
        static_cast<uint64_t>(bbPtr->tilingIns_->shapeInfo.orgHi * bbPtr->tilingIns_->shapeInfo.orgWi));
    bbPtr->l1TilingParams.nBL1Value = bbPtr->basicBlockInfoPtr->nTile;
}

void ConvTilingAlgorithmBBmode::L1LoadStrategyKFullLoad::GetL1LoadTilingWeightFullLoad(ConvTilingAlgorithmBBmode* bbPtr,
                                                                                       uint64_t maxNBL1Iter) const
{
    bbPtr->basicBlockInfoPtr->iterateMNOrder = IterateMNOrder::ITER_M_FST;
    bbPtr->multiM = 1;
    bbPtr->multiN = maxNBL1Iter;
    bbPtr->l1TilingParams.mAL1Value = bbPtr->basicBlockInfoPtr->mIn;
    bbPtr->l1TilingParams.nBL1Value = min(static_cast<uint64_t>(bbPtr->multiN) * bbPtr->basicBlockInfoPtr->nTile,
                                          static_cast<uint64_t>(bbPtr->tilingIns_->shapeInfo.singleCo));
}

void ConvTilingAlgorithmBBmode::L1LoadStrategyKFullLoad::GetL1LoadTilingWithoutFullLoad(
    ConvTilingAlgorithmBBmode* bbPtr) const
{
    bbPtr->multiM = 1;
    bbPtr->multiN = 1;
    bbPtr->l1TilingParams.mAL1Value = bbPtr->basicBlockInfoPtr->mIn;
    bbPtr->l1TilingParams.nBL1Value = bbPtr->basicBlockInfoPtr->nTile;
}

// level 1 strategy pattern class implement
bool ConvTilingAlgorithmBBmode::L1LoadStrategyKFullLoad::GetL1LoadTilingParams(ConvTilingAlgorithmBBmode* bbPtr)
{
    // KAL1 and KBL1 fullLoad set
    bbPtr->l1TilingParams.kAL1 = bbPtr->singleCi1xC0 * bbPtr->fullLoadFactor;
    bbPtr->l1TilingParams.kBL1 = bbPtr->singleCi1xC0 * bbPtr->tilingIns_->shapeInfo.orgkH *
                                 bbPtr->tilingIns_->shapeInfo.orgkW * bbPtr->fullLoadFactor;

    uint64_t maxNBL1Iter = CeilDiv(bbPtr->basicBlockInfoPtr->nCut, bbPtr->basicBlockInfoPtr->nDim);
    // 单core上weight放BB的Multi1大小
    int64_t weightLoadBBSizeMultix1 = min(bbPtr->basicBlockInfoPtr->nTile,
                                          static_cast<uint32_t>(bbPtr->tilingIns_->shapeInfo.singleCo)) *
                                      bbPtr->singleCi1xC0 * bbPtr->tilingIns_->shapeInfo.orgkH *
                                      bbPtr->tilingIns_->shapeInfo.orgkW * bbPtr->weightDTypeSize *
                                      static_cast<int64_t>(bbPtr->fullLoadFactor);
    // 单core上weight放BB的总大小
    int64_t singleBBWeightSize = min(maxNBL1Iter * bbPtr->basicBlockInfoPtr->nTile,
                                     static_cast<uint64_t>(bbPtr->tilingIns_->shapeInfo.singleCo)) *
                                 bbPtr->singleCi1xC0 * bbPtr->tilingIns_->shapeInfo.orgkH *
                                 bbPtr->tilingIns_->shapeInfo.orgkW * bbPtr->weightDTypeSize *
                                 static_cast<int64_t>(bbPtr->fullLoadFactor);

    uint64_t maxMAL1Iter = CeilDiv(bbPtr->basicBlockInfoPtr->mCut, bbPtr->basicBlockInfoPtr->mDim);
    // 单core上fmap放BB的Multi1大小
    int64_t fmapLoadSizeMultix1 = bbPtr->basicBlockInfoPtr->mIn * bbPtr->singleCi1xC0 * bbPtr->fMapDTypeSize *
                                  bbPtr->tilingIns_->innerBatch * bbPtr->fullLoadFactor;
    // 单core上fmap放BB的总大小
    int64_t singleBBFmapSize = maxMAL1Iter * fmapLoadSizeMultix1;

    OP_LOGD(bbPtr->tilingIns_->nodeType,
            "singleBBFmapSize: %lu, singleBBWeightSize: %lu, fmapSizeMultix1: %lu, weightBBSizeMultix1: %lu",
            singleBBFmapSize, singleBBWeightSize, fmapLoadSizeMultix1, weightLoadBBSizeMultix1);
    if (GetAOrBFullLoadL1TilingParams(bbPtr, fmapLoadSizeMultix1, weightLoadBBSizeMultix1, singleBBFmapSize,
                                      singleBBWeightSize)) {
        return true;
    }

    if (GetNoneFullLoadL1TilingParams(bbPtr, fmapLoadSizeMultix1, weightLoadBBSizeMultix1)) {
        return true;
    }
    return false;
}

bool ConvTilingAlgorithmBBmode::L1LoadStrategyKFullLoad::GetAOrBFullLoadL1TilingParams(ConvTilingAlgorithmBBmode* bbPtr,
                                                                                       int64_t fmapLoadSizeMultix1,
                                                                                       int64_t weightLoadBBSizeMultix1,
                                                                                       int64_t singleBBFmapSize,
                                                                                       int64_t singleBBWeightSize) const
{
    uint64_t maxMAL1Iter = CeilDiv(bbPtr->basicBlockInfoPtr->mCut, bbPtr->basicBlockInfoPtr->mDim);
    uint64_t maxNBL1Iter = CeilDiv(bbPtr->basicBlockInfoPtr->nCut, bbPtr->basicBlockInfoPtr->nDim);
    // Fmap and weight fullLoad in L1
    if (singleBBFmapSize + singleBBWeightSize <= bbPtr->availableL1Size) {
        GetL1LoadTilingBothFullLoad(bbPtr, maxMAL1Iter, maxNBL1Iter);
        return true;
    }
    if (bbPtr->basicBlockInfoPtr->iterateMNOrder == IterateMNOrder::ITER_N_FST) {
        // Fmap fullLoad in L1
        if (singleBBFmapSize + weightLoadBBSizeMultix1 * static_cast<int64_t>(DOUBLE_BUFFER_NUM) <=
            bbPtr->availableL1Size) {
            GetL1LoadTilingFmapFullLoad(bbPtr, maxMAL1Iter);
            return true;
        }
        // weight fullLoad in L1
        else if (fmapLoadSizeMultix1 * static_cast<int64_t>(DOUBLE_BUFFER_NUM) + singleBBWeightSize <=
                 bbPtr->availableL1Size) {
            GetL1LoadTilingWeightFullLoad(bbPtr, maxNBL1Iter);
            return true;
        }
    } else {
        // weight fullLoad in L1
        if (fmapLoadSizeMultix1 * static_cast<int64_t>(DOUBLE_BUFFER_NUM) + singleBBWeightSize <=
            bbPtr->availableL1Size) {
            GetL1LoadTilingWeightFullLoad(bbPtr, maxNBL1Iter);
            return true;
        }
        // Fmap fullLoad in L1
        else if (singleBBFmapSize + weightLoadBBSizeMultix1 * static_cast<int64_t>(DOUBLE_BUFFER_NUM) <=
                 bbPtr->availableL1Size) {
            GetL1LoadTilingFmapFullLoad(bbPtr, maxMAL1Iter);
            return true;
        }
    }
    return false;
}

bool ConvTilingAlgorithmBBmode::L1LoadStrategyKFullLoad::GetNoneFullLoadL1TilingParams(
    ConvTilingAlgorithmBBmode* bbPtr, int64_t fmapLoadSizeMultix1, int64_t weightLoadBBSizeMultix1) const
{
    // 4. No one FullLoad, DB on
    if (fmapLoadSizeMultix1 * static_cast<int64_t>(DOUBLE_BUFFER_NUM) +
            weightLoadBBSizeMultix1 * static_cast<int64_t>(DOUBLE_BUFFER_NUM) <=
        bbPtr->availableL1Size) {
        GetL1LoadTilingWithoutFullLoad(bbPtr);
        return true;
    }

    // 5. No one FullLoad, DB off
    if (fmapLoadSizeMultix1 + weightLoadBBSizeMultix1 <= bbPtr->availableL1Size) {
        bbPtr->aL1DBNeedClose = true;
        bbPtr->bL1DBNeedClose = true;
        GetL1LoadTilingWithoutFullLoad(bbPtr);
        return true;
    }
    return false;
}

void ConvTilingAlgorithmBBmode::GetKABFullLoadL1TilingParams()
{
    this->l1TilingParams.kAL1 = singleCi1xC0 * fullLoadFactor;
    this->l1TilingParams.kBL1 = singleCi1xC0 * tilingIns_->shapeInfo.orgkH * tilingIns_->shapeInfo.orgkW *
                                fullLoadFactor;

    tilingIns_->shapeInfo.orgDi = tilingIns_->shapeInfo.orgDi > 0 ? static_cast<uint64_t>(tilingIns_->shapeInfo.orgDi) :
                                                                    1;

    // singlecore fmap fullload size
    int64_t singleCoreFmapSize = CeilDiv(tilingIns_->shapeInfo.orgDi * tilingIns_->shapeInfo.orgHi *
                                             tilingIns_->shapeInfo.orgWi * basicBlockInfoPtr->batch,
                                         basicBlockInfoPtr->fDim) *
                                 singleCi1xC0;
    // singlecore weight fullload size
    int64_t singleCoreWeightSize = fullLoadFactor * tilingIns_->shapeInfo.orgkH * tilingIns_->shapeInfo.orgkW *
                                   CeilDiv(tilingIns_->shapeInfo.singleCo, basicBlockInfoPtr->nDim) * singleCi1xC0;

    if (basicBlockInfoPtr->mAl1FullLoad && basicBlockInfoPtr->nBl1FullLoad) {
        basicBlockInfoPtr->iterateMNOrder = IterateMNOrder::ITER_M_FST;
        this->dbValue.pbAL1 = SINGLE_BUFFER_NUM;
        if (this->tilingIns_->enableInnerBatch &&
            this->tilingIns_->innerBatch < static_cast<uint32_t>(this->tilingIns_->shapeInfo.singleBatch)) {
            this->dbValue.pbAL1 = DOUBLE_BUFFER_NUM;
        }
        this->dbValue.pbBL1 = SINGLE_BUFFER_NUM;
        l1LoadStrategyType = L1LoadStrategyType::K_AND_MAL1_FULL_LOAD;
    } else if (basicBlockInfoPtr->mAl1FullLoad) {
        basicBlockInfoPtr->iterateMNOrder = IterateMNOrder::ITER_N_FST;
        this->dbValue.pbAL1 = SINGLE_BUFFER_NUM;
        if (this->tilingIns_->enableInnerBatch &&
            this->tilingIns_->innerBatch < static_cast<uint32_t>(this->tilingIns_->shapeInfo.singleBatch)) {
            this->dbValue.pbAL1 = DOUBLE_BUFFER_NUM;
        }
        l1LoadStrategyType = L1LoadStrategyType::K_AND_MAL1_FULL_LOAD;
    } else if (basicBlockInfoPtr->nBl1FullLoad) {
        basicBlockInfoPtr->iterateMNOrder = IterateMNOrder::ITER_M_FST;
        this->dbValue.pbBL1 = SINGLE_BUFFER_NUM;
        l1LoadStrategyType = L1LoadStrategyType::K_AND_NBL1_FULL_LOAD;
    } else {
        basicBlockInfoPtr->iterateMNOrder = singleCoreFmapSize > singleCoreWeightSize ? IterateMNOrder::ITER_M_FST :
                                                                                        IterateMNOrder::ITER_N_FST;
        l1LoadStrategyType = L1LoadStrategyType::K_AND_NONE_FULL_LOAD;
    }
}

bool ConvTilingAlgorithmBBmode::L1LoadStrategyNFirst::GetL1LoadTilingParams(ConvTilingAlgorithmBBmode* bbPtr)
{
    // KAL1 fullLoad set
    bbPtr->l1TilingParams.kAL1 = bbPtr->singleCi1xC0 * bbPtr->fullLoadFactor;

    auto aL1size = bbPtr->l1TilingParams.kAL1 * bbPtr->l1TilingParams.mAL1Value * bbPtr->dbValue.pbAL1 *
                   bbPtr->fMapDTypeSize * bbPtr->tilingIns_->innerBatch;
    bbPtr->availableL1Size -= aL1size;
    // Fmap resides in L1, BL1 DB ON
    int64_t cinBL1FullLoadSize = bbPtr->singleCi1xC0;
    int64_t cinBL1 = min(
        (bbPtr->availableL1Size / static_cast<int64_t>(bbPtr->weightDTypeSize) /
         (static_cast<int64_t>(bbPtr->tilingIns_->shapeInfo.orgkH) *
          static_cast<int64_t>(bbPtr->tilingIns_->shapeInfo.orgkW)) /
         static_cast<int64_t>(DOUBLE_BUFFER_NUM) / static_cast<int64_t>(bbPtr->basicBlockInfoPtr->nTile)),
        cinBL1FullLoadSize); // 这个场景放大KBL1, MultiN=1，所以用nTile
    if (cinBL1 < bbPtr->tilingIns_->cubeInfo.k0) {
        bbPtr->dbValue.pbBL1 = SINGLE_BUFFER_NUM;
        bbPtr->bL1DBNeedClose = true;
        cinBL1 = min((bbPtr->availableL1Size / static_cast<int64_t>(bbPtr->weightDTypeSize) /
                      (static_cast<int64_t>(bbPtr->tilingIns_->shapeInfo.orgkH) *
                       static_cast<int64_t>(bbPtr->tilingIns_->shapeInfo.orgkW)) /
                      static_cast<int64_t>(bbPtr->basicBlockInfoPtr->nTile)),
                     cinBL1FullLoadSize);
    }
    cinBL1 = FloorAlign(cinBL1, bbPtr->tilingIns_->cubeInfo.k0);
    if (cinBL1 <= 0) {
        OP_LOGE(bbPtr->tilingIns_->nodeType, "L1LoadStrategyNFirst still cannot generate L1 tiling.");
        return false;
    }
    bbPtr->l1TilingParams.kBL1 = cinBL1 * bbPtr->tilingIns_->shapeInfo.orgkH * bbPtr->tilingIns_->shapeInfo.orgkW;

    bool ret = MultiLoadKL1(bbPtr, Kl1MultiAxis::KBL1); // KBL1 set
    if (!ret) {
        return ret;
    }
    if (optiling::conv_ops_tiling::IsDAV920Arch(bbPtr->tilingIns_->platformInfo.npuArch)) {
        uint64_t kBL1CinSlice = bbPtr->l1TilingParams.kBL1 /
                                (bbPtr->tilingIns_->shapeInfo.orgkH * bbPtr->tilingIns_->shapeInfo.orgkW);
        uint64_t kdValueFromKBL1 = bbPtr->apiStrategy_->CalcDinFromCinSlice(kBL1CinSlice, bbPtr->singleCi1xC0);
        bbPtr->l1TilingParams.kBL1 = bbPtr->apiStrategy_->ValidateAndAdjustKL1(kBL1CinSlice, kdValueFromKBL1) *
                                     (bbPtr->tilingIns_->shapeInfo.orgkH * bbPtr->tilingIns_->shapeInfo.orgkW);
    }
    return true;
}

bool ConvTilingAlgorithmBBmode::FmapFullLoad::GetL1LoadTilingParams(ConvTilingAlgorithmBBmode* bbPtr)
{
    bbPtr->dbValue.pbAL1 = SINGLE_BUFFER_NUM;
    if (bbPtr->tilingIns_->enableInnerBatch &&
        bbPtr->tilingIns_->innerBatch < static_cast<uint32_t>(bbPtr->tilingIns_->shapeInfo.singleBatch)) {
        bbPtr->dbValue.pbAL1 = DOUBLE_BUFFER_NUM;
    }
    uint64_t maxMAL1Iter = CeilDiv(bbPtr->basicBlockInfoPtr->mCut, bbPtr->basicBlockInfoPtr->mDim);
    bbPtr->multiM = maxMAL1Iter;
    bbPtr->multiN = 1;
    bbPtr->l1TilingParams.mAL1Value = min(
        static_cast<uint64_t>(bbPtr->multiM) * bbPtr->basicBlockInfoPtr->mIn,
        static_cast<uint64_t>(bbPtr->tilingIns_->shapeInfo.orgHi * bbPtr->tilingIns_->shapeInfo.orgWi));
    bbPtr->l1TilingParams.nBL1Value = bbPtr->basicBlockInfoPtr->nTile;

    bool ret = L1LoadStrategyNFirst::GetL1LoadTilingParams(bbPtr);
    return ret;
}

bool ConvTilingAlgorithmBBmode::FmapKFullLoad::GetL1LoadTilingParams(ConvTilingAlgorithmBBmode* bbPtr)
{
    bbPtr->dbValue.pbAL1 = DOUBLE_BUFFER_NUM;
    bbPtr->multiM = 1;
    bbPtr->multiN = 1;
    bbPtr->l1TilingParams.mAL1Value = bbPtr->basicBlockInfoPtr->mIn;
    bbPtr->l1TilingParams.nBL1Value = bbPtr->basicBlockInfoPtr->nTile;

    bool ret = L1LoadStrategyNFirst::GetL1LoadTilingParams(bbPtr);
    return ret;
}

bool ConvTilingAlgorithmBBmode::L1LoadStrategyMFirst::GetL1LoadTilingParams(ConvTilingAlgorithmBBmode* bbPtr)
{
    // KBL1 fullLoad set
    bbPtr->l1TilingParams.kBL1 = bbPtr->singleCi1xC0 * bbPtr->tilingIns_->shapeInfo.orgkH *
                                 bbPtr->tilingIns_->shapeInfo.orgkW * bbPtr->fullLoadFactor;

    int64_t kAL1FullLoadSize = bbPtr->singleCi1xC0 * bbPtr->fullLoadFactor;
    int64_t bL1size = bbPtr->l1TilingParams.kBL1 * bbPtr->l1TilingParams.nBL1Value * bbPtr->dbValue.pbBL1 *
                      bbPtr->weightDTypeSize;
    bbPtr->availableL1Size -= bL1size;
    // Fmap resides in L1, AL1 DB ON
    int64_t cinAL1 = min((bbPtr->availableL1Size / static_cast<int64_t>(bbPtr->basicBlockInfoPtr->mIn) /
                          static_cast<int64_t>(DOUBLE_BUFFER_NUM) / static_cast<int64_t>(bbPtr->fMapDTypeSize) /
                          bbPtr->tilingIns_->innerBatch),
                         kAL1FullLoadSize);
    if (cinAL1 < bbPtr->tilingIns_->cubeInfo.k0 ||
        (optiling::conv_ops_tiling::IsDAV920Arch(bbPtr->tilingIns_->platformInfo.npuArch) &&
         cinAL1 < kAL1FullLoadSize && bbPtr->dbValue.pbAL1 == DOUBLE_BUFFER_NUM)) {
        bbPtr->dbValue.pbAL1 = SINGLE_BUFFER_NUM;
        bbPtr->aL1DBNeedClose = true;
        cinAL1 = min((bbPtr->availableL1Size / static_cast<int64_t>(bbPtr->basicBlockInfoPtr->mIn) /
                      static_cast<int64_t>(bbPtr->fMapDTypeSize) / bbPtr->tilingIns_->innerBatch),
                     kAL1FullLoadSize);
    }
    cinAL1 = FloorAlign(cinAL1, bbPtr->tilingIns_->cubeInfo.k0);
    if (cinAL1 <= 0) {
        OP_LOGE(bbPtr->tilingIns_->nodeType, "L1LoadStrategyMFirst still cannot generate L1 tiling.");
        return false;
    }
    bbPtr->l1TilingParams.kAL1 = cinAL1;

    bool ret = MultiLoadKL1(bbPtr, Kl1MultiAxis::KAL1); // KAL1 set
    if (!ret) {
        return ret;
    }
    if (optiling::conv_ops_tiling::IsDAV920Arch(bbPtr->tilingIns_->platformInfo.npuArch)) {
        uint64_t kdValue = bbPtr->apiStrategy_->CalcDinFromCinSlice(bbPtr->l1TilingParams.kAL1, bbPtr->singleCi1xC0);
        bbPtr->l1TilingParams.kAL1 = bbPtr->apiStrategy_->ValidateAndAdjustKL1(bbPtr->l1TilingParams.kAL1, kdValue);
    }
    return true;
}

bool ConvTilingAlgorithmBBmode::WeightFullLoad::GetL1LoadTilingParams(ConvTilingAlgorithmBBmode* bbPtr)
{
    bbPtr->dbValue.pbBL1 = SINGLE_BUFFER_NUM;
    uint64_t maxNBL1Iter = CeilDiv(bbPtr->basicBlockInfoPtr->nCut, bbPtr->basicBlockInfoPtr->nDim);
    bbPtr->multiN = maxNBL1Iter;
    bbPtr->multiM = 1;
    bbPtr->l1TilingParams.nBL1Value = min(static_cast<uint64_t>(bbPtr->multiN) * bbPtr->basicBlockInfoPtr->nTile,
                                          static_cast<uint64_t>(bbPtr->tilingIns_->shapeInfo.singleCo));
    bbPtr->l1TilingParams.mAL1Value = bbPtr->basicBlockInfoPtr->mIn;
    bool ret = L1LoadStrategyMFirst::GetL1LoadTilingParams(bbPtr);
    return ret;
}

bool ConvTilingAlgorithmBBmode::WeightKFullLoad::GetL1LoadTilingParams(ConvTilingAlgorithmBBmode* bbPtr)
{
    bbPtr->dbValue.pbBL1 = DOUBLE_BUFFER_NUM;
    bbPtr->multiN = 1;
    bbPtr->multiM = 1;
    bbPtr->l1TilingParams.nBL1Value = bbPtr->basicBlockInfoPtr->nTile;
    bbPtr->l1TilingParams.mAL1Value = bbPtr->basicBlockInfoPtr->mIn;
    bool ret = L1LoadStrategyMFirst::GetL1LoadTilingParams(bbPtr);
    return ret;
}

bool ConvTilingAlgorithmBBmode::KAllSplit::MultiLoadKAllSplit(ConvTilingAlgorithmBBmode* bbPtr) const
{
    bool ret = false;
    if (bbPtr->basicBlockInfoPtr->nBl1FullLoad) {
        ret = MultiLoadKAllSplit(bbPtr, Kl1MultiAxis::KBL1);
    } else if (bbPtr->basicBlockInfoPtr->mAl1FullLoad) {
        ret = MultiLoadKAllSplit(bbPtr, Kl1MultiAxis::KAL1);
    } else {
        if (bbPtr->basicBlockInfoPtr->nTile <= bbPtr->basicBlockInfoPtr->mTile) {
            ret = MultiLoadKAllSplit(bbPtr, Kl1MultiAxis::KAL1);
        } else {
            ret = MultiLoadKAllSplit(bbPtr, Kl1MultiAxis::KBL1);
        }
    }
    return ret;
}

bool ConvTilingAlgorithmBBmode::KAllSplit::GetL1LoadTilingParams(ConvTilingAlgorithmBBmode* bbPtr)
{
    bbPtr->multiM = 1;
    bbPtr->multiN = 1;
    bbPtr->l1TilingParams.nBL1Value = bbPtr->basicBlockInfoPtr->nTile;
    bbPtr->l1TilingParams.mAL1Value = bbPtr->basicBlockInfoPtr->mIn;

    if (bbPtr->basicBlockInfoPtr->nBl1FullLoad) {
        bbPtr->basicBlockInfoPtr->iterateMNOrder = IterateMNOrder::ITER_M_FST;
    } else if (bbPtr->basicBlockInfoPtr->mAl1FullLoad) {
        bbPtr->basicBlockInfoPtr->iterateMNOrder = IterateMNOrder::ITER_N_FST;
    }
    uint32_t cinL1 = 0;
    const vector<pair<int, int>> iterations = {
        {DOUBLE_BUFFER_NUM, DOUBLE_BUFFER_NUM},
        {(bbPtr->basicBlockInfoPtr->iterateMNOrder == IterateMNOrder::ITER_M_FST) ? DOUBLE_BUFFER_NUM :
                                                                                    SINGLE_BUFFER_NUM,
         (bbPtr->basicBlockInfoPtr->iterateMNOrder == IterateMNOrder::ITER_M_FST) ? SINGLE_BUFFER_NUM :
                                                                                    DOUBLE_BUFFER_NUM},
        {SINGLE_BUFFER_NUM, SINGLE_BUFFER_NUM}};
    for (uint64_t i = 0; i < iterations.size(); i++) {
        pair<int, int> pbABL1 = iterations[i];
        bbPtr->dbValue.pbAL1 = pbABL1.first;
        bbPtr->dbValue.pbBL1 = pbABL1.second;
        cinL1 = static_cast<uint32_t>(GetCinL1(bbPtr));
        if (cinL1 != 0) {
            break;
        }
    }
    bbPtr->aL1DBNeedClose = bbPtr->dbValue.pbAL1 == SINGLE_BUFFER_NUM ? true : false;
    bbPtr->bL1DBNeedClose = bbPtr->dbValue.pbBL1 == SINGLE_BUFFER_NUM ? true : false;
    if (cinL1 == 0) {
        OP_LOGE(bbPtr->tilingIns_->nodeType, "K all split still cannot generate L1 tiling.");
        return false;
    }
    uint32_t cinTile = cinL1;
    while (cinTile >= bbPtr->tilingIns_->cubeInfo.k0 && (bbPtr->singleCi1xC0 * bbPtr->fullLoadFactor) % cinTile != 0) {
        cinTile -= bbPtr->tilingIns_->cubeInfo.k0;
    }

    if (cinTile <= bbPtr->tilingIns_->cubeInfo.k0 || cinTile == bbPtr->singleCi1xC0 * bbPtr->fullLoadFactor) {
        cinTile = cinL1;
    }
    uint32_t kAL1Max = static_cast<uint32_t>(
        MAX_16_BIT_NUM / (bbPtr->tilingIns_->shapeInfo.singlekH * bbPtr->tilingIns_->shapeInfo.singlekW));
    if (cinTile >= kAL1Max) {
        DivideAndAlign(cinTile, kAL1Max, bbPtr->tilingIns_->cubeInfo.k0);
    }
    bbPtr->l1TilingParams.kAL1 = cinTile;
    bbPtr->l1TilingParams.kBL1 = cinTile * bbPtr->tilingIns_->shapeInfo.orgkH * bbPtr->tilingIns_->shapeInfo.orgkW;
    if (!MultiLoadKAllSplit(bbPtr)) {
        return false;
    }

    if (optiling::conv_ops_tiling::IsDAV920Arch(bbPtr->tilingIns_->platformInfo.npuArch)) {
        return AdjustKL1(bbPtr);
    }
    return true;
}

bool ConvTilingAlgorithmBBmode::KAllSplit::AdjustKL1(ConvTilingAlgorithmBBmode* bbPtr) const
{
    uint64_t kdValue = bbPtr->apiStrategy_->CalcDinFromCinSlice(bbPtr->l1TilingParams.kAL1, bbPtr->singleCi1xC0);
    bbPtr->l1TilingParams.kAL1 = bbPtr->apiStrategy_->ValidateAndAdjustKL1(bbPtr->l1TilingParams.kAL1, kdValue);
    uint64_t kBL1CinSlice = bbPtr->l1TilingParams.kBL1 /
                            (bbPtr->tilingIns_->shapeInfo.orgkH * bbPtr->tilingIns_->shapeInfo.orgkW);
    uint64_t kdValueFromKBL1 = bbPtr->apiStrategy_->CalcDinFromCinSlice(kBL1CinSlice, bbPtr->singleCi1xC0);
    bbPtr->l1TilingParams.kBL1 = bbPtr->apiStrategy_->ValidateAndAdjustKL1(kBL1CinSlice, kdValueFromKBL1) *
                                 (bbPtr->tilingIns_->shapeInfo.orgkH * bbPtr->tilingIns_->shapeInfo.orgkW);
    return true;
}

int64_t ConvTilingAlgorithmBBmode::KAllSplit::GetCinL1(ConvTilingAlgorithmBBmode* bbPtr) const
{
    auto cinL1 = FloorAlign(
        (bbPtr->availableL1Size) /
            (bbPtr->basicBlockInfoPtr->mIn * bbPtr->fMapDTypeSize * bbPtr->dbValue.pbAL1 *
                 bbPtr->tilingIns_->innerBatch +
             bbPtr->basicBlockInfoPtr->nTile * bbPtr->tilingIns_->shapeInfo.orgkH * bbPtr->tilingIns_->shapeInfo.orgkW *
                 bbPtr->weightDTypeSize * bbPtr->dbValue.pbBL1),
        bbPtr->tilingIns_->cubeInfo.k0);
    return cinL1;
}

bool ConvTilingAlgorithmBBmode::KAllSplit::MultiLoadKAllSplit(ConvTilingAlgorithmBBmode* bbPtr,
                                                              const Kl1MultiAxis& kL1MultiAxis) const
{
    bool ret = false;
    if (kL1MultiAxis == Kl1MultiAxis::KAL1) {
        bbPtr->basicBlockInfoPtr->iterateMNOrder = IterateMNOrder::ITER_N_FST;
        auto bL1size = bbPtr->l1TilingParams.kBL1 * bbPtr->l1TilingParams.nBL1Value * bbPtr->dbValue.pbBL1 *
                       bbPtr->weightDTypeSize;
        bbPtr->availableL1Size -= bL1size;
        ret = MultiLoadKL1(bbPtr, Kl1MultiAxis::KAL1);
    } else {
        bbPtr->basicBlockInfoPtr->iterateMNOrder = IterateMNOrder::ITER_M_FST;
        auto aL1size = bbPtr->l1TilingParams.kAL1 * bbPtr->l1TilingParams.mAL1Value * bbPtr->dbValue.pbAL1 *
                       bbPtr->fMapDTypeSize * bbPtr->tilingIns_->innerBatch;
        bbPtr->availableL1Size -= aL1size;
        ret = MultiLoadKL1(bbPtr, Kl1MultiAxis::KBL1);
    }
    return ret;
}

bool ConvTilingAlgorithmBBmode::L1LoadStrategyBase::MultiLoadKL1(ConvTilingAlgorithmBBmode* bbPtr,
                                                                 const Kl1MultiAxis& kL1MultiAxis) const
{
    if (kL1MultiAxis == Kl1MultiAxis::INVALID) {
        return false;
    }

    int64_t l1TempSize = 0;
    if (kL1MultiAxis == Kl1MultiAxis::KAL1) {
        // when you decide to scale KAL1, this means that KAL1 has to have double buffer
        l1TempSize = bbPtr->l1TilingParams.kAL1 * bbPtr->l1TilingParams.mAL1Value * bbPtr->fMapDTypeSize *
                     bbPtr->dbValue.pbAL1 * bbPtr->tilingIns_->innerBatch;
        // DB ON cannot Load in L1
        int64_t maxsingleCoreFmapSize = static_cast<int64_t>(bbPtr->l1TilingParams.mAL1Value * bbPtr->singleCi1xC0 *
                                                             bbPtr->fMapDTypeSize * bbPtr->fullLoadFactor) *
                                        static_cast<int64_t>(bbPtr->tilingIns_->innerBatch);
        if (l1TempSize > min(bbPtr->availableL1Size, bbPtr->fmapL1FullLoadSize)) { // multi is 1
            l1TempSize /= bbPtr->dbValue.pbAL1;
            bbPtr->dbValue.pbAL1 = SINGLE_BUFFER_NUM;

            if (bbPtr->l1TilingParams.kAL1 * bbPtr->tilingIns_->shapeInfo.singlekH *
                    bbPtr->tilingIns_->shapeInfo.singlekW >
                MAX_16_BIT_NUM) {
                OP_LOGD(bbPtr->tilingIns_->nodeType, "Exceeded kStartPos limit");
                return false;
            }
            if (FindMaxProductUnderLimit(l1TempSize, min(bbPtr->availableL1Size, maxsingleCoreFmapSize)) == 0) {
                OP_LOGE(bbPtr->tilingIns_->nodeType, "AL1 space is not enough for Basic Block mode in this case.");
                return false;
            }
            return true;
        }
        // DB ON can Load in L1, try to find the max K value
        uint64_t kAL1Multi = FindMaxProductUnderLimit(l1TempSize, min(bbPtr->availableL1Size, maxsingleCoreFmapSize));
        uint64_t kAL1MultiMax = MAX_16_BIT_NUM /
                                (bbPtr->tilingIns_->shapeInfo.singlekH * bbPtr->tilingIns_->shapeInfo.singlekW);

        bbPtr->l1TilingParams.kAL1 *= min(kAL1Multi, kAL1MultiMax);
    } else if (kL1MultiAxis == Kl1MultiAxis::KBL1) {
        // when you decide to scale KBL1, this means that KBL1 has to have double buffer
        l1TempSize = bbPtr->l1TilingParams.kBL1 * bbPtr->l1TilingParams.nBL1Value * bbPtr->weightDTypeSize *
                     bbPtr->dbValue.pbBL1;
        int64_t maxSingleCoreWeightSize = bbPtr->tilingIns_->shapeInfo.orgkH * bbPtr->tilingIns_->shapeInfo.orgkW *
                                          bbPtr->l1TilingParams.nBL1Value * bbPtr->singleCi1xC0 *
                                          bbPtr->weightDTypeSize * bbPtr->fullLoadFactor;
        // DB ON cannot Load in L1
        if (l1TempSize > min(bbPtr->availableL1Size, bbPtr->weightL1FullLoadSize)) { // multi is 1
            l1TempSize /= bbPtr->dbValue.pbBL1;
            bbPtr->dbValue.pbBL1 = SINGLE_BUFFER_NUM;

            if (FindMaxProductUnderLimit(l1TempSize, min(bbPtr->availableL1Size, maxSingleCoreWeightSize)) == 0) {
                OP_LOGE(bbPtr->tilingIns_->nodeType, "BL1 space is not enough for Basic Block mode in this case.");
                return false;
            }
            return true;
        }
        // DB ON can Load in L1, try to find the max K value
        bbPtr->l1TilingParams.kBL1 *= FindMaxProductUnderLimit(l1TempSize,
                                                               min(bbPtr->availableL1Size, maxSingleCoreWeightSize));
    }
    return true;
}

void ConvTilingAlgorithmBBmode::UpdateFinalFullLoadFlag()
{
    if (!tilingIns_->enableInnerBatch &&
        static_cast<int64_t>(this->multiM * basicBlockInfoPtr->mTile) >= tilingIns_->shapeInfo.singleM) {
        basicBlockInfoPtr->mAl1FullLoad = true;
    } else if (tilingIns_->enableInnerBatch &&
               tilingIns_->innerBatch >= static_cast<uint32_t>(tilingIns_->shapeInfo.singleBatch)) {
        basicBlockInfoPtr->mAl1FullLoad = true;
    } else {
        basicBlockInfoPtr->mAl1FullLoad = false;
    }

    if (static_cast<int64_t>(this->multiN * basicBlockInfoPtr->nTile) >= tilingIns_->shapeInfo.singleCo) {
        basicBlockInfoPtr->nBl1FullLoad = true;
    } else {
        basicBlockInfoPtr->nBl1FullLoad = false;
    }

    if (l1TilingParams.kAL1 == singleCi1xC0 * fullLoadFactor) {
        basicBlockInfoPtr->kAl1FullLoad = true;
    } else {
        basicBlockInfoPtr->kAl1FullLoad = false;
    }

    if (l1TilingParams.kBL1 ==
        singleCi1xC0 * tilingIns_->shapeInfo.orgkH * tilingIns_->shapeInfo.orgkW * fullLoadFactor) {
        basicBlockInfoPtr->kBl1FullLoad = true;
    } else {
        basicBlockInfoPtr->kBl1FullLoad = false;
    }
    UpdateFinalDb();
    UpdateFinalMNOrder();
}

void ConvTilingAlgorithmBBmode::UpdateFinalDb()
{
    const uint8_t pbAL1Temp = (basicBlockInfoPtr->kAl1FullLoad && basicBlockInfoPtr->mAl1FullLoad) ? SINGLE_BUFFER_NUM :
                                                                                                     DOUBLE_BUFFER_NUM;
    const uint8_t pbBL1Temp = (basicBlockInfoPtr->kBl1FullLoad && basicBlockInfoPtr->nBl1FullLoad) ? SINGLE_BUFFER_NUM :
                                                                                                     DOUBLE_BUFFER_NUM;
    dbValue.pbAL1 = aL1DBNeedClose ? SINGLE_BUFFER_NUM : pbAL1Temp;
    dbValue.pbBL1 = bL1DBNeedClose ? SINGLE_BUFFER_NUM : pbBL1Temp;
}

void ConvTilingAlgorithmBBmode::UpdateFinalMNOrder()
{
    if (dbValue.pbAL1 == SINGLE_BUFFER_NUM && dbValue.pbBL1 == DOUBLE_BUFFER_NUM) {
        basicBlockInfoPtr->iterateMNOrder = IterateMNOrder::ITER_N_FST;
    } else if (dbValue.pbAL1 == DOUBLE_BUFFER_NUM && dbValue.pbBL1 == SINGLE_BUFFER_NUM) {
        basicBlockInfoPtr->iterateMNOrder = IterateMNOrder::ITER_M_FST;
    }
    // fmap full load and weight not full load, set iterateMNOrder = IterateMNOrder::ITER_N_FST
    if (basicBlockInfoPtr->kAl1FullLoad && basicBlockInfoPtr->mAl1FullLoad &&
        !(basicBlockInfoPtr->kBl1FullLoad && basicBlockInfoPtr->nBl1FullLoad)) {
        basicBlockInfoPtr->iterateMNOrder = IterateMNOrder::ITER_N_FST;
    }
}

bool ConvTilingAlgorithmBBmode::GetL1TilingParams(ConvTilingAlgorithmBBmode* bbPtr, L1LoadStrategyType strategyType)
{
    auto ret = false;
    ret = l1LoadStrategies[strategyType]->GetL1LoadTilingParams(bbPtr);
    return ret;
}

void ConvTilingAlgorithmBBmode::ScaleBiasUbTilingDecision() const
{
    uint32_t mL0 = basicBlockInfoPtr->mTile;
    uint32_t nL0 = basicBlockInfoPtr->nTile;
    if (mL0 == 0 || nL0 == 0) {
        return;
    }
    if (!tilingIns_->isScaleBiasInUb) {
        return;
    }

    uint64_t mUb = 0;
    std::vector<uint64_t> mUbRange;
    CalcCommFactor(mL0, mL0, mUbRange);

    // output + scale + bias(fp16/bf16) + bias(fp32) <= UB_SIZE
    // output + scale + bias(fp32) <= UB_SIZE
    uint64_t totalScaleBiasTypeSize = tilingIns_->descInfo.biasType.dtype == ConvDtype::FLOAT32 ?
                                          TOTAL_SCALE_BIAS_32_TYPE_SIZE :
                                          TOTAL_SCALE_BIAS_16_TYPE_SIZE;
    for (int32_t i = mUbRange.size() - 1; i >= 0; i--) {
        if (nL0 * (mUbRange[i] * MAX_OUT_TYPE_SIZE + totalScaleBiasTypeSize) <= tilingIns_->platformInfo.ubSize) {
            mUb = mUbRange[i];
            break;
        }
    }

    // if mUb fullload mL0, split mL0 to use more vector
    if (mUb == mL0) {
        mUb = CeilDiv(mL0, tilingIns_->platformInfo.aivPerAic);
    }
    tilingIns_->ubTilingInfo.mUb = mUb;
    tilingIns_->ubTilingInfo.nUb = nL0; // nUb fullload nL0 by default;
}

void ConvTilingAlgorithmBBmode::GetUbTiling() const { ScaleBiasUbTilingDecision(); }
} // namespace conv_tiling_algo_bb
