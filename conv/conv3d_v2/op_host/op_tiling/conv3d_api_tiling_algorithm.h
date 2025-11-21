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
 * \file conv3d_tiling_algorithm.h
 * \brief
 */

#ifndef ASCENDC_TIKCFW_TILING_CONV3D_TILING_ALGORITHM_H
#define ASCENDC_TIKCFW_TILING_CONV3D_TILING_ALGORITHM_H

#include <cstdint>
#include "conv3d_api_tiling_base.h"

namespace Conv3dApiTiling {

enum class L1TilingMode : std::uint8_t {
    FULL_LOAD_BL1 = 0,
    FULL_LOAD_AL1,
    ALL_FULL_LOAD,
    NONE_FULL_LOAD,
    INVALID
};

struct L1TilingFlag {
    L1TilingMode abL1Mode = L1TilingMode::INVALID;
    IterateMNOrder iterateMNOrder = IterateMNOrder::INVALID;
    bool kAL1CanFullLoadFlag = false;
    bool kBL1CanFullLoadFlag = false;
    bool kABL1CanFullLoadFlag = false;
    bool isBiasFullLoad = false;
    bool isWeightBypass = false;
};

struct DoubleBufferTilingValue {
    uint8_t pbAL1 = PING_PONG_AL1;
    uint8_t pbBL1 = INITIAL_PING_PONG_BL1;
    uint8_t pbAL0 = INITIAL_PING_PONG_BL0;
    uint8_t pbBL0 = INITIAL_PING_PONG_BL0;
    uint8_t pbCL0 = INITIAL_PING_PONG_CL0;
    uint8_t pbUB = INITIAL_PING_PONG_UB;
    uint64_t pBufferFlag;
};

struct L1TilingIdx {
    uint64_t kAL1Idx = INITIAL_INDEX;
    uint64_t kBL1Idx = INITIAL_INDEX;
    uint64_t mAL1Idx = INITIAL_INDEX;
    uint64_t hoAL1Idx = INITIAL_INDEX;
    uint64_t woAL1Idx = INITIAL_INDEX;
    uint64_t nBL1Idx = INITIAL_INDEX;

    void SetIdx(uint64_t kAl1, uint64_t kBl1, uint64_t mAL1, uint64_t nBL1)
    {
        kAL1Idx = kAl1;
        kBL1Idx = kBl1;
        mAL1Idx = mAL1;
        nBL1Idx = nBL1;
    }

    void SetIdx(uint64_t kAl1, uint64_t kBl1, uint64_t hoAL1, uint64_t woAL1, uint64_t nBL1)
    {
        kAL1Idx = kAl1;
        kBL1Idx = kBl1;
        hoAL1Idx = hoAL1;
        woAL1Idx = woAL1;
        nBL1Idx = nBL1;
    }
};

struct L0TilingRange {
    std::vector<uint64_t> mL0Range;
    std::vector<uint64_t> hoL0Range;
    std::vector<uint64_t> woL0Range;
    std::vector<uint64_t> kL0Range;
    std::vector<uint64_t> nL0Range;
};

struct L0TilingParams {
    uint64_t mL0 = INITIAL_SIZE;
    uint64_t hoL0 = INITIAL_SIZE;
    uint64_t woL0 = INITIAL_SIZE;
    uint64_t kL0 = INITIAL_SIZE;
    uint64_t nL0 = INITIAL_SIZE;
    uint64_t orgCoAlignN0 = INITIAL_SIZE;
};

struct L0TilingIdx {
    uint64_t mL0Idx = INITIAL_INDEX;
    uint64_t hoL0Idx = INITIAL_INDEX;
    uint64_t woL0Idx = INITIAL_INDEX;
    uint64_t nL0Idx = INITIAL_INDEX;
    uint64_t kL0Idx = INITIAL_INDEX;
};

struct L1TilingRange {
    std::vector<uint64_t> mAL1ValueRange;
    std::vector<uint64_t> hoAL1ValueRange;
    std::vector<uint64_t> woAL1ValueRange;
    std::vector<uint64_t> nBL1ValueRange;
    std::vector<uint64_t> kAL1Range;
    std::vector<uint64_t> kBL1Range;
};

struct L1TilingParams {
    uint64_t kAL1 = INITIAL_SIZE;
    uint64_t kAL1Tail = INITIAL_SIZE;
    uint64_t kBL1 = INITIAL_SIZE;
    uint64_t kBL1Tail = INITIAL_SIZE;
    uint64_t mAL1Value = INITIAL_SIZE;
    uint64_t hoAL1Value = INITIAL_SIZE;
    uint64_t woAL1Value = INITIAL_SIZE;
    uint64_t nBL1Value = INITIAL_SIZE;
};

struct L1TilingCalc {
    uint64_t ci0HkWk = INITIAL_SIZE;
    uint64_t alignCinKhKwKd = INITIAL_SIZE;
    uint64_t fmapFullLoadL1Size = INITIAL_SIZE;
    uint64_t weightFullLoadL1Size = INITIAL_SIZE;
    uint64_t fmapKL1FullLoadSize = INITIAL_SIZE;
    uint64_t weightKL1FullLoadSize = INITIAL_SIZE;
    uint64_t fmapMinLoadL1Size = INITIAL_SIZE;
    uint64_t weightMinLoadL1Size = INITIAL_SIZE;
    uint64_t biasMinLoadL1Size = INITIAL_SIZE;
};

class Conv3dTilingAlgorithm {
public:
    explicit Conv3dTilingAlgorithm(Conv3dTilingBase *tilingIns);
    virtual ~Conv3dTilingAlgorithm()
    {
        tilingIns_ = nullptr;
    }
    int64_t Process();

protected:
    // weight bypass
    virtual bool CoreL1TilingMinWeightBypass() const;
    virtual bool NoneKABL1FullLoadWeightBypass() const;

    virtual uint64_t L1NoFullLoadFmapSize() const;
    // L1 Tiling
    int64_t GetL1Tiling();
    int64_t PreProcessingL1Tiling();
    virtual bool CheckL1Buffer() const;
    virtual uint64_t CalcCurrFmapL1Size() const;
    virtual int64_t InitCalcL1Params();
    virtual int64_t InitCalcL1ParamsForFmap();
    int64_t InitCalcL1ParamsForWeight();
    uint64_t InferHiL1(uint64_t inputHoL1, uint64_t hi) const;
    uint64_t InferWiL1(uint64_t inputWoL1, uint64_t wi) const;
    virtual void GetL1TilingRange();
    virtual void GetL1TilingRangeForM();
    void InitL1TiLing();
    virtual void InitL1TiLingMap();
    virtual void InitABL1TilingMode();
    int64_t CoreL1TilingDecision();
    virtual int64_t ProcessAllL1FullLoad();
    virtual int64_t ProcessFmapL1FullLoad();
    int64_t ProcessWeightL1FullLoad();
    int64_t ProcessNoneL1FullLoad();
    virtual void MAL1IdxIter();
    void FmapL1FullLoadIter();
    void WeightL1FullLoadIter();
    void InitKL1LoadFlag();
    int64_t L1NoFullLoadIter();
    void KAL1FullLoadIter();
    void KBL1FullLoadIter();
    virtual int64_t KABL1FullLoadIter();
    virtual uint64_t KABL1FullLoadIterN();
    uint64_t KABL1FullLoadIterM();
    void NoneKABL1FullLoadIter();
    void BiasL1TilingDecision();
    bool FixL0PingpongDecision();
    virtual void GetKL0TilingDecision();
    virtual void WeightBypassDecision();
    virtual void SetKAL1KBL1TailRes();
    virtual void SetMAL1NBL1ValueAndMode();
    void SetL1TilingRes();

     // L0 tiling
    int64_t GetL0Tiling();
    virtual void InitPingPong();
    virtual void GetL0TilingRange();
    virtual void L0TilingDecision();
    bool CheckL0Buffer(uint64_t currmL0, uint64_t currkL0, uint64_t currnL0) const;
    uint64_t CalcAL0Size(uint64_t currmL0, uint64_t currkL0) const;
    uint64_t CalcBL0Size(uint64_t currkL0, uint64_t currnL0) const;
    uint64_t CalcCL0Size(uint64_t currmL0, uint64_t currnL0) const;
    virtual uint64_t CalcL1SizeForL0Tiling(uint64_t currmL0, uint64_t currnL0) const;
    virtual uint64_t CalcBTSize(uint64_t currnL0) const;
    void CheckL0CDoubleBuffer();

    // UB tiling
    void GetVecTiling() const;

    // set db
    void SetPBufferFlag();

protected:
    Conv3dTilingBase* tilingIns_ = nullptr;

    L0TilingRange l0TilingRange;
    L0TilingParams l0TilingParams;
    L0TilingIdx l0TilingIdx;

    L1TilingRange l1TilingRange;
    L1TilingParams l1TilingParams;
    L1TilingCalc l1TilingCalc;
    L1TilingIdx l1TilingIdx;
    L1TilingFlag l1TilingFlag;
    DoubleBufferTilingValue doubleBufferValue;
    std::map<L1TilingMode, L1TilingIdx> l1TilingInitMap;

    uint64_t fMapDTypeSize = {0};
    uint64_t weightDTypeSize = {0};
    uint64_t biasDTypeSize = {0};
    uint64_t ubInDTypeSize = {0};
    uint64_t scaleDTypeSize = {0};
    uint64_t outputDTypeSize = {0};
};

} // namespace Conv3dApiTiling

#endif // ASCENDC_TIKCFW_TILING_CONV3D_TILING_ALGORITHM_H