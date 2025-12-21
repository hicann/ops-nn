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
 * \file conv_api_tiling_algorithm_Mmode.h
 * \brief
 */

#ifndef ASCENDC_TIKCFW_TILING_CONV_API_TILING_ALGORITHM_MMODE_H
#define ASCENDC_TIKCFW_TILING_CONV_API_TILING_ALGORITHM_MMODE_H

#include <cstdint>
#include <unordered_map>
#include "conv_api_tiling_algorithm_base.h"
#include "conv_api_tiling_base.h"
using namespace conv_tiling;
namespace conv_tiling_algo_m {
struct L1TilingFlag {
    L1TilingMode abL1Mode = L1TilingMode::INVALID;
    IterateMNOrder iterateMNOrder = IterateMNOrder::INVALID;
    bool kAL1CanFullLoadFlag = false;
    bool kBL1CanFullLoadFlag = false;
    bool kABL1CanFullLoadFlag = false;
    bool isBiasFullLoad = false;
    bool isFixpFullLoad = false;
};

struct L1TilingIdx {
    uint64_t kAL1Idx = 0;
    uint64_t kBL1Idx = 0;
    uint64_t mAL1Idx = 0;
    uint64_t nBL1Idx = 0;

    void SetIdx(uint64_t kAl1, uint64_t kBl1, uint64_t mAL1, uint64_t nBL1)
    {
        kAL1Idx = kAl1;
        kBL1Idx = kBl1;
        mAL1Idx = mAL1;
        nBL1Idx = nBL1;
    }
};

struct L0TilingRange {
    std::vector<uint64_t> mL0Range;
    std::vector<uint64_t> kL0Range;
    std::vector<uint64_t> nL0Range;
};

struct L0TilingParams {
    uint64_t mL0 = 0;
    uint64_t kL0 = 0;
    uint64_t nL0 = 0;
};

struct L0TilingIdx {
    uint64_t mL0Idx = 0;
    uint64_t nL0Idx = 0;
    uint64_t kL0Idx = 0;
};

struct L1TilingRange {
    std::vector<uint64_t> mAL1ValueRange;
    std::vector<uint64_t> nBL1ValueRange;
    std::vector<uint64_t> kAL1Range;
    std::vector<uint64_t> kBL1Range;
};

struct L1TilingParams {
    uint64_t kAL1 = 0;
    uint64_t kBL1 = 0;
    uint64_t mAL1Value = 0;
    uint64_t nBL1Value = 0;
};

struct L1TilingCalc {
    uint64_t ci0HkWk = 0;
    uint64_t c04KSizeAlign = 0;
    uint64_t fmapFullLoadL1Size = 0;
    uint64_t weightFullLoadL1Size = 0;
    uint64_t fmapKL1FullLoadSize = 0;
    uint64_t weightKL1FullLoadSize = 0;
    uint64_t fmapMinLoadL1Size = 0;
    uint64_t weightMinLoadL1Size = 0;
    uint64_t biasMinLoadL1Size = 0;
    uint64_t fixpMinLoadL1Size = 0;
    uint64_t kAL1FullLoadSize = 0;
    uint64_t kBL1FullLoadSize = 0;
};

class ConvTilingAlgorithmMmode : public ConvTilingAlgorithmBase {
public:
    explicit ConvTilingAlgorithmMmode(ConvTilingBase *tilingIns) : ConvTilingAlgorithmBase(tilingIns) {};
    ~ConvTilingAlgorithmMmode() override {};
    int64_t Process() override;
    void SetL1TilingRes() override;
    void SetL0TilingRes() override;
private:
    // L1 Tiling
    int64_t GetL1Tiling();
    bool CheckMinL1Tiling();
    int64_t PreProcessingL1Tiling();
    bool CheckL1Buffer();
    int64_t InitCalcL1Params();
    void GetL1TilingRange();
    void InitL1TiLing();
    void InitABL1TilingMode();
    void CoreL1TilingDecision();
    void FmapL1FullLoadIter();
    void WeightL1FullLoadIter();
    void InitKL1LoadFlag();
    void L1NoFullLoadIter();
    void KAL1FullLoadIter();
    void KBL1FullLoadIter();
    uint64_t KABL1FullLoadIterN();
    uint64_t KABL1FullLoadIterM();
    void NoneKABL1FullLoadIter();
    void BiasL1TilingDecision();
    void GetKL0TilingDecision();
    void UpdateL1DoubelBuffer();
    void CalFormulaicInnerBatch();
     // L0 tiling
    int64_t GetL0Tiling();
    void InitPingPong();
    void GetL0TilingRange();
    void L0TilingDecision();
    uint64_t CalcL1SizeForL0Tiling(uint64_t currmL0, uint64_t currnL0, uint64_t reserveWeightSize = 0);
    void CheckL0CDoubleBuffer();
    bool IsKAL1orKBL1FullLoad();
    bool IsKAL1andKBL1FullLoad();

    L0TilingRange l0TilingRange;
    L0TilingParams l0TilingParams;
    L0TilingIdx l0TilingIdx;

    L1TilingRange l1TilingRange;
    L1TilingParams l1TilingParams;
    L1TilingCalc l1TilingCalc;
    L1TilingIdx l1TilingIdx;
    L1TilingFlag l1TilingFlag;
    unordered_map<L1TilingMode, L1TilingIdx> l1TilingInitMap;
};
} // namespace conv_tiling_algo_m

#endif