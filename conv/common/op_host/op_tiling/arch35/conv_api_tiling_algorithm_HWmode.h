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
 * \file conv_api_tiling_algorithm_HWmode.h
 * \brief
 */

#ifndef ASCENDC_TILING_CONV_API_TILING_ALGORITHM_HWMODE_H
#define ASCENDC_TILING_CONV_API_TILING_ALGORITHM_HWMODE_H

#include <cstdint>
#include <unordered_map>
#include "conv_api_tiling_algorithm_base.h"
#include "conv_api_tiling_base.h"

namespace conv_tiling_algo_hw {
using namespace conv_tiling;

struct L1TilingParams {
    uint64_t kAL1 = 0; // without khkw
    uint64_t kBL1 = 0;
    uint64_t hoAL1 = 0;
    uint64_t woAL1 = 0;
    uint64_t nBL1 = 0;

    uint64_t khL1 = 0; // dma kernel split
    uint64_t kwL1 = 0;
};

struct L0TilingParams {
    uint64_t hoL0Index = 0;
    uint64_t woL0Index = 0;
    uint64_t kL0Index = 0;
    uint64_t nL0Index = 0;
 
    uint64_t hoL0 = 0;
    uint64_t woL0 = 0;
    uint64_t kL0 = 0;
    uint64_t nL0 = 0;
};

struct L1TilingRange {
    std::vector<uint64_t> hoAL1Range;
    std::vector<uint64_t> woAL1Range;
    std::vector<uint64_t> nBL1Range;
    std::vector<uint64_t> kAL1Range;
    std::vector<uint64_t> kBL1Range;

    std::vector<uint64_t> hoAL1StrictRange;
    std::vector<uint64_t> woAL1StrictRange;
    std::vector<uint64_t> nBL1StrictRange;
    std::vector<uint64_t> kAL1StrictRange;
    std::vector<uint64_t> kBL1StrictRange;
};

struct L0TilingRange {
    std::vector<uint64_t> hoL0Range;
    std::vector<uint64_t> woL0Range;
    std::vector<uint64_t> kL0Range;
    std::vector<uint64_t> nL0Range;
};

struct L1TilingCalc {
    uint64_t aL1MaxSize = 0;
    uint64_t bL1MaxSize = 0;
    uint64_t biasL1MaxSize = 0;
    uint64_t fixpParamsL1MaxSize = 0;
    uint64_t hiAL1Min = 0;
    uint64_t wiAL1Min = 0;
    uint64_t hoAL1Min = 0;
    uint64_t woAL1Min = 0;
    uint64_t aL1Minsize = 0;
    uint64_t bL1Minsize = 0;
    uint64_t biasL1MinSize = 0;
    uint64_t fixpParamsL1MinSize = 0;
    uint64_t kAL1MaxSize = 0;
    uint64_t kBL1MaxSize = 0;
    uint64_t woAL1MaxSize = 0;
    uint64_t nBL1MaxSize = 0;
    uint64_t ci0HkWk = 0;
    uint64_t c04KSizeAlign = 0;
    uint64_t ubMaxSize = 0;
    uint64_t khL1 = 0;
    uint64_t kwL1 = 0;
    uint64_t ci0KhL1KwL1 = 0;
};

struct L1TilingInit {
    uint64_t kAL1Init = 0;
    uint64_t kBL1Init = 0;
    uint64_t hoAL1Init = 0;
    uint64_t woAL1Init = 0;
    uint64_t nBL1Init = 0;
};

struct L1TilingFlag {
    L1TilingMode abL1Mode = L1TilingMode::INVALID;
    IterateMNOrder iterateMNOrder = IterateMNOrder::INVALID;
    bool isBiasFullLoad = false;
    bool isFixpParamsFullLoad = false;
    bool isWoAL1FullLoad = false;
    bool useWoAL1SpareRange = false;
    bool usehoAL1SpareRange = false;
    bool isWoL1MustFullLoadFlagC04 = true;
    bool isKernelSplitFlag = false;
};

class ConvTilingAlgorithmHWmode : public ConvTilingAlgorithmBase {
public:
    explicit ConvTilingAlgorithmHWmode(ConvTilingBase *tilingIns) : ConvTilingAlgorithmBase(tilingIns)
    {
        this->minBurstNum = MIN_BURST_SIZE / DTYPE_SIZE_TAB.at(tilingIns_->descInfo.fMapType.dtype);
    };
    ~ConvTilingAlgorithmHWmode() override {};
    int64_t Process() override;
    void SetL1TilingRes() override;
    void SetL0TilingRes() override;
private:
    // L1 tiling
    void InitPingPong();
    uint64_t CalcCurL1Size(const L1TilingParams &inputTiling, uint64_t biasFixpParamNB);
    uint64_t CalcCurUbSize(uint64_t hoL1, uint64_t woL1, uint64_t curKh, uint64_t curKw);
    int64_t GetL1Tiling();
    void InitCalcL1Params();
    void InitCalcL1ParamsC04Mode();
    void GetL1TilingRange();
    void InitL1TiLing();
    void CoreL1TilingDecision();
    void CoreL1TilingNoneFullLoadDecision();
    bool IsAL1FullFload(const uint64_t currHoAL1, const uint64_t currWoAL1, const uint64_t currKAL1) const;
    bool IsBL1FullFload(const uint64_t currNBL1, const uint64_t currKBL1) const;
    void UpdateAL1DoubelBuffer();
    void UpdateBiasFixpParamsL1Fullload();
    void UpdateAL1DoubleBufferFirst();
    void UpdateAL1DoubleBufferSecond();
    int64_t CheckMinL1Tiling();
    uint64_t GetFixpParamsL1FullLoadSize() const;
    uint64_t GetFixpParamsL1MinLoadSize() const;
    int64_t InferWiL1(uint64_t inputWoL1, int64_t wi) const;

    bool CheckBL1FullLoad();
    bool CheckHoWoL1FullLoad();
    bool CheckAL1FullLoad();
    bool CheckABL1FullLoad();
    bool CheckHinOverLoad3dv2Limit(uint64_t hout) const;
    bool CheckWinOverLoad3dv2Limit(uint64_t wout) const;
    bool IsBL1FullLoadFirst();
    bool IsKBL1FullLoadFirst();
    void KAL1FullLoadFirst(uint64_t &tmpKAL1, uint64_t &tmpKBL1, uint64_t &tmpHoAL1,
                           uint64_t &tmpWoAL1, uint64_t &tmpNBL1);
    void KBL1FullLoadFirst(uint64_t &tmpKAL1, uint64_t &tmpKBL1, uint64_t &tmpHoAL1,
                           uint64_t &tmpWoAL1, uint64_t &tmpNBL1);
    bool CheckKL1FullLoad(uint64_t kAL1, uint64_t kBL1);
    bool CheckWoL0FullLoadC04Mode() const;

    void RestrictRange(const std::vector<uint64_t> &inputRange, std::vector<uint64_t> &strictRange,
                       uint64_t restriction) const;
    void InitABL1TilingMode();

    void IterKAL1(uint64_t &kAL1, uint64_t kBL1, uint64_t hoAL1, uint64_t woAL1, uint64_t nBL1);
    void IterKBL1(uint64_t kAL1, uint64_t &kBL1, uint64_t hoAL1, uint64_t woAL1, uint64_t nBL1);
    void IterNBL1(uint64_t kAL1, uint64_t kBL1, uint64_t hoAL1, uint64_t woAL1, uint64_t &nBL1);
    void IterHoAL1(uint64_t &hoAL1Res, const std::vector<uint64_t> &inputHoAL1Range, L1TilingParams &inputTiling);
    void IterWoAL1(uint64_t &woAL1Res, const std::vector<uint64_t> &inputWoAL1Range, L1TilingParams &inputTiling);
    void IterHoWoAL1(uint64_t kAL1, uint64_t kBL1, uint64_t &hoAL1, uint64_t &woAL1, uint64_t nBL1);
    void IterKABL1(uint64_t& tmpKAL1, uint64_t& tmpKBL1, uint64_t tmpHoAL1, uint64_t tmpWoAL1, uint64_t tmpNBL1);
    void GetCaseStatus();
    void GetHoWoNL0Tiling();
    void GetHoWoNL0TilingRange();
    void HoWoNL0TilingDecision();
    void UpdateL0TilingParamsIndex(bool& indexOverflowTag);
    void SetHoL0ForC04();
    void GetKL0Tiling();
    uint64_t CalcL1SizeForL0Tiling(uint64_t currHoL0, uint64_t currWoL0, uint64_t currNL0);
     // L0 tiling
    void UpdateNL0();
    void UpdateHoL0WoL0(uint64_t& value, uint64_t& index, std::vector<uint64_t>& range, uint64_t valueMax,
                        uint64_t anotherValue);
    void L0TilingRest(bool kFullLoadFlag);
    void CheckL0DoubleBuffer();
    void Printl1TilingCalc();
    void GetDmaL1Tiling();
    void ScaleBiasUbTilingDecision();
    void GetUbTiling();
    L1TilingCalc l1TilingCalc;
    L1TilingInit l1TilingInit;
    unordered_map<L1TilingMode, L1TilingInit> l1TilingInitMap;
    L1TilingRange l1TilingRange;
    L1TilingParams l1Params;
    L1TilingFlag l1Flags;
    L0TilingRange l0TilingRange;
    L0TilingParams l0Params;

    uint32_t minBurstNum = 0;
};

} // namespace conv_tiling_algo_hw

#endif // ASCENDC_TILING_CONV_API_TILING_ALGORITHM_HWMODE_H