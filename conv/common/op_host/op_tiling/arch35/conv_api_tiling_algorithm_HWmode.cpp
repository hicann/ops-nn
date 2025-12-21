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

#include <cstdint>
#include "conv_api_tiling_algorithm_HWmode.h"

using namespace std;

namespace conv_tiling_algo_hw {
void ConvTilingAlgorithmHWmode::InitPingPong()
{
    this->dbValue.pbAL1 = tilingIns_->isDmaFlag ? 1 : DOUBLE_BUFFER_NUM;
    this->dbValue.pbBL1 = tilingIns_->isDmaFlag ? 1 : DOUBLE_BUFFER_NUM;
    this->dbValue.pbAL0 = DOUBLE_BUFFER_NUM;
    this->dbValue.pbBL0 = DOUBLE_BUFFER_NUM;
    this->dbValue.pbCL0 = 1;

    L1TilingParams inputTiling;
    inputTiling.kAL1 = tilingIns_->cubeInfo.k0;
    inputTiling.kBL1 = tilingIns_->shapeInfo.singlekH *
        tilingIns_->shapeInfo.singlekW * tilingIns_->cubeInfo.k0;
    inputTiling.nBL1 = tilingIns_->cubeInfo.n0;
    inputTiling.hoAL1 = 1;
    inputTiling.woAL1 = tilingIns_->cubeInfo.m0;
    if (CalcCurL1Size(inputTiling, tilingIns_->cubeInfo.n0) > tilingIns_->platformInfo.l1Size) {
        this->dbValue.pbAL1 = 1;
        this->dbValue.pbBL1 = 1;
    }
}

int64_t ConvTilingAlgorithmHWmode::Process()
{
    InitPingPong();
    GetHoWoNL0Tiling();
    GetDmaL1Tiling();
    if (GetL1Tiling() == -1) {
        return -1;
    }
    GetKL0Tiling();
    SetL0TilingRes();
    CheckL0DoubleBuffer();
    SetPBufferRes();
    return 0;
}

void ConvTilingAlgorithmHWmode::GetKL0Tiling()
{
    if (tilingIns_->isDmaFlag) {
        CalcCommFactorOfTwoNum(l1Params.kAL1, l1Params.kBL1,  l0TilingRange.kL0Range);
    } else {
        uint64_t kAL1WithKhKw = l1Params.kAL1 * tilingIns_->shapeInfo.singlekH * tilingIns_->shapeInfo.singlekW;
        CalcCommFactorOfTwoNum(kAL1WithKhKw, l1Params.kBL1,  l0TilingRange.kL0Range);
    }
    uint64_t validKL0 = 0;
    while (l0Params.kL0Index < l0TilingRange.kL0Range.size()) {
        if (l0TilingRange.kL0Range[l0Params.kL0Index] % tilingIns_->cubeInfo.k0 == 0) {
            uint64_t tmpKL0 = l0TilingRange.kL0Range[l0Params.kL0Index];
            if (CalcAL0Size(l0Params.hoL0 * l0Params.woL0, tmpKL0) <= tilingIns_->platformInfo.l0ASize &&
                CalcBL0Size(tmpKL0, l0Params.nL0) <= tilingIns_->platformInfo.l0BSize) {
                validKL0 = tmpKL0;
                l0Params.kL0Index++;
            } else {
                break;
            }
        } else {
            l0Params.kL0Index++;
        }
    }
    l0Params.kL0 = validKL0 > tilingIns_->cubeInfo.k0 ? validKL0 : tilingIns_->cubeInfo.k0;
    // try to close double buffer to make KL0 full load
    bool bMNFullLoadRet = static_cast<uint64_t>(tilingIns_->shapeInfo.singleHo) <= l0Params.hoL0 &&
        static_cast<uint64_t>(tilingIns_->shapeInfo.singleWo) <= l0Params.woL0 &&
        tilingIns_->shapeInfo.singleCo1 * static_cast<uint64_t>(tilingIns_->cubeInfo.n0) <= l0Params.nL0;
    bool bKFullLoadRet = l1Params.kBL1 == this->l1TilingCalc.kBL1MaxSize &&
        l1Params.kAL1 == this->l1TilingCalc.kAL1MaxSize;
    bool bCheckL0SizeRet =
        (CalcAL0Size(l0Params.hoL0 * l0Params.woL0, this->l1TilingCalc.kBL1MaxSize) / this->dbValue.pbAL0 <=
            tilingIns_->platformInfo.l0ASize) &&
        (CalcBL0Size(this->l1TilingCalc.kBL1MaxSize, l0Params.nL0) / this->dbValue.pbBL0 <=
            tilingIns_->platformInfo.l0BSize);
    if (bMNFullLoadRet && bKFullLoadRet && bCheckL0SizeRet) {
        l0Params.kL0 = this->l1TilingCalc.kBL1MaxSize;
    }
}

void ConvTilingAlgorithmHWmode::GetDmaL1Tiling()
{
    if (!tilingIns_->isDmaFlag) {
        return;
    }
    std::vector<uint64_t> kwL1Range;
    std::vector<uint64_t> khL1Range;
    CalcCommFactor(tilingIns_->shapeInfo.singlekW, tilingIns_->shapeInfo.singlekW, kwL1Range);
    CalcCommFactor(tilingIns_->shapeInfo.singlekH, tilingIns_->shapeInfo.singlekH, khL1Range);
    PrintRanges(khL1Range, "khL1Range");
    PrintRanges(kwL1Range, "kwL1Range");
    uint32_t kwL1Idx = 0;
    uint32_t khL1Idx = 0;
    L1TilingParams inputTiling;
    bool isExceedL1Size = false;
    uint64_t curKwL1Value = 0;
    uint64_t curKhL1Value = 0;
    inputTiling.hoAL1 = l0Params.hoL0;
    inputTiling.woAL1 = l0Params.woL0;
    inputTiling.nBL1 = l0Params.nL0;
    while (kwL1Idx < kwL1Range.size()) {
        curKwL1Value = kwL1Range[kwL1Idx];
        inputTiling.kAL1 = tilingIns_->cubeInfo.k0 * curKwL1Value;
        inputTiling.kBL1 = inputTiling.kAL1;
        isExceedL1Size = CalcCurL1Size(inputTiling, inputTiling.nBL1) > tilingIns_->platformInfo.l1Size;
        if (kwL1Idx > 0 && isExceedL1Size) {
            curKwL1Value = kwL1Range[kwL1Idx - 1];
            break;
        }
        if (kwL1Idx == 0 && isExceedL1Size) {
            this->l1TilingCalc.kwL1 = 0;
            return;
        }
        kwL1Idx++;
    }
    while (khL1Idx < khL1Range.size()) {
        curKhL1Value = khL1Range[khL1Idx];
        inputTiling.kAL1 = tilingIns_->cubeInfo.k0 * curKwL1Value * curKhL1Value;
        inputTiling.kBL1 = inputTiling.kAL1;
        isExceedL1Size = CalcCurL1Size(inputTiling, inputTiling.nBL1) > tilingIns_->platformInfo.l1Size;
        if (khL1Idx > 0 && isExceedL1Size) {
            curKhL1Value = khL1Range[khL1Idx - 1];
            break;
        }
        if (khL1Idx == 0 && isExceedL1Size) {
            this->l1TilingCalc.khL1 = 0;
            return;
        }
        khL1Idx++;
    }
    this->l1TilingCalc.kwL1 = curKwL1Value;
    this->l1TilingCalc.khL1 = curKhL1Value;
    this->l1Flags.isKernelSplitFlag = this->l1TilingCalc.kwL1 < static_cast<uint64_t>(tilingIns_->shapeInfo.singlekW) ||
                                      this->l1TilingCalc.khL1 < static_cast<uint64_t>(tilingIns_->shapeInfo.singlekH);
}

void ConvTilingAlgorithmHWmode::GetHoWoNL0Tiling()
{
    GetHoWoNL0TilingRange();
    HoWoNL0TilingDecision();
    // set L0C double buffer
    this->dbValue.pbCL0 = DOUBLE_BUFFER_NUM;
    if (CalcCL0Size(l0Params.hoL0 * l0Params.woL0, l0Params.nL0) > tilingIns_->platformInfo.l0CSize) {
        this->dbValue.pbCL0 = 1;
    }
}

void ConvTilingAlgorithmHWmode::GetHoWoNL0TilingRange()
{
    uint64_t mL0Max = min(tilingIns_->platformInfo.l0ASize /
        (tilingIns_->cubeInfo.k0 * this->dbValue.pbAL0 * fMapDTypeSize),
        tilingIns_->platformInfo.l0CSize / (tilingIns_->cubeInfo.n0 *
        this->dbValue.pbCL0 * DTYPE_SIZE_TAB.at(tilingIns_->cubeInfo.madType)));
    // Get hoL0 range
    uint64_t hoL0Max = min(mL0Max / static_cast<uint64_t>(tilingIns_->shapeInfo.singleWo),
                        static_cast<uint64_t>(tilingIns_->shapeInfo.singleHo));
    hoL0Max = hoL0Max > 1 ? hoL0Max : 1;
    CalcCommFactorWithPowerOfTwo(tilingIns_->shapeInfo.singleHo, hoL0Max, l0TilingRange.hoL0Range);

    // Get woL0 range
    uint64_t woL0Max = min(mL0Max, static_cast<uint64_t>(tilingIns_->shapeInfo.singleWo));
    CalcCommFactorWithPowerOfTwo(CeilDiv(tilingIns_->shapeInfo.singleWo, tilingIns_->cubeInfo.m0),
                                CeilDiv(woL0Max, tilingIns_->cubeInfo.m0), l0TilingRange.woL0Range);
    VectorElementMultip(l0TilingRange.woL0Range, tilingIns_->cubeInfo.m0);

    // Get nL0 range
    uint64_t nL0Max = min(tilingIns_->platformInfo.l0BSize /
        (tilingIns_->cubeInfo.k0 * this->dbValue.pbBL0 * weightDTypeSize),
        tilingIns_->platformInfo.l0CSize / (tilingIns_->cubeInfo.m0 *
        this->dbValue.pbCL0 * DTYPE_SIZE_TAB.at(tilingIns_->cubeInfo.madType)));
    CalcCommFactorWithPowerOfTwo(tilingIns_->shapeInfo.singleCo1, nL0Max / tilingIns_->cubeInfo.n0,
        l0TilingRange.nL0Range);
    VectorElementMultip(l0TilingRange.nL0Range, tilingIns_->cubeInfo.n0);
}

void ConvTilingAlgorithmHWmode::SetHoL0ForC04()
{
    // In C04 mode, if orgHi > 1
    // wiL1 = orgWi AND woL1 = orgWo
    // if orgWo % woL0 != 0, hoL0 must be equal to 1
    if (tilingIns_->isC04Flag && tilingIns_->shapeInfo.orgHi > 1) {
        if (tilingIns_->shapeInfo.orgWo % l0Params.woL0 != 0) {
            l0Params.hoL0 = 1;
        }    
    }
}

void ConvTilingAlgorithmHWmode::UpdateL0TilingParamsIndex(bool& indexOverflowTag)
{
    if (l0Params.nL0 < l0Params.hoL0 * l0Params.woL0) {
        if (l0Params.nL0Index < l0TilingRange.nL0Range.size() - 1) {
            l0Params.nL0Index++;
        } else if (l0Params.woL0Index < l0TilingRange.woL0Range.size() - 1) {
            l0Params.woL0Index++;
        } else if (l0Params.hoL0Index < l0TilingRange.hoL0Range.size() - 1) {
            l0Params.hoL0Index++;
        } else {
            indexOverflowTag = true;
        }
    } else { // nL0 >= hoL0 * woL0
        if (l0Params.woL0Index < l0TilingRange.woL0Range.size() - 1) {
            l0Params.woL0Index++;
        } else if (l0Params.hoL0Index < l0TilingRange.hoL0Range.size() - 1) {
            l0Params.hoL0Index++;
        } else if (l0Params.nL0Index < l0TilingRange.nL0Range.size() - 1) {
            l0Params.nL0Index++;
        } else {
            indexOverflowTag = true;
        }
    }
}

void ConvTilingAlgorithmHWmode::HoWoNL0TilingDecision()
{
    l0Params.hoL0 = l0TilingRange.hoL0Range[l0Params.hoL0Index];
    l0Params.woL0 = l0TilingRange.woL0Range[l0Params.woL0Index];
    l0Params.nL0 = l0TilingRange.nL0Range[l0Params.nL0Index];
    l0Params.kL0 = tilingIns_->cubeInfo.k0;
    bool l0BufferNotOverflowFlag = CheckL0Buffer(l0Params.hoL0 * l0Params.woL0, l0Params.kL0, l0Params.nL0);
    uint64_t usedL1Size = CalcL1SizeForL0Tiling(l0Params.hoL0, l0Params.woL0, l0Params.nL0);
    uint64_t usedBTSize = CalcBTSize(l0Params.nL0);
    uint64_t usedFBSize = CalcFBSize(l0Params.nL0);
    uint64_t usedUBSize = CalcCurUbSize(l0Params.hoL0, l0Params.woL0, 1, 1);
    bool indexOverflow = false;
    L0TilingParams preL0Params; // Records the results before the cursor is updated
    while (l0BufferNotOverflowFlag && usedL1Size <= tilingIns_->platformInfo.l1Size &&
        usedBTSize <= tilingIns_->platformInfo.btSize && usedFBSize <= tilingIns_->platformInfo.fbSize &&
        usedUBSize <= tilingIns_->platformInfo.ubSize) {
        preL0Params = l0Params;
        UpdateL0TilingParamsIndex(indexOverflow);
        if (indexOverflow) {
            break;
        }
        l0Params.nL0 = l0TilingRange.nL0Range[l0Params.nL0Index];
        l0Params.woL0 = l0TilingRange.woL0Range[l0Params.woL0Index];
        l0Params.hoL0 = l0TilingRange.hoL0Range[l0Params.hoL0Index];
 
        l0BufferNotOverflowFlag = CheckL0Buffer(l0Params.hoL0 * l0Params.woL0, l0Params.kL0, l0Params.nL0);
        usedL1Size = CalcL1SizeForL0Tiling(l0Params.hoL0, l0Params.woL0, l0Params.nL0);
        usedBTSize = CalcBTSize(l0Params.nL0);
        usedFBSize = CalcFBSize(l0Params.nL0);
        usedUBSize = CalcCurUbSize(l0Params.hoL0, l0Params.woL0, 1, 1);
    }
    if (!indexOverflow) { // The loop ends because the space limit is exceeded
        l0Params = preL0Params;
        l0Params.nL0 = l0TilingRange.nL0Range[preL0Params.nL0Index];
        l0Params.woL0 = l0TilingRange.woL0Range[preL0Params.woL0Index];
        l0Params.hoL0 = l0TilingRange.hoL0Range[preL0Params.hoL0Index];
    } // else --- The loop ends because the index is completely traversed. l0Params updated in the loop.
}

uint64_t ConvTilingAlgorithmHWmode::CalcL1SizeForL0Tiling(uint64_t currHoL0, uint64_t currWoL0, uint64_t currNL0)
{
    uint64_t aL1Size = 0;
    uint64_t wiL1 = 0;
    uint64_t hiL1 = 0;
    if (tilingIns_->isDmaFlag) {
        aL1Size = tilingIns_->cubeInfo.k0 * currWoL0 * currHoL0 * this->fMapDTypeSize;
    } else {
        wiL1 = InferWiL1(currWoL0, tilingIns_->shapeInfo.orgWi);
        hiL1 = InferHiL1(currHoL0, tilingIns_->shapeInfo.orgHi);
        aL1Size = tilingIns_->cubeInfo.k0 * wiL1 * hiL1 * this->fMapDTypeSize;
    }

    uint64_t bL1Size = tilingIns_->isDmaFlag ? tilingIns_->cubeInfo.k0 * currNL0 * this->weightDTypeSize :
            tilingIns_->shapeInfo.orgkH * tilingIns_->shapeInfo.orgkW * tilingIns_->cubeInfo.k0 * currNL0 *
                this->weightDTypeSize;
    uint64_t biasSize = tilingIns_->hasBias ? currNL0 * this->biasDTypeSize : 0;
    uint64_t scaleSize = currNL0 * tilingIns_->shapeInfo.channelWiseCoeff * FP16_DTYPE_SIZE;
    // C04 should be considered in L0Tiling Decision, to avoid fullload failure in K-direction of L1 tiling.
    if (tilingIns_->isC04Flag) {
        if (this->l1Flags.isWoL1MustFullLoadFlagC04) {
            wiL1 = tilingIns_->shapeInfo.orgWi;
        }
        aL1Size = AlignB(hiL1 * wiL1, C0_SIZE / (this->fMapDTypeSize * C04_CIN_SIZE)) *
                    C04_CIN_SIZE * this->fMapDTypeSize;
        bL1Size = AlignB(tilingIns_->shapeInfo.orgkH * tilingIns_->shapeInfo.orgkW * C04_CIN_SIZE,
                    tilingIns_->cubeInfo.k0) * currNL0 * this->weightDTypeSize;
    }
    uint64_t usedL1Size = aL1Size + bL1Size + biasSize + scaleSize;
    return usedL1Size;
}

void ConvTilingAlgorithmHWmode::SetL1TilingRes()
{
    // real kAL1 = kAL1(spilt_ci1 * ci0) * kh * kw * kd
    if (tilingIns_->isC04Flag) {
        tilingIns_->l1TilingInfo.kAL1 = l1Params.kAL1 * this->l1TilingCalc.c04KSizeAlign;
    } else {
        if (tilingIns_->isDmaFlag) {
            tilingIns_->l1TilingInfo.kAL1 = l1Params.kAL1;
        } else {
        // No need to multiply kd，since kd is contained in kAL1
        tilingIns_->l1TilingInfo.kAL1 = l1Params.kAL1 * tilingIns_->shapeInfo.singlekH * tilingIns_->shapeInfo.singlekW;
        }
    }
    tilingIns_->l1TilingInfo.kBL1 = l1Params.kBL1;
    tilingIns_->l1TilingInfo.hoAL1 = l1Params.hoAL1;
    tilingIns_->l1TilingInfo.woAL1 = l1Params.woAL1;
    tilingIns_->l1TilingInfo.nBL1 = l1Params.nBL1;
    tilingIns_->l1TilingInfo.khL1 = this->l1TilingCalc.khL1;
    tilingIns_->l1TilingInfo.kwL1 = this->l1TilingCalc.kwL1;
    tilingIns_->l1TilingInfo.iterateMNOrder = l1Flags.iterateMNOrder;
    tilingIns_->l1TilingInfo.biasFullLoadFlag = l1Flags.isBiasFullLoad;
    tilingIns_->l1TilingInfo.fixpParamsFullLoadFlag = l1Flags.isFixpParamsFullLoad;
}

void ConvTilingAlgorithmHWmode::SetL0TilingRes()
{
    tilingIns_->l0TilingInfo.hoL0 = l0Params.hoL0;
    tilingIns_->l0TilingInfo.woL0 = l0Params.woL0;
    tilingIns_->l0TilingInfo.kL0 = l0Params.kL0;
    tilingIns_->l0TilingInfo.nL0 = l0Params.nL0;
}

bool ConvTilingAlgorithmHWmode::CheckWoL0FullLoadC04Mode() const
{
    return static_cast<uint64_t>(tilingIns_->shapeInfo.singleWo * DOUBLE_BUFFER_NUM * C0_SIZE) <=
        tilingIns_->platformInfo.l0ASize;
}

void ConvTilingAlgorithmHWmode::UpdateNL0()
{
    for (auto it = l0TilingRange.nL0Range.begin(); it != l0TilingRange.nL0Range.end();) {
        if (*it > l1Params.nBL1 || l1Params.nBL1 % *it != 0) {
            it = l0TilingRange.nL0Range.erase(it);
        } else {
            ++it;
        }
    }

    l0Params.nL0Index = 0;
    while (l0Params.nL0Index < l0TilingRange.nL0Range.size() - 1) {
        l0Params.nL0 = l0TilingRange.nL0Range[l0Params.nL0Index + 1];

        if (CalcBTSize(l0Params.nL0) > tilingIns_->platformInfo.btSize ||
            CalcFBSize(l0Params.nL0) > tilingIns_->platformInfo.fbSize ||
            CalcBL0Size(l0Params.kL0, l0Params.nL0) > tilingIns_->platformInfo.l0BSize ||
            CalcCL0Size(l0Params.hoL0 * l0Params.woL0, l0Params.nL0) > tilingIns_->platformInfo.l0CSize) {
            l0Params.nL0 = l0TilingRange.nL0Range[l0Params.nL0Index];
            break;
        }

        l0Params.nL0Index++;

        if (l0Params.nL0 >= l0Params.woL0 * l0Params.hoL0) {
            break;
        }
    }
}

void ConvTilingAlgorithmHWmode::UpdateHoL0WoL0(uint64_t& value, uint64_t& index, std::vector<uint64_t>& range,
                                               uint64_t valueMax, uint64_t anotherValue)
{
    for (auto it = range.begin(); it != range.end();) {
        if (*it > valueMax || valueMax % *it != 0) {
            it = range.erase(it);
        } else {
            ++it;
        }
    }

    index = 0;
    while (index < range.size() - 1) {
        value = range[index + 1];
        if (CalcAL0Size(l0Params.hoL0 * l0Params.woL0, l0Params.kL0) > tilingIns_->platformInfo.l0ASize ||
            CalcCL0Size(l0Params.hoL0 * l0Params.woL0, l0Params.nL0) > tilingIns_->platformInfo.l0CSize) {
            value = range[index];
            break;
        }

        index++;
 
        if (value * anotherValue >= l0Params.nL0) {
            break;
        }
    }
}

void ConvTilingAlgorithmHWmode::L0TilingRest(bool kFullLoadFlag)
{
    // when al0 & bl0 open db but kl0 tiling not fullLoad Kl1, need close db and update l0 tiling
    bool resetFlag = this->dbValue.pbAL0 == DOUBLE_BUFFER_NUM &&
                     this->dbValue.pbBL0 == DOUBLE_BUFFER_NUM && kFullLoadFlag;
    if (!resetFlag) {
        return;
    }

    if (tilingIns_->l1TilingInfo.iterateMNOrder == IterateMNOrder::ITER_M_FST) {
        uint64_t orginNL0 = l0Params.nL0;
        UpdateNL0();
        if (l0Params.nL0 > orginNL0) {
            this->dbValue.pbBL0 = static_cast<uint8_t>(1);
        }
    } else {
        uint64_t orginHoL0 = l0Params.hoL0;
        uint64_t orginWoL0 = l0Params.woL0;
        UpdateHoL0WoL0(l0Params.woL0, l0Params.woL0Index, l0TilingRange.woL0Range, l1Params.woAL1, l0Params.hoL0);
        UpdateHoL0WoL0(l0Params.hoL0, l0Params.hoL0Index, l0TilingRange.hoL0Range, l1Params.hoAL1, l0Params.woL0);
        if (orginHoL0 < l0Params.hoL0 || orginWoL0 < l0Params.woL0) {
            this->dbValue.pbAL0 = static_cast<uint8_t>(1);
        }
    }
    SetL0TilingRes();
}

void ConvTilingAlgorithmHWmode::CheckL0DoubleBuffer()
{
    bool kFullLoadFlag = tilingIns_->l0TilingInfo.kL0 == (tilingIns_->isC04Flag ? this->l1TilingCalc.c04KSizeAlign :
                                                          this->l1TilingCalc.kBL1MaxSize) ? true : false;
    if (static_cast<uint64_t>(tilingIns_->shapeInfo.singleHo) <= tilingIns_->l0TilingInfo.hoL0 &&
        static_cast<uint64_t>(tilingIns_->shapeInfo.singleWo) <= tilingIns_->l0TilingInfo.woL0 &&
        kFullLoadFlag) {
        this->dbValue.pbAL0 = 1;
    }
 
    if (kFullLoadFlag &&
        tilingIns_->shapeInfo.singleCo1 * tilingIns_->cubeInfo.n0 == tilingIns_->l0TilingInfo.nL0) {
        this->dbValue.pbBL0 = 1;
    }

    L0TilingRest(kFullLoadFlag);

    if (CalcCL0Size(tilingIns_->l0TilingInfo.hoL0 * tilingIns_->l0TilingInfo.woL0, tilingIns_->l0TilingInfo.nL0) <=
        (tilingIns_->platformInfo.l0CSize / DOUBLE_BUFFER_NUM)) {
        this->dbValue.pbCL0 = DOUBLE_BUFFER_NUM;
    }

    if (this->dbValue.pbAL0 == DOUBLE_BUFFER_NUM && this->dbValue.pbBL0 == 1) {
        tilingIns_->l1TilingInfo.iterateMNOrder = IterateMNOrder::ITER_M_FST;
    } else if (this->dbValue.pbAL0 == 1 && this->dbValue.pbBL0 == DOUBLE_BUFFER_NUM) {
        tilingIns_->l1TilingInfo.iterateMNOrder = IterateMNOrder::ITER_N_FST;
    }
}

uint64_t ConvTilingAlgorithmHWmode::GetFixpParamsL1FullLoadSize() const
{
    uint64_t fullLoadFixpParamsSize = 0;
    fullLoadFixpParamsSize += tilingIns_->shapeInfo.singleCo1 * tilingIns_->cubeInfo.n0 *
        tilingIns_->shapeInfo.channelWiseCoeff * FP16_DTYPE_SIZE;

    return fullLoadFixpParamsSize;
}

uint64_t ConvTilingAlgorithmHWmode::GetFixpParamsL1MinLoadSize() const
{
    uint64_t minLoadFixpParamsSize = 0;
    minLoadFixpParamsSize += tilingIns_->cubeInfo.n0 * tilingIns_->shapeInfo.channelWiseCoeff * FP16_DTYPE_SIZE;

    return minLoadFixpParamsSize;
}

bool ConvTilingAlgorithmHWmode::CheckHinOverLoad3dv2Limit(uint64_t hout) const
{
    if (tilingIns_->isDmaFlag) {
        return false;
    }
    // load3dv2 hin limit: [0, 32767]
    uint64_t hiAL1 = InferHiL1(hout, tilingIns_->shapeInfo.orgHi);
    return hiAL1 > LOAD3DV2_HIN_WIN_LIMIT_VALUE;
}

bool ConvTilingAlgorithmHWmode::CheckWinOverLoad3dv2Limit(uint64_t wout) const
{
    if (tilingIns_->isDmaFlag) {
        return false;
    }
    // load3dv2 win limit: [0, 32767]
    uint64_t wiAL1 = static_cast<uint64_t>(InferWiL1(wout, tilingIns_->shapeInfo.orgWi));
    return wiAL1 > LOAD3DV2_HIN_WIN_LIMIT_VALUE;
}

void ConvTilingAlgorithmHWmode::GetL1TilingRange()
{
    if (!tilingIns_->isC04Flag) {
        CalcCommFactor(tilingIns_->shapeInfo.singleCi1, tilingIns_->shapeInfo.singleCi1, this->l1TilingRange.kAL1Range);
        VectorElementMultip(this->l1TilingRange.kAL1Range, tilingIns_->cubeInfo.k0);
        if (tilingIns_->isDmaFlag) {
            VectorElementMultip(this->l1TilingRange.kAL1Range, this->l1TilingCalc.kwL1 * this->l1TilingCalc.khL1);
        }
        uint64_t khkw = tilingIns_->shapeInfo.singlekH * tilingIns_->shapeInfo.singlekW;
        if (!tilingIns_->isDmaFlag) {
            auto effectiveCount = count_if(this->l1TilingRange.kAL1Range.begin(), this->l1TilingRange.kAL1Range.end(),
                                        [khkw](uint64_t x) { return x * khkw <= MAX_16_BIT_NUM; });
            this->l1TilingRange.kAL1Range.resize(effectiveCount);
        }

        CalcCommFactor(tilingIns_->shapeInfo.singleCi1, tilingIns_->shapeInfo.singleCi1, this->l1TilingRange.kBL1Range);
        if (tilingIns_->isDmaFlag) {
            VectorElementMultip(this->l1TilingRange.kBL1Range, this->l1TilingCalc.ci0KhL1KwL1);
        } else {
            VectorElementMultip(this->l1TilingRange.kBL1Range, this->l1TilingCalc.ci0HkWk);
        }
    } else {
        this->l1TilingRange.kAL1Range.emplace_back(this->l1TilingCalc.kAL1MaxSize);
        this->l1TilingRange.kBL1Range.emplace_back(this->l1TilingCalc.kBL1MaxSize);
    }

    uint64_t multiNBL1Max = CeilDiv(tilingIns_->shapeInfo.singleCo1 * tilingIns_->cubeInfo.n0, l0Params.nL0);
    CalcCommFactor(multiNBL1Max, multiNBL1Max, this->l1TilingRange.nBL1Range);
    VectorElementMultip(this->l1TilingRange.nBL1Range, l0Params.nL0);

    if (tilingIns_->isC04Flag && this->l1Flags.isWoL1MustFullLoadFlagC04) {
        this->l1TilingRange.woAL1Range.emplace_back(AlignB(tilingIns_->shapeInfo.singleWo, tilingIns_->cubeInfo.m0));
    } else {
        uint64_t multiWoL1Max = CeilDiv(CeilDiv(tilingIns_->shapeInfo.singleWo, tilingIns_->cubeInfo.m0) *
                            tilingIns_->cubeInfo.m0, l0Params.woL0);
        CalcCommFactor(multiWoL1Max, multiWoL1Max, this->l1TilingRange.woAL1Range);
        VectorElementMultip(this->l1TilingRange.woAL1Range, l0Params.woL0);
    }
    uint64_t multiHoL1Max = tilingIns_->shapeInfo.singleHo / l0Params.hoL0;
    CalcCommFactor(multiHoL1Max, multiHoL1Max, this->l1TilingRange.hoAL1Range);
    VectorElementMultip(this->l1TilingRange.hoAL1Range, l0Params.hoL0);
        // remove value which is over load3dv2's hin/win limit [0, 32767]
    for (auto it = this->l1TilingRange.hoAL1Range.begin(); it != this->l1TilingRange.hoAL1Range.end();) {
        if (CheckHinOverLoad3dv2Limit(*it)) {
            TILING_LOG_DEBUG("Hin over Load3dv2 limit, erase it.");
            it = this->l1TilingRange.hoAL1Range.erase(it);
        } else {
            ++it;
        }
    }
    for (auto it = this->l1TilingRange.woAL1Range.begin(); it != this->l1TilingRange.woAL1Range.end();) {
        if (CheckWinOverLoad3dv2Limit(*it)) {
            TILING_LOG_DEBUG("Win over Load3dv2 limit, erase it.");
            it = this->l1TilingRange.woAL1Range.erase(it);
        } else {
            ++it;
        }
    }
}

bool ConvTilingAlgorithmHWmode::CheckBL1FullLoad()
{
    if (tilingIns_->isDmaFlag && this->l1Flags.isKernelSplitFlag) {
        return false;
    }
    // if weight is fullload, bias and fixpParams must be fulload
    return this->l1TilingCalc.aL1Minsize + this->l1TilingCalc.bL1MaxSize + this->l1TilingCalc.biasL1MaxSize +
           this->l1TilingCalc.fixpParamsL1MaxSize <= tilingIns_->platformInfo.l1Size;
}

bool ConvTilingAlgorithmHWmode::CheckAL1FullLoad()
{
    if (tilingIns_->isDmaFlag &&
        (this->l1TilingCalc.ubMaxSize > tilingIns_->platformInfo.ubSize || this->l1Flags.isKernelSplitFlag)) {
        return false;
    }
 
    uint64_t hoAL1Full = tilingIns_->shapeInfo.singleHo;
    uint64_t woAL1Full = AlignB(tilingIns_->shapeInfo.singleWo, tilingIns_->cubeInfo.m0);
    // load3d m start position <= LOAD3D_M_START_POS_LIMIT
    if (!tilingIns_->isDmaFlag && hoAL1Full * woAL1Full > LOAD3D_M_START_POS_LIMIT) {
        return false;
    }
    if (!tilingIns_->isDmaFlag && tilingIns_->shapeInfo.singleCi1 * tilingIns_->shapeInfo.singlekH *
        tilingIns_->shapeInfo.singlekW * tilingIns_->cubeInfo.k0 > MAX_16_BIT_NUM) {
        return false;
    }
    return this->l1TilingCalc.aL1MaxSize + this->l1TilingCalc.bL1Minsize + this->l1TilingCalc.biasL1MinSize +
           this->l1TilingCalc.fixpParamsL1MinSize <= tilingIns_->platformInfo.l1Size;
}
 
bool ConvTilingAlgorithmHWmode::CheckABL1FullLoad()
{
    if (tilingIns_->isDmaFlag &&
        (this->l1TilingCalc.ubMaxSize > tilingIns_->platformInfo.ubSize || this->l1Flags.isKernelSplitFlag)) {
        return false;
    }
 
    uint64_t hoAL1Full = tilingIns_->shapeInfo.singleHo;
    uint64_t woAL1Full = AlignB(tilingIns_->shapeInfo.singleWo, tilingIns_->cubeInfo.m0);
    // load3d m start position <= LOAD3D_M_START_POS_LIMIT
    if (!tilingIns_->isDmaFlag && hoAL1Full * woAL1Full > LOAD3D_M_START_POS_LIMIT) {
        return false;
    }
    return this->l1TilingCalc.aL1MaxSize + this->l1TilingCalc.bL1MaxSize + this->l1TilingCalc.biasL1MaxSize +
           this->l1TilingCalc.fixpParamsL1MaxSize <= tilingIns_->platformInfo.l1Size;
}

bool ConvTilingAlgorithmHWmode::IsBL1FullLoadFirst()
{
    return tilingIns_->platformInfo.abL1mte2BandWidthCof * this->l1TilingCalc.bL1MaxSize +
           this->l1TilingCalc.biasL1MaxSize + this->l1TilingCalc.fixpParamsL1MaxSize >= this->l1TilingCalc.aL1MaxSize;
}

void ConvTilingAlgorithmHWmode::InitABL1TilingMode()
{
    if (CheckABL1FullLoad()) {
        this->l1Flags.abL1Mode = L1TilingMode::ALL_FULL_LOAD;
        return;
    }

    if (IsBL1FullLoadFirst()) {
        if (CheckBL1FullLoad()) {
            this->l1Flags.abL1Mode = L1TilingMode::FULL_LOAD_BL1;
        } else if (CheckAL1FullLoad()) {
            this->l1Flags.abL1Mode = L1TilingMode::FULL_LOAD_AL1;
        } else {
            this->l1Flags.abL1Mode = L1TilingMode::NONE_FULL_LOAD;
        }
    } else {
        if (CheckAL1FullLoad()) {
            this->l1Flags.abL1Mode = L1TilingMode::FULL_LOAD_AL1;
        } else if (CheckBL1FullLoad()) {
            this->l1Flags.abL1Mode = L1TilingMode::FULL_LOAD_BL1;
        } else {
            this->l1Flags.abL1Mode = L1TilingMode::NONE_FULL_LOAD;
        }
    }
}

int64_t ConvTilingAlgorithmHWmode::InferWiL1(uint64_t inputWoL1, int64_t wi) const
{
    if (inputWoL1 == static_cast<uint64_t>(tilingIns_->shapeInfo.singleWo) && tilingIns_->isC04Flag) {
        return tilingIns_->shapeInfo.orgWi;
    }

    int64_t kwDilated = (tilingIns_->shapeInfo.singlekW - 1) * tilingIns_->attrInfo.dilationW + 1;
    int64_t tmpWiL1 = (inputWoL1 - 1) * tilingIns_->attrInfo.strideW + kwDilated;
    if (tmpWiL1 > wi) {
        tmpWiL1 = wi;
    }

    return tmpWiL1;
}

void ConvTilingAlgorithmHWmode::RestrictRange(const std::vector<uint64_t> &inputRange,
                                              std::vector<uint64_t> &strictRange,
                                              uint64_t restriction) const
{
    uint32_t legalStart = inputRange.size();
    for (uint32_t i = 0; i < inputRange.size(); i++) {
        if (inputRange[i] >= restriction) {
            legalStart = i;
            break;
        }
    }
    strictRange.clear();
    strictRange.assign(inputRange.begin() + legalStart, inputRange.end());
    if (strictRange.size() == 0) {
        strictRange.emplace_back(restriction);
    } else if (restriction < strictRange[0]) {
        strictRange.insert(strictRange.begin(), restriction);
    }
}

uint64_t ConvTilingAlgorithmHWmode::CalcCurL1Size(const L1TilingParams &inputTiling, uint64_t biasFixpParamNB)
{
    // when decide L1 tiling, biasFixpParamNB = nBL1
    auto biasL1Size = tilingIns_->hasBias ? AlignB(biasFixpParamNB * biasDTypeSize, C0_SIZE) : 0;
    auto fixpParamsSize = AlignB(biasFixpParamNB * tilingIns_->shapeInfo.channelWiseCoeff * FP16_DTYPE_SIZE, C0_SIZE);

    auto hiAL1 = InferHiL1(inputTiling.hoAL1, tilingIns_->shapeInfo.orgHi);
    auto wiAL1 = InferWiL1(inputTiling.woAL1, tilingIns_->shapeInfo.orgWi);
    uint64_t feaMapSize = 0;
    uint64_t weightSize = 0;
    if (tilingIns_->isC04Flag) {
        feaMapSize = AlignB(hiAL1 * wiAL1, C0_SIZE / (fMapDTypeSize * C04_CIN_SIZE)) *
                     this->dbValue.pbAL1 * C04_CIN_SIZE * fMapDTypeSize;
        weightSize = inputTiling.kBL1 * inputTiling.nBL1 * this->dbValue.pbBL1 * weightDTypeSize;
    } else {
        if (tilingIns_->isDmaFlag) {
            feaMapSize = AlignB(inputTiling.hoAL1 * inputTiling.woAL1 * inputTiling.kAL1 *
                                fMapDTypeSize * this->dbValue.pbAL1, C0_SIZE);
        } else {
            feaMapSize = AlignB(hiAL1 * wiAL1 * inputTiling.kAL1 * fMapDTypeSize * this->dbValue.pbAL1, C0_SIZE);
        }
        weightSize = AlignB(inputTiling.nBL1 * inputTiling.kBL1 * weightDTypeSize * this->dbValue.pbBL1, C0_SIZE);
    }
    TILING_LOG_DEBUG("CalcCurL1Size: hoAL1: %lu, woAL1: %lu, kAL1: %lu, kBL1: %lu, nBL1: %lu, hiAL1: %ld, wiAL1: %ld, \
        fmSize: %lu, weightSize: %lu, biasSize: %lu, fixpSize: %lu, curL1Size: %lu",
        inputTiling.hoAL1, inputTiling.woAL1, inputTiling.kAL1, inputTiling.kBL1, inputTiling.nBL1, hiAL1, wiAL1,
        feaMapSize, weightSize, biasL1Size, fixpParamsSize, feaMapSize + weightSize + biasL1Size + fixpParamsSize);
    return feaMapSize + weightSize + biasL1Size + fixpParamsSize;
}

uint64_t ConvTilingAlgorithmHWmode::CalcCurUbSize(uint64_t hoL1, uint64_t woL1, uint64_t curKh, uint64_t curKw)
{
    if (!tilingIns_->isDmaFlag) {
        return 0;
    }
 
    return (hoL1 * woL1 * curKh * curKw * tilingIns_->cubeInfo.k0 * fMapDTypeSize);
}

void ConvTilingAlgorithmHWmode::IterKAL1(uint64_t &kAL1, uint64_t kBL1, uint64_t hoAL1, uint64_t woAL1, uint64_t nBL1)
{
    if (tilingIns_->isC04Flag) {
        kAL1 = l1TilingInit.kAL1Init;
        return;
    }
    TILING_LOG_DEBUG("IterKAL1");
    uint32_t curIdx = 0;
    uint64_t curValue = 0;
    bool isExceedL1Size = false;
    L1TilingParams inputTiling;
    inputTiling.kBL1 = kBL1;
    inputTiling.nBL1 = nBL1;
    inputTiling.hoAL1 = hoAL1;
    inputTiling.woAL1 = woAL1;
    while (curIdx < l1TilingRange.kAL1StrictRange.size()) {
        curValue = l1TilingRange.kAL1StrictRange[curIdx];
        inputTiling.kAL1 = curValue;
        isExceedL1Size = CalcCurL1Size(inputTiling, nBL1) > tilingIns_->platformInfo.l1Size;
        if (curIdx > 0 && isExceedL1Size) {
            curValue = l1TilingRange.kAL1StrictRange[curIdx - 1];
            break;
        }
        if (curIdx == 0 && isExceedL1Size) {
            kAL1 = 0;
            return;
        }
        curIdx++;
    }
    kAL1 = curValue;
    if (curIdx == l1TilingRange.kAL1StrictRange.size()) {
        for (int64_t multi = 1; multi <= tilingIns_->shapeInfo.singlekD; ++multi) {
            kAL1 = multi * curValue;
            inputTiling.kAL1 = kAL1;
            isExceedL1Size = CalcCurL1Size(inputTiling, nBL1) > tilingIns_->platformInfo.l1Size;
            if (isExceedL1Size) {
                kAL1 = (multi - 1) * curValue;
                break;
            }
        }
    }
}

void ConvTilingAlgorithmHWmode::IterKBL1(uint64_t kAL1, uint64_t &kBL1, uint64_t hoAL1, uint64_t woAL1, uint64_t nBL1)
{
    TILING_LOG_DEBUG("IterKBL1");
    if (tilingIns_->isC04Flag) {
        kBL1 = l1TilingInit.kBL1Init;
        return;
    }
    uint32_t curIdx = 0;
    uint64_t curValue = 0;
    bool isExceedL1Size = false;
    L1TilingParams inputTiling;
    inputTiling.kAL1 = kAL1;
    inputTiling.nBL1 = nBL1;
    inputTiling.hoAL1 = hoAL1;
    inputTiling.woAL1 = woAL1;
    while (curIdx < l1TilingRange.kBL1StrictRange.size()) {
        curValue = l1TilingRange.kBL1StrictRange[curIdx];
        inputTiling.kBL1 = curValue;
        isExceedL1Size = CalcCurL1Size(inputTiling, nBL1) > tilingIns_->platformInfo.l1Size;
        if (curIdx > 0 && isExceedL1Size) {
            curValue = l1TilingRange.kBL1StrictRange[curIdx - 1];
            break;
        }
        if (curIdx == 0 && isExceedL1Size) {
            kBL1 = 0;
            return;
        }
        curIdx++;
    }
    kBL1 = curValue;
    if (curIdx == l1TilingRange.kBL1StrictRange.size()) {
        for (int64_t multi = 1; multi <= tilingIns_->shapeInfo.singlekD; ++multi) {
            kBL1 = multi * curValue;
            inputTiling.kBL1 = kBL1;
            isExceedL1Size = CalcCurL1Size(inputTiling, nBL1) > tilingIns_->platformInfo.l1Size;
            if (isExceedL1Size) {
                kBL1 = (multi - 1) * curValue;
                break;
            }
        }
    }
}

void ConvTilingAlgorithmHWmode::IterNBL1(uint64_t kAL1, uint64_t kBL1, uint64_t hoAL1, uint64_t woAL1, uint64_t &nBL1)
{
    TILING_LOG_DEBUG("IterNBL1");
    uint32_t curIdx = 0;
    uint64_t curValue = 0;
    bool isExceedL1Size = false;
    L1TilingParams inputTiling;

    while (curIdx < l1TilingRange.nBL1StrictRange.size()) {
        curValue = l1TilingRange.nBL1StrictRange[curIdx];
        inputTiling.kAL1 = kAL1;
        inputTiling.kBL1 = kBL1;
        inputTiling.nBL1 = curValue;
        inputTiling.hoAL1 = hoAL1;
        inputTiling.woAL1 = woAL1;
        isExceedL1Size = CalcCurL1Size(inputTiling, curValue) > tilingIns_->platformInfo.l1Size;
        if (curIdx > 0 && isExceedL1Size) {
            curValue = l1TilingRange.nBL1StrictRange[curIdx - 1];
            break;
        }
        if (curIdx == 0 && isExceedL1Size) {
            nBL1 = 0;
            return;
        }
        curIdx++;
    }

    nBL1 = curValue;
    if (nBL1 != 0 && nBL1 == this->l1TilingCalc.nBL1MaxSize) {
        l1Flags.isBiasFullLoad = true;
        l1Flags.isFixpParamsFullLoad = true;
    }
}

void ConvTilingAlgorithmHWmode::IterHoAL1(uint64_t &hoAL1Res, const std::vector<uint64_t> &inputHoAL1Range,
                                          L1TilingParams &inputTiling)
{
    TILING_LOG_DEBUG("IterHoAL1");
    uint32_t curHoIdx = 0;
    uint64_t curHoValue = 0;
    bool isExceedL1Size = false;
    bool isExceedUbSize = false;
    bool isExceedInstrLimit = false;
    while (curHoIdx < inputHoAL1Range.size()) {
        curHoValue = inputHoAL1Range[curHoIdx];
        inputTiling.hoAL1 = curHoValue;
        isExceedL1Size = CalcCurL1Size(inputTiling, inputTiling.nBL1) > tilingIns_->platformInfo.l1Size;
        isExceedUbSize = CalcCurUbSize(inputTiling.hoAL1, inputTiling.woAL1, 1, 1) > tilingIns_->platformInfo.ubSize;
        isExceedInstrLimit = tilingIns_->isDmaFlag ? false : curHoValue * inputTiling.woAL1 > LOAD3D_M_START_POS_LIMIT;
        if (curHoIdx > 0 && (isExceedL1Size || isExceedInstrLimit || isExceedUbSize)) {
            curHoValue = inputHoAL1Range[curHoIdx - 1];
            break;
        }
        if (curHoIdx == 0 && (isExceedL1Size || isExceedInstrLimit)) {
            hoAL1Res = 0;
            return;
        }
        curHoIdx++;
    }

    hoAL1Res = curHoValue;
}

void ConvTilingAlgorithmHWmode::IterWoAL1(uint64_t &woAL1Res, const std::vector<uint64_t> &inputWoAL1Range,
                                          L1TilingParams &inputTiling)
{
    TILING_LOG_DEBUG("IterWoAL1");
    inputTiling.hoAL1 = l1TilingRange.hoAL1StrictRange[0];
    uint32_t curWoIdx = 0;
    uint64_t curWoValue = 0;
    bool isExceedL1Size = false;
    bool isExceedUbSize = false;
    bool isExceedInstrLimit = false;
    while (curWoIdx < inputWoAL1Range.size()) {
        curWoValue = inputWoAL1Range[curWoIdx];
        inputTiling.woAL1 = curWoValue;
        isExceedL1Size = CalcCurL1Size(inputTiling, inputTiling.nBL1) > tilingIns_->platformInfo.l1Size;
        isExceedUbSize = CalcCurUbSize(inputTiling.hoAL1, inputTiling.woAL1, 1, 1) > tilingIns_->platformInfo.ubSize;
        isExceedInstrLimit = tilingIns_->isDmaFlag ? false : curWoValue * inputTiling.hoAL1 > LOAD3D_M_START_POS_LIMIT;
        if (curWoIdx > 0 && (isExceedL1Size || isExceedInstrLimit || isExceedUbSize)) {
            curWoValue = inputWoAL1Range[curWoIdx - 1];
            break;
        }
        if (curWoIdx == 0 && (isExceedL1Size || isExceedInstrLimit)) {
            woAL1Res = 0;
            return;
        }
        curWoIdx++;
    }

    woAL1Res = curWoValue;
}

void ConvTilingAlgorithmHWmode::IterHoWoAL1(uint64_t kAL1, uint64_t kBL1, uint64_t &hoAL1, uint64_t &woAL1,
                                            uint64_t nBL1)
{
    L1TilingParams inputTiling;
    inputTiling.kAL1 = kAL1;
    inputTiling.kBL1 = kBL1;
    inputTiling.nBL1 = nBL1;
    IterWoAL1(woAL1, l1TilingRange.woAL1StrictRange, inputTiling);
    inputTiling.woAL1 = woAL1;
    IterHoAL1(hoAL1, l1TilingRange.hoAL1StrictRange, inputTiling);
}

bool ConvTilingAlgorithmHWmode::IsKBL1FullLoadFirst()
{
    return tilingIns_->platformInfo.abL1mte2BandWidthCof * this->l1TilingCalc.bL1MaxSize >=
           this->l1TilingCalc.aL1MaxSize;
}

bool ConvTilingAlgorithmHWmode::CheckKL1FullLoad(uint64_t kAL1, uint64_t kBL1)
{
    TILING_LOG_DEBUG("CheckKL1FullLoad");
    if (!tilingIns_->isDmaFlag &&
        kAL1 * tilingIns_->shapeInfo.singlekH * tilingIns_->shapeInfo.singlekW > MAX_16_BIT_NUM) {
        TILING_LOG_DEBUG("Exceeded kStartPos limit");
        return false;
    }
    L1TilingParams inputTiling;
    inputTiling.nBL1 = l1TilingRange.nBL1StrictRange[0];
    inputTiling.kAL1 = kAL1;
    inputTiling.kBL1 = kBL1;

    // check minHo minWo can make L1 fullLoad
    inputTiling.hoAL1 = l1TilingRange.hoAL1StrictRange[0];
    inputTiling.woAL1 = l1TilingRange.woAL1StrictRange[0];
    if (CalcCurL1Size(inputTiling, inputTiling.nBL1) <= tilingIns_->platformInfo.l1Size) {
        return true;
    }

    return false;
}

void ConvTilingAlgorithmHWmode::IterKABL1(uint64_t& tmpKAL1, uint64_t& tmpKBL1, uint64_t tmpHoAL1,
                                          uint64_t tmpWoAL1, uint64_t tmpNBL1)
{
    TILING_LOG_DEBUG("IterKABL1");
    if (tilingIns_->isC04Flag) {
        tmpKAL1 = l1TilingInit.kAL1Init;
        tmpKBL1 = l1TilingInit.kBL1Init;
        return;
    }
    bool isExceedL1Size = false;
    uint32_t kAL1Idx = 0;
    uint32_t kBL1Idx = 0;
    L1TilingParams inputTiling;
    inputTiling.nBL1 = tmpNBL1;
    inputTiling.hoAL1 = tmpHoAL1;
    inputTiling.woAL1 = tmpWoAL1;
    bool enlargeKAL1Tag = false;
    while (kAL1Idx < l1TilingRange.kAL1StrictRange.size() &&
           kBL1Idx < l1TilingRange.kBL1StrictRange.size()) {
        inputTiling.kAL1 = l1TilingRange.kAL1StrictRange[kAL1Idx];
        inputTiling.kBL1 = l1TilingRange.kBL1StrictRange[kBL1Idx];
        isExceedL1Size = CalcCurL1Size(inputTiling, tmpNBL1) > tilingIns_->platformInfo.l1Size;
        if (isExceedL1Size) {
            if (kAL1Idx == 0 && kBL1Idx == 0) {
                TILING_LOG_DEBUG("no valid value in IterKABL1");
            }
            break;
        }
        tmpKAL1 = l1TilingRange.kAL1StrictRange[kAL1Idx];
        tmpKBL1 = l1TilingRange.kBL1StrictRange[kBL1Idx];
        if (enlargeKAL1Tag) {
            kAL1Idx++;
            enlargeKAL1Tag = false;
        } else {
            kBL1Idx++;
            enlargeKAL1Tag = true;
        }
    }
}

void ConvTilingAlgorithmHWmode::KBL1FullLoadFirst(uint64_t &tmpKAL1, uint64_t &tmpKBL1, uint64_t &tmpHoAL1,
                                                  uint64_t &tmpWoAL1, uint64_t &tmpNBL1)
{
    if (CheckKL1FullLoad(l1TilingRange.kAL1StrictRange[0], l1TilingCalc.kBL1MaxSize)) {
        l1Flags.iterateMNOrder = IterateMNOrder::ITER_M_FST;
        tmpKBL1 = l1TilingCalc.kBL1MaxSize;
        tmpHoAL1 = l0Params.hoL0;
        tmpWoAL1 = l0Params.woL0;
        IterKAL1(tmpKAL1, tmpKBL1, tmpHoAL1, tmpWoAL1, l1TilingRange.nBL1StrictRange[0]);
        IterNBL1(tmpKAL1, tmpKBL1, tmpHoAL1, tmpWoAL1, tmpNBL1);
    } else if (CheckKL1FullLoad(l1TilingCalc.kAL1MaxSize, l1TilingRange.kBL1StrictRange[0])) {
        l1Flags.iterateMNOrder = IterateMNOrder::ITER_N_FST;
        tmpKAL1 = l1TilingCalc.kAL1MaxSize;
        tmpNBL1 = l0Params.nL0;
        IterKBL1(tmpKAL1, tmpKBL1, l1TilingRange.hoAL1StrictRange[0],
                l1TilingRange.woAL1StrictRange[0], l1TilingRange.nBL1StrictRange[0]);
        IterHoWoAL1(tmpKAL1, tmpKBL1, tmpHoAL1, tmpWoAL1, tmpNBL1);
    } else {
        l1Flags.iterateMNOrder = IterateMNOrder::ITER_N_FST;
        tmpHoAL1 = l0Params.hoL0;
        tmpWoAL1 = tilingIns_->isC04Flag && l1Flags.isWoL1MustFullLoadFlagC04 ?
            AlignB(tilingIns_->shapeInfo.singleWo, tilingIns_->cubeInfo.m0) : l0Params.woL0;
        tmpNBL1 = l0Params.nL0;
        IterKABL1(tmpKAL1, tmpKBL1, tmpHoAL1, tmpWoAL1, tmpNBL1);
    }
}

void ConvTilingAlgorithmHWmode::KAL1FullLoadFirst(uint64_t &tmpKAL1, uint64_t &tmpKBL1, uint64_t &tmpHoAL1,
                                                  uint64_t &tmpWoAL1, uint64_t &tmpNBL1)
{
    if (CheckKL1FullLoad(l1TilingCalc.kAL1MaxSize, l1TilingRange.kBL1StrictRange[0])) {
        l1Flags.iterateMNOrder = IterateMNOrder::ITER_N_FST;
        tmpKAL1 = l1TilingCalc.kAL1MaxSize;
        tmpNBL1 = l0Params.nL0;
        IterKBL1(tmpKAL1, tmpKBL1, l1TilingRange.hoAL1StrictRange[0], l1TilingRange.woAL1StrictRange[0], tmpNBL1);
        IterHoWoAL1(tmpKAL1, tmpKBL1, tmpHoAL1, tmpWoAL1, tmpNBL1);
    } else if (CheckKL1FullLoad(l1TilingRange.kAL1StrictRange[0], l1TilingCalc.kBL1MaxSize)) {
        l1Flags.iterateMNOrder = IterateMNOrder::ITER_M_FST;
        tmpKBL1 = l1TilingCalc.kBL1MaxSize;
        tmpHoAL1 = l0Params.hoL0;
        tmpWoAL1 = l0Params.woL0;
        IterKAL1(tmpKAL1, tmpKBL1, tmpHoAL1, tmpWoAL1, l1TilingRange.nBL1StrictRange[0]);
        IterNBL1(tmpKAL1, tmpKBL1, tmpHoAL1, tmpWoAL1, tmpNBL1);
    } else {
        l1Flags.iterateMNOrder = IterateMNOrder::ITER_M_FST;
        tmpHoAL1 = l0Params.hoL0;
        tmpWoAL1 = tilingIns_->isC04Flag && l1Flags.isWoL1MustFullLoadFlagC04 ?
            AlignB(tilingIns_->shapeInfo.singleWo, tilingIns_->cubeInfo.m0) : l0Params.woL0;
        tmpNBL1 = l0Params.nL0;
        IterKABL1(tmpKAL1, tmpKBL1, tmpHoAL1, tmpWoAL1, tmpNBL1);
    }
}

bool ConvTilingAlgorithmHWmode::IsAL1FullFload(const uint64_t currHoAL1, const uint64_t currWoAL1, 
                                               const uint64_t currKAL1) const
{
    if (currHoAL1 >= static_cast<uint64_t>(tilingIns_->shapeInfo.singleHo) &&
        currWoAL1 >= static_cast<uint64_t>(tilingIns_->shapeInfo.singleWo) &&
        currKAL1 >= static_cast<uint64_t>(this->l1TilingCalc.kAL1MaxSize)) {
        return true;
    }
    return false;
}
 
bool ConvTilingAlgorithmHWmode::IsBL1FullFload(const uint64_t currNBL1, const uint64_t currKBL1) const
{
    if (currNBL1 >= static_cast<uint64_t>(tilingIns_->shapeInfo.singleCo) &&
        currKBL1 >= static_cast<uint64_t>(this->l1TilingCalc.kBL1MaxSize)) {
        return true;
    }
    return false;
}

void ConvTilingAlgorithmHWmode::CoreL1TilingNoneFullLoadDecision()
{
    uint64_t tmpKAL1 = 0;
    uint64_t tmpKBL1 = 0;
    uint64_t tmpHoAL1 = 0;
    uint64_t tmpWoAL1 = 0;
    uint64_t tmpNBL1 = 0;

    if (CheckKL1FullLoad(l1TilingCalc.kAL1MaxSize, l1TilingCalc.kBL1MaxSize)) {
        tmpKAL1 = l1TilingCalc.kAL1MaxSize;
        tmpKBL1 = l1TilingCalc.kBL1MaxSize;
        l1Flags.iterateMNOrder = IterateMNOrder::ITER_M_FST;
        IterNBL1(tmpKAL1, tmpKBL1, l0Params.hoL0, l0Params.woL0, tmpNBL1);
        IterHoWoAL1(tmpKAL1, tmpKBL1, tmpHoAL1, tmpWoAL1, tmpNBL1);
    } else {
        if (IsKBL1FullLoadFirst()) { // try fullload KBL1 first
            KBL1FullLoadFirst(tmpKAL1, tmpKBL1, tmpHoAL1, tmpWoAL1, tmpNBL1);
        } else { // try fullload KAL1 first
            KAL1FullLoadFirst(tmpKAL1, tmpKBL1, tmpHoAL1, tmpWoAL1, tmpNBL1);
        }
    }
    L1TilingParams tmpL1Params = {tmpKAL1, tmpKBL1, tmpHoAL1, tmpWoAL1, tmpNBL1,
                                  this->l1TilingCalc.khL1, this->l1TilingCalc.kwL1};
    bool isExceedL1Size = CalcCurL1Size(tmpL1Params, tmpNBL1) > tilingIns_->platformInfo.l1Size;
    if (tmpKAL1 == 0 || tmpKBL1 == 0 || tmpHoAL1 == 0 || tmpWoAL1 == 0 || tmpNBL1 == 0 || isExceedL1Size) {
        this->dbValue.pbAL1 = 1;
        this->dbValue.pbBL1 = 1;

        l1Flags.iterateMNOrder = IterateMNOrder::ITER_M_FST;
        tmpHoAL1 = l0Params.hoL0;
        tmpWoAL1 = tilingIns_->isC04Flag && l1Flags.isWoL1MustFullLoadFlagC04 ?
            AlignB(tilingIns_->shapeInfo.singleWo, tilingIns_->cubeInfo.m0) : l0Params.woL0;
        tmpNBL1 = l0Params.nL0;
        IterKABL1(tmpKAL1, tmpKBL1, tmpHoAL1, tmpWoAL1, tmpNBL1);
    }
    l1Params = {tmpKAL1, tmpKBL1, tmpHoAL1, tmpWoAL1, tmpNBL1, this->l1TilingCalc.khL1, this->l1TilingCalc.kwL1};
    if (IsAL1FullFload(tmpHoAL1, tmpWoAL1, tmpKAL1) && (!IsBL1FullFload(tmpNBL1, tmpKBL1))) {
        l1Flags.iterateMNOrder = IterateMNOrder::ITER_N_FST;
    } else if ((!IsAL1FullFload(tmpHoAL1, tmpWoAL1, tmpKAL1)) && IsBL1FullFload(tmpNBL1, tmpKBL1)) {
        l1Flags.iterateMNOrder = IterateMNOrder::ITER_M_FST;
    }
    l1Flags.isBiasFullLoad = false;
    l1Flags.isFixpParamsFullLoad = false;
}

void ConvTilingAlgorithmHWmode::UpdateBiasFixpParamsL1Fullload()
{
    TILING_LOG_DEBUG("UpdateBiasFixpParamsL1Fullload");
    uint64_t biasFixpFullLoadSize = tilingIns_->shapeInfo.singleCo1 * tilingIns_->cubeInfo.n0;
    bool updateBiasFixpFullLoadFlag = CalcCurL1Size(l1Params, l1Params.nBL1) < tilingIns_->platformInfo.l1Size &&
                                      !l1Flags.isBiasFullLoad && !l1Flags.isFixpParamsFullLoad &&
                                      CalcCurL1Size(l1Params, biasFixpFullLoadSize) <= tilingIns_->platformInfo.l1Size;
    if (updateBiasFixpFullLoadFlag) {
        l1Flags.isBiasFullLoad = true;
        l1Flags.isFixpParamsFullLoad = true;
    }
    bool exceedDataCopyLimits = tilingIns_->shapeInfo.singleCo1 * tilingIns_->cubeInfo.n0 * this->biasDTypeSize >
                                DATACOPYPARAMS_BURSTLEN_MAX;
    if (l1Flags.isBiasFullLoad && l1Flags.isFixpParamsFullLoad && exceedDataCopyLimits) {
        l1Flags.isBiasFullLoad = false;
        l1Flags.isFixpParamsFullLoad = false;
    }
}

void ConvTilingAlgorithmHWmode::UpdateAL1DoubleBufferFirst()
{
    if (this->l1Flags.abL1Mode == L1TilingMode::ALL_FULL_LOAD) {
        this->dbValue.pbAL1 = 1;
        this->dbValue.pbBL1 = 1;
    } else if (this->l1Flags.abL1Mode == L1TilingMode::FULL_LOAD_AL1) {
        this->dbValue.pbAL1 = 1;
    } else if (this->l1Flags.abL1Mode == L1TilingMode::FULL_LOAD_BL1) {
        this->dbValue.pbBL1 = 1;
    }
}

void ConvTilingAlgorithmHWmode::UpdateAL1DoubleBufferSecond()
{
    TILING_LOG_DEBUG("UpdateAL1DoubleBufferSecond");
    bool kAl1FullLoadFlag = tilingIns_->isC04Flag ? true :
        l1Params.kAL1 == this->l1TilingCalc.kAL1MaxSize ? true : false;
    if (static_cast<uint64_t>(tilingIns_->shapeInfo.singleHo) <= l1Params.hoAL1 &&
        static_cast<uint64_t>(tilingIns_->shapeInfo.singleWo) <= l1Params.woAL1 &&
        kAl1FullLoadFlag) {
        this->dbValue.pbAL1 = 1;
        return;
    }
}

void ConvTilingAlgorithmHWmode::CoreL1TilingDecision()
{
    if (l1Flags.abL1Mode == L1TilingMode::ALL_FULL_LOAD) {
        TILING_LOG_DEBUG("Attributed to ALL_FULL_LOAD");
        l1Params = {l1TilingInit.kAL1Init, l1TilingInit.kBL1Init, l1TilingInit.hoAL1Init, l1TilingInit.woAL1Init,
                    l1TilingInit.nBL1Init, this->l1TilingCalc.khL1, this->l1TilingCalc.kwL1};
        l1Flags.iterateMNOrder = IterateMNOrder::ITER_M_FST;
        l1Flags.isBiasFullLoad = true;
        l1Flags.isFixpParamsFullLoad = true;
        return;
    }

    if (l1Flags.abL1Mode == L1TilingMode::FULL_LOAD_BL1) {
        TILING_LOG_DEBUG("Attributed to FULL_LOAD_BL1");
        uint64_t tmpKAL1 = 0;
        uint64_t tmpHoAL1 = 0;
        uint64_t tmpWoAL1 = 0;
        std::vector<uint64_t> inputHoAL1Range = l1TilingRange.hoAL1StrictRange.empty() ?
            std::vector<uint64_t> {l1TilingInit.hoAL1Init} : l1TilingRange.hoAL1StrictRange;
        std::vector<uint64_t> inputWoAL1Range = l1TilingRange.woAL1StrictRange.empty() ?
            std::vector<uint64_t> {l1TilingInit.woAL1Init} : l1TilingRange.woAL1StrictRange;
        IterKAL1(tmpKAL1, l1TilingInit.kBL1Init, inputHoAL1Range[0], inputWoAL1Range[0], l1TilingInit.nBL1Init);
        IterHoWoAL1(tmpKAL1, l1TilingInit.kBL1Init, tmpHoAL1, tmpWoAL1, l1TilingInit.nBL1Init);
        l1Params = {tmpKAL1, l1TilingInit.kBL1Init, tmpHoAL1, tmpWoAL1, l1TilingInit.nBL1Init,
                    this->l1TilingCalc.khL1, this->l1TilingCalc.kwL1};
        l1Flags.iterateMNOrder = IterateMNOrder::ITER_M_FST;
        l1Flags.isBiasFullLoad = true;
        l1Flags.isFixpParamsFullLoad = true;
        return;
    }

    if (l1Flags.abL1Mode == L1TilingMode::FULL_LOAD_AL1) {
        TILING_LOG_DEBUG("Attributed to FULL_LOAD_AL1");
        uint64_t tmpKBL1 = 0;
        uint64_t tmpNBL1 = 0;
        IterKBL1(l1TilingInit.kAL1Init, tmpKBL1, l1TilingInit.hoAL1Init, l1TilingInit.woAL1Init,
                 l1TilingRange.nBL1StrictRange[0]);
        IterNBL1(l1TilingInit.kAL1Init, tmpKBL1, l1TilingInit.hoAL1Init, l1TilingInit.woAL1Init, tmpNBL1);
        l1Params = {l1TilingInit.kAL1Init, tmpKBL1, l1TilingInit.hoAL1Init, l1TilingInit.woAL1Init,
                    tmpNBL1, this->l1TilingCalc.khL1, this->l1TilingCalc.kwL1};
        l1Flags.iterateMNOrder = IterateMNOrder::ITER_N_FST;
        l1Flags.isBiasFullLoad = false;
        l1Flags.isFixpParamsFullLoad = false;
        return;
    }
    TILING_LOG_DEBUG("Attributed to NONE_FULL_LOAD");
    CoreL1TilingNoneFullLoadDecision();
}

void ConvTilingAlgorithmHWmode::InitL1TiLing()
{
    uint64_t kAL1Spilt = tilingIns_->isC04Flag ? this->l1TilingCalc.kAL1MaxSize : tilingIns_->isDmaFlag ?
                         this->l1TilingCalc.ci0KhL1KwL1 : tilingIns_->cubeInfo.k0;
    uint64_t kBL1Spilt = tilingIns_->isC04Flag ? this->l1TilingCalc.kBL1MaxSize : tilingIns_->isDmaFlag ?
                         this->l1TilingCalc.ci0KhL1KwL1 : this->l1TilingCalc.ci0HkWk;
    uint64_t hoAL1Spilt = l0Params.hoL0;
    uint64_t woAL1Spilt = (tilingIns_->isC04Flag && this->l1Flags.isWoL1MustFullLoadFlagC04) ?
                          AlignB(tilingIns_->shapeInfo.singleWo, tilingIns_->cubeInfo.m0) : l0Params.woL0;
    uint64_t nBL1Spilt = l0Params.nL0;
    // If kAL1 needs to be amplified, kd will be taken into account.
    // Thus, if kAL1 is fullload, the value of kd needs to be multiplied directly.
    // Note: the result of KAL1 is multiplied by khkw.
    uint64_t kAL1Full = this->l1TilingCalc.kAL1MaxSize * tilingIns_->shapeInfo.singlekD;
    uint64_t kBL1Full = this->l1TilingCalc.kBL1MaxSize;
    uint64_t hoAL1Full = tilingIns_->shapeInfo.singleHo;
    uint64_t woAL1Full = AlignB(tilingIns_->shapeInfo.singleWo, tilingIns_->cubeInfo.m0);
    uint64_t nBL1Full = tilingIns_->cubeInfo.n0 * tilingIns_->shapeInfo.singleCo1;

    l1TilingInitMap[L1TilingMode::NONE_FULL_LOAD] = {kAL1Spilt, kBL1Spilt, hoAL1Spilt, woAL1Spilt, nBL1Spilt};
    l1TilingInitMap[L1TilingMode::FULL_LOAD_AL1] = {kAL1Full, kBL1Spilt, hoAL1Full, woAL1Full, nBL1Spilt};
    l1TilingInitMap[L1TilingMode::FULL_LOAD_BL1] = {kAL1Spilt, kBL1Full, hoAL1Spilt, woAL1Spilt, nBL1Full};
    l1TilingInitMap[L1TilingMode::ALL_FULL_LOAD] = {kAL1Full, kBL1Full, hoAL1Full, woAL1Full, nBL1Full};

    InitABL1TilingMode();
    this->l1TilingInit = this->l1TilingInitMap.at(this->l1Flags.abL1Mode);
    RestrictRange(l1TilingRange.hoAL1Range, l1TilingRange.hoAL1StrictRange, l1TilingInit.hoAL1Init);
    RestrictRange(l1TilingRange.woAL1Range, l1TilingRange.woAL1StrictRange, l1TilingInit.woAL1Init);
    RestrictRange(l1TilingRange.kAL1Range, l1TilingRange.kAL1StrictRange, l1TilingInit.kAL1Init);
    RestrictRange(l1TilingRange.kBL1Range, l1TilingRange.kBL1StrictRange, l1TilingInit.kBL1Init);
    RestrictRange(l1TilingRange.nBL1Range, l1TilingRange.nBL1StrictRange, l1TilingInit.nBL1Init);

    PrintRanges(l1TilingRange.hoAL1StrictRange, "hoAL1StrictRange");
    PrintRanges(l1TilingRange.woAL1StrictRange, "woAL1StrictRange");
    PrintRanges(l1TilingRange.kAL1StrictRange, "kAL1StrictRange");
    PrintRanges(l1TilingRange.kBL1StrictRange, "kBL1StrictRange");
    PrintRanges(l1TilingRange.nBL1StrictRange, "nBL1StrictRange");
}

void ConvTilingAlgorithmHWmode::InitCalcL1Params()
{
    tilingIns_->shapeInfo.singleCi1 = CeilDiv(tilingIns_->shapeInfo.singleCi, tilingIns_->cubeInfo.k0);
    tilingIns_->shapeInfo.singleCo1 = CeilDiv(tilingIns_->shapeInfo.singleCo, tilingIns_->cubeInfo.n0);
    this->l1TilingCalc.ci0HkWk = tilingIns_->shapeInfo.singlekH * tilingIns_->shapeInfo.singlekW *
                                 tilingIns_->cubeInfo.k0;
    this->l1TilingCalc.ci0KhL1KwL1 = tilingIns_->cubeInfo.k0 * this->l1TilingCalc.khL1 * this->l1TilingCalc.kwL1;
    this->l1TilingCalc.woAL1Min = AlignB(min(static_cast<uint64_t>(tilingIns_->shapeInfo.singleWo), l0Params.woL0),
                                  tilingIns_->cubeInfo.m0);
    this->l1TilingCalc.hoAL1Min = min(static_cast<uint64_t>(tilingIns_->shapeInfo.singleHo), l0Params.hoL0);

    this->l1TilingCalc.kBL1MaxSize = tilingIns_->shapeInfo.singleCi1 * this->l1TilingCalc.ci0HkWk *
                                     tilingIns_->shapeInfo.singlekD;
    this->l1TilingCalc.nBL1MaxSize = tilingIns_->shapeInfo.singleCo1 * tilingIns_->cubeInfo.n0;

    this->l1TilingCalc.bL1MaxSize = this->l1TilingCalc.kBL1MaxSize *
                                    tilingIns_->shapeInfo.singleCo1 * tilingIns_->cubeInfo.n0 * weightDTypeSize;
    this->l1TilingCalc.biasL1MaxSize = tilingIns_->hasBias ? tilingIns_->shapeInfo.singleCo1 * tilingIns_->cubeInfo.n0 *
                                       biasDTypeSize : 0;
    this->l1TilingCalc.fixpParamsL1MaxSize = GetFixpParamsL1FullLoadSize();

    this->l1TilingCalc.bL1Minsize = this->l1TilingCalc.ci0HkWk * l0Params.nL0 * weightDTypeSize * this->dbValue.pbBL1;
    this->l1TilingCalc.biasL1MinSize = tilingIns_->hasBias ? l0Params.nL0 * biasDTypeSize : 0;
    this->l1TilingCalc.fixpParamsL1MinSize = l0Params.nL0 * tilingIns_->shapeInfo.channelWiseCoeff * FP16_DTYPE_SIZE;
    this->l1TilingCalc.kAL1MaxSize = tilingIns_->isDmaFlag ?
                                     tilingIns_->shapeInfo.singleCi1 * this->l1TilingCalc.ci0HkWk :
                                     tilingIns_->shapeInfo.singleCi1 * tilingIns_->cubeInfo.k0; // without khkw
    if (tilingIns_->isDmaFlag) {
        uint64_t woAL1Max = AlignB(tilingIns_->shapeInfo.singleWo, tilingIns_->cubeInfo.m0);
        this->l1TilingCalc.aL1MaxSize = tilingIns_->shapeInfo.singleHo * woAL1Max * tilingIns_->shapeInfo.singlekH *
                                        tilingIns_->shapeInfo.singlekW * this->l1TilingCalc.kAL1MaxSize * fMapDTypeSize;
        this->l1TilingCalc.aL1Minsize = this->l1TilingCalc.hoAL1Min * this->l1TilingCalc.woAL1Min *
                                        this->l1TilingCalc.ci0KhL1KwL1 * fMapDTypeSize * this->dbValue.pbAL1;
        this->l1TilingCalc.ubMaxSize = tilingIns_->shapeInfo.singleHo * woAL1Max *
                                       tilingIns_->cubeInfo.k0 * fMapDTypeSize;
        this->l1TilingCalc.bL1Minsize = this->l1TilingCalc.ci0KhL1KwL1 * l0Params.nL0 * weightDTypeSize *
                                        this->dbValue.pbBL1;
    } else {
        auto tmpSingleHi = InferHiL1(tilingIns_->shapeInfo.singleHo, tilingIns_->shapeInfo.orgHi);
        auto tmpSingleWi = InferWiL1(AlignB(tilingIns_->shapeInfo.singleWo, tilingIns_->cubeInfo.m0),
                            tilingIns_->shapeInfo.orgWi);
        this->l1TilingCalc.aL1MaxSize = tilingIns_->shapeInfo.singlekD * tmpSingleHi * tmpSingleWi *
                                        this->l1TilingCalc.kAL1MaxSize * fMapDTypeSize;
        this->l1TilingCalc.wiAL1Min = InferWiL1(this->l1TilingCalc.woAL1Min, tilingIns_->shapeInfo.orgWi);
        this->l1TilingCalc.hiAL1Min = InferHiL1(this->l1TilingCalc.hoAL1Min, tilingIns_->shapeInfo.orgHi);
        this->l1TilingCalc.aL1Minsize = this->l1TilingCalc.wiAL1Min * this->l1TilingCalc.hiAL1Min *
                                        tilingIns_->cubeInfo.k0 * fMapDTypeSize * this->dbValue.pbAL1;
        this->l1TilingCalc.bL1Minsize = this->l1TilingCalc.ci0HkWk * l0Params.nL0 * weightDTypeSize *
                                        this->dbValue.pbBL1;
    }
    Printl1TilingCalc();
}

void ConvTilingAlgorithmHWmode::Printl1TilingCalc()
{
    TILING_LOG_DEBUG("In Printl1TilingCalc");
    TILING_LOG_DEBUG("khL1: %lu, kwL1: %lu", this->l1TilingCalc.khL1, this->l1TilingCalc.kwL1);
    TILING_LOG_DEBUG("l0Params woL0: %lu, hoL0: %lu, nL0: %lu", l0Params.woL0, l0Params.hoL0, l0Params.nL0);
    TILING_LOG_DEBUG("cubeInfo m0: %u, k0: %u, n0: %u",
                     tilingIns_->cubeInfo.m0, tilingIns_->cubeInfo.k0, tilingIns_->cubeInfo.n0);
    TILING_LOG_DEBUG("shapeInfo.singleCi: %ld,  shapeInfo.singleCo: %ld, shapeInfo.singleCi1: %lu, \
                      shapeInfo.singleCo1: %lu, shapeInfo.singleWo: %ld, shapeInfo.singleHo: %ld",
                      tilingIns_->shapeInfo.singleCi, tilingIns_->shapeInfo.singleCo, tilingIns_->shapeInfo.singleCi1,
                      tilingIns_->shapeInfo.singleCo1, tilingIns_->shapeInfo.singleWo, tilingIns_->shapeInfo.singleHo);
    TILING_LOG_DEBUG("dbValue.pbAL1: %lu, dbValue.pbBL1: %lu",
                     static_cast<uint64_t>(this->dbValue.pbAL1), static_cast<uint64_t>(this->dbValue.pbBL1));
    TILING_LOG_DEBUG("tmpSingleHi: %lu, tmpSingleWi: %ld",
        InferHiL1(tilingIns_->shapeInfo.singleHo, tilingIns_->shapeInfo.orgHi),
        InferWiL1(static_cast<uint64_t>(tilingIns_->shapeInfo.singleWo), tilingIns_->shapeInfo.orgWi));
    TILING_LOG_DEBUG("l1TilingCalc ci0HkWk: %lu, woAL1Min: %lu, hoAL1Min: %lu, kAL1MaxSize: %lu, kBL1MaxSize: %lu, \
        aL1MaxSize: %lu, bL1MaxSize: %lu, biasL1MaxSize: %lu, fixpParamsL1MaxSize: %lu, \
        bL1Minsize: %lu, biasL1MinSize: %lu, fixpParamsL1MinSize: %lu, wiAL1Min: %lu, hiAL1Min: %lu, aL1Minsize: %lu,",
    this->l1TilingCalc.ci0HkWk,
    this->l1TilingCalc.woAL1Min,
    this->l1TilingCalc.hoAL1Min,
    this->l1TilingCalc.kAL1MaxSize,
    this->l1TilingCalc.kBL1MaxSize,
    this->l1TilingCalc.aL1MaxSize,
    this->l1TilingCalc.bL1MaxSize,
    this->l1TilingCalc.biasL1MaxSize,
    this->l1TilingCalc.fixpParamsL1MaxSize,
    this->l1TilingCalc.bL1Minsize,
    this->l1TilingCalc.biasL1MinSize,
    this->l1TilingCalc.fixpParamsL1MinSize,
    this->l1TilingCalc.wiAL1Min,
    this->l1TilingCalc.hiAL1Min,
    this->l1TilingCalc.aL1Minsize);
}


void ConvTilingAlgorithmHWmode::InitCalcL1ParamsC04Mode()
{
    tilingIns_->shapeInfo.singleCi1 = CONST_VALUE_1;
    tilingIns_->shapeInfo.singleCo1 = CeilDiv(tilingIns_->shapeInfo.singleCo, tilingIns_->cubeInfo.n0);
    this->l1TilingCalc.ci0HkWk = tilingIns_->shapeInfo.singlekH * tilingIns_->shapeInfo.singlekW *
                                 tilingIns_->cubeInfo.k0;
    this->l1TilingCalc.c04KSizeAlign = AlignB(C04_CIN_SIZE * tilingIns_->shapeInfo.singlekH *
                                              tilingIns_->shapeInfo.singlekW,
                                              tilingIns_->cubeInfo.k0);
    auto tmpWoAL1Min = min(static_cast<uint64_t>(tilingIns_->shapeInfo.singleWo), l0Params.woL0);
    tmpWoAL1Min = this->l1Flags.isWoL1MustFullLoadFlagC04 ? tilingIns_->shapeInfo.singleWo : tmpWoAL1Min;
    this->l1TilingCalc.woAL1Min = AlignB(tmpWoAL1Min, tilingIns_->cubeInfo.m0);
    this->l1TilingCalc.hoAL1Min = min(static_cast<uint64_t>(tilingIns_->shapeInfo.singleHo), l0Params.hoL0);
    this->l1TilingCalc.kAL1MaxSize = 1;
    this->l1TilingCalc.kBL1MaxSize = this->l1TilingCalc.c04KSizeAlign;

    auto tmpSingleHi = InferHiL1(tilingIns_->shapeInfo.singleHo, tilingIns_->shapeInfo.orgHi);
    auto tmpSingleWi = this->l1Flags.isWoL1MustFullLoadFlagC04 ? tilingIns_->shapeInfo.orgWi :
         InferWiL1(AlignB(tilingIns_->shapeInfo.singleWo, tilingIns_->cubeInfo.m0), tilingIns_->shapeInfo.orgWi);
    this->l1TilingCalc.aL1MaxSize = AlignB(tmpSingleHi * tmpSingleWi, C0_SIZE / (fMapDTypeSize * C04_CIN_SIZE)) *
                                    C04_CIN_SIZE * fMapDTypeSize;
    this->l1TilingCalc.biasL1MaxSize = tilingIns_->hasBias ?
        tilingIns_->shapeInfo.singleCo1 * tilingIns_->cubeInfo.n0 * biasDTypeSize : 0;
    this->l1TilingCalc.fixpParamsL1MaxSize = tilingIns_->shapeInfo.singleCo1 * tilingIns_->cubeInfo.n0 *
        tilingIns_->shapeInfo.channelWiseCoeff * FP16_DTYPE_SIZE;
    this->l1TilingCalc.bL1MaxSize =
        this->l1TilingCalc.kBL1MaxSize * tilingIns_->shapeInfo.singleCo1 * tilingIns_->cubeInfo.n0 * weightDTypeSize;

    this->l1TilingCalc.bL1Minsize = this->l1TilingCalc.kBL1MaxSize * l0Params.nL0 *
                                    weightDTypeSize  * this->dbValue.pbBL1;
    this->l1TilingCalc.biasL1MinSize = tilingIns_->hasBias ? l0Params.nL0 * biasDTypeSize : 0;
    this->l1TilingCalc.fixpParamsL1MinSize = l0Params.nL0 * tilingIns_->shapeInfo.channelWiseCoeff * FP16_DTYPE_SIZE;

    this->l1TilingCalc.wiAL1Min = this->l1Flags.isWoL1MustFullLoadFlagC04 ?
        tilingIns_->shapeInfo.orgWi : InferWiL1(this->l1TilingCalc.woAL1Min, tilingIns_->shapeInfo.orgWi);
    this->l1TilingCalc.hiAL1Min = InferHiL1(this->l1TilingCalc.hoAL1Min, tilingIns_->shapeInfo.orgHi);
    this->l1TilingCalc.aL1Minsize =
        AlignB(this->l1TilingCalc.hiAL1Min * this->l1TilingCalc.wiAL1Min, C0_SIZE / (fMapDTypeSize * C04_CIN_SIZE)) *
        C04_CIN_SIZE * fMapDTypeSize  * this->dbValue.pbAL1;
}

int64_t ConvTilingAlgorithmHWmode::CheckMinL1Tiling()
{
    TILING_LOG_DEBUG("CheckMinL1Tiling");
    uint64_t minKBL1 = 0;
    uint64_t minwoAL1 = tilingIns_->shapeInfo.singleWo < tilingIns_->cubeInfo.m0 ?
                        tilingIns_->shapeInfo.singleWo : tilingIns_->cubeInfo.m0;
    if (tilingIns_->isC04Flag) {
        minKBL1 = AlignB(C04_CIN_SIZE * static_cast<uint64_t>(tilingIns_->shapeInfo.singlekH) *
                         static_cast<uint64_t>(tilingIns_->shapeInfo.singlekW),
                         tilingIns_->cubeInfo.k0);
        minwoAL1 = this->l1Flags.isWoL1MustFullLoadFlagC04 ? tilingIns_->shapeInfo.singleWo : tilingIns_->cubeInfo.m0;
    } else {
        minKBL1 = tilingIns_->cubeInfo.k0 * static_cast<uint64_t>(tilingIns_->shapeInfo.singlekH) *
                  static_cast<uint64_t>(tilingIns_->shapeInfo.singlekW);
    }
    uint64_t minHoAL1 = tilingIns_->shapeInfo.singleWo < tilingIns_->cubeInfo.m0 ?
                        CeilDiv(tilingIns_->cubeInfo.m0, tilingIns_->shapeInfo.singleWo) : 1;
    L1TilingParams inputTiling = {tilingIns_->cubeInfo.k0, minKBL1, minHoAL1, minwoAL1, tilingIns_->cubeInfo.n0, 1, 1};
    if (CalcCurL1Size(inputTiling, inputTiling.nBL1) > tilingIns_->platformInfo.l1Size) {
        TILING_LOG_ERROR("L1 tiling failed, l1 minLoadSize is exceeding L1size.");
        return -1;
    }

    return 0;
}

void ConvTilingAlgorithmHWmode::GetCaseStatus()
{
    if (tilingIns_->isC04Flag) {
        this->l1Flags.isWoL1MustFullLoadFlagC04 = !(tilingIns_->shapeInfo.orgHi == 1);
    }
}

int64_t ConvTilingAlgorithmHWmode::GetL1Tiling()
{
    GetCaseStatus();
    if (tilingIns_->isC04Flag) {
        InitCalcL1ParamsC04Mode();
    } else {
        InitCalcL1Params();
    }
    GetL1TilingRange();
    TILING_LOG_DEBUG("Before InitL1TiLing this->dbValue.pbAL1: %lu", static_cast<uint64_t>(this->dbValue.pbAL1));
    TILING_LOG_DEBUG("Before InitL1TiLing this->dbValue.pbBL1: %lu", static_cast<uint64_t>(this->dbValue.pbBL1));
    InitL1TiLing();
    TILING_LOG_DEBUG("After InitL1TiLing this->dbValue.pbAL1: %lu", static_cast<uint64_t>(this->dbValue.pbAL1));
    TILING_LOG_DEBUG("After InitL1TiLing this->dbValue.pbBL1: %lu", static_cast<uint64_t>(this->dbValue.pbBL1));
    UpdateAL1DoubleBufferFirst();
    CoreL1TilingDecision();
    UpdateAL1DoubleBufferSecond();
    UpdateBiasFixpParamsL1Fullload();
    SetL1TilingRes();
    return 0;
}
} // namespace conv_tiling_algo_hw