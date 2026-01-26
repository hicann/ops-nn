/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file foreach_non_finite_check_and_unscale_regbase.h
 * \brief
 */
#ifndef FOREACH_NON_FINITE_CHECK_AND_UNSCALE_REGBASE_H
#define FOREACH_NON_FINITE_CHECK_AND_UNSCALE_REGBASE_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "../inc/kernel_utils.h"
#include "../inc/platform.h"
#include "../inc/load_store_utils.h"

namespace ForeachNonFiniteCheckAndUnscaleRegbase {
using namespace AscendC;
using AscendC::MicroAPI::CreateMask;
using AscendC::MicroAPI::LoadDist;
using AscendC::MicroAPI::LocalMemBar;
using AscendC::MicroAPI::MaskPattern;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::MemType;
using AscendC::MicroAPI::RegTensor;
using AscendC::MicroAPI::StoreDist;
using AscendC::MicroAPI::UpdateMask;

__aicore__ inline constexpr uint32_t GetUbBlockSize()
{
    return 32U;
}

constexpr int32_t BUFFER_NUM = 2;
constexpr uint32_t BYTE_BLOCK = GetUbBlockSize();
constexpr uint32_t TEM_BUFFER_REMAIN = 32;

template <typename T>
class ForeachNonFiniteCheckAndUnscaleNDRegbase
{
public:
    __aicore__ inline ForeachNonFiniteCheckAndUnscaleNDRegbase(){};
    __aicore__ inline void Init(
        GM_ADDR scaled_grads, GM_ADDR found_inf, GM_ADDR inv_scale, GM_ADDR workspace,
        const ForeachNonFiniteCheckAndUnscaleRegbaseTilingData* __restrict tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ParseTilingData();
    __aicore__ inline void SingleTensorProcess(int64_t dataCount);
    __aicore__ inline void CopyIn(uint16_t index, int64_t dataCount);
    __aicore__ inline void Compute(uint16_t index, int64_t dataCount);
    __aicore__ inline void CopyOut(uint16_t index, int64_t dataCount);
    __aicore__ inline bool IsNonFinite(float value);
    __aicore__ inline __gm__ T* GetTensorAddr(uint16_t index);

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> copyInQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> copyOutQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> tempValQueue_;
    TBuf<TPosition::VECCALC> maxBuf_;
    TBuf<TPosition::VECCALC> minBuf_;
    GlobalTensor<T> scaledGradsGM_;
    GlobalTensor<float> foundInfGM_;
    GlobalTensor<float> invScaleGM_;
    LocalTensor<float> maxLocal_;
    LocalTensor<float> minLocal_;
    GM_ADDR scaledGradsPtr_ = nullptr;
    int64_t blockIdx_ = 0;
    float invScaleVal_ = 0;
    int32_t perBlockCount_ = 0;
    uint32_t maxDataCount_ = 0;
    // tiling params
    const ForeachNonFiniteCheckAndUnscaleRegbaseTilingData* __restrict tilingDataInClass_ = nullptr;
    uint32_t scaledGradsUbSize_ = 0;
    uint32_t reduceTempValUbSize_ = 0;
    const int64_t* __restrict tensorDataCountList_ = nullptr;
    uint16_t tensorStart_ = 0;
    uint16_t tensorEnd_ = 0;
    int64_t tensorStartOffset_ = 0;
    int64_t tensorEndOffset_ = 0;
};

template <typename T>
__aicore__ inline void ForeachNonFiniteCheckAndUnscaleNDRegbase<T>::Init(
    GM_ADDR scaled_grads, GM_ADDR found_inf, GM_ADDR inv_scale, GM_ADDR workspace,
    const ForeachNonFiniteCheckAndUnscaleRegbaseTilingData* __restrict tilingData)
{
    tilingDataInClass_ = tilingData;
    blockIdx_ = GetBlockIdx();
    scaledGradsPtr_ = scaled_grads;
    ParseTilingData();
    foundInfGM_.SetGlobalBuffer((__gm__ float*)found_inf, 1);
    invScaleGM_.SetGlobalBuffer((__gm__ float*)inv_scale, 1);
    pipe_.InitBuffer(copyInQueue_, BUFFER_NUM, scaledGradsUbSize_);
    pipe_.InitBuffer(copyOutQueue_, BUFFER_NUM, scaledGradsUbSize_);
    pipe_.InitBuffer(tempValQueue_, BUFFER_NUM, reduceTempValUbSize_);
    pipe_.InitBuffer(maxBuf_, TEM_BUFFER_REMAIN);
    pipe_.InitBuffer(minBuf_, TEM_BUFFER_REMAIN);
    invScaleVal_ = invScaleGM_.GetValue(0);
    perBlockCount_ = BYTE_BLOCK / sizeof(T);
    maxDataCount_ = scaledGradsUbSize_ / sizeof(T);
}

template <typename T>
__aicore__ inline void ForeachNonFiniteCheckAndUnscaleNDRegbase<T>::Process()
{
    maxLocal_ = maxBuf_.Get<float>();
    Duplicate(maxLocal_, float(0.0), TEM_BUFFER_REMAIN / sizeof(float));
    minLocal_ = minBuf_.Get<float>();
    Duplicate(minLocal_, float(0.0), TEM_BUFFER_REMAIN / sizeof(float));

    for (uint16_t i = tensorStart_; i <= tensorEnd_; i++) {
        int64_t cursorStart = 0;
        int64_t cursorEnd = tensorDataCountList_[i] - 1;
        int64_t dataCount = 0;
        if (i == tensorStart_) {
            cursorStart = tensorStartOffset_;
        }
        if (i == tensorEnd_ && tensorEndOffset_ < cursorEnd) {
            cursorEnd = tensorEndOffset_;
        }
        dataCount = cursorEnd - cursorStart + 1;
        scaledGradsGM_.SetGlobalBuffer(GetTensorAddr(i) + cursorStart);
        SingleTensorProcess(dataCount);
    }

    event_t eventIDVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIDVToS);
    WaitFlag<HardEvent::V_S>(eventIDVToS);
    if (IsNonFinite(maxLocal_.GetValue(0)) || IsNonFinite(minLocal_.GetValue(0))) {
        foundInfGM_.SetValue(0, 1.0);
        DataCacheCleanAndInvalid<float, CacheLine::SINGLE_CACHE_LINE>(foundInfGM_);
    }
}

template <typename T>
__aicore__ inline void ForeachNonFiniteCheckAndUnscaleNDRegbase<T>::SingleTensorProcess(int64_t dataCount)
{
    // Batch handling and calculation.
    uint32_t copyTimes = CeilDivision(dataCount, static_cast<int64_t>(maxDataCount_));
    for (uint32_t i = 0; i < copyTimes; i++) {
        int64_t tempCount = maxDataCount_;
        if ((i + 1 == copyTimes) && (dataCount % maxDataCount_)) {
            tempCount = dataCount % maxDataCount_;
        }
        CopyIn(i, tempCount);
        int64_t alignCount = CeilDivision(tempCount, static_cast<int64_t>(perBlockCount_)) * perBlockCount_;
        Compute(i, alignCount);
        CopyOut(i, tempCount);
    }
}

template <typename T>
__aicore__ inline void ForeachNonFiniteCheckAndUnscaleNDRegbase<T>::ParseTilingData()
{
    scaledGradsUbSize_ = tilingDataInClass_->scaledGradsUbSize;
    reduceTempValUbSize_ = tilingDataInClass_->reduceTempValUbSize;
    tensorDataCountList_ = tilingDataInClass_->tensorDataCountList;
    tensorStart_ = tilingDataInClass_->tensorStartList[blockIdx_];
    tensorEnd_ = tilingDataInClass_->tensorEndList[blockIdx_];
    tensorStartOffset_ = tilingDataInClass_->tensorStartOffsetList[blockIdx_];
    tensorEndOffset_ = tilingDataInClass_->tensorEndOffsetList[blockIdx_];
}

template <typename T>
__aicore__ inline void ForeachNonFiniteCheckAndUnscaleNDRegbase<T>::CopyIn(uint16_t index, int64_t dataCount)
{
    LocalTensor<T> copyInLT = copyInQueue_.AllocTensor<T>();
    if (dataCount % perBlockCount_) {
        struct DataCopyParams copyParams = {1, 0, 0, 0};
        copyParams.blockLen = dataCount * sizeof(T);
        struct DataCopyPadParams padParams = {true, 0, 0, 0};
        int64_t alignDataCount = CeilDivision(dataCount, static_cast<int64_t>(perBlockCount_)) * perBlockCount_;
        padParams.rightPadding = alignDataCount - dataCount;
        DataCopyPad(copyInLT, scaledGradsGM_[1ULL * index * maxDataCount_], copyParams, padParams);
    } else {
        DataCopy(copyInLT, scaledGradsGM_[1ULL * index * maxDataCount_], dataCount);
    }
    copyInQueue_.EnQue(copyInLT);
}

template <typename T>
__aicore__ inline void ForeachNonFiniteCheckAndUnscaleNDRegbase<T>::Compute(uint16_t index, int64_t dataCount)
{
    LocalTensor<T> computeInLT = copyInQueue_.DeQue<T>();
    LocalTensor<T> computeOutLT = copyOutQueue_.AllocTensor<T>();

    __local_mem__ T* inUbAddr = (__ubuf__ T*)computeInLT.GetPhyAddr();
    __local_mem__ T* outUbAddr = (__ubuf__ T*)computeOutLT.GetPhyAddr();
    __local_mem__ float* minUbAddr = (__ubuf__ float*)minLocal_.GetPhyAddr();
    __local_mem__ float* maxUbAddr = (__ubuf__ float*)maxLocal_.GetPhyAddr();

    uint32_t dataCountPerLoop = platform::GetVRegSize() / sizeof(float);
    uint16_t repeatTimes = CeilDivision(dataCount, dataCountPerLoop);
    uint32_t sreg = (uint32_t)dataCount;
    float invScaleVal = invScaleVal_;

    __VEC_SCOPE__
    {
        RegTensor<float> min;
        RegTensor<float> max;
        RegTensor<float> x;
        RegTensor<float> y;
        RegTensor<float> lastMin;
        RegTensor<float> lastMax;
        MaskReg pregMain = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregMerge = CreateMask<float, MaskPattern::VL1>();
        MaskReg maskReg;
        Duplicate(min, float(0.0), pregMain);
        Duplicate(max, float(0.0), pregMain);
        for (uint16_t i = 0; i < repeatTimes - 1; i++) {
            maskReg = UpdateMask<float>(sreg);
            ops::LoadOneTensorForDtypeT<T>(inUbAddr, x, maskReg, i * dataCountPerLoop);
            Max(max, max, x, maskReg);
            Min(min, min, x, maskReg);
            Muls(y, x, invScaleVal, maskReg);
            ops::StoreOneTensorForDtypeT<T>(outUbAddr, y, maskReg, i * dataCountPerLoop);
        }
        {
            maskReg = UpdateMask<float>(sreg);
            ops::LoadOneTensorForDtypeT<T>(inUbAddr, x, maskReg, (repeatTimes - 1) * dataCountPerLoop);
            Adds(x, x, float(0.0), maskReg);
            Max(max, max, x, pregMain);
            Min(min, min, x, pregMain);
            Muls(y, x, invScaleVal, maskReg);
            ops::StoreOneTensorForDtypeT<T>(outUbAddr, y, maskReg, (repeatTimes - 1) * dataCountPerLoop);
        }
        DataCopy(lastMin, (__local_mem__ float*)(minUbAddr));
        DataCopy(lastMax, (__local_mem__ float*)(maxUbAddr));
        ReduceMax(max, max, pregMain);
        ReduceMin(min, min, pregMain);
        Max(lastMax, lastMax, max, pregMain);
        Min(lastMin, lastMin, min, pregMain);
        DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(((__local_mem__ float*)minUbAddr), lastMin, pregMerge);
        DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(((__local_mem__ float*)maxUbAddr), lastMax, pregMerge);
    }
    copyOutQueue_.EnQue(computeOutLT);
    copyInQueue_.FreeTensor(computeInLT);
}

template <typename T>
__aicore__ inline void ForeachNonFiniteCheckAndUnscaleNDRegbase<T>::CopyOut(uint16_t index, int64_t dataCount)
{
    LocalTensor<T> copyOutLT = copyOutQueue_.DeQue<T>();
    if (dataCount % perBlockCount_) {
        struct DataCopyParams copyParams = {1, 0, 0, 0};
        copyParams.blockLen = dataCount * sizeof(T);
        DataCopyPad(scaledGradsGM_[1ULL * index * maxDataCount_], copyOutLT, copyParams);
    } else {
        DataCopy(scaledGradsGM_[1ULL * index * maxDataCount_], copyOutLT, dataCount);
    }

    copyOutQueue_.FreeTensor(copyOutLT);
}

template <typename T>
__aicore__ inline bool ForeachNonFiniteCheckAndUnscaleNDRegbase<T>::IsNonFinite(float value)
{
    uint32_t tempValue = *((uint32_t*)&value);
    if ((tempValue & 0x7FFFFFFF) >> 23 == 0xFF) {
        return true;
    } else {
        return false;
    }
}

template <typename T>
__aicore__ inline __gm__ T* ForeachNonFiniteCheckAndUnscaleNDRegbase<T>::GetTensorAddr(uint16_t index)
{
    __gm__ uint64_t* dataAddr = reinterpret_cast<__gm__ uint64_t*>(scaledGradsPtr_);
    uint64_t tensorPtrOffset = *dataAddr; // The offset of the data address from the first address.
    // Moving 3 bits to the right means dividing by sizeof(uint64 t).
    __gm__ uint64_t* tensorPtr = dataAddr + (tensorPtrOffset >> 3);
    return reinterpret_cast<__gm__ T*>(*(tensorPtr + index));
}

} // namespace ForeachNonFiniteCheckAndUnscaleRegbase

#endif // FOREACH_NON_FINITE_CHECK_AND_UNSCALE_REGBASE_H