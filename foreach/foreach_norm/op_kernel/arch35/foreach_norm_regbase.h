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
 * \file foreach_norm_regbase.h
 * \brief
 */

#ifndef FOREACH_NORM_N_D_H
#define FOREACH_NORM_N_D_H

#include "../../foreach_utils/arch35/foreach_regbase_common.h"
#include "../../inc/load_store_utils.h"
#include "../../inc/platform.h"

namespace ForeachNorm {

using namespace AscendC;
constexpr int32_t VL_SIZE = platform::GetVRegSize();

constexpr uint16_t MAX_TENSOR_CONT = 256;
constexpr uint16_t MAX_CORE_CONT = 80;
constexpr uint8_t ZERO_SCALAR_NORM_MODEL_CODE = 0;
constexpr uint8_t ONE_SCALAR_NORM_MODEL_CODE = 1;
constexpr uint8_t TWO_SCALAR_NORM_MODEL_CODE = 2;
constexpr uint8_t POSITIVE_INF_SCALAR_NORM_MODEL_CODE = 3;
constexpr uint8_t NEGATIVE_INF_SCALAR_NORM_MODEL_CODE = 4;
constexpr uint8_t DEFAULT_SCALAR_NORM_MODEL_CODE = 5;

/**
 * modelCode:
 * 0 p=else... the default operator(not used now)
 * 1 p=0 Calculate the count of not-zero element in each tensor. (not used now)
 * 2 p=1 AbsAndNotNeedPower NotNeedSqrt(now is default as we now only consider p=1 || p=2)
 * 3 p=2 MulSelf(we don't need abs this time) Sqrt(not Power(self,1/p))
 * 4 p=+inf Calculate the max Abs(element) in each tensor.
 * 5 p=-inf Calculate the min Abs(element) in each tensor. (not used now)
 */
template <typename T>
__aicore__ inline void SetValueAdapter(LocalTensor<T>& outLocal, float value, uint16_t calCount)
{
    Duplicate<T>(outLocal, T(value), calCount);
};

template <>
__aicore__ inline void SetValueAdapter<bfloat16_t>(LocalTensor<bfloat16_t>& outLocal, float value, uint16_t calCount)
{
    Duplicate<bfloat16_t>(outLocal, ToBfloat16(value), calCount);
};

template <typename T, uint8_t modelCode, typename Tiling>
class ForeachNormNDRegbase
{
public:
    __aicore__ inline ForeachNormNDRegbase(){};
    __aicore__ inline void Init(
        GM_ADDR inputs, GM_ADDR outputs, GM_ADDR workspace, const Tiling* tilingData, TPipe* tPipe)
    {
        workTensorGM_.SetGlobalBuffer((__gm__ float*)(workspace), MAX_TENSOR_CONT + MAX_CORE_CONT);
        blockIdx_ = GetBlockIdx();
        inDesc_ = ListTensorDesc((__gm__ void*)inputs);
        outDesc_ = ListTensorDesc((__gm__ void*)outputs);
        ParseTilingData(tilingData);
        tPipe->InitBuffer(inQueue_, BUFFER_NUM_, inputsTensorUbSize_ * sizeof(T));
        tPipe->InitBuffer(outQueue_, 1, maxTensorNumPerCore_ * sizeof(float));
        maxDataCount_ = inputsTensorUbSize_;
    }

    __aicore__ inline void Process()
    {
        LocalTensor<float> outLocal = outQueue_.AllocTensor<float>();
        for (uint16_t i = tensorStart_; i <= tensorEnd_; i++) {
            int64_t cursorStart = 0;
            int64_t cursorEnd = tensorDataCountList_[i] - 1;
            int64_t dataCount = 0;
            if (i == tensorStart_) {
                cursorStart = tensorStartOffset_;
            }
            if (i == tensorEnd_) {
                cursorEnd = tensorEndOffset_;
            }
            dataCount = cursorEnd - cursorStart + 1;
            inTensorGM_.SetGlobalBuffer(GetGlobalTensorAddr(i, inDesc_) + cursorStart);
            SingleTensorProcess(outLocal, dataCount, i - tensorStart_);
        }
        outQueue_.EnQue(outLocal);
        outLocal = outQueue_.DeQue<float>();

        uint16_t tensorNum = tensorEnd_ - tensorStart_ + 1;
        DataCopyExtParams copyParams{1, tensorNum * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
        DataCopyPad(workTensorGM_[1ULL * coreMiddleOffset_], outLocal, copyParams);

        outQueue_.FreeTensor(outLocal);

        SyncAll();

        // Stage2 Reduce2 and sqrt
        for (uint16_t i = blockIdx_; i < totalTensorCount_; i += needCoreNum_) {
            outTensorGM_.SetGlobalBuffer(GetGlobalTensorAddr(i, outDesc_));
            if (tensorDataCountList_[i] == 0) {
                OutputZero();
                continue;
            }
            ReduceCompute(tensorMiddleCountList_[i], tensorMiddleStartList_[i]);
            CopyOut();
        }
    }
    __aicore__ inline void ReduceCompute(uint16_t dataCount, uint16_t offset)
    {
        LocalTensor<float> dataLocal = inQueue_.AllocTensor<float>();

        // 结构体DataCopyExtParams最后一个参数是rsv保留位
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(dataCount * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> padParams{true, 0, 0, 0};
        DataCopyPad(dataLocal, workTensorGM_[1ULL * offset], copyParams, padParams);

        inQueue_.EnQue(dataLocal);
        dataLocal = inQueue_.DeQue<float>();
        LocalTensor<T> outLocal = outQueue_.AllocTensor<T>();

        if (dataCount > 1) {
            ReduceComputeImpl<true, modelCode>(dataLocal, outLocal, dataCount);
        } else {
            ReduceComputeImpl<false, modelCode>(dataLocal, outLocal, dataCount);
        }
        inQueue_.FreeTensor(dataLocal);
        outQueue_.EnQue(outLocal);
    }

    template <bool isMulitNum, uint8_t modelOrd>
    __aicore__ inline void ReduceComputeImpl(LocalTensor<float> inLocal, LocalTensor<T> outLocal, uint16_t dataCount)
    {
        __local_mem__ float* inUbAddr = (__ubuf__ float*)inLocal.GetPhyAddr();
        __local_mem__ T* outUbAddr = (__ubuf__ T*)outLocal.GetPhyAddr();

        uint32_t dataCountPerLoop = VL_SIZE / sizeof(float);
        uint16_t repeatTimes = CeilDivision(dataCount, dataCountPerLoop);
        uint32_t sreg = (uint32_t)dataCount;

        __VEC_SCOPE__
        {
            MicroAPI::MaskReg maskReg;
            MicroAPI::RegTensor<float> inRegToFloat;
            MicroAPI::RegTensor<float> reduceFloat;
            MicroAPI::MaskReg pregMain = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
            Duplicate(reduceFloat, (float)0.0, pregMain);
            for (uint16_t i = 0; i < (uint16_t)repeatTimes; i++) {
                maskReg = MicroAPI::UpdateMask<float>(sreg);
                ops::LoadOneTensorForDtypeT<float>(inUbAddr, inRegToFloat, maskReg, i * dataCountPerLoop);
                if constexpr (isMulitNum) {
                    if constexpr (modelOrd == POSITIVE_INF_SCALAR_NORM_MODEL_CODE) {
                        Max(reduceFloat, reduceFloat, inRegToFloat, pregMain);
                    } else {
                        Add(reduceFloat, reduceFloat, inRegToFloat, pregMain);
                    }
                }
            }
            if constexpr (isMulitNum) {
                if constexpr (modelOrd == POSITIVE_INF_SCALAR_NORM_MODEL_CODE) {
                    ReduceMax<float>(inRegToFloat, reduceFloat, maskReg);
                } else {
                    ReduceSum<float>(inRegToFloat, reduceFloat, maskReg);
                }
            }
            if constexpr (modelOrd == TWO_SCALAR_NORM_MODEL_CODE) {
                Sqrt(inRegToFloat, inRegToFloat, pregMain);
            }
            ops::StoreOneTensorForDtypeT<T>(outUbAddr, inRegToFloat, pregMain, 0);
        }
    }

    __aicore__ inline void SingleTensorProcess(LocalTensor<float> outLocal, int64_t dataCount, uint16_t outOffset)
    {
        // Batch handling and calculation.
        int64_t quotient = CeilDivision(dataCount, maxDataCount_);
        for (int64_t i = 0; i < quotient; i++) {
            int64_t currentDataCount =
                (i == (quotient - 1)) ? (dataCount - (quotient - 1) * maxDataCount_) : maxDataCount_;
            CopyIn(i, currentDataCount);
            Compute(outLocal, currentDataCount, i, outOffset);
        }
    }

    template <bool isFirst, uint8_t modelOrd>
    __aicore__ inline void DoCompute(
        LocalTensor<T> inLocal, LocalTensor<float> outLocal, int64_t dataCount, uint16_t outOffset)
    {
        __local_mem__ T* inUbAddr = (__ubuf__ T*)inLocal.GetPhyAddr();
        __local_mem__ float* outUbAddr = (__ubuf__ float*)outLocal.GetPhyAddr();

        uint32_t dataCountPerLoop = VL_SIZE / sizeof(float);
        uint16_t repeatTimes = CeilDivision(dataCount, dataCountPerLoop);
        uint32_t sreg = (uint32_t)dataCount;

        __VEC_SCOPE__
        {
            MicroAPI::MaskReg maskReg;
            MicroAPI::RegTensor<float> inRegToFloat;
            MicroAPI::RegTensor<float> reduceFloat;
            MicroAPI::RegTensor<float> outFloat;
            MicroAPI::MaskReg pregMain = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
            Duplicate(reduceFloat, (float)0.0, pregMain);
            for (uint16_t i = 0; i < (uint16_t)repeatTimes; i++) {
                maskReg = MicroAPI::UpdateMask<float>(sreg);
                ops::LoadOneTensorForDtypeT<T>(inUbAddr, inRegToFloat, maskReg, i * dataCountPerLoop);
                if constexpr (modelOrd == POSITIVE_INF_SCALAR_NORM_MODEL_CODE) {
                    Abs(inRegToFloat, inRegToFloat, maskReg);
                    Max(reduceFloat, reduceFloat, inRegToFloat, pregMain);
                } else if constexpr (modelOrd == ONE_SCALAR_NORM_MODEL_CODE) {
                    Abs(inRegToFloat, inRegToFloat, maskReg);
                    Add(reduceFloat, reduceFloat, inRegToFloat, pregMain);
                } else {
                    Mul(inRegToFloat, inRegToFloat, inRegToFloat, maskReg);
                    Add(reduceFloat, reduceFloat, inRegToFloat, pregMain);
                }
            }
            if constexpr (modelOrd == POSITIVE_INF_SCALAR_NORM_MODEL_CODE) {
                ReduceMax(reduceFloat, reduceFloat, pregMain);
            } else {
                ReduceSum(reduceFloat, reduceFloat, pregMain);
            }
            MicroAPI::MaskReg pregMarge = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::VL1>();
            if constexpr (isFirst) {
                DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                    (__local_mem__ float*)(outUbAddr + outOffset), reduceFloat, pregMarge);
            } else {
                DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(
                    outFloat, (__local_mem__ float*)(outUbAddr + outOffset));
                if constexpr (modelOrd == POSITIVE_INF_SCALAR_NORM_MODEL_CODE) {
                    Max(outFloat, outFloat, reduceFloat, pregMarge);
                } else {
                    Add(outFloat, outFloat, reduceFloat, pregMarge);
                }
                DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                    (__local_mem__ float*)(outUbAddr + outOffset), outFloat, pregMarge);
            }
        }
    }

    __aicore__ inline void Compute(LocalTensor<float> outLocal, int64_t dataCount, int64_t index, uint16_t outOffset)
    {
        LocalTensor<T> inLocal = inQueue_.DeQue<T>();
        if (index == 0) {
            DoCompute<true, modelCode>(inLocal, outLocal, dataCount, outOffset);
        } else {
            DoCompute<false, modelCode>(inLocal, outLocal, dataCount, outOffset);
        }
        inQueue_.FreeTensor<T>(inLocal);
    }

    __aicore__ inline void OutputZero()
    {
        LocalTensor<T> outLocal = outQueue_.AllocTensor<T>();

        SetValueAdapter(outLocal, float(0.0), 1);

        outQueue_.EnQue(outLocal);
        outLocal = outQueue_.DeQue<T>();

        // 结构体DataCopyExtParams最后一个参数是rsv保留位
        DataCopyExtParams copyParams2{1, static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
        DataCopyPad(outTensorGM_, outLocal, copyParams2);

        outQueue_.FreeTensor(outLocal);
    }

    __aicore__ inline void ParseTilingData(const Tiling* tilingData)
    {
        maxTensorNumPerCore_ = tilingData->maxTensorNumPerCore;
        inputsTensorUbSize_ = tilingData->inputsTensorUbSize;
        tensorDataCountList_ = (int64_t*)tilingData->tensorDataCountList;
        needCoreNum_ = tilingData->needCoreNum;
        totalTensorCount_ = tilingData->totalTensorCount;
        tensorStart_ = tilingData->tensorStartList[blockIdx_];
        tensorEnd_ = tilingData->tensorEndList[blockIdx_];
        tensorStartOffset_ = tilingData->tensorStartOffsetList[blockIdx_];
        tensorEndOffset_ = tilingData->tensorEndOffsetList[blockIdx_];
        // for reduce
        tensorMiddleStartList_ = tilingData->tensorMiddleStartList;
        tensorMiddleCountList_ = tilingData->tensorMiddleCountList;
        coreMiddleOffset_ = tilingData->coreMiddleOffsetList[blockIdx_];
    }
    __aicore__ inline void CopyIn(int64_t index, int64_t dataCount)
    {
        LocalTensor<T> dataLocal = inQueue_.AllocTensor<T>();
        DataCopyPadExtParams<T> dataCopyPadExtParams;
        dataCopyPadExtParams.isPad = false;
        dataCopyPadExtParams.leftPadding = 0;
        dataCopyPadExtParams.rightPadding = 0;
        dataCopyPadExtParams.paddingValue = 0;
        DataCopyExtParams copyInParams;
        copyInParams.blockCount = 1;
        copyInParams.blockLen = dataCount * sizeof(T);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;
        DataCopyPad(dataLocal, inTensorGM_[1ULL * index * maxDataCount_], copyInParams, dataCopyPadExtParams);
        inQueue_.EnQue(dataLocal);
    }

    __aicore__ inline void CopyOut()
    {
        LocalTensor<T> retLocal = outQueue_.DeQue<T>();
        DataCopyExtParams copyInParams;
        copyInParams.blockCount = 1;
        copyInParams.blockLen = sizeof(T);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;
        DataCopyPad(outTensorGM_, retLocal, copyInParams);
        outQueue_.FreeTensor(retLocal);
    }

    __aicore__ inline __gm__ T* GetGlobalTensorAddr(uint16_t index, ListTensorDesc desc)
    {
        return (__gm__ T*)desc.GetDataPtr<__gm__ T>(index);
    }

protected:
    static constexpr int32_t BUFFER_NUM_ = 2;
    TQue<QuePosition::VECIN, BUFFER_NUM_> inQueue_;
    TQue<QuePosition::VECOUT, 1> outQueue_;

    GlobalTensor<T> inTensorGM_;
    GlobalTensor<T> outTensorGM_;
    GlobalTensor<float> workTensorGM_;

    int64_t blockIdx_ = 0;

    uint32_t maxDataCount_ = {0};
    // tiling params
    uint16_t maxTensorNumPerCore_ = {0};
    uint64_t inputsTensorUbSize_ = 0;
    int64_t* tensorDataCountList_ = nullptr;
    uint16_t needCoreNum_ = 0;
    uint16_t totalTensorCount_ = 0;
    uint16_t tensorStart_ = {0};
    uint16_t tensorEnd_ = {0};
    int64_t tensorStartOffset_ = {0};
    int64_t tensorEndOffset_ = {0};
    // tiling param for Reduce Op
    const uint16_t* tensorMiddleCountList_ = nullptr;
    const uint16_t* tensorMiddleStartList_ = nullptr;
    uint16_t coreMiddleOffset_ = {0};

    ListTensorDesc inDesc_;
    ListTensorDesc outDesc_;
};

} // namespace ForeachNorm

#endif // FOREACH_NORM_N_D_H