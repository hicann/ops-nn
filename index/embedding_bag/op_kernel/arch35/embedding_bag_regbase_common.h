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
 * \file embedding_bag_regbase_common.h
 * \brief
 */

#ifndef EMBEDDING_BAG_REGBASE_COMMON_H
#define EMBEDDING_BAG_REGBASE_COMMON_H

#include "kernel_operator.h"
#include "../inc/platform.h"
#include "../inc/kernel_utils.h"
#include "../inc/load_store_utils.h"
using namespace AscendC;
using namespace ops;

constexpr uint32_t TENSOR_COUNT = 8;
constexpr uint32_t DOUBLE_BUFFER = 2;
constexpr uint32_t MODE_SUM = 0;
constexpr uint32_t MODE_MEAN = 1;
constexpr uint32_t MODE_MAX = 2;

constexpr uint64_t UB_AGLIN_VALUE = 32;
constexpr uint32_t WEIGHT_INPUT_IDX = 0;
constexpr uint32_t INDICES_INPUT_IDX = 1;
constexpr uint32_t OFFSETS_INPUT_IDX = 2;
constexpr uint32_t PERSAMPWEIGHT_INPUT_IDX = 3;
constexpr uint32_t Y_OUTPUT_IDX = 4;
constexpr uint32_t OFFSET2BAG_OUTPUT_IDX = 5;
constexpr uint32_t BAGSIZE_OUTPUT_IDX = 6;
constexpr uint32_t MAXINDICES_OUTPUT_IDX = 7;

constexpr int64_t DIGIT_1 = 1;
constexpr int64_t DIGIT_2 = 2;
constexpr int64_t DIGIT_4 = 4;
constexpr uint32_t USED_THREAD_NUM = 2048;
constexpr uint32_t BLOCK_DIM_0 = 32;
constexpr uint32_t BLOCK_DIM_1 = USED_THREAD_NUM / BLOCK_DIM_0;

template <typename T>
__aicore__ inline void DuplicateNegInf(LocalTensor<T>& localTensor, int32_t dataLen)
{
    // -inf
    constexpr uint32_t FLOAT32_NEG_INF = 0xFF800000;
    constexpr uint16_t FLOAT16_NEG_INF = 0xFC00;
    constexpr uint16_t BFLOAT16_NEG_INF = 0xFF80;
    using computeType = std::conditional_t<std::is_same<T, float>::value, uint32_t, uint16_t>;

    if constexpr (std::is_same<T, float>::value) {
        Duplicate(localTensor.template ReinterpretCast<computeType>(), (FLOAT32_NEG_INF), dataLen);
    } else if constexpr (std::is_same<T, half>::value) {
        Duplicate(localTensor.template ReinterpretCast<computeType>(), (FLOAT16_NEG_INF), dataLen);
    } else {
        Duplicate(localTensor.template ReinterpretCast<computeType>(), (BFLOAT16_NEG_INF), dataLen);
    }
}

template <typename T>
__aicore__ inline void ComputeSumWithWeight(LocalTensor<T> weightLocal,  LocalTensor<T> sumLocal,
        float val, int64_t rowNum, int64_t colNum, int64_t colNumAlign)
{
    __local_mem__ T* sumLocalAddr = (__ubuf__ T*)sumLocal.GetPhyAddr();
    __local_mem__ T* weightLocalAddr = (__ubuf__ T*)weightLocal.GetPhyAddr();

    uint32_t vfLen = platform::GetVRegSize() / sizeof(float);
    uint16_t loopCnt = (uint16_t)((colNum + vfLen) / vfLen);
    uint32_t counter = colNum;
    uint16_t numRows = static_cast<uint16_t>(rowNum);
    
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::MaskReg maskRegUpdate;
        AscendC::MicroAPI::RegTensor<float> sumCastReg;
        AscendC::MicroAPI::RegTensor<float> weightCastReg;
        for (uint16_t i = 0; i < loopCnt; ++i) {
            uint32_t colOffset = i * vfLen; //
            maskRegUpdate = AscendC::MicroAPI::UpdateMask<float>(counter);
            ops::LoadOneTensorForDtypeT<T>(sumLocalAddr, sumCastReg, maskRegUpdate, colOffset);
            for (uint16_t j = 0; j < numRows; ++j) {
                uint32_t weightOffset = colOffset + j * colNumAlign;
                ops::LoadOneTensorForDtypeT<T>(weightLocalAddr, weightCastReg, maskRegUpdate, weightOffset);
                AscendC::MicroAPI::Muls(weightCastReg, weightCastReg, val, maskRegUpdate);
                AscendC::MicroAPI::Add<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(
                    sumCastReg, sumCastReg, weightCastReg, maskRegUpdate);
            }
            ops::StoreOneTensorForDtypeT<T>(sumLocalAddr, sumCastReg, maskRegUpdate, colOffset);
        }
    }
}

template <typename T>
__aicore__ inline void ComputeSum(LocalTensor<T> weightLocal,  LocalTensor<T> sumLocal,
            int64_t rowNum, int64_t colNum, int64_t colNumAlign)
{
    __local_mem__ T* weightLocalAddr = (__ubuf__ T*)weightLocal.GetPhyAddr();
    __local_mem__ T* sumLocalAddr = (__ubuf__ T*)sumLocal.GetPhyAddr();

    uint32_t vfLen = platform::GetVRegSize() / sizeof(float);
    uint16_t loopCnt = (uint16_t)((colNum + vfLen) / vfLen);
    uint32_t counter = colNum;
    uint16_t numRows = static_cast<uint16_t>(rowNum);
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<float> weightCastReg;
        AscendC::MicroAPI::RegTensor<float> sumCastReg;

        AscendC::MicroAPI::MaskReg maskRegUpdate;
        for (uint16_t i = 0; i < loopCnt; ++i) {
            maskRegUpdate = AscendC::MicroAPI::UpdateMask<float>(counter);
            uint32_t colOffset = i * vfLen; //
            ops::LoadOneTensorForDtypeT<T>(sumLocalAddr, sumCastReg, maskRegUpdate, colOffset);
            for (uint16_t j = 0; j < numRows; ++j) {
                uint32_t weightOffset = colOffset + j * colNumAlign;
                ops::LoadOneTensorForDtypeT<T>(weightLocalAddr, weightCastReg, maskRegUpdate, weightOffset);
                AscendC::MicroAPI::Add<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(
                    sumCastReg, sumCastReg, weightCastReg, maskRegUpdate);
            }
            ops::StoreOneTensorForDtypeT<T>(sumLocalAddr, sumCastReg, maskRegUpdate, colOffset);
        }
    }
}

template <typename T>
__aicore__ inline void DivForMean(LocalTensor<T> sumLocal, int64_t count, int64_t dataLen)
{
    __local_mem__ T* sumAddr = (__local_mem__ T*)sumLocal.GetPhyAddr();

    uint32_t vfLen = platform::GetVRegSize() / sizeof(float);
    uint16_t loopSize = ops::CeilDiv(static_cast<uint32_t>(dataLen), vfLen);

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<float> castReg;
        AscendC::MicroAPI::RegTensor<float> countReg;
        AscendC::MicroAPI::MaskReg maskReg;
        uint32_t maskLen = static_cast<uint32_t>(dataLen);
        AscendC::MicroAPI::MaskReg countMask = AscendC::MicroAPI::CreateMask<float>();
        AscendC::MicroAPI::Duplicate(countReg, static_cast<float>(count), countMask);
        for (uint16_t j = 0; j < static_cast<uint16_t>(loopSize); j++) {
            maskReg = AscendC::MicroAPI::UpdateMask<float>(maskLen);
            auto sumAddrOffet = j * vfLen;
            ops::LoadOneTensorForDtypeT<T>(sumAddr, castReg, maskReg, sumAddrOffet);
            AscendC::MicroAPI::Div(castReg, castReg, countReg, maskReg);
            ops::StoreOneTensorForDtypeT<T>(sumAddr, castReg, maskReg, sumAddrOffet);
        }
    }
}

template <typename T, typename U, typename E, typename I>
class EmbeddingBagRegBaseIndx1dBase {
public:
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> inQueueWeight_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> inQueueIndices_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> inQueueOffsets_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> inQueueperSampleWeights_;

    TQue<QuePosition::VECOUT, DOUBLE_BUFFER> outQueueY_;
    TBuf<QuePosition::VECOUT> outQueueOffset2bag_;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER> outQueueBagSize_;
    TBuf<QuePosition::VECOUT> outQueueMaxIndices_;

    TBuf<QuePosition::VECCALC> maxIndicesCalcBuf_;
    TBuf<QuePosition::VECCALC> compareMaskBuf_;
    TBuf<QuePosition::VECCALC> perSampleWeightsBuf_;

    GlobalTensor<T> weightGm_;
    GlobalTensor<U> indicesGm_;
    GlobalTensor<E> offsetsGm_;
    GlobalTensor<T> perSampleWeightsGm_;

    GlobalTensor<T> yGm_;
    GlobalTensor<I> offset2bagGm_;
    GlobalTensor<I> bagSizeGm_;
    GlobalTensor<I> maxIndicesGm_;
    static constexpr uint32_t vfOffsetsNum_ = platform::GetVRegSize() / sizeof(E);
    static constexpr uint32_t vfOutYNum_ = platform::GetVRegSize() / sizeof(T);
    static constexpr uint32_t vfISizeNum_ = platform::GetVRegSize() / sizeof(I);
    static constexpr uint32_t floatSizeNum_ = platform::GetVRegSize() / sizeof(float);

    static constexpr AscendC::MicroAPI::CastTrait castTraitInt32Int64 = {
        AscendC::MicroAPI::RegLayout::ZERO,
        AscendC::MicroAPI::SatMode::UNKNOWN,
        AscendC::MicroAPI::MaskMergeMode::ZEROING,
        AscendC::RoundMode::UNKNOWN,
    };
    using OffsetRegType = typename std::conditional<IsSameType<I, E>::value, E, int64_t>::type;
    int64_t blockId_{0};
    int64_t blockNums_{0};

    int64_t embeddingDim_{0};
    int64_t indicesSize_{0};

    int64_t weightDimFactor_{0};
    int64_t indiceFactor_{0};
    int64_t offsetsFactor_{0};

    int64_t curCoreEmbedDim_{0};
    int64_t curCoreBag_{0};

    int64_t weightCoreOfset_{0};
    int64_t offsetStart_{0};

    int64_t indicesOfset_{0};

    int64_t indicesNumel_{0};

    int64_t numBags_{0};

    int64_t weightLoop_{0};
    int64_t tailLoopWeightNmber_{0};

    int64_t copyIndicesStart_{0};
    int64_t copyIndicesEnd_{0};

    __aicore__ inline void CopyOffsetsFromGm(int64_t offsen, int64_t offsenNumber)
    {
        LocalTensor<E> offsetsLocal_ = inQueueOffsets_.AllocTensor<E>();
        DataCopyExtParams copyExtParamsGrad;
        copyExtParamsGrad.blockCount = 1;
        copyExtParamsGrad.blockLen = static_cast<uint32_t>(offsenNumber * sizeof(E));
        copyExtParamsGrad.srcStride = 0;
        copyExtParamsGrad.dstStride = 0;
        DataCopyPadExtParams<E> padParamsGrad{false, 0, 0, 0};
        DataCopyPad(offsetsLocal_, offsetsGm_[offsen], copyExtParamsGrad, padParamsGrad);
        inQueueOffsets_.EnQue(offsetsLocal_);
    }

    __aicore__ inline void CopyIndicesFromGm(int64_t indiceOffsen, int64_t indicesFactorNumber)
    {
        LocalTensor<U> indicesLocal_ = inQueueIndices_.DeQue<U>();
        DataCopyExtParams copyExtParamsGrad;
        copyExtParamsGrad.blockCount = 1;
        copyExtParamsGrad.blockLen = static_cast<uint32_t>(indicesFactorNumber * sizeof(U));
        copyExtParamsGrad.srcStride = 0;
        copyExtParamsGrad.dstStride = 0;
        DataCopyPadExtParams<U> padParamsGrad{false, 0, 0, 0};
        DataCopyPad(indicesLocal_, indicesGm_[indiceOffsen], copyExtParamsGrad, padParamsGrad);
        inQueueIndices_.EnQue(indicesLocal_);
    }

    __aicore__ inline void CopyPerSampleWeightsFromGm(int64_t indiceOffsen, int64_t dataLen)
    {
        if (dataLen <= 0) {
            return;
        }
        LocalTensor<T> perSampleWeightsLocal_ = this->inQueueperSampleWeights_.template DeQue<T>();
        DataCopyExtParams copyExtParamsGrad;
        copyExtParamsGrad.blockCount = 1;
        copyExtParamsGrad.blockLen = static_cast<uint32_t>(dataLen * sizeof(T));
        copyExtParamsGrad.srcStride = 0;
        copyExtParamsGrad.dstStride = 0;
        DataCopyPadExtParams<T> padParamsGrad{false, 0, 0, 0};
        DataCopyPad(perSampleWeightsLocal_, perSampleWeightsGm_[indiceOffsen], copyExtParamsGrad, padParamsGrad);
        this->inQueueperSampleWeights_.template EnQue(perSampleWeightsLocal_);
    }

    __aicore__ inline void CopyIndicesNoPerSampleFromGm(
        int64_t indiceOffsen, int64_t indicesFactorNumber, LocalTensor<U> indicesLocal_)
    {
        DataCopyExtParams copyExtParamsGrad;
        copyExtParamsGrad.blockCount = 1;
        copyExtParamsGrad.blockLen = static_cast<uint32_t>(indicesFactorNumber * sizeof(U));
        copyExtParamsGrad.srcStride = 0;
        copyExtParamsGrad.dstStride = 0;
        DataCopyPadExtParams<U> padParamsGrad{false, 0, 0, 0};
        DataCopyPad(indicesLocal_, indicesGm_[indiceOffsen], copyExtParamsGrad, padParamsGrad);
    }

    __aicore__ inline void CopyWeightFromGm(int64_t weightOffset, int64_t colNormalNum, LocalTensor<T> weightLocal_)
    {
        DataCopyExtParams copyExtParamsGrad;
        copyExtParamsGrad.blockCount = 1;
        copyExtParamsGrad.blockLen = static_cast<uint32_t>(colNormalNum * sizeof(T));
        copyExtParamsGrad.srcStride = 0;
        copyExtParamsGrad.dstStride = 0;
        DataCopyPadExtParams<T> padParamsGrad{false, 0, 0, 0};
        DataCopyPad(weightLocal_, weightGm_[weightOffset], copyExtParamsGrad, padParamsGrad);
    }

    __aicore__ inline void CopyWeightNoPerSampleFromGm(
        int64_t weightOffset, int64_t colNormalNum, LocalTensor<T> weightLocal_)
    {
        DataCopyExtParams copyExtParamsGrad;
        copyExtParamsGrad.blockCount = 1;
        copyExtParamsGrad.blockLen = static_cast<uint32_t>(colNormalNum * sizeof(T));
        copyExtParamsGrad.srcStride = 0;
        copyExtParamsGrad.dstStride = 0;
        DataCopyPadExtParams<T> padParamsGrad{false, 0, 0, 0};
        DataCopyPad(weightLocal_, weightGm_[weightOffset], copyExtParamsGrad, padParamsGrad);
    }

    __aicore__ inline void CopyOffset2bagToGm(
        int64_t offset2bagOffset, int64_t outOffset2bagNumber, int64_t curOffsetStart)
    {
        LocalTensor<I> offset2bagLocal_ = outQueueOffset2bag_.Get<I>();
        Duplicate(offset2bagLocal_, static_cast<I>(curOffsetStart), static_cast<int32_t>(outOffset2bagNumber));
        int32_t eventIDVToMTE3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
        WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
        DataCopyExtParams copyParam{1, static_cast<uint32_t>(outOffset2bagNumber * sizeof(I)), 0, 0, 0};
        DataCopyPad(offset2bagGm_[offset2bagOffset], offset2bagLocal_, copyParam);
    }

    __aicore__ inline void ComputeBagSize(int64_t number, LocalTensor<E> offsetsLocal)
    {
        LocalTensor<OffsetRegType> bagSizeLocal = outQueueBagSize_.AllocTensor<OffsetRegType>();
        __local_mem__ E* offsetsLocalAddr = (__ubuf__ E*)offsetsLocal.GetPhyAddr();
        __local_mem__ OffsetRegType* bagSizeLocalAddr = (__ubuf__ OffsetRegType*)bagSizeLocal.GetPhyAddr();
        uint16_t loopCnt = (uint16_t)((number + vfISizeNum_) / vfISizeNum_);

        int32_t scalar = 0;
        uint32_t counter = number;
        uint32_t counter1 = number;

        __VEC_SCOPE__
        {
            AscendC::MicroAPI::RegTensor<E> offsetsReg;
            AscendC::MicroAPI::RegTensor<E> offsetsShiftOneReg;
            AscendC::MicroAPI::RegTensor<OffsetRegType> bagSizeReg;

            AscendC::MicroAPI::MaskReg maskERegUpdate;
            AscendC::MicroAPI::MaskReg maskIRegUpdate;
            AscendC::MicroAPI::UnalignReg u0;

            for (uint16_t i = 0; i < loopCnt; ++i) {
                maskIRegUpdate = AscendC::MicroAPI::UpdateMask<I>(counter1);
                if constexpr (sizeof(I) / sizeof(E) == DIGIT_2) {
                    AscendC::MicroAPI::MaskPack(maskERegUpdate, maskIRegUpdate);
                } else {
                     maskERegUpdate = AscendC::MicroAPI::UpdateMask<E>(counter);
                }
                auto offsetsLocalAddrUpdate = offsetsLocalAddr + i * vfISizeNum_;
                auto bagSizeLocalAddrUpdate = bagSizeLocalAddr + i * vfISizeNum_;
                AscendC::MicroAPI::DataCopy(offsetsReg, offsetsLocalAddrUpdate);
                AscendC::MicroAPI::DataCopyUnAlignPre(u0, offsetsLocalAddrUpdate + 1);
                AscendC::MicroAPI::DataCopyUnAlign<E>(offsetsShiftOneReg, u0, offsetsLocalAddrUpdate + 1);
                AscendC::MicroAPI::Sub<E, AscendC::MicroAPI::MaskMergeMode::ZEROING>(
                    offsetsShiftOneReg, offsetsShiftOneReg, offsetsReg, maskERegUpdate);

                if constexpr (sizeof(I) / sizeof(E) == DIGIT_2) {
                    AscendC::MicroAPI::UnPack(
                        (AscendC::MicroAPI::RegTensor<I>&)offsetsShiftOneReg,
                        (AscendC::MicroAPI::RegTensor<E>&)offsetsShiftOneReg);
                    AscendC::MicroAPI::Cast<I, E, castTraitInt32Int64>(bagSizeReg, offsetsShiftOneReg, maskIRegUpdate);
                    AscendC::MicroAPI::DataCopy(bagSizeLocalAddrUpdate, bagSizeReg, maskIRegUpdate);
                } else {
                    AscendC::MicroAPI::DataCopy(bagSizeLocalAddrUpdate,  offsetsShiftOneReg, maskERegUpdate);
                }
            }
        }
        outQueueBagSize_.EnQue(bagSizeLocal);
    }

    __aicore__ inline void ComputeAdd(
        int64_t number, uint16_t indicesNumber, LocalTensor<T> outYLocal, LocalTensor<T> weightLocal)
    {
        int32_t eventIDMTE2ToV = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);

        __local_mem__ T* weightLocalAddr = (__ubuf__ T*)weightLocal.GetPhyAddr();
        __local_mem__ T* outYLocalAddr = (__ubuf__ T*)outYLocal.GetPhyAddr();

        uint16_t loopCnt = (uint16_t)((number + floatSizeNum_) / floatSizeNum_);
        uint32_t counter = number;

        __VEC_SCOPE__
        {
            AscendC::MicroAPI::RegTensor<T> weightReg;
            AscendC::MicroAPI::RegTensor<T> outYReg;
            AscendC::MicroAPI::RegTensor<float> weightCastReg;
            AscendC::MicroAPI::RegTensor<float> outYCastReg;

            AscendC::MicroAPI::MaskReg maskRegUpdate;
            for (uint16_t i = 0; i < loopCnt; ++i) {
                maskRegUpdate = AscendC::MicroAPI::UpdateMask<float>(counter);
                uint32_t colOffset = i * floatSizeNum_; //
                ops::LoadOneTensorForDtypeT<T>(outYLocalAddr, outYCastReg, maskRegUpdate, colOffset);
                for (uint16_t j = 0; j < indicesNumber; ++j) {
                    uint32_t weightOffset = i * floatSizeNum_ + j * weightDimFactor_; // weight
                    ops::LoadOneTensorForDtypeT<T>(weightLocalAddr, weightCastReg, maskRegUpdate, weightOffset);
                    AscendC::MicroAPI::Add<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(
                        outYCastReg, outYCastReg, weightCastReg, maskRegUpdate);
                }
                ops::StoreOneTensorForDtypeT<T>(outYLocalAddr, outYCastReg, maskRegUpdate, colOffset);
            }
        }
    }

    __aicore__ inline void ComputeAddPerSample(
        int64_t number, uint16_t indicesNumber, LocalTensor<T> outYLocal, LocalTensor<T> weightLocal,
        LocalTensor<float> perSampleLocal)
    {
        int32_t eventIDMTE2ToV = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);

        int32_t eventIDSToV = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventIDSToV);
        WaitFlag<HardEvent::S_V>(eventIDSToV);

        __local_mem__ T* weightLocalAddr = (__ubuf__ T*)weightLocal.GetPhyAddr();
        __local_mem__ T* outYLocalAddr = (__ubuf__ T*)outYLocal.GetPhyAddr();
        __local_mem__ float* perSampleLocalAddr = (__ubuf__ float*)perSampleLocal.GetPhyAddr();

        uint16_t loopCnt = (uint16_t)((number + floatSizeNum_) / floatSizeNum_);

        uint32_t counter = number;

        __VEC_SCOPE__
        {
            AscendC::MicroAPI::RegTensor<T> weightReg;
            AscendC::MicroAPI::RegTensor<T> outYReg;
            AscendC::MicroAPI::RegTensor<float> weightCastReg;
            AscendC::MicroAPI::RegTensor<float> outYCastReg;
            AscendC::MicroAPI::RegTensor<float> perSampleCastReg;

            AscendC::MicroAPI::MaskReg maskRegUpdate;
            for (uint16_t i = 0; i < loopCnt; ++i) {
                maskRegUpdate = AscendC::MicroAPI::UpdateMask<float>(counter);
                uint32_t colOffset = i * floatSizeNum_; //
                ops::LoadOneTensorForDtypeT<T>(outYLocalAddr, outYCastReg, maskRegUpdate, colOffset);
                for (uint16_t j = 0; j < indicesNumber; ++j) {
                    uint32_t weightOffset = i * floatSizeNum_ + j * weightDimFactor_;
                    ops::LoadOneTensorForDtypeT<T>(weightLocalAddr, weightCastReg, maskRegUpdate, weightOffset);
                    AscendC::MicroAPI::Muls<float, float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(
                        weightCastReg, weightCastReg, perSampleLocalAddr[j], maskRegUpdate);
                    AscendC::MicroAPI::Add<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(
                        outYCastReg, outYCastReg, weightCastReg, maskRegUpdate);
                }
                ops::StoreOneTensorForDtypeT<T>(outYLocalAddr, outYCastReg, maskRegUpdate, colOffset);
            }
        }
    }

    __aicore__ inline void ComputeMax(
        int64_t number, uint16_t indicesNumber, LocalTensor<T>& outYLocal, LocalTensor<T>& weightLocal,
        LocalTensor<I>& maxIndicesLocal)
    {
        int32_t eventIDSToV = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventIDSToV);
        WaitFlag<HardEvent::S_V>(eventIDSToV);

        int32_t eventIDMTE2ToV = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
        LocalTensor<I> maxIndicesCalcLocal = maxIndicesCalcBuf_.Get<I>();

        __local_mem__ T* weightLocalAddr = (__ubuf__ T*)weightLocal.GetPhyAddr();
        __local_mem__ T* outYLocalAddr = (__ubuf__ T*)outYLocal.GetPhyAddr();
        __local_mem__ I* maxIndicesLocalAddr = (__ubuf__ I*)maxIndicesLocal.GetPhyAddr();
        __local_mem__ I* maxIndicesCalcLocalAddr = (__ubuf__ I*)maxIndicesCalcLocal.GetPhyAddr();
        uint16_t loopCnt = (uint16_t)((number + vfISizeNum_ - 1) / vfISizeNum_);
        uint32_t counter = number;
        uint32_t counter1 = number;

        __VEC_SCOPE__
        {
            AscendC::MicroAPI::RegTensor<T> weightReg;
            AscendC::MicroAPI::RegTensor<T> outYReg;
            AscendC::MicroAPI::RegTensor<I> maxIndicesCalcReg;
            AscendC::MicroAPI::RegTensor<I> maxIndicesReg;

            AscendC::MicroAPI::MaskReg cmpMask;
            AscendC::MicroAPI::MaskReg cmpMaskT2;
            AscendC::MicroAPI::MaskReg cmpMaskT4;

            AscendC::MicroAPI::MaskReg tMaskRegUpdate;
            AscendC::MicroAPI::MaskReg iMaskRegUpdate;

            for (uint16_t i = 0; i < loopCnt; ++i) {
                iMaskRegUpdate = AscendC::MicroAPI::UpdateMask<I>(counter1);

                if constexpr (sizeof(I) / sizeof(T) == DIGIT_1) {
                    tMaskRegUpdate = AscendC::MicroAPI::UpdateMask<T>(counter);
                } else if constexpr (sizeof(I) / sizeof(T) == DIGIT_2) {
                    AscendC::MicroAPI::MaskPack(tMaskRegUpdate, iMaskRegUpdate);
                } else if (sizeof(I) / sizeof(T) == DIGIT_4) {
                    AscendC::MicroAPI::MaskPack(tMaskRegUpdate, iMaskRegUpdate);
                    AscendC::MicroAPI::MaskPack(tMaskRegUpdate, tMaskRegUpdate);
                }

                auto outYLocalAddrUpdate = outYLocalAddr + i * vfISizeNum_;
                AscendC::MicroAPI::DataCopy(outYReg, outYLocalAddrUpdate);
                auto maxIndicesLocalAddrUpdate = maxIndicesLocalAddr + i * vfISizeNum_;
                AscendC::MicroAPI::DataCopy(maxIndicesReg, maxIndicesLocalAddrUpdate);

                for (uint16_t j = 0; j < indicesNumber; ++j) {
                    auto weightLocalAddrUpdate = weightLocalAddr + i * vfISizeNum_ + j * weightDimFactor_;
                    AscendC::MicroAPI::DataCopy(weightReg, weightLocalAddrUpdate);
                    AscendC::MicroAPI::Duplicate(maxIndicesCalcReg, maxIndicesCalcLocalAddr[j]); // 当前处理行的索引
                    AscendC::MicroAPI::Compare<T, CMPMODE::GT>(cmpMask, weightReg, outYReg, tMaskRegUpdate);
                    if constexpr (sizeof(I) / sizeof(T) == DIGIT_1) {
                        AscendC::MicroAPI::Select(maxIndicesReg, maxIndicesCalcReg, maxIndicesReg, cmpMask);
                    } else if constexpr (sizeof(I) / sizeof(T) == DIGIT_2) {
                        AscendC::MicroAPI::MaskUnPack(cmpMaskT2, cmpMask);
                        AscendC::MicroAPI::Select(maxIndicesReg, maxIndicesCalcReg, maxIndicesReg, cmpMaskT2);
                    } else if (sizeof(I) / sizeof(T) == DIGIT_4) {
                        AscendC::MicroAPI::MaskUnPack(cmpMaskT2, cmpMask);
                        AscendC::MicroAPI::MaskUnPack(cmpMaskT4, cmpMaskT2);
                        AscendC::MicroAPI::Select(maxIndicesReg, maxIndicesCalcReg, maxIndicesReg, cmpMaskT4);
                    }

                    AscendC::MicroAPI::Max<T, AscendC::MicroAPI::MaskMergeMode::ZEROING>(
                        outYReg, weightReg, outYReg, tMaskRegUpdate);
                }
                AscendC::MicroAPI::DataCopy(outYLocalAddrUpdate, outYReg, tMaskRegUpdate);
                AscendC::MicroAPI::DataCopy(maxIndicesLocalAddrUpdate, maxIndicesReg, iMaskRegUpdate);
            }
        }

        int32_t eventIDVToS = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventIDVToS);
        WaitFlag<HardEvent::V_S>(eventIDVToS);
    }

    __aicore__ inline void ComputeWeightAdd(
        int64_t weightNumber, int64_t updateIndiceOffsen, LocalTensor<T> weightLocal_,
        LocalTensor<T> perSampleWeightsLocal_, LocalTensor<T> outYLocal_)
    {
        int32_t eventIDMTE2ToS = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(eventIDMTE2ToS);
        WaitFlag<HardEvent::MTE2_S>(eventIDMTE2ToS);
        T perSampleWeights = perSampleWeightsLocal_(updateIndiceOffsen);

        AscendC::Muls(weightLocal_, weightLocal_, perSampleWeights, weightNumber);
        AscendC::Add(outYLocal_, weightLocal_, outYLocal_, weightNumber);
    }

    __aicore__ inline void ComputeWeightMean(int64_t weightNumber)
    {
        LocalTensor<T> weightLocal_ = inQueueWeight_.DeQue<T>();
        LocalTensor<T> outYLocal_ = outQueueY_.DeQue<T>();

        AscendC::Add(outYLocal_, weightLocal_, outYLocal_, weightNumber);
        inQueueWeight_.EnQue(weightLocal_);
        outQueueY_.EnQue(outYLocal_);
    }

    __aicore__ inline void CopyYToGm(int64_t weightOffset, int64_t weightNumber, LocalTensor<T> outYLocal)
    {
        DataCopyExtParams copyParam{1, static_cast<uint32_t>(weightNumber * sizeof(T)), 0, 0, 0};
        DataCopyPad(yGm_[weightOffset], outYLocal, copyParam);
    }

    __aicore__ inline void CopyMaxIndicesToGm(
        int64_t weightOffset, int64_t weightNumber, LocalTensor<I> maxIndicesLocal)
    {
        DataCopyExtParams copyParam{1, static_cast<uint32_t>(weightNumber * sizeof(U)), 0, 0, 0};
        DataCopyPad(maxIndicesGm_[weightOffset], maxIndicesLocal, copyParam);
    }

    __aicore__ inline void DisposalBagSize(GlobalTensor<I> globalTensor, int64_t inclueLastOfst)
    {
        if (inclueLastOfst){
            LocalTensor<I> dataDupLocal = maxIndicesCalcBuf_.Get<I>();
            Duplicate(dataDupLocal, (I)0, 1);
            DataCopyExtParams copyParam{1, static_cast<uint32_t>(1 * sizeof(I)), 0, 0, 0};
            DataCopyPad(globalTensor, dataDupLocal, copyParam);
        }
    }
};

#endif // EMBEDDING_BAG_H_REGBASE_COMMON_H

