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
 * \file index_full_load.h
 * \brief Ascendc index kernel
 */

#ifndef ASCENDC_INDEX_FULL_LOAD_H_
#define ASCENDC_INDEX_FULL_LOAD_H_

#include "kernel_operator.h"
#include "op_kernel/platform_util.h"
#include "op_kernel/math_util.h"
#include "kernel_operator_list_tensor_intf.h"
#include "index_tiling_data.h"

namespace Index {
using namespace AscendC;

constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t B8 = 1;
constexpr uint32_t B16 = 2;
constexpr uint32_t B32 = 4;
constexpr uint32_t B64 = 8;
constexpr uint8_t MASK_MODE_0 = 0;
constexpr uint8_t MASK_MODE_1 = 1;
constexpr uint8_t INPUT_AXIS_6 = 6;
constexpr uint8_t INPUT_AXIS_7 = 7;
constexpr uint8_t DIM1 = 1;
constexpr uint8_t DIM2 = 2;

template <typename T>
struct GetGatherType {
    using type = typename std::conditional<
        IsSameType<T, int8_t>::value, int16_t,
        typename std::conditional<IsSameType<T, uint8_t>::value, uint16_t, T>::type>::type;
};

template <typename T, typename U, typename SrcU, uint8_t MASK_MODE, uint8_t DIM_NUM>
class KernelIndexFullLoad {
public:
    __aicore__ inline KernelIndexFullLoad(TPipe& pipeIn, const IndexFullLoadTilingData& tilingData)
        : pipe_(pipeIn), tilingData_(tilingData){};
    __aicore__ inline void Init(GM_ADDR inputX, GM_ADDR indexedSizes, GM_ADDR indices, GM_ADDR output);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyInX();
    __aicore__ inline void CopyInIndices(int64_t loopIdx, int64_t length);
    __aicore__ inline __gm__ SrcU* GetTensorAddr(int64_t index, int64_t offset);
    __aicore__ inline void Compute(LocalTensor<T>& inputLocal, uint32_t indicesCount);
    __aicore__ inline void ComputeDim1Each(Reg::MaskReg& maskValue, Reg::MaskReg& maskIndex, uint16_t loopIdx,
                                           uint16_t repeatElem, __ubuf__ T* inputAddr, __ubuf__ U* indicesAddr,
                                           __ubuf__ T* outputAddr, int32_t dim0Shape, Reg::MaskReg& inputB16MaskValue);
    __aicore__ inline void ComputeDim1(uint32_t maskCount, uint32_t tailMaskCount, uint16_t repeatTimes,
                                       uint16_t repeatElem, __ubuf__ T* inputAddr, __ubuf__ U* indicesAddr,
                                       __ubuf__ T* outputAddr, int32_t dim0Shape);
    __aicore__ inline void ComputeDim2Each(Reg::MaskReg& inputMaskValue, Reg::MaskReg& indexMaskValue, uint16_t loopIdx,
                                           uint16_t indexRepeatElem, uint16_t inputRepeatElem, __ubuf__ T* inputAddr,
                                           __ubuf__ U* indicesAddr, __ubuf__ T* outputAddr, uint32_t indiceOffset,
                                           uint32_t stride0, int32_t dim0Shape, int32_t dim1Shape, uint32_t inputValue);
    __aicore__ inline void ComputeDim2(uint32_t maskCount, uint32_t tailMaskCount, uint32_t indexMaskCount,
                                       uint32_t tailIndexMaskCount, uint16_t repeatTimes, uint16_t indexRepeatElem,
                                       uint16_t inputRepeatElem, __ubuf__ T* inputAddr, __ubuf__ U* indicesAddr,
                                       __ubuf__ T* outputAddr, uint32_t indiceOffset, uint32_t stride0,
                                       int32_t dim0Shape, int32_t dim1Shape);
    __aicore__ inline void CopyOut(int64_t loopIdx, int64_t length);

private:
    TPipe& pipe_;
    const IndexFullLoadTilingData& tilingData_;
    GlobalTensor<T> inputXGm_;
    GlobalTensor<T> outputGm_;
    GlobalTensor<SrcU> indicesGm_;
    TQue<QuePosition::VECIN, 1> inQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> indicesQueue_;
    TBuf<QuePosition::VECCALC> castBuff_;
    int64_t currentCoreLoop_ = 0;
    int64_t currentCoreTail_ = 0;
    uint32_t blockIdx_ = 0;
    uint32_t blockSize_ = 0;
    uint32_t indicesQueOffset_ = 0;
    ListTensorDesc inputList_;
};
template <typename T, typename U, typename SrcU, uint8_t MASK_MODE, uint8_t DIM_NUM>
__aicore__ inline void KernelIndexFullLoad<T, U, SrcU, MASK_MODE, DIM_NUM>::Init(GM_ADDR inputX, GM_ADDR indexedSizes,
                                                                                 GM_ADDR indices, GM_ADDR output)
{
    blockIdx_ = GetBlockIdx();
    blockSize_ = Ops::Base::GetUbBlockSize();
    inputList_ = ListTensorDesc(reinterpret_cast<__gm__ void*>(indices));
    indicesQueOffset_ = Ops::Base::CeilAlign(tilingData_.ubIndices * sizeof(U), static_cast<uint64_t>(blockSize_)) /
                        sizeof(U);
    if (blockIdx_ == tilingData_.usedCoreNum - 1) {
        currentCoreLoop_ = tilingData_.tailCoreLoop;
        currentCoreTail_ = tilingData_.tailCoreTail;
    } else {
        currentCoreLoop_ = tilingData_.normalCoreLoop;
        currentCoreTail_ = tilingData_.normalCoreTail;
    }
    inputXGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(inputX));
    outputGm_.SetGlobalBuffer((__gm__ T*)(output) + blockIdx_ * tilingData_.eachCoreSize * tilingData_.lastDimSize);
    pipe_.InitBuffer(inQueue_, 1, tilingData_.inputQueSize);
    pipe_.InitBuffer(outQueue_, BUFFER_NUM, tilingData_.outputQueSize);
    pipe_.InitBuffer(indicesQueue_, BUFFER_NUM, tilingData_.indicesQueSize);
    if constexpr (std::is_same<SrcU, int64_t>::value) {
        pipe_.InitBuffer(castBuff_,
                         Ops::Base::CeilAlign(tilingData_.ubIndices * sizeof(SrcU), static_cast<uint64_t>(blockSize_)));
    }
    return;
}
template <typename T, typename U, typename SrcU, uint8_t MASK_MODE, uint8_t DIM_NUM>
__aicore__ inline void KernelIndexFullLoad<T, U, SrcU, MASK_MODE, DIM_NUM>::Process()
{
    if (blockIdx_ >= tilingData_.usedCoreNum) {
        return;
    }
    CopyInX();
    LocalTensor<T> inputLocal = inQueue_.DeQue<T>();
    for (int64_t i = 0; i < currentCoreLoop_ - 1; i++) {
        CopyInIndices(i, tilingData_.ubIndices);
        Compute(inputLocal, static_cast<uint32_t>(tilingData_.ubIndices));
        CopyOut(i, tilingData_.ubIndices);
    }
    CopyInIndices(currentCoreLoop_ - 1, currentCoreTail_);
    Compute(inputLocal, static_cast<uint32_t>(currentCoreTail_));
    CopyOut(currentCoreLoop_ - 1, currentCoreTail_);
    inQueue_.FreeTensor(inputLocal);
    return;
}
template <typename T, typename U, typename SrcU, uint8_t MASK_MODE, uint8_t DIM_NUM>
__aicore__ inline void KernelIndexFullLoad<T, U, SrcU, MASK_MODE, DIM_NUM>::CopyInX()
{
    LocalTensor<T> inputLocal = inQueue_.AllocTensor<T>();
    DataCopyExtParams copyParams = {static_cast<uint16_t>(1),
                                    static_cast<uint32_t>(tilingData_.inputShapeSize * sizeof(T)),
                                    static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    DataCopyPadExtParams<T> padParams = {false, 0, 0, 0};
    DataCopyPad(inputLocal, inputXGm_, copyParams, padParams);
    inQueue_.EnQue(inputLocal);
    return;
}
template <typename T, typename U, typename SrcU, uint8_t MASK_MODE, uint8_t DIM_NUM>
__aicore__ inline void KernelIndexFullLoad<T, U, SrcU, MASK_MODE, DIM_NUM>::CopyInIndices(int64_t loopIdx,
                                                                                          int64_t length)
{
    LocalTensor<U> indicesLocal = indicesQueue_.AllocTensor<U>();
    DataCopyPadExtParams<SrcU> padParams = {false, 0, 0, 0};
    DataCopyExtParams copyParams = {static_cast<uint16_t>(1), static_cast<uint32_t>(length * sizeof(SrcU)),
                                    static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    int64_t indicesGmOffset = blockIdx_ * tilingData_.eachCoreSize + loopIdx * tilingData_.ubIndices;
    for (int64_t j = 0; j < tilingData_.indicesNum; j++) {
        uint32_t ubOffset = j * indicesQueOffset_;
        indicesGm_.SetGlobalBuffer(GetTensorAddr(j, indicesGmOffset));
        if constexpr (std::is_same<SrcU, int64_t>::value) {
            LocalTensor<SrcU> castBuffer = castBuff_.Get<SrcU>();
            DataCopyPad(castBuffer, indicesGm_, copyParams, padParams);
            event_t eventIdMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(eventIdMTE2ToV);
            WaitFlag<HardEvent::MTE2_V>(eventIdMTE2ToV);
            Cast<U, SrcU>(indicesLocal[ubOffset], castBuffer, RoundMode::CAST_NONE, length);
            event_t eventIdVToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
            SetFlag<HardEvent::V_MTE2>(eventIdVToMTE2);
            WaitFlag<HardEvent::V_MTE2>(eventIdVToMTE2);
        } else {
            DataCopyPad(indicesLocal[ubOffset], indicesGm_, copyParams, padParams);
        }
    }
    indicesQueue_.EnQue(indicesLocal);
    return;
}
template <typename T, typename U, typename SrcU, uint8_t MASK_MODE, uint8_t DIM_NUM>
__aicore__ inline __gm__ SrcU* KernelIndexFullLoad<T, U, SrcU, MASK_MODE, DIM_NUM>::GetTensorAddr(int64_t index,
                                                                                                  int64_t offset)
{
    return inputList_.GetDataPtr<SrcU>(index) + offset;
}
template <typename T, typename U, typename SrcU, uint8_t MASK_MODE, uint8_t DIM_NUM>
__aicore__ inline void KernelIndexFullLoad<T, U, SrcU, MASK_MODE, DIM_NUM>::Compute(LocalTensor<T>& inputLocal,
                                                                                    uint32_t indicesCount)
{
    LocalTensor<U> indicesLocal = indicesQueue_.DeQue<U>();
    LocalTensor<T> outputLocal = outQueue_.AllocTensor<T>();
    __ubuf__ T* inputAddr = (__ubuf__ T*)inputLocal.GetPhyAddr();
    __ubuf__ U* indicesAddr = (__ubuf__ U*)indicesLocal.GetPhyAddr();
    __ubuf__ T* outputAddr = (__ubuf__ T*)outputLocal.GetPhyAddr();
    uint16_t repeatTimes = 0;
    uint16_t inputRepeatElem = 0;
    uint16_t indexRepeatElem = 0;
    uint16_t tail = 0;
    uint32_t maskCount = 0;
    uint32_t tailMaskCount = 0;
    uint32_t indexMaskCount = 0;
    uint32_t tailIndexMaskCount = 0;
    uint32_t vlLength = Ops::Base::GetVRegSize() / sizeof(U);
    if constexpr (MASK_MODE == MASK_MODE_0) {
        uint16_t indexBatchCount = vlLength / tilingData_.lastDimSize;
        repeatTimes = Ops::Base::CeilDiv(indicesCount, static_cast<uint32_t>(indexBatchCount));
        inputRepeatElem = indexBatchCount * tilingData_.lastDimSize;
        tail = (indicesCount - (repeatTimes - 1) * indexBatchCount) * tilingData_.lastDimSize;
        maskCount = inputRepeatElem;
        tailMaskCount = tail;
        indexRepeatElem = indexBatchCount;
        indexMaskCount = inputRepeatElem;
        tailIndexMaskCount = tail;
    } else if constexpr (MASK_MODE == MASK_MODE_1) {
        repeatTimes = Ops::Base::CeilDiv(indicesCount, vlLength);
        inputRepeatElem = vlLength;
        tail = indicesCount - (repeatTimes - 1) * inputRepeatElem;
        maskCount = inputRepeatElem;
        tailMaskCount = tail;
        indexRepeatElem = inputRepeatElem;
        indexMaskCount = inputRepeatElem;
        tailIndexMaskCount = tail;
    }
    if constexpr (DIM_NUM == DIM1) {
        ComputeDim1(maskCount, tailMaskCount, repeatTimes, inputRepeatElem, inputAddr, indicesAddr, outputAddr,
                    tilingData_.inputShape[INPUT_AXIS_7]);
    } else if constexpr (DIM_NUM == DIM2) {
        ComputeDim2(maskCount, tailMaskCount, indexMaskCount, tailIndexMaskCount, repeatTimes, indexRepeatElem,
                    inputRepeatElem, inputAddr, indicesAddr, outputAddr, indicesQueOffset_,
                    tilingData_.inputStride[INPUT_AXIS_6], tilingData_.inputShape[INPUT_AXIS_6],
                    tilingData_.inputShape[INPUT_AXIS_7]);
    }
    indicesQueue_.FreeTensor(indicesLocal);
    outQueue_.EnQue(outputLocal);
    return;
}
template <typename T, typename U, typename SrcU, uint8_t MASK_MODE, uint8_t DIM_NUM>
__aicore__ inline void KernelIndexFullLoad<T, U, SrcU, MASK_MODE, DIM_NUM>::ComputeDim1Each(
    Reg::MaskReg& maskValue, Reg::MaskReg& maskIndex, uint16_t loopIdx, uint16_t repeatElem, __ubuf__ T* inputAddr,
    __ubuf__ U* indicesAddr, __ubuf__ T* outputAddr, int32_t dim0Shape, Reg::MaskReg& inputB16MaskValue)
{
    Reg::RegTensor<U> indexReg;
    Reg::RegTensor<U> negIndexReg;
    Reg::RegTensor<T> valueReg;
    Reg::MaskReg dstMaskReg;
    Reg::LoadAlign(indexReg, indicesAddr + loopIdx * repeatElem);
    Reg::Compares<U, CMPMODE::LT>(dstMaskReg, indexReg, static_cast<U>(0), maskIndex);
    Reg::Adds(negIndexReg, indexReg, dim0Shape, maskIndex);
    Reg::Select(indexReg, negIndexReg, indexReg, dstMaskReg);
    if constexpr (sizeof(T) == B8) {
        using dstType = typename GetGatherType<T>::type;
        Reg::RegTensor<dstType> gatherDstReg;
        Reg::RegTensor<uint16_t> indexConvert;
        Reg::Pack(indexConvert, indexReg);
        Reg::Gather(gatherDstReg, inputAddr, indexConvert, inputB16MaskValue);
        Reg::Pack((Reg::RegTensor<uint8_t>&)valueReg, gatherDstReg);
        Reg::StoreAlign(outputAddr + loopIdx * repeatElem, valueReg, maskValue);
    } else if constexpr (sizeof(T) == B16) {
        Reg::RegTensor<T> gatherDstReg;
        Reg::RegTensor<uint16_t> indexConvert;
        Reg::Pack(indexConvert, indexReg);
        Reg::Gather(gatherDstReg, inputAddr, indexConvert, maskValue);
        Reg::StoreAlign(outputAddr + loopIdx * repeatElem, gatherDstReg, maskValue);
    } else if constexpr (sizeof(T) == B32) {
        Reg::RegTensor<T> gatherDstReg;
        Reg::Gather(gatherDstReg, inputAddr, (Reg::RegTensor<uint32_t>&)indexReg, maskValue);
        Reg::StoreAlign(outputAddr + loopIdx * repeatElem, gatherDstReg, maskValue);
    } else {
        Reg::RegTensor<T, Reg::RegTraitNumTwo> gatherDstReg;
        Reg::Gather(gatherDstReg, inputAddr, (Reg::RegTensor<uint32_t>&)indexReg, maskValue);
        Reg::StoreAlign(outputAddr + loopIdx * repeatElem, gatherDstReg, maskValue);
    }
    return;
}
template <typename T, typename U, typename SrcU, uint8_t MASK_MODE, uint8_t DIM_NUM>
__aicore__ inline void KernelIndexFullLoad<T, U, SrcU, MASK_MODE, DIM_NUM>::ComputeDim1(
    uint32_t maskCount, uint32_t tailMaskCount, uint16_t repeatTimes, uint16_t repeatElem, __ubuf__ T* inputAddr,
    __ubuf__ U* indicesAddr, __ubuf__ T* outputAddr, int32_t dim0Shape)
{
    __VEC_SCOPE__
    {
        Reg::MaskReg maskValue;
        Reg::MaskReg maskIndex;
        Reg::MaskReg maskValueB16;
        uint32_t maskCountIndex = maskCount;
        uint32_t tailMaskCountIndex = tailMaskCount;
        uint32_t maskCountB16 = maskCount;
        uint32_t tailMaskCountB16 = tailMaskCount;
        if constexpr (sizeof(T) == B64) {
            maskValue = Reg::UpdateMask<T, Reg::RegTraitNumTwo>(maskCount);
        } else {
            maskValue = Reg::UpdateMask<T>(maskCount);
        }
        maskIndex = Reg::UpdateMask<U>(maskCountIndex);
        maskValueB16 = Reg::UpdateMask<uint16_t>(maskCountB16);
        for (uint16_t i = 0; i < static_cast<uint16_t>(repeatTimes - 1); i++) {
            ComputeDim1Each(maskValue, maskIndex, i, repeatElem, inputAddr, indicesAddr, outputAddr, dim0Shape,
                            maskValueB16);
        }
        if constexpr (sizeof(T) == B64) {
            maskValue = Reg::UpdateMask<T, Reg::RegTraitNumTwo>(tailMaskCount);
        } else {
            maskValue = Reg::UpdateMask<T>(tailMaskCount);
        }
        maskIndex = Reg::UpdateMask<U>(tailMaskCountIndex);
        maskValueB16 = Reg::UpdateMask<uint16_t>(tailMaskCountB16);
        ComputeDim1Each(maskValue, maskIndex, repeatTimes - 1, repeatElem, inputAddr, indicesAddr, outputAddr,
                        dim0Shape, maskValueB16);
    }
    return;
}
template <typename T, typename U, typename SrcU, uint8_t MASK_MODE, uint8_t DIM_NUM>
__aicore__ inline void KernelIndexFullLoad<T, U, SrcU, MASK_MODE, DIM_NUM>::ComputeDim2Each(
    Reg::MaskReg& inputMaskValue, Reg::MaskReg& indexMaskValue, uint16_t loopIdx, uint16_t indexRepeatElem,
    uint16_t inputRepeatElem, __ubuf__ T* inputAddr, __ubuf__ U* indicesAddr, __ubuf__ T* outputAddr,
    uint32_t indiceOffset, uint32_t stride0, int32_t dim0Shape, int32_t dim1Shape, uint32_t inputValue)
{
    Reg::RegTensor<U> indexReg;
    Reg::RegTensor<U> indexReg1;
    Reg::RegTensor<U> indexReg2;
    Reg::RegTensor<U> negIndexReg;
    Reg::MaskReg dstMaskReg;
    Reg::RegTensor<T> valueReg;
    auto curOutputAddr = outputAddr + loopIdx * inputRepeatElem;
    if constexpr (MASK_MODE == MASK_MODE_0) {
        Reg::RegTensor<U> vciReg;
        Reg::RegTensor<U> duplicatReg;
        Reg::RegTensor<U> tmpReg;
        Reg::Duplicate(duplicatReg, static_cast<U>(dim1Shape));
        Reg::Arange(vciReg, static_cast<U>(0));
        Reg::Div(tmpReg, vciReg, duplicatReg, indexMaskValue);
        Reg::Mul(indexReg2, tmpReg, duplicatReg, indexMaskValue);
        Reg::Sub(indexReg2, vciReg, indexReg2, indexMaskValue);
        Reg::Adds(tmpReg, tmpReg, loopIdx * indexRepeatElem, indexMaskValue);
        Reg::Gather(indexReg1, indicesAddr, (Reg::RegTensor<uint32_t>&)(tmpReg), indexMaskValue);
        Reg::Compares<U, CMPMODE::LT>(dstMaskReg, indexReg1, static_cast<U>(0), indexMaskValue);
        Reg::Adds(negIndexReg, indexReg1, dim0Shape, indexMaskValue);
        Reg::Select(indexReg1, negIndexReg, indexReg1, dstMaskReg);
        Reg::Muls(indexReg1, indexReg1, stride0, indexMaskValue);
        Reg::Add(indexReg, indexReg1, indexReg2, indexMaskValue);
    } else if constexpr (MASK_MODE == MASK_MODE_1) {
        Reg::LoadAlign(indexReg1, indicesAddr + loopIdx * indexRepeatElem);
        Reg::Compares<U, CMPMODE::LT>(dstMaskReg, indexReg1, static_cast<U>(0), indexMaskValue);
        Reg::Adds(negIndexReg, indexReg1, dim0Shape, indexMaskValue);
        Reg::Select(indexReg1, negIndexReg, indexReg1, dstMaskReg);
        Reg::Muls(indexReg, indexReg1, stride0, indexMaskValue);
        Reg::LoadAlign(indexReg2, indicesAddr + indiceOffset + loopIdx * indexRepeatElem);
        Reg::Compares<U, CMPMODE::LT>(dstMaskReg, indexReg2, static_cast<U>(0), indexMaskValue);
        Reg::Adds(negIndexReg, indexReg2, dim1Shape, indexMaskValue);
        Reg::Select(indexReg2, negIndexReg, indexReg2, dstMaskReg);
        Reg::Add(indexReg, indexReg, indexReg2, indexMaskValue);
    }
    if constexpr (sizeof(T) == B8) {
        using dstType = typename GetGatherType<T>::type;
        Reg::RegTensor<dstType> gatherDstReg;
        Reg::RegTensor<uint16_t> indexConvert;
        Reg::Pack(indexConvert, indexReg);
        uint32_t inputValueTmp = inputValue;
        Reg::MaskReg inputB16MaskValue = Reg::UpdateMask<uint16_t>(inputValueTmp);
        Reg::Gather(gatherDstReg, inputAddr, indexConvert, inputB16MaskValue);
        Reg::Pack((Reg::RegTensor<uint8_t>&)valueReg, gatherDstReg);
        if constexpr (MASK_MODE == MASK_MODE_1) {
            Reg::StoreAlign(curOutputAddr, valueReg, inputMaskValue);
        } else if constexpr (MASK_MODE == MASK_MODE_0) {
            Reg::UnalignRegForStore u0;
            Reg::StoreUnAlign(curOutputAddr, valueReg, u0, inputValue);
            Reg::StoreUnAlignPost(curOutputAddr, u0, 0);
        }
    } else if constexpr (sizeof(T) == B16) {
        Reg::RegTensor<T> gatherDstReg;
        Reg::RegTensor<uint16_t> indexConvert;
        Reg::Pack(indexConvert, indexReg);
        Reg::Gather(gatherDstReg, inputAddr, indexConvert, inputMaskValue);
        if constexpr (MASK_MODE == MASK_MODE_1) {
            Reg::StoreAlign(curOutputAddr, gatherDstReg, inputMaskValue);
        } else if constexpr (MASK_MODE == MASK_MODE_0) {
            Reg::UnalignRegForStore u0;
            Reg::StoreUnAlign(curOutputAddr, gatherDstReg, u0, inputValue);
            Reg::StoreUnAlignPost(curOutputAddr, u0, 0);
        }
    } else if constexpr (sizeof(T) == B32) {
        Reg::RegTensor<T> gatherDstReg;
        Reg::Gather(gatherDstReg, inputAddr, (Reg::RegTensor<uint32_t>&)indexReg, inputMaskValue);
        if constexpr (MASK_MODE == MASK_MODE_1) {
            Reg::StoreAlign(curOutputAddr, gatherDstReg, inputMaskValue);
        } else if constexpr (MASK_MODE == MASK_MODE_0) {
            Reg::UnalignRegForStore u0;
            Reg::StoreUnAlign(curOutputAddr, gatherDstReg, u0, inputValue);
            Reg::StoreUnAlignPost(curOutputAddr, u0, 0);
        }
    } else {
        Reg::RegTensor<T, Reg::RegTraitNumTwo> gatherDstReg;
        Reg::Gather(gatherDstReg, inputAddr, (Reg::RegTensor<uint32_t>&)indexReg, inputMaskValue);
        if constexpr (MASK_MODE == MASK_MODE_1) {
            Reg::StoreAlign(curOutputAddr, gatherDstReg, inputMaskValue);
        } else if constexpr (MASK_MODE == MASK_MODE_0) {
            Reg::UnalignRegForStore u0;
            Reg::StoreUnAlign(curOutputAddr, gatherDstReg, u0, inputValue);
            Reg::StoreUnAlignPost(curOutputAddr, u0, 0);
        }
    }
    return;
}
template <typename T, typename U, typename SrcU, uint8_t MASK_MODE, uint8_t DIM_NUM>
__aicore__ inline void KernelIndexFullLoad<T, U, SrcU, MASK_MODE, DIM_NUM>::ComputeDim2(
    uint32_t maskCount, uint32_t tailMaskCount, uint32_t indexMaskCount, uint32_t tailIndexMaskCount,
    uint16_t repeatTimes, uint16_t indexRepeatElem, uint16_t inputRepeatElem, __ubuf__ T* inputAddr,
    __ubuf__ U* indicesAddr, __ubuf__ T* outputAddr, uint32_t indiceOffset, uint32_t stride0, int32_t dim0Shape,
    int32_t dim1Shape)
{
    __VEC_SCOPE__
    {
        Reg::MaskReg maskValue;
        Reg::MaskReg maskValueB16;
        uint32_t inputValue = maskCount;
        uint32_t tailInputValue = tailMaskCount;
        Reg::MaskReg maskIndex = Reg::UpdateMask<U>(indexMaskCount);
        if constexpr (sizeof(T) == B64) {
            maskValue = Reg::UpdateMask<T, Reg::RegTraitNumTwo>(maskCount);
        } else {
            maskValue = Reg::UpdateMask<T>(maskCount);
        }
        for (uint16_t i = 0; i < static_cast<uint16_t>(repeatTimes - 1); i++) {
            ComputeDim2Each(maskValue, maskIndex, i, indexRepeatElem, inputRepeatElem, inputAddr, indicesAddr,
                            outputAddr, indiceOffset, stride0, dim0Shape, dim1Shape, inputValue);
        }
        if constexpr (sizeof(T) == B64) {
            maskValue = Reg::UpdateMask<T, Reg::RegTraitNumTwo>(tailMaskCount);
        } else {
            maskValue = Reg::UpdateMask<T>(tailMaskCount);
        }
        maskIndex = Reg::UpdateMask<U>(tailIndexMaskCount);
        ComputeDim2Each(maskValue, maskIndex, repeatTimes - 1, indexRepeatElem, inputRepeatElem, inputAddr, indicesAddr,
                        outputAddr, indiceOffset, stride0, dim0Shape, dim1Shape, tailInputValue);
    }
    return;
}
template <typename T, typename U, typename SrcU, uint8_t MASK_MODE, uint8_t DIM_NUM>
__aicore__ inline void KernelIndexFullLoad<T, U, SrcU, MASK_MODE, DIM_NUM>::CopyOut(int64_t loopIdx, int64_t length)
{
    LocalTensor<T> outputLocal = outQueue_.DeQue<T>();
    int64_t outputGmOffset = loopIdx * tilingData_.ubIndices * tilingData_.lastDimSize;
    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = length * tilingData_.lastDimSize * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;

    DataCopyPad(outputGm_[outputGmOffset], outputLocal, copyParams);
    outQueue_.FreeTensor(outputLocal);
    return;
}
} // namespace Index
#endif
