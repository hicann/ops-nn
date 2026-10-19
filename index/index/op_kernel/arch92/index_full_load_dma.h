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
 * \file index_full_load_dma.h
 * \brief DV121(960) DMA path: input 留 GM 用 GatherDataCopy, 仅单维索引 (DIM1/DIM2 MODE_0) 适用.
 *        仅在 __NPU_ARCH__ == 9201/9202 (960dt/960pr) 平台编译.
 */

#ifndef ASCENDC_INDEX_FULL_LOAD_DMA_H_
#define ASCENDC_INDEX_FULL_LOAD_DMA_H_

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 9201 || __NPU_ARCH__ == 9202)

#include "../arch35/index_full_load.h"

namespace Index {
using namespace AscendC;

constexpr static AscendC::MicroAPI::CastTrait castTraitB32B64 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
};

template <typename IDX_T>
__simd_vf__ inline void GenOffsetVF(__ubuf__ IDX_T* indicesAddr, __ubuf__ int64_t* offsetAddr, uint32_t vfLen,
                                    uint16_t loopCnt, uint32_t size, uint32_t sizeTwo, int64_t dim0Shape,
                                    int64_t lastDimBytes, IDX_T zero)
{
    MicroAPI::RegTensor<IDX_T> indicesReg;
    MicroAPI::RegTensor<IDX_T> tmpReg;
    MicroAPI::RegTensor<int64_t, MicroAPI::RegTraitNumTwo> offsetReg;
    MicroAPI::MaskReg maskReg;
    MicroAPI::MaskReg maskRegTwo;
    MicroAPI::MaskReg negMask;

    for (uint16_t i = 0; i < loopCnt; i++) {
        maskReg = MicroAPI::UpdateMask<IDX_T>(size);
        MicroAPI::DataCopy<IDX_T, MicroAPI::LoadDist::DIST_NORM>(indicesReg, indicesAddr + i * vfLen);
        MicroAPI::Compares<IDX_T, CMPMODE::LT>(negMask, indicesReg, zero, maskReg);
        MicroAPI::Adds(tmpReg, indicesReg, static_cast<IDX_T>(dim0Shape), negMask);
        Copy<IDX_T, AscendC::MicroAPI::MaskMergeMode::MERGING>(indicesReg, tmpReg, negMask);
        maskRegTwo = MicroAPI::UpdateMask<int64_t, MicroAPI::RegTraitNumTwo>(sizeTwo);
        MicroAPI::Cast<int64_t, int32_t, castTraitB32B64>(offsetReg, indicesReg, maskRegTwo);
        MicroAPI::Muls(offsetReg, offsetReg, lastDimBytes, maskRegTwo);
        MicroAPI::DataCopy(offsetAddr + i * vfLen, offsetReg, maskRegTwo);
    }
}

template <typename T, typename U, typename SrcU, uint8_t MASK_MODE, uint8_t DIM_NUM>
class KernelIndexFullLoadDma {
public:
    __aicore__ inline KernelIndexFullLoadDma(TPipe& pipeIn, const IndexFullLoadTilingData& tilingData)
        : pipe_(pipeIn), tilingData_(tilingData){};
    __aicore__ inline void Init(GM_ADDR inputX, GM_ADDR indexedSizes, GM_ADDR indices, GM_ADDR output);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyInIndices(int64_t loopIdx, int64_t length);
    __aicore__ inline __gm__ SrcU* GetTensorAddr(int64_t index, int64_t offset);
    __aicore__ inline void Compute(uint32_t indicesCount);
    __aicore__ inline void ComputeDma(uint32_t indicesCount, LocalTensor<U>& indicesLocal, LocalTensor<T>& outputLocal,
                                      int32_t dim0Shape);
    __aicore__ inline void PrepareGatherOffset(uint32_t indicesCount, LocalTensor<U>& indicesLocal,
                                               LocalTensor<int64_t>& offsetLocal, int64_t dim0Shape,
                                               uint32_t lastDimBytes);
    __aicore__ inline void CopyOut(int64_t loopIdx, int64_t length);

private:
    TPipe& pipe_;
    const IndexFullLoadTilingData& tilingData_;
    GlobalTensor<int8_t> inputXGmInt8_;
    GlobalTensor<T> outputGm_;
    GlobalTensor<SrcU> indicesGm_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> indicesQueue_;
    TBuf<QuePosition::VECCALC> castBuff_;
    TBuf<QuePosition::VECCALC> offsetBuf_;
    int64_t currentCoreLoop_ = 0;
    int64_t currentCoreTail_ = 0;
    uint32_t blockIdx_ = 0;
    uint32_t blockSize_ = 0;
    uint32_t indicesQueOffset_ = 0;
    ListTensorDesc inputList_;
};

template <typename T, typename U, typename SrcU, uint8_t MASK_MODE, uint8_t DIM_NUM>
__aicore__ inline void KernelIndexFullLoadDma<T, U, SrcU, MASK_MODE, DIM_NUM>::Init(GM_ADDR inputX,
                                                                                    GM_ADDR indexedSizes,
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
    inputXGmInt8_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(inputX));
    outputGm_.SetGlobalBuffer((__gm__ T*)(output) + blockIdx_ * tilingData_.eachCoreSize * tilingData_.lastDimSize);
    pipe_.InitBuffer(outQueue_, BUFFER_NUM, tilingData_.outputQueSize);
    pipe_.InitBuffer(indicesQueue_, BUFFER_NUM, tilingData_.indicesQueSize);
    if constexpr (std::is_same<SrcU, int64_t>::value) {
        pipe_.InitBuffer(castBuff_,
                         Ops::Base::CeilAlign(tilingData_.ubIndices * sizeof(SrcU), static_cast<uint64_t>(blockSize_)));
    }
    pipe_.InitBuffer(offsetBuf_,
                     Ops::Base::CeilAlign(tilingData_.ubIndices * sizeof(int64_t), static_cast<uint64_t>(blockSize_)));
    return;
}

template <typename T, typename U, typename SrcU, uint8_t MASK_MODE, uint8_t DIM_NUM>
__aicore__ inline void KernelIndexFullLoadDma<T, U, SrcU, MASK_MODE, DIM_NUM>::Process()
{
    if (blockIdx_ >= tilingData_.usedCoreNum) {
        return;
    }
    for (int64_t i = 0; i < currentCoreLoop_ - 1; i++) {
        CopyInIndices(i, tilingData_.ubIndices);
        Compute(static_cast<uint32_t>(tilingData_.ubIndices));
        CopyOut(i, tilingData_.ubIndices);
    }
    CopyInIndices(currentCoreLoop_ - 1, currentCoreTail_);
    Compute(static_cast<uint32_t>(currentCoreTail_));
    CopyOut(currentCoreLoop_ - 1, currentCoreTail_);
    return;
}

template <typename T, typename U, typename SrcU, uint8_t MASK_MODE, uint8_t DIM_NUM>
__aicore__ inline void KernelIndexFullLoadDma<T, U, SrcU, MASK_MODE, DIM_NUM>::CopyInIndices(int64_t loopIdx,
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
__aicore__ inline __gm__ SrcU* KernelIndexFullLoadDma<T, U, SrcU, MASK_MODE, DIM_NUM>::GetTensorAddr(int64_t index,
                                                                                                     int64_t offset)
{
    return inputList_.GetDataPtr<SrcU>(index) + offset;
}

template <typename T, typename U, typename SrcU, uint8_t MASK_MODE, uint8_t DIM_NUM>
__aicore__ inline void KernelIndexFullLoadDma<T, U, SrcU, MASK_MODE, DIM_NUM>::Compute(uint32_t indicesCount)
{
    LocalTensor<U> indicesLocal = indicesQueue_.DeQue<U>();
    LocalTensor<T> outputLocal = outQueue_.AllocTensor<T>();
    int32_t dim0Shape = (DIM_NUM == DIM1) ? tilingData_.inputShape[INPUT_AXIS_7] : tilingData_.inputShape[INPUT_AXIS_6];
    ComputeDma(indicesCount, indicesLocal, outputLocal, dim0Shape);
    indicesQueue_.FreeTensor(indicesLocal);
    outQueue_.EnQue(outputLocal);
    return;
}

template <typename T, typename U, typename SrcU, uint8_t MASK_MODE, uint8_t DIM_NUM>
__aicore__ inline void KernelIndexFullLoadDma<T, U, SrcU, MASK_MODE, DIM_NUM>::PrepareGatherOffset(
    uint32_t indicesCount, LocalTensor<U>& indicesLocal, LocalTensor<int64_t>& offsetLocal, int64_t dim0Shape,
    uint32_t lastDimBytes)
{
    constexpr uint32_t vfLen = VECTOR_REG_WIDTH / sizeof(U);
    uint16_t loopCnt = static_cast<uint16_t>((indicesCount + vfLen - 1) / vfLen);
    __local_mem__ U* indicesAddr = (__local_mem__ U*)indicesLocal.GetPhyAddr();
    __local_mem__ int64_t* offsetAddr = (__local_mem__ int64_t*)offsetLocal.GetPhyAddr();
    GenOffsetVF<U>(indicesAddr, offsetAddr, vfLen, loopCnt, indicesCount, indicesCount, dim0Shape,
                   static_cast<int64_t>(lastDimBytes), static_cast<U>(0));
}

template <typename T, typename U, typename SrcU, uint8_t MASK_MODE, uint8_t DIM_NUM>
__aicore__ inline void KernelIndexFullLoadDma<T, U, SrcU, MASK_MODE, DIM_NUM>::ComputeDma(uint32_t indicesCount,
                                                                                          LocalTensor<U>& indicesLocal,
                                                                                          LocalTensor<T>& outputLocal,
                                                                                          int32_t dim0Shape)
{
    const uint32_t lastDimBytes = static_cast<uint32_t>(tilingData_.lastDimSize * sizeof(T));
    const uint32_t dstStride = static_cast<uint32_t>(
        Ops::Base::CeilAlign(static_cast<uint64_t>(lastDimBytes), static_cast<uint64_t>(blockSize_)));
    LocalTensor<int64_t> offsetLocal = offsetBuf_.Get<int64_t>();

    PrepareGatherOffset(indicesCount, indicesLocal, offsetLocal, static_cast<int64_t>(dim0Shape), lastDimBytes);

    LocalTensor<int8_t> outputInt8 = outputLocal.template ReinterpretCast<int8_t>();
    Duplicate<int8_t>(outputInt8, 0, indicesCount * (dstStride / sizeof(int8_t)));

    event_t eventIdVtoMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    SetFlag<HardEvent::V_MTE2>(eventIdVtoMTE2);
    WaitFlag<HardEvent::V_MTE2>(eventIdVtoMTE2);

    GatherParams gatherParams;
    gatherParams.offsetNum = static_cast<uint16_t>(indicesCount);
    gatherParams.burstLen = lastDimBytes;
    gatherParams.dstStride = dstStride;
    GatherPadParams<int8_t> gatherPadParams{GatherPaddingMode::CONSTANT_PADDING_MODE, static_cast<int8_t>(0)};
    GatherDataCopy(outputInt8, inputXGmInt8_, offsetLocal, gatherParams, gatherPadParams);

    event_t eventIdMTE2toV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIdMTE2toV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMTE2toV);
    return;
}

template <typename T, typename U, typename SrcU, uint8_t MASK_MODE, uint8_t DIM_NUM>
__aicore__ inline void KernelIndexFullLoadDma<T, U, SrcU, MASK_MODE, DIM_NUM>::CopyOut(int64_t loopIdx, int64_t length)
{
    LocalTensor<T> outputLocal = outQueue_.DeQue<T>();
    int64_t outputGmOffset = loopIdx * tilingData_.ubIndices * tilingData_.lastDimSize;
    uint32_t lastDimBytes = static_cast<uint32_t>(tilingData_.lastDimSize * sizeof(T));
    DataCopyExtParams copyParams;
    copyParams.blockCount = static_cast<uint16_t>(length);
    copyParams.blockLen = lastDimBytes;
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPad(outputGm_[outputGmOffset], outputLocal, copyParams);
    outQueue_.FreeTensor(outputLocal);
    return;
}
} // namespace Index

#endif // defined(__NPU_ARCH__) && (__NPU_ARCH__ == 9201 || __NPU_ARCH__ == 9202)
#endif // ASCENDC_INDEX_FULL_LOAD_DMA_H_
