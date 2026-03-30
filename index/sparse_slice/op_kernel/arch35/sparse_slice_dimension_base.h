/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file sparse_slice_dimension_base.h
 * \brief
 */

#ifndef SPARSE_SLICE_DIMENSION_BASE_H
#define SPARSE_SLICE_DIMENSION_BASE_H

#include "sparse_slice_common.h"
#include "../inc/kernel_utils.h"
#include "../inc/platform.h"

namespace SparseSlice {
using namespace AscendC;
constexpr uint32_t VECTOR_REG_WIDTH_ONE = 256;
constexpr int64_t NUM_ZERO = 0;
constexpr int64_t BUFFER_NUM_ONE = 1;
constexpr int64_t BUFFER_NUM_TWO = 2;
constexpr int64_t INT64_ALIGN_NUM = 4;
constexpr int64_t NUM_EIGHT = 8;
constexpr int64_t NUM_ONE_K = 1024;
constexpr int64_t REDUCE_OUTPUT_NUM = 32;
constexpr int64_t INDICES_TENSOR_RANK = 2;
constexpr int64_t Y_SHAPE_TILING_DATA_OFFSET = 64;
constexpr int64_t START_TILING_DATA_OFFSET = 64 + 24 * sizeof(int64_t);
constexpr int64_t END_TILING_DATA_OFFSET = 64 + 48 * sizeof(int64_t);
constexpr int64_t TILING_DATA_ARRAY_SIZE = 24 * sizeof(int64_t);

static constexpr MultiCopyConfig config = { false };

template <typename T>
class SparseSliceDimension : public SparseSliceBase {
public:
    __aicore__ inline SparseSliceDimension(){};
    __aicore__ inline void Init(GM_ADDR indices, GM_ADDR values,
                                GM_ADDR shape, GM_ADDR start, GM_ADDR size,
                                GM_ADDR yIndices, GM_ADDR yValues, GM_ADDR yShape, GM_ADDR outputShape1,
                                GM_ADDR workspace, const SparseSliceTilingData& tilingData, TPipe* pipeIn);
    __aicore__ inline void Process();
    __aicore__ inline void InitOut();
    __aicore__ inline void Compute(const LocalTensor<int64_t> &sliceStartUb, const LocalTensor<int64_t> &sliceEndUb,
                                   LocalTensor<int32_t> &counterUb, int32_t count);
    __aicore__ inline void CopyOut(int64_t offset, int32_t count);
    __aicore__ inline void CopyOutNum(const LocalTensor<int32_t> &counterUb);
    __aicore__ inline void CopyOutMask(int64_t offset ,int32_t count);
    __aicore__ inline void CopyInMask(int64_t offset ,int32_t count);

    __aicore__ inline void CalcFinalOffset();

    template<bool isPad=false>
    __aicore__ inline void CopyInIndices(int64_t offset, int32_t count);

    __aicore__ inline void CompareMultiDim(
        const LocalTensor<int64_t> &sliceStartUb, const LocalTensor<int64_t> &sliceEndUb,
        const LocalTensor<int64_t> &indicesUb, LocalTensor<int8_t> &maskUb, LocalTensor<int32_t> &counterUb, int32_t count);
    __aicore__ inline void CopyInValues(int64_t offset, int32_t count);

    template <bool isTwoDim=false>
    __aicore__ inline void GatherOutputCopyOut(LocalTensor<int64_t> &sliceStartUb);

    template<bool isPad=false, bool isTwoDim=false>
    __aicore__ inline void GatherIndicesAndValues(
        const LocalTensor<int8_t> &maskUb, LocalTensor<int64_t> &yIndicesTransposed, LocalTensor<int64_t> &sliceStartUb, int32_t count);
    __aicore__ inline void GatherIndicesMultiDim(
        __ubuf__ int8_t* maskUbAddr, __ubuf__ int64_t* indicesUbAddr,
        __ubuf__ int64_t* yIndicesUbAddr, __ubuf__ int64_t* sliceStartAddr, int32_t count);

    template <typename T1>
    __aicore__ inline void GatherValues(__ubuf__ int8_t *maskUbAddr, __ubuf__ T1 *valuesUbAddr, __ubuf__ T1 *yValuesUbAddr, int32_t count);
    __aicore__ inline void CopyOutIndicesAndValues();
    __aicore__ inline void Transpose2D(LocalTensor<int64_t> &yIndicesTransposed, int64_t rowNum, int64_t colNum);
    __aicore__ inline void Transpose2D2Dim(LocalTensor<int64_t> &yIndicesTransposed, int64_t colNum);
    __aicore__ inline void CopyOutShapes();

private:
    TPipe* pipe;

    TQue<TPosition::VECIN, BUFFER_NUM_TWO> indicesQueue_, valuesQueue_, maskInQueue_;
    TQue<TPosition::VECOUT, BUFFER_NUM_TWO> yIndicesQueue_, yValuesQueue_, maskOutQueue_, yIndicesTransposedQueue_;
    TQue<TPosition::VECIN, BUFFER_NUM_ONE> outNumQueue_, inNumQueue_;
    TBuf<TPosition::VECCALC> countBuf_;
    TBuf<TPosition::VECCALC> yShapeBuf_, yIndicesShapeBuf_, yValuesShapeBuf_, yShapeShapeBuf_;
    TBuf<TPosition::VECCALC> sliceStartBuf_, sliceEndBuf_;

    int64_t curCoreProcessNum_ = 0;     // current core process point num
    int64_t valuePerUbAligned_ = 0;     // maximum process points per loop in ub
    int64_t ubLoopNum_ = 0;         // ub loop num
    int64_t ubLoopTailCount_ = 0;   // tail process number in the last ub loop
    int64_t curCoreOutNum_ = 0;     // current core output sparse point number
    uint64_t finalOffset_ = 0;      // sparse points final offset globally
    uint64_t specialArNum_ = 0;     // store output number per loop in ub
    uint64_t curCoreOffset_ = 0;    // update current core offset per loop in ub

    GlobalTensor<int64_t> indicesGm_;
    GlobalTensor<T> valuesGm_;
    GlobalTensor<int64_t> shapeGm_;
    GlobalTensor<int64_t> startGm_;
    GlobalTensor<int64_t> sizeGm_;
    GlobalTensor<int64_t> yIndicesGm_;
    GlobalTensor<T> yValuesGm_;
    GlobalTensor<int64_t> yShapeGm_;
    GlobalTensor<int64_t> outputShapeGm_;

    GlobalTensor<int64_t> outNumGm_;
    GlobalTensor<int8_t> maskGm_;
    SparseSliceTilingData tilingData_;
    bool isOutShapeEmpty_ = false;

    GM_ADDR yIndicesAddr_;
    GM_ADDR yValuesAddr_;

    uint32_t repeatElmB64_ = platform::GetVRegSize() / sizeof(int64_t);
    uint32_t repeatElmB32_ = platform::GetVRegSize() / sizeof(int32_t);
    uint32_t repeatElmB16_ = platform::GetVRegSize() / sizeof(uint16_t);
    uint32_t repeatElmB8_ = platform::GetVRegSize() / sizeof(uint8_t);
};

template <typename T>
__aicore__ inline void SparseSliceDimension<T>::Init(
    GM_ADDR indices, GM_ADDR values, GM_ADDR shape, GM_ADDR start, GM_ADDR size,
    GM_ADDR yIndices, GM_ADDR yValues, GM_ADDR yShape, GM_ADDR outputShape1,
    GM_ADDR workspace, const SparseSliceTilingData& tilingData, TPipe* pipeIn)
{
    pipe = pipeIn;
    // parse tiling data
    SparseSliceBase::ParseTilingData(tilingData);
    blockIdx_ = GetBlockIdx();
    curCoreProcessNum_ = (blockIdx_ == usedCoreNum_ - 1) ? valuePerTail_ : valuePerCore_;
    valuePerUbAligned_ = ops::CeilDiv<int64_t>(valuePerUb_, INT64_ALIGN_NUM) * INT64_ALIGN_NUM;
    tilingData_ = tilingData;

    // copy gm addr 
    yIndicesAddr_ = yIndices;
    yValuesAddr_ = yValues;
    if (blockIdx_ >= usedCoreNum_) {
        return;
    }

    uint64_t intraCoreOffsetIndices = blockIdx_ * valuePerCore_ * rankNumbers_;
    uint64_t intraCoreOffsetValues = blockIdx_ * valuePerCore_;

    indicesGm_.SetGlobalBuffer((__gm__ int64_t*)indices + intraCoreOffsetIndices, curCoreProcessNum_ * rankNumbers_);
    valuesGm_.SetGlobalBuffer((__gm__ T*)values + intraCoreOffsetValues, curCoreProcessNum_);
    shapeGm_.SetGlobalBuffer((__gm__ int64_t*)shape, rankNumbers_);
    startGm_.SetGlobalBuffer((__gm__ int64_t*)start, rankNumbers_);
    sizeGm_.SetGlobalBuffer((__gm__ int64_t*)size, rankNumbers_);

    yShapeGm_.SetGlobalBuffer((__gm__ int64_t*)yShape, rankNumbers_);
    outputShapeGm_.SetGlobalBuffer((__gm__ int64_t*)outputShape1, DIGIT_NINE * DIGIT_THREE);

    // should use block num
    // 设置per core偏移
    outNumGm_.SetGlobalBuffer((__gm__ int64_t*)workspace + usedCoreNum_, usedCoreNum_ * NUM_EIGHT);
    maskGm_.SetGlobalBuffer((__gm__ int8_t*)workspace + intraCoreOffsetValues + NUM_ONE_K, curCoreProcessNum_);

    pipe->InitBuffer(sliceStartBuf_, TILING_DATA_ARRAY_SIZE);
    pipe->InitBuffer(sliceEndBuf_, TILING_DATA_ARRAY_SIZE);
    pipe->InitBuffer(countBuf_, REDUCE_OUTPUT_NUM * sizeof(int64_t));

    pipe->InitBuffer(indicesQueue_, BUFFER_NUM_TWO, rankNumbers_ * valuePerUbAligned_ * sizeof(int64_t));
    pipe->InitBuffer(yIndicesTransposedQueue_, BUFFER_NUM_TWO, rankNumbers_ * valuePerUbAligned_ * sizeof(int64_t));
    pipe->InitBuffer(valuesQueue_, BUFFER_NUM_TWO, valuePerUbAligned_ * sizeof(int64_t));

    pipe->InitBuffer(maskInQueue_, BUFFER_NUM_TWO, valuePerUbAligned_ * sizeof(int8_t));
    pipe->InitBuffer(maskOutQueue_, BUFFER_NUM_TWO, valuePerUbAligned_ * sizeof(int8_t));

    pipe->InitBuffer(outNumQueue_, BUFFER_NUM_ONE, 1 * sizeof(int64_t));
    pipe->InitBuffer(inNumQueue_, BUFFER_NUM_ONE, usedCoreNum_ * sizeof(int64_t));

    pipe->InitBuffer(yShapeBuf_, rankNumbers_ * sizeof(int64_t));
    pipe->InitBuffer(yIndicesShapeBuf_, EMPTY_RESERVED * sizeof(int64_t));
    pipe->InitBuffer(yValuesShapeBuf_, EMPTY_RESERVED * sizeof(int64_t));
    pipe->InitBuffer(yShapeShapeBuf_, EMPTY_RESERVED * sizeof(int64_t));
};

template <typename T>
__aicore__ inline void SparseSliceDimension<T>::InitOut()
{
    yIndicesGm_.SetGlobalBuffer((__gm__ int64_t*)yIndicesAddr_ + finalOffset_ * rankNumbers_, curCoreOutNum_ * rankNumbers_);
    yValuesGm_.SetGlobalBuffer((__gm__ T*)yValuesAddr_ + finalOffset_, curCoreOutNum_);
    pipe->InitBuffer(yIndicesQueue_, BUFFER_NUM_TWO, rankNumbers_ * valuePerUbAligned_ * sizeof(int64_t));
    pipe->InitBuffer(yValuesQueue_, BUFFER_NUM_TWO, valuePerUbAligned_ * sizeof(T));

    if (blockIdx_ == usedCoreNum_ - 1) {
        CopyOutShapes();
    }
}

template <typename T>
__aicore__ inline void SparseSliceDimension<T>::CopyOutShapes()
{
    // tail core only
    LocalTensor<int64_t> yShapeTensor = yShapeBuf_.Get<int64_t>();
    for (int32_t i = 0; i < rankNumbers_; i++) {
        yShapeTensor.SetValue(i, yShape_[i]);
    }

    event_t eventId1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(eventId1);
    WaitFlag<HardEvent::S_MTE3>(eventId1);
    DataCopyPad(yShapeGm_[0], yShapeTensor[0], {1, static_cast<uint16_t>(rankNumbers_ * sizeof(int64_t)), 0, 0});

    LocalTensor<int64_t> outputIndicesShapeTensor = yIndicesShapeBuf_.Get<int64_t>();
    outputIndicesShapeTensor.SetValue(0, Y_INDICES_SHAPE_DIM_BASE);
    outputIndicesShapeTensor.SetValue(1, finalOffset_ + curCoreOutNum_); // 这边如果有有效输出要改成有效输出的值
    outputIndicesShapeTensor.SetValue(2, rankNumbers_);
    event_t eventId2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(eventId2);
    WaitFlag<HardEvent::S_MTE3>(eventId2);
    DataCopyPad(outputShapeGm_[0], outputIndicesShapeTensor[0], {1, static_cast<uint16_t>(DIGIT_THREE * sizeof(int64_t)), 0, 0});

    LocalTensor<int64_t> outputValuesShapeTensor = yValuesShapeBuf_.Get<int64_t>();
    outputValuesShapeTensor.SetValue(0, DIGIT_ONE);
    outputValuesShapeTensor.SetValue(1, finalOffset_ + curCoreOutNum_); // 这边如果有有效输出要改成有效输出的值
    event_t eventId3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(eventId3);
    WaitFlag<HardEvent::S_MTE3>(eventId3);
    DataCopyPad(outputShapeGm_[DIGIT_NINE], outputValuesShapeTensor[0], {1, static_cast<uint16_t>(DIGIT_TWO * sizeof(int64_t)), 0, 0});

    LocalTensor<int64_t> outputShapeShapeTensor = yShapeShapeBuf_.Get<int64_t>();
    outputShapeShapeTensor.SetValue(0, DIGIT_ONE);
    outputShapeShapeTensor.SetValue(1, rankNumbers_);
    event_t eventId4 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(eventId4);
    WaitFlag<HardEvent::S_MTE3>(eventId4);
    DataCopyPad(outputShapeGm_[DIGIT_EIGHTEEN], outputShapeShapeTensor[0], {1, static_cast<uint16_t>(DIGIT_TWO * sizeof(int64_t)), 0, 0});
}

template <typename T>
__aicore__ inline void SparseSliceDimension<T>::Process()
{
    if (blockIdx_ >= usedCoreNum_) {
        SyncAll();
        return;
    }

    ubLoopNum_ = ops::CeilDiv<int64_t>(curCoreProcessNum_, valuePerUbAligned_);
    ubLoopTailCount_ = curCoreProcessNum_ - (ubLoopNum_ - 1) * valuePerUbAligned_;
    int64_t offset = 0;
    uint64_t copyMask = 24;

    LocalTensor<int64_t> startTemp(TPosition::VECIN, START_TILING_DATA_OFFSET, TILING_DATA_ARRAY_SIZE);
    LocalTensor<int64_t> endTemp(TPosition::VECIN, END_TILING_DATA_OFFSET, TILING_DATA_ARRAY_SIZE);

    LocalTensor<int64_t> sliceStartUb = sliceStartBuf_.Get<int64_t>();
    LocalTensor<int64_t> sliceEndUb = sliceEndBuf_.Get<int64_t>();
    LocalTensor<int32_t> counterUb = countBuf_.Get<int32_t>(REDUCE_OUTPUT_NUM);

    DataCopy(sliceStartUb, startTemp, copyMask);
    DataCopy(sliceEndUb, endTemp, copyMask);
    Duplicate(counterUb, 0, REDUCE_OUTPUT_NUM);

    
    for (int64_t loopIndex = 0; loopIndex < ubLoopNum_ - 1; loopIndex++) {
        CopyInIndices<false>(offset, valuePerUbAligned_);
        Compute(sliceStartUb, sliceEndUb, counterUb, valuePerUbAligned_);
        CopyOutMask(offset, valuePerUbAligned_);
        offset += valuePerUbAligned_;
    }
    // tail block calculation, if loopNum == 1, no need to CopyOut mask to gm
    {
        // tail block copy in should consider padding along the zero-th dim (pad along number of points)
        CopyInIndices<true>(offset, ubLoopTailCount_);
        Compute(sliceStartUb, sliceEndUb, counterUb, ubLoopTailCount_);
        CopyOutMask(offset, ubLoopTailCount_);
    }

    // copyOut curCoreProcessNum to gm
    CopyOutNum(counterUb);
    // SyncAll vector cores

    PipeBarrier<PIPE_ALL>();
    SyncAll();

    // calculate gm copyOut offset
    CalcFinalOffset();
    // init output gm Addr
    InitOut();
    // gather output
    if (rankNumbers_ == 2) {
        GatherOutputCopyOut<true>(sliceStartUb);
    } else {
        GatherOutputCopyOut<false>(sliceStartUb);
    }
}

template <typename T>
__aicore__ inline void SparseSliceDimension<T>::CopyOutNum(const LocalTensor<int32_t> &counterUb)
{
    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);

    LocalTensor<int64_t> outNumTensor = outNumQueue_.AllocTensor<int64_t>();
    curCoreOutNum_ = static_cast<int64_t>(counterUb.GetValue(0));
    outNumTensor.SetValue<int64_t>(0, curCoreOutNum_);

    event_t eventIdSToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(eventIdSToMte3);
    WaitFlag<HardEvent::S_MTE3>(eventIdSToMte3);

    DataCopyExtParams copyParams{ static_cast<uint16_t>(1), static_cast<uint32_t>(1 * static_cast<int32_t>(sizeof(int64_t))),
        static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0) };
    DataCopyPad<int64_t>(outNumGm_[blockIdx_], outNumTensor, copyParams);
    outNumQueue_.FreeTensor(outNumTensor);
}

template <typename T>
__aicore__ inline void SparseSliceDimension<T>::CopyOutMask(int64_t offset ,int32_t count)
{
    LocalTensor<int8_t> mask = maskOutQueue_.DeQue<int8_t>();
    DataCopyParams copyParams{ static_cast<uint16_t>(1), static_cast<uint16_t>(count * static_cast<int32_t>(sizeof(int8_t))),
        static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    DataCopyPad<int8_t>(maskGm_[offset], mask, copyParams);
    maskOutQueue_.FreeTensor(mask);
}

template <typename T>
__aicore__ inline void SparseSliceDimension<T>::CopyInMask(int64_t offset ,int32_t count)
{
    LocalTensor<int8_t> maskUb = maskInQueue_.AllocTensor<int8_t>();

    DataCopyExtParams copyParams{ static_cast<uint16_t>(1), static_cast<uint32_t>(count * static_cast<int32_t>(sizeof(int8_t))),
        static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0) };
    DataCopyPadExtParams<int8_t> padParams{false, static_cast<uint8_t>(0), static_cast<uint8_t>(0),
        static_cast<int8_t>(0) };
    DataCopyPad<int8_t>(maskUb, maskGm_[offset], copyParams, padParams);
    maskInQueue_.EnQue(maskUb);
}


template <typename T>
__aicore__ inline void SparseSliceDimension<T>::CalcFinalOffset()
{
    LocalTensor<int64_t> inNumsUb = inNumQueue_.AllocTensor<int64_t>();
    DataCopyExtParams copyParams{ static_cast<uint16_t>(1), static_cast<uint32_t>(usedCoreNum_ * static_cast<int32_t>(sizeof(int64_t))),
        static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0) };
    DataCopyPadExtParams<int64_t> padParams{true, static_cast<uint8_t>(0), static_cast<uint8_t>(0),
        static_cast<int64_t>(0) };

    DataCopyPad<int64_t>(inNumsUb, outNumGm_[0], copyParams, padParams);

    event_t eventIdMte2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
    WaitFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
    uint64_t finalOffset = 0;
    for (int64_t idx = 0; idx < blockIdx_; idx++) {
        finalOffset += inNumsUb.GetValue(idx);
    }
    finalOffset_ = finalOffset;
    curCoreOutNum_ = inNumsUb.GetValue(blockIdx_);
    inNumQueue_.FreeTensor(inNumsUb);
}

template <typename T>
template <bool isPad>
__aicore__ inline void SparseSliceDimension<T>::CopyInIndices(int64_t offset, int32_t count)
{
    LocalTensor<int64_t> indicesUb = indicesQueue_.AllocTensor<int64_t>();
    uint8_t rightPaddingNum = 0;
    if constexpr(isPad) {
        rightPaddingNum = static_cast<uint8_t>(ops::CeilDiv<int64_t>(count, INT64_ALIGN_NUM) * INT64_ALIGN_NUM - count);
    }
    MultiCopyLoopInfo<INDICES_TENSOR_RANK> loopInfo{
        {(uint64_t)1, (uint64_t)rankNumbers_},
        {(uint32_t)count + rightPaddingNum, (uint32_t)1},
        {(uint32_t)rankNumbers_, (uint32_t)count},
        {(uint8_t)0, (uint8_t)0},
        {(uint8_t)rightPaddingNum, (uint8_t)0}, 
    };
    MultiCopyParams<int64_t, INDICES_TENSOR_RANK> params{
        loopInfo,
        NUM_ZERO
    };
    DataCopy<int64_t, INDICES_TENSOR_RANK, config>(indicesUb, indicesGm_[offset * rankNumbers_], params);
    indicesQueue_.EnQue(indicesUb);
}

template <typename T>
__aicore__ inline void SparseSliceDimension<T>::Compute(
    const LocalTensor<int64_t> &sliceStartUb, const LocalTensor<int64_t> &sliceEndUb,
    LocalTensor<int32_t> &counterUb, int32_t count)
{
    LocalTensor<int64_t> indicesUb = indicesQueue_.DeQue<int64_t>();
    LocalTensor<int8_t> maskUb = maskOutQueue_.AllocTensor<int8_t>();

    Duplicate(maskUb, (int8_t)1, valuePerUbAligned_);
    CompareMultiDim(sliceStartUb, sliceEndUb, indicesUb, maskUb, counterUb, count);
    indicesQueue_.FreeTensor(indicesUb);
    maskOutQueue_.EnQue(maskUb);
}

template <typename T>
__aicore__ inline void SparseSliceDimension<T>::CompareMultiDim(
    const LocalTensor<int64_t> &sliceStartUb, const LocalTensor<int64_t> &sliceEndUb,
    const LocalTensor<int64_t> &indicesUb, LocalTensor<int8_t> &maskUb, LocalTensor<int32_t> &counterUb, int32_t count)
{
    uint32_t dtypeSize = sizeof(int64_t);

    uint32_t vlSize =  VECTOR_REG_WIDTH_ONE / dtypeSize;        // 32
    uint32_t vl = vlSize * DIGIT_TWO;                                   // 64
    uint16_t loopNum = ops::CeilDiv<uint32_t>(count, vl);
    
    int64_t offsetPerLoop = ops::CeilDiv<int64_t>(count, INT64_ALIGN_NUM) * INT64_ALIGN_NUM;

    __ubuf__ int64_t* sliceStartAddr = (__ubuf__ int64_t*)sliceStartUb.GetPhyAddr();
    __ubuf__ int64_t* sliceEndAddr = (__ubuf__ int64_t*)sliceEndUb.GetPhyAddr();

    __ubuf__ int64_t* indicesUbAddr = (__ubuf__ int64_t*)indicesUb.GetPhyAddr();
    __ubuf__ int8_t* maskUbAddr = (__ubuf__ int8_t*)maskUb.GetPhyAddr();
    __ubuf__ int32_t* counterUbAddr = (__ubuf__ int32_t*)counterUb.GetPhyAddr();

    uint16_t rn = static_cast<uint16_t>(rankNumbers_);

    __VEC_SCOPE__ {
        uint16_t rankNum = rn;
        MicroAPI::RegTensor<int64_t> vregInputOne;
        MicroAPI::RegTensor<int64_t> vregInputTwo;
        
        MicroAPI::RegTensor<uint32_t> vregLowerBoundLowerHalf;
        MicroAPI::RegTensor<int32_t> vregLowerBoundHigherHalf;
        MicroAPI::RegTensor<uint32_t> vregUpperBoundLowerHalf;
        MicroAPI::RegTensor<int32_t> vregUpperBoundHigherHalf;

        MicroAPI::RegTensor<int32_t> vregInputLowerHalf;        // defined as int32, but use as uint32 in comparison
        MicroAPI::RegTensor<int32_t> vregInputHigherHalf;

        MicroAPI::RegTensor<int8_t> vregInMask;
        MicroAPI::RegTensor<int8_t> vregOutMask;
        MicroAPI::RegTensor<int32_t> vregZeros;

        MicroAPI::RegTensor<int32_t> vregCounterIn;
        MicroAPI::RegTensor<int32_t> reduceResult;

        MicroAPI::MaskReg flagHigherEqual;
        MicroAPI::MaskReg flagHigherCmp;
        MicroAPI::MaskReg flagLowerCmp;
        MicroAPI::MaskReg flagResultB32;
        MicroAPI::MaskReg flagResultB32Two;
        MicroAPI::MaskReg pregB32All = MicroAPI::CreateMask<int32_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg pregB32First = MicroAPI::CreateMask<int32_t, MicroAPI::MaskPattern::VL1>();
        MicroAPI::MaskReg pregIndicesB32;

        MicroAPI::Duplicate<int32_t>(vregZeros, 0);

        for (uint16_t dim = 0; dim < rankNum; dim++) {
            uint32_t sreg0 = count;
            MicroAPI::DataCopy<uint32_t, MicroAPI::LoadDist::DIST_BRC_B32>(vregLowerBoundLowerHalf, (__ubuf__ uint32_t*)(sliceStartAddr) + dim * DIGIT_TWO);
            MicroAPI::DataCopy<int32_t, MicroAPI::LoadDist::DIST_BRC_B32>(vregLowerBoundHigherHalf, (__ubuf__ int32_t*)(sliceStartAddr) + dim * DIGIT_TWO + 1);
            MicroAPI::DataCopy<uint32_t, MicroAPI::LoadDist::DIST_BRC_B32>(vregUpperBoundLowerHalf, (__ubuf__ uint32_t*)(sliceEndAddr) + dim * DIGIT_TWO);
            MicroAPI::DataCopy<int32_t, MicroAPI::LoadDist::DIST_BRC_B32>(vregUpperBoundHigherHalf, (__ubuf__ int32_t*)(sliceEndAddr) + dim * DIGIT_TWO + 1);

            AscendC::MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
            for (uint16_t loopIndex = 0; loopIndex < loopNum; loopIndex++) {
                // use b32 updatemask since there's twice copyin of int64 from ub to register
                pregIndicesB32 = MicroAPI::UpdateMask<int32_t>(sreg0);
                MicroAPI::DataCopy(vregInputOne, (__ubuf__ int64_t*)(indicesUbAddr + dim * offsetPerLoop + loopIndex * vl));
                MicroAPI::DataCopy(vregInputTwo, (__ubuf__ int64_t*)(indicesUbAddr + dim * offsetPerLoop + loopIndex * vl + vlSize));

                // copyin b8 mask and unpack it to b32 mask
                MicroAPI::DataCopy<int8_t, MicroAPI::LoadDist::DIST_UNPACK4_B8>(vregInMask, (__ubuf__ int8_t*)(maskUbAddr + loopIndex * vl));

                // split lower part and higher part to compare
                MicroAPI::DeInterleave<int32_t>(
                    vregInputLowerHalf, vregInputHigherHalf,
                    (MicroAPI::RegTensor<int32_t>&)vregInputOne, (MicroAPI::RegTensor<int32_t>&)vregInputTwo);
                // lower bound compare
                MicroAPI::Compare<uint32_t, CMPMODE::GE>(
                    flagLowerCmp, (MicroAPI::RegTensor<uint32_t>&)vregInputLowerHalf, vregLowerBoundLowerHalf, pregB32All);
                MicroAPI::Compare<int32_t, CMPMODE::GE>(flagHigherCmp, vregInputHigherHalf, vregLowerBoundHigherHalf, pregB32All);
                MicroAPI::Compare<int32_t, CMPMODE::EQ>(flagHigherEqual, vregInputHigherHalf, vregLowerBoundHigherHalf, pregB32All);
                MicroAPI::MaskSel(flagResultB32, flagLowerCmp, flagHigherCmp, flagHigherEqual);
                // upper bound compare
                MicroAPI::Compare<uint32_t, CMPMODE::LT>(
                    flagLowerCmp, (MicroAPI::RegTensor<uint32_t>&)vregInputLowerHalf, vregUpperBoundLowerHalf, pregB32All);
                MicroAPI::Compare<int32_t, CMPMODE::LT>(flagHigherCmp, vregInputHigherHalf, vregUpperBoundHigherHalf, pregB32All);
                MicroAPI::Compare<int32_t, CMPMODE::EQ>(flagHigherEqual, vregInputHigherHalf, vregUpperBoundHigherHalf, pregB32All);
                MicroAPI::MaskSel(flagResultB32Two, flagLowerCmp, flagHigherCmp, flagHigherEqual);

                MicroAPI::MaskAnd(flagResultB32, flagResultB32, flagResultB32Two, pregIndicesB32);

                MicroAPI::Select((MicroAPI::RegTensor<int32_t>&)vregOutMask, (MicroAPI::RegTensor<int32_t>&)vregInMask, vregZeros, flagResultB32);
                MicroAPI::DataCopy<int8_t, MicroAPI::StoreDist::DIST_PACK4_B32>((__ubuf__ int8_t*)(maskUbAddr + loopIndex * vl), vregOutMask, pregIndicesB32);
            }
        }
        // reduce part
        AscendC::MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
        uint32_t sreg0 = count;
        MicroAPI::DataCopy<int32_t, MicroAPI::LoadDist::DIST_NORM>(vregCounterIn, (__ubuf__ int32_t*)(counterUbAddr));
        for (uint16_t loopIndex = 0; loopIndex < loopNum; loopIndex++) {
            pregIndicesB32 = MicroAPI::UpdateMask<int32_t>(sreg0);
            MicroAPI::DataCopy<int8_t, MicroAPI::LoadDist::DIST_UNPACK4_B8>(vregInMask, (__ubuf__ int8_t*)(maskUbAddr + loopIndex * vl));
            MicroAPI::ReduceSum<int32_t>(reduceResult, (MicroAPI::RegTensor<int32_t>&)vregInMask, pregIndicesB32);
            MicroAPI::Add(vregCounterIn, vregCounterIn, reduceResult, pregB32First);
        }
        MicroAPI::DataCopy<int32_t, MicroAPI::StoreDist::DIST_NORM>((__ubuf__ int32_t*)(counterUbAddr), vregCounterIn, pregB32First);
    } // vf ends
}


template <typename T>
__aicore__ inline void SparseSliceDimension<T>::CopyInValues(int64_t offset, int32_t count)
{
    LocalTensor<T> valuesUb = valuesQueue_.AllocTensor<T>();
    DataCopyExtParams copyParams{ static_cast<uint16_t>(1), static_cast<uint32_t>(count * static_cast<int32_t>(sizeof(T))),
        static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0) };
    DataCopyPadExtParams<T> padParams{false, static_cast<uint8_t>(0), static_cast<uint8_t>(0),
        static_cast<T>(0) };
    DataCopyPad<T>(valuesUb, valuesGm_[offset], copyParams, padParams);
    valuesQueue_.EnQue(valuesUb);
}

template <typename T>
template <bool isTwoDim>
__aicore__ inline void SparseSliceDimension<T>::GatherOutputCopyOut(LocalTensor<int64_t> &sliceStartUb)
{
    if (ubLoopNum_ > 1) {
        CopyInIndices<false>(0, valuePerUbAligned_);
        CopyInValues(0, valuePerUbAligned_);
        CopyInMask(0, valuePerUbAligned_);
    }
    for (int64_t loopIndex = 1; loopIndex < ubLoopNum_ - 1; loopIndex++) {
        CopyInIndices<false>(valuePerUbAligned_ * loopIndex, valuePerUbAligned_);
        CopyInValues(valuePerUbAligned_ * loopIndex, valuePerUbAligned_);
        CopyInMask(valuePerUbAligned_ * loopIndex, valuePerUbAligned_);
        LocalTensor<int8_t> maskUb = maskInQueue_.DeQue<int8_t>();
        LocalTensor<int64_t> yIndicesTransposed = yIndicesTransposedQueue_.AllocTensor<int64_t>();
        GatherIndicesAndValues<false, isTwoDim>(maskUb, yIndicesTransposed, sliceStartUb, valuePerUbAligned_);
        CopyOutIndicesAndValues();
        maskInQueue_.FreeTensor(maskUb);
        yIndicesTransposedQueue_.FreeTensor(yIndicesTransposed);
    }
    if (ubLoopNum_ > 1) {
        LocalTensor<int8_t> maskUb = maskInQueue_.DeQue<int8_t>();
        LocalTensor<int64_t> yIndicesTransposed = yIndicesTransposedQueue_.AllocTensor<int64_t>();
        GatherIndicesAndValues<false, isTwoDim>(maskUb, yIndicesTransposed, sliceStartUb, valuePerUbAligned_);
        CopyOutIndicesAndValues();
        maskInQueue_.FreeTensor(maskUb);
        yIndicesTransposedQueue_.FreeTensor(yIndicesTransposed);
    }

    CopyInIndices<true>(valuePerUbAligned_ * (ubLoopNum_ - 1), ubLoopTailCount_);
    CopyInValues(valuePerUbAligned_ * (ubLoopNum_ - 1), ubLoopTailCount_);
    CopyInMask(valuePerUbAligned_ * (ubLoopNum_ - 1), ubLoopTailCount_);
    LocalTensor<int8_t> maskUb = maskInQueue_.DeQue<int8_t>();
    LocalTensor<int64_t> yIndicesTransposed = yIndicesTransposedQueue_.AllocTensor<int64_t>();
    GatherIndicesAndValues<true, isTwoDim>(maskUb, yIndicesTransposed, sliceStartUb, ubLoopTailCount_);
    CopyOutIndicesAndValues();
    maskInQueue_.FreeTensor(maskUb);
    yIndicesTransposedQueue_.FreeTensor(yIndicesTransposed);
}

template <typename T>
template <bool isPad, bool isTwoDim>
__aicore__ inline void SparseSliceDimension<T>::GatherIndicesAndValues(
    const LocalTensor<int8_t> &maskUb, LocalTensor<int64_t> &yIndicesTransposed, LocalTensor<int64_t> &sliceStartUb, int32_t count)
{
    LocalTensor<int64_t> indicesUb = indicesQueue_.DeQue<int64_t>();

    __ubuf__ int8_t* maskUbAddr = (__ubuf__ int8_t*)maskUb.GetPhyAddr();
    int32_t offsetPerLoop = count;
    if constexpr(isPad) {
        offsetPerLoop = ops::CeilDiv<int64_t>(count, INT64_ALIGN_NUM) * INT64_ALIGN_NUM;
    }
    __ubuf__ int64_t* indicesUbAddr = (__ubuf__ int64_t*)indicesUb.GetPhyAddr();
    __ubuf__ int64_t* yIndicesTransposedAddr = (__ubuf__ int64_t*)yIndicesTransposed.GetPhyAddr();
    __ubuf__ int64_t* sliceStartUbAddr = (__ubuf__ int64_t*)sliceStartUb.GetPhyAddr();
    
    GatherIndicesMultiDim(maskUbAddr, indicesUbAddr, yIndicesTransposedAddr, sliceStartUbAddr, offsetPerLoop);

    if constexpr(isTwoDim) {
        Transpose2D2Dim(yIndicesTransposed, offsetPerLoop);
    } else {
        Transpose2D(yIndicesTransposed, rankNumbers_, offsetPerLoop);
    }

    // gather values
    LocalTensor<T> yValuesUb = yValuesQueue_.AllocTensor<T>();
    LocalTensor<T> valuesUb = valuesQueue_.DeQue<T>();

    if constexpr(sizeof(T) == sizeof(int8_t)) {
        __ubuf__ int8_t* valuesUbAddr = (__ubuf__ int8_t*)valuesUb.GetPhyAddr();
        __ubuf__ int8_t* yValuesUbAddr = (__ubuf__ int8_t*)yValuesUb.GetPhyAddr();
        GatherValues<int8_t>(maskUbAddr, valuesUbAddr, yValuesUbAddr, count);
    } else if constexpr(sizeof(T) == sizeof(int16_t)) {
        __ubuf__ int16_t* valuesUbAddr = (__ubuf__ int16_t*)valuesUb.GetPhyAddr();
        __ubuf__ int16_t* yValuesUbAddr = (__ubuf__ int16_t*)yValuesUb.GetPhyAddr();
        GatherValues<int16_t>(maskUbAddr, valuesUbAddr, yValuesUbAddr, count);
    } else if constexpr(sizeof(T) == sizeof(int32_t)) {
        __ubuf__ int32_t* valuesUbAddr = (__ubuf__ int32_t*)valuesUb.GetPhyAddr();
        __ubuf__ int32_t* yValuesUbAddr = (__ubuf__ int32_t*)yValuesUb.GetPhyAddr();
        GatherValues<int32_t>(maskUbAddr, valuesUbAddr, yValuesUbAddr, count);
    } else {
        __ubuf__ int64_t* valuesUbAddr = (__ubuf__ int64_t*)valuesUb.GetPhyAddr();
        __ubuf__ int64_t* yValuesUbAddr = (__ubuf__ int64_t*)yValuesUb.GetPhyAddr();
        GatherValues<int64_t>(maskUbAddr, valuesUbAddr, yValuesUbAddr, count);
    }
    yValuesQueue_.EnQue(yValuesUb);

    indicesQueue_.FreeTensor(indicesUb);
    valuesQueue_.FreeTensor(valuesUb);
}

template <typename T>
__aicore__ inline void SparseSliceDimension<T>::GatherIndicesMultiDim(
    __ubuf__ int8_t* maskUbAddr, __ubuf__ int64_t* indicesUbAddr,
    __ubuf__ int64_t* yIndicesUbAddr, __ubuf__ int64_t* sliceStartAddr, int32_t count)
{
    uint32_t dtypeSize = sizeof(int64_t);
    uint32_t vlSize = VECTOR_REG_WIDTH_ONE / dtypeSize;
    uint32_t vl = vlSize * DIGIT_TWO;
    uint16_t loopNum = ops::CeilDiv<uint32_t>(count, vl);
    uint16_t rn = static_cast<uint16_t>(rankNumbers_);

    __VEC_SCOPE__ {
        // clear special AR for GatherMask
        MicroAPI::RegTensor<int64_t> vregInputOne;
        MicroAPI::RegTensor<int64_t> vregInputTwo;

        MicroAPI::RegTensor<uint32_t> vregLowerBoundLowerHalf;
        MicroAPI::RegTensor<int32_t> vregLowerBoundHigherHalf;

        MicroAPI::RegTensor<uint32_t> vregInputLowerHalf;
        MicroAPI::RegTensor<int32_t> vregInputHigherHalf;

        MicroAPI::RegTensor<uint32_t> vregInputSubLower;
        MicroAPI::RegTensor<int32_t> vregInputSubHigher;


        MicroAPI::RegTensor<int32_t> vregGatherLowerHalf;
        MicroAPI::RegTensor<int32_t> vregGatherHigherHalf;

        MicroAPI::RegTensor<int32_t> vregGatherOutput0;
        MicroAPI::RegTensor<int32_t> vregGatherOutput1;
        MicroAPI::RegTensor<int32_t> vregGatherResult0;
        MicroAPI::RegTensor<int32_t> vregGatherResult1;

        MicroAPI::RegTensor<int8_t> vregInMask;
        MicroAPI::RegTensor<int32_t> vregInMaskGathered;

        MicroAPI::RegTensor<int32_t> vregOnes;

        MicroAPI::MaskReg pregIndices;
        MicroAPI::MaskReg pregSelect;
        MicroAPI::MaskReg carryOut;
        MicroAPI::MaskReg pregSelectFinal0;
        MicroAPI::MaskReg pregSelectFinal1;

        MicroAPI::MaskReg pregGathered;
        MicroAPI::MaskReg pregGatheredTwo;

        uint16_t rankNum = rn;
        MicroAPI::UnalignReg ureg0;

        MicroAPI::Duplicate(vregOnes, 1);

        for (uint16_t dim = 0; dim < rankNum; dim++) {
            MicroAPI::ClearSpr<SpecialPurposeReg::AR>();
            uint32_t sreg0 = count;
            __ubuf__ int64_t* yIndicesCurDimAddr = (__ubuf__ int64_t*)(yIndicesUbAddr) + dim * count;
            MicroAPI::DataCopy<uint32_t, MicroAPI::LoadDist::DIST_BRC_B32>(vregLowerBoundLowerHalf, (__ubuf__ uint32_t*)(sliceStartAddr) + dim * DIGIT_TWO);
            MicroAPI::DataCopy<int32_t, MicroAPI::LoadDist::DIST_BRC_B32>(vregLowerBoundHigherHalf, (__ubuf__ int32_t*)(sliceStartAddr) + dim * DIGIT_TWO + 1);
            carryOut = MicroAPI::CreateMask<uint8_t>();
            for (uint16_t loopIndex = 0; loopIndex < loopNum; loopIndex++) {
                pregIndices = MicroAPI::UpdateMask<int32_t>(sreg0);
                MicroAPI::DataCopy(vregInputOne, (__ubuf__ int64_t*)(indicesUbAddr + dim * count + loopIndex * vl));
                MicroAPI::DataCopy(vregInputTwo, (__ubuf__ int64_t*)(indicesUbAddr + dim * count + loopIndex * vl + vlSize));
                MicroAPI::DataCopy<int8_t, MicroAPI::LoadDist::DIST_UNPACK4_B8>(vregInMask, (__ubuf__ int8_t*)(maskUbAddr + loopIndex * vl));
                MicroAPI::Compare<int32_t, CMPMODE::EQ>(pregSelect, (MicroAPI::RegTensor<int32_t>&)vregInMask, vregOnes, pregIndices);

                MicroAPI::GatherMask<int32_t, MicroAPI::GatherMaskMode::NO_STORE_REG>(
                    vregInMaskGathered, (MicroAPI::RegTensor<int32_t>&)vregInMask, pregSelect);
                MicroAPI::Compare<int32_t, CMPMODE::EQ>(pregGathered, vregInMaskGathered, vregOnes, pregIndices);

                // deinterleave -> gathermask -> interleave to implement b64 gathermask
                MicroAPI::DeInterleave<int32_t>(
                    (MicroAPI::RegTensor<int32_t>&)vregInputLowerHalf, vregInputHigherHalf,
                    (MicroAPI::RegTensor<int32_t>&)vregInputOne, (MicroAPI::RegTensor<int32_t>&)vregInputTwo);

                // do int64 sub before gathermask
                MicroAPI::SubCarryOut(carryOut, vregInputLowerHalf, vregInputLowerHalf, vregLowerBoundLowerHalf, pregSelect);
                MicroAPI::SubCarryOuts(carryOut, vregInputHigherHalf, vregInputHigherHalf, vregLowerBoundHigherHalf, carryOut, pregSelect);

                // gather mask
                MicroAPI::GatherMask<int32_t, MicroAPI::GatherMaskMode::NO_STORE_REG>(
                    vregGatherLowerHalf, (MicroAPI::RegTensor<int32_t>&)vregInputLowerHalf, pregSelect);
                MicroAPI::GatherMask<int32_t, MicroAPI::GatherMaskMode::NO_STORE_REG>(
                    vregGatherHigherHalf, vregInputHigherHalf, pregSelect);
                
                // vreg interleave and mask interleave
                MicroAPI::Interleave<int32_t>(vregGatherResult0, vregGatherResult1, vregGatherLowerHalf, vregGatherHigherHalf);
                MicroAPI::MaskInterleave<int32_t>(pregSelectFinal0, pregSelectFinal1, pregGathered, pregGathered);

                // additional gathermask to save real copyout data len
                MicroAPI::GatherMask<int32_t, MicroAPI::GatherMaskMode::STORE_REG>(vregGatherOutput0, vregGatherResult0, pregSelectFinal0);
                MicroAPI::DataCopyUnAlign<int64_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>((__ubuf__ int64_t*)(yIndicesCurDimAddr), (MicroAPI::RegTensor<int64_t>&)vregGatherResult0, ureg0);

                MicroAPI::GatherMask<int32_t, MicroAPI::GatherMaskMode::STORE_REG>(vregGatherOutput1, vregGatherResult1, pregSelectFinal1);
                MicroAPI::DataCopyUnAlign<int64_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>((__ubuf__ int64_t*)(yIndicesCurDimAddr), (MicroAPI::RegTensor<int64_t>&)vregGatherResult1, ureg0);
            }
            AscendC::MicroAPI::DataCopyUnAlignPost(yIndicesCurDimAddr, ureg0);
        }
    }
}

template <typename T>
template <typename T1>
__aicore__ inline void SparseSliceDimension<T>::GatherValues(
    __ubuf__ int8_t *maskUbAddr, __ubuf__ T1 *valuesUbAddr, __ubuf__ T1 *yValuesUbAddr, int32_t count)
{
    uint32_t repeatElm;
    uint32_t sreg0 = count;
    MicroAPI::RegTensor<T1> vregInput;
    MicroAPI::RegTensor<int8_t> vregMask;

    MicroAPI::RegTensor<int32_t> vregInputLowerHalf;
    MicroAPI::RegTensor<int32_t> vregInputHigherHalf;
    MicroAPI::RegTensor<int32_t> vregGatherLowerHalf;
    MicroAPI::RegTensor<int32_t> vregGatherHigherHalf;
    MicroAPI::RegTensor<int32_t> vregGatherTemp;
    MicroAPI::RegTensor<T1> vregGathered;

    MicroAPI::MaskReg pregInput;
    MicroAPI::MaskReg pregCompare;
    MicroAPI::MaskReg pregSelectB8;
    MicroAPI::MaskReg pregSelected;
    MicroAPI::MaskReg pregSelectFinal1;
    MicroAPI::MaskReg pregSelectFinal2;

    if constexpr(sizeof(T1) == sizeof(int8_t)) {
        repeatElm = repeatElmB8_;
    } else if constexpr(sizeof(T1) == sizeof(int16_t)) {
        repeatElm = repeatElmB16_;
    } else if constexpr(sizeof(T1) == sizeof(int32_t)) {
        repeatElm = repeatElmB32_;
    } else if constexpr(sizeof(T1) == sizeof(int64_t)) {
        repeatElm = repeatElmB64_;
    }
    uint16_t loopNum = ops::CeilDiv<uint32_t>(sreg0, repeatElm);

    __VEC_SCOPE__ {
        MicroAPI::UnalignReg ureg0;
        AscendC::MicroAPI::ClearSpr<SpecialPurposeReg::AR>();
        for (uint16_t loopIdx = 0; loopIdx < loopNum; loopIdx++) {
            pregInput = MicroAPI::UpdateMask<T1>(sreg0);
            MicroAPI::DataCopy(vregInput, valuesUbAddr + loopIdx * repeatElm);
            MicroAPI::DataCopy(vregMask, maskUbAddr + loopIdx * repeatElm);
            if constexpr(sizeof(T1) == sizeof(int8_t)) {
                pregCompare = pregInput;
            } else if constexpr(sizeof(T1) == sizeof(int16_t)) {
                MicroAPI::MaskPack<MicroAPI::HighLowPart::LOWEST>(pregCompare, pregInput);
            } else if constexpr(sizeof(T1) == sizeof(int32_t)) {
                MicroAPI::MaskPack<MicroAPI::HighLowPart::LOWEST>(pregCompare, pregInput);
                MicroAPI::MaskPack<MicroAPI::HighLowPart::LOWEST>(pregCompare, pregCompare);
            } else if constexpr(sizeof(T1) == sizeof(int64_t)) {
                MicroAPI::MaskPack<MicroAPI::HighLowPart::LOWEST>(pregCompare, pregInput);
                MicroAPI::MaskPack<MicroAPI::HighLowPart::LOWEST>(pregCompare, pregCompare);
                MicroAPI::MaskPack<MicroAPI::HighLowPart::LOWEST>(pregCompare, pregCompare);
            }
            MicroAPI::CompareScalar<int8_t, CMPMODE::EQ>(pregSelectB8, vregMask, (int8_t)1, pregCompare);
            if constexpr(sizeof(T1) == sizeof(int64_t)) {
                MicroAPI::MaskUnPack<MicroAPI::HighLowPart::LOWEST>(pregSelected, pregSelectB8);
                MicroAPI::MaskUnPack<MicroAPI::HighLowPart::LOWEST>(pregSelected, pregSelected);
                MicroAPI::DeInterleave<int32_t>(vregInputLowerHalf, vregInputHigherHalf,
                    (MicroAPI::RegTensor<int32_t>&)vregInput,
                    (MicroAPI::RegTensor<int32_t>&)vregInput);
                MicroAPI::GatherMask<int32_t, MicroAPI::GatherMaskMode::NO_STORE_REG>(
                    vregGatherLowerHalf, vregInputLowerHalf, pregSelected);
                MicroAPI::GatherMask<int32_t, MicroAPI::GatherMaskMode::NO_STORE_REG>(
                    vregGatherHigherHalf, vregInputHigherHalf, pregSelected);
                MicroAPI::Interleave<int32_t>((MicroAPI::RegTensor<int32_t>&)vregGathered, vregGatherTemp, vregGatherLowerHalf, vregGatherHigherHalf);
            } else if constexpr(sizeof(T1) == sizeof(int8_t)) {
                pregSelected = pregSelectB8;
            } else if constexpr(sizeof(T1) == sizeof(int16_t)) {
                MicroAPI::MaskUnPack<MicroAPI::HighLowPart::LOWEST>(pregSelected, pregSelectB8);
            } else if constexpr(sizeof(T1) == sizeof(int32_t)) {
                MicroAPI::MaskUnPack<MicroAPI::HighLowPart::LOWEST>(pregSelected, pregSelectB8);
                MicroAPI::MaskUnPack<MicroAPI::HighLowPart::LOWEST>(pregSelected, pregSelected);
            }
            if constexpr(sizeof(T1) == sizeof(int64_t)) {
                MicroAPI::MaskInterleave<int32_t>(pregSelectFinal1, pregSelectFinal2, pregSelected, pregSelected);
                MicroAPI::GatherMask<int32_t, MicroAPI::GatherMaskMode::STORE_REG>(vregGatherTemp, vregInputHigherHalf, pregSelectFinal1);
                MicroAPI::DataCopyUnAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    yValuesUbAddr, (MicroAPI::RegTensor<T1>&)vregGathered, ureg0);
            } else {
                MicroAPI::GatherMask<T1, MicroAPI::GatherMaskMode::STORE_REG>(vregGathered, vregInput, pregSelected);
                MicroAPI::DataCopyUnAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(yValuesUbAddr, vregGathered, ureg0);
            }
        }
        MicroAPI::DataCopyUnAlignPost(yValuesUbAddr, ureg0);
    }
    specialArNum_ = AscendC::MicroAPI::GetSpr<SpecialPurposeReg::AR>() / sizeof(T1);
}

template <typename T>
__aicore__ inline void SparseSliceDimension<T>::CopyOutIndicesAndValues()
{
    LocalTensor<int64_t> yIndicesUb = yIndicesQueue_.DeQue<int64_t>();
    LocalTensor<T> yValuesUb = yValuesQueue_.DeQue<T>();

    int64_t curLoopOutNum = specialArNum_;
    if (curLoopOutNum > 0) {
        DataCopyExtParams copyParamsIndices{ static_cast<uint16_t>(1), static_cast<uint32_t>(rankNumbers_ * curLoopOutNum * static_cast<int32_t>(sizeof(int64_t))),
            static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0) };
        DataCopyExtParams copyParamsValues{ static_cast<uint16_t>(1), static_cast<uint32_t>(curLoopOutNum * sizeof(T)),
            static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0) };
        DataCopyPad<int64_t>(yIndicesGm_[curCoreOffset_ * rankNumbers_], yIndicesUb, copyParamsIndices);
        DataCopyPad<T>(yValuesGm_[curCoreOffset_], yValuesUb, copyParamsValues);
        curCoreOffset_ += curLoopOutNum;
    }
    yIndicesQueue_.FreeTensor(yIndicesUb);
    yValuesQueue_.FreeTensor(yValuesUb);
}


template <typename T>
__aicore__ inline void SparseSliceDimension<T>::Transpose2D(
    LocalTensor<int64_t> &yIndicesTransposed, int64_t rowNum, int64_t colNum)
{
    uint32_t sreg0 = rowNum * colNum;
    uint32_t vl = repeatElmB64_;
    uint16_t loopNum = ops::CeilDiv<uint32_t>(sreg0, vl);
    uint32_t vlSize = vl;
    __ubuf__ int64_t* yIndicesTransposedAddr = (__ubuf__ int64_t*)yIndicesTransposed.GetPhyAddr();
    LocalTensor<int64_t> yIndicesUb = yIndicesQueue_.AllocTensor<int64_t>();
    __ubuf__ int64_t* yIndicesAddr = (__ubuf__ int64_t*)yIndicesUb.GetPhyAddr();

    __VEC_SCOPE__ {
        MicroAPI::RegTensor<int64_t> vregInput;
        MicroAPI::RegTensor<int32_t> vregSrcIndex;
        MicroAPI::RegTensor<int32_t> vregDstIndex;
        MicroAPI::RegTensor<int64_t> vregDstIndexB64;
        MicroAPI::RegTensor<int32_t> vreg0;
        MicroAPI::RegTensor<int32_t> vreg1;
        MicroAPI::RegTensor<int32_t> vreg2;
        MicroAPI::RegTensor<int32_t> vreg3;
        MicroAPI::RegTensor<int32_t> vregCol;
        MicroAPI::RegTensor<int32_t> vregRow;
        MicroAPI::MaskReg preg;
        MicroAPI::MaskReg pregB32;
        MicroAPI::MaskReg pregB32ALL = MicroAPI::CreateMask<int32_t, MicroAPI::MaskPattern::ALL>();

        MicroAPI::Duplicate(vregCol, colNum, pregB32ALL);
        MicroAPI::Duplicate(vregRow, rowNum, pregB32ALL);

        for (uint16_t loopIndex = 0; loopIndex < loopNum; loopIndex++) {
            preg = MicroAPI::UpdateMask<int64_t>(sreg0);
            MicroAPI::MaskPack<MicroAPI::HighLowPart::LOWEST>(pregB32, preg);
            MicroAPI::DataCopy(vregInput, yIndicesTransposedAddr + loopIndex * vlSize);
            MicroAPI::Arange<int32_t>(vregSrcIndex, (int32_t)(loopIndex * vlSize));
            MicroAPI::Div(vreg0, vregSrcIndex, vregCol, pregB32);
            MicroAPI::Mul(vreg1, vreg0, vregCol, pregB32);
            MicroAPI::Sub(vreg2, vregSrcIndex, vreg1, pregB32);
            MicroAPI::Mul(vreg3, vreg2, vregRow, pregB32);
            MicroAPI::Add(vregDstIndex, vreg3, vreg0, pregB32);
            MicroAPI::UnPack<int64_t, int32_t, MicroAPI::HighLowPart::LOWEST>(vregDstIndexB64, vregDstIndex);
            MicroAPI::DataCopyScatter(yIndicesAddr, vregInput, (MicroAPI::RegTensor<uint64_t>&)vregDstIndexB64, preg);
        }
    }
    yIndicesQueue_.EnQue(yIndicesUb);
}


template <typename T>
__aicore__ inline void SparseSliceDimension<T>::Transpose2D2Dim(
    LocalTensor<int64_t> &yIndicesTransposed, int64_t colNum)
{
    uint32_t sreg0 = colNum * DIGIT_TWO;
    // vl is regtrait two
    uint32_t vl = repeatElmB64_ * DIGIT_TWO;
    uint16_t loopNum = ops::CeilDiv<uint32_t>(sreg0, vl);
    uint32_t vlSize = vl;
    __ubuf__ int64_t* yIndicesTransposedAddr = (__ubuf__ int64_t*)yIndicesTransposed.GetPhyAddr();
    LocalTensor<int64_t> yIndicesUb = yIndicesQueue_.AllocTensor<int64_t>();
    __ubuf__ int64_t* yIndicesAddr = (__ubuf__ int64_t*)yIndicesUb.GetPhyAddr();

    MicroAPI::RegTensor<int64_t, MicroAPI::RegTraitNumTwo> vregInputDim0;
    MicroAPI::RegTensor<int64_t, MicroAPI::RegTraitNumTwo> vregInputDim1;
    MicroAPI::RegTensor<int64_t, MicroAPI::RegTraitNumTwo> vregOutputPart0;
    MicroAPI::RegTensor<int64_t, MicroAPI::RegTraitNumTwo> vregOutputPart1;
    MicroAPI::MaskReg preg;

    __VEC_SCOPE__ {
        for (uint16_t loopIndex = 0; loopIndex < loopNum; loopIndex++) {
            preg = MicroAPI::UpdateMask<int64_t, MicroAPI::RegTraitNumTwo>(sreg0);
            MicroAPI::DataCopy(vregInputDim0, yIndicesTransposedAddr + loopIndex * vlSize);
            MicroAPI::DataCopy(vregInputDim1, yIndicesTransposedAddr + loopIndex * vlSize + colNum);
            MicroAPI::Interleave(vregOutputPart0, vregOutputPart1, vregInputDim0, vregInputDim1);
            MicroAPI::DataCopy<int64_t, MicroAPI::StoreDist::DIST_NORM>(yIndicesAddr + loopIndex * vlSize * DIGIT_TWO, vregOutputPart0, preg);
            MicroAPI::DataCopy<int64_t, MicroAPI::StoreDist::DIST_NORM>(yIndicesAddr + loopIndex * vlSize * DIGIT_TWO + vlSize, vregOutputPart1, preg);
        }
    }
    yIndicesQueue_.EnQue(yIndicesUb);
}

} // namespace SparseSlice

#endif // SPARSE_SLICE_DIMENSION_BASE_H