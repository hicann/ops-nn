/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file scatter_nd_update_common.h
 * \brief scatter_nd_update
 */
#ifndef ASCENDC_SCATTER_ND_UPDATE_COMMON_H_
#define ASCENDC_SCATTER_ND_UPDATE_COMMON_H_

#include "kernel_operator.h"
#include "op_kernel/platform_util.h"
#include "../inc/load_store_utils.h"
#include "scatter_nd_update_struct.h"
#include "indices_sort_utils.h"
#include "op_kernel/math_util.h"
namespace ScatterNdUpdate {
using namespace AscendC;

constexpr int64_t DOUBLE_BUFFER = 2;
constexpr uint64_t UB_AGLIN_VALUE = 32;
constexpr uint64_t SORT_PAD_NUM = 2;
constexpr uint16_t MAX_RANK_COUNT = 7;
constexpr uint16_t MIN_SAME_IDX_ACCM_COUNT = 256;
constexpr uint32_t THREAD_NUM = 1024;
constexpr uint32_t THREAD_NUM_LAUNCH_BOUND = 1024;
constexpr uint16_t MAX_SHAPE_RANK = 8;
constexpr uint16_t INDICE_RANK_TWO = 2;
constexpr float SORT_HIST_THRESHOLD = 0.01f;
constexpr uint32_t HASH_SCORE_BUF_SIZE = 128;
constexpr uint32_t MASK_DEFAULT = 0;

static constexpr SortConfig sortConfig{SortType::RADIX_SORT, false};

template <typename T, typename U>
class ScatterNdUpdateBase {
public:
    int64_t indicesFactor_ = 0;
    int64_t afterAxisFactor_ = 0;
    int64_t afterAxis_ = 0;
    int64_t indexRankSize_ = 0;
    int64_t eachCoreAfterAxisCount_ = 0;
    int64_t shiftOffset_ = UB_AGLIN_VALUE / sizeof(U);
    uint32_t uniqueIdNum_ = 0;
    float maxScore_ = 0;

    AscendC::GlobalTensor<U> indicesGm_;
    AscendC::GlobalTensor<T> updatesGm_;
    AscendC::GlobalTensor<T> yGm_;

    TBuf<QuePosition::VECCALC> indicesBuf_;
    TBuf<QuePosition::VECCALC> outOfstBuf_;
    TBuf<QuePosition::VECCALC> strideBuf_;
    TBuf<QuePosition::VECCALC> maxScoreBuf_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, DOUBLE_BUFFER> dataQueue_;

    TBuf<QuePosition::VECCALC> sortIndicesQue_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> updatesOriginIdexQue_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> uniqueIdCountQue_;

    using IndexRegType = typename std::conditional<IsSameType<U, int64_t>::value,
                         typename AscendC::MicroAPI::RegTensor<uint64_t, AscendC::MicroAPI::RegTraitNumTwo>,
                         typename AscendC::MicroAPI::RegTensor<uint32_t>>::type;  
    using InnerRegType = typename std::conditional<IsSameType<U, int64_t>::value,
                         typename AscendC::MicroAPI::RegTensor<int64_t, AscendC::MicroAPI::RegTraitNumTwo>,
                         typename AscendC::MicroAPI::RegTensor<int32_t>>::type;

    using selRegType = typename std::conditional<IsSameType<T, bool>::value, int8_t, T>::type;

    __aicore__ inline void InitBaseBuffer(
        TPipe &pipe, uint32_t indicesNumber, GM_ADDR indices, GM_ADDR updates, GM_ADDR y)
    {
        indicesGm_.SetGlobalBuffer((__gm__ U *)(indices));
        updatesGm_.SetGlobalBuffer((__gm__ T *)(updates));
        yGm_.SetGlobalBuffer((__gm__ T *)(y));

        pipe.InitBuffer(strideBuf_, MAX_RANK_COUNT * sizeof(U));
        pipe.InitBuffer(dataQueue_, DOUBLE_BUFFER, indicesFactor_ * afterAxisFactor_ * sizeof(T));
        pipe.InitBuffer(outOfstBuf_, indicesFactor_ * sizeof(U));
        pipe.InitBuffer(indicesBuf_, indicesFactor_ * indexRankSize_ * sizeof(U));
        pipe.InitBuffer(maxScoreBuf_, HASH_SCORE_BUF_SIZE * sizeof(float));

        pipe.InitBuffer(
            sortIndicesQue_, Ops::Base::CeilAlign(indicesFactor_ * sizeof(U) + SORT_PAD_NUM * UB_AGLIN_VALUE, UB_AGLIN_VALUE));
        pipe.InitBuffer(updatesOriginIdexQue_, DOUBLE_BUFFER, indicesFactor_ * sizeof(uint32_t));
        pipe.InitBuffer(
            uniqueIdCountQue_, DOUBLE_BUFFER, Ops::Base::CeilAlign((indicesFactor_ + 1) * sizeof(int32_t), UB_AGLIN_VALUE));
    }

    template <typename PARAM_T>
    __aicore__ inline void CopyIn(const LocalTensor<PARAM_T>& dstTensor, const GlobalTensor<PARAM_T>& srcTensor, int64_t dataLen)
    {
        DataCopyExtParams copyParams = {static_cast<uint16_t>(1), 
                                        static_cast<uint32_t>(dataLen * sizeof(PARAM_T)),
                                        static_cast<uint32_t>(0),
                                        static_cast<uint32_t>(0),
                                        static_cast<uint32_t>(0)};
        DataCopyPadExtParams<PARAM_T> padParams = {
            false, static_cast<uint8_t>(0), static_cast<uint8_t>(0), static_cast<PARAM_T>(0)};
        DataCopyPad(dstTensor, srcTensor, copyParams, padParams);
    }

    template <typename PARAM_T>
    __aicore__ inline void CopyOut(const GlobalTensor<PARAM_T>& dstTensor, const LocalTensor<PARAM_T>& srcTensor, int64_t dataLen)
    {
        DataCopyExtParams copyParams = {static_cast<uint16_t>(1),
                                        static_cast<uint32_t>(dataLen * sizeof(PARAM_T)),
                                        static_cast<uint32_t>(0),
                                        static_cast<uint32_t>(0),
                                        static_cast<uint32_t>(0)};
        DataCopyPad(dstTensor, srcTensor, copyParams);
    }

    __aicore__ inline void ComputeOutOfset(
        const LocalTensor<U> indicesLocal, const LocalTensor<U> outOfstLocal, int32_t indicesLen, int32_t rankSize)
    {
        LocalTensor<U> strideLocal = strideBuf_.Get<U>();
        __local_mem__ U* indicesLocalPtr = ((__local_mem__ U*)indicesLocal.GetPhyAddr());
        __local_mem__ U* outOfstLocalPtr = ((__local_mem__ U*)outOfstLocal.GetPhyAddr());

        uint32_t dataLen = indicesLen;
        uint32_t vfLen = Ops::Base::GetVRegSize() / sizeof(int32_t);
        uint32_t indicesLenTimes = Ops::Base::CeilDiv(dataLen, vfLen);
        uint16_t loopCnt = static_cast<uint16_t>(indicesLenTimes);
        uint16_t rankSizeLoops = static_cast<uint16_t>(rankSize);

        __VEC_SCOPE__
        {
            InnerRegType inReg;
            InnerRegType outReg;
            InnerRegType orderReg;
            IndexRegType indexReg;
            AscendC::MicroAPI::MaskReg pregLoop;

            for (uint16_t i = 0; i < loopCnt; i++) {
                if constexpr (IsSameType<U, int64_t>::value) {
                    pregLoop = AscendC::MicroAPI::UpdateMask<U, AscendC::MicroAPI::RegTraitNumTwo>(dataLen);
                } else {
                    pregLoop = AscendC::MicroAPI::UpdateMask<U>(dataLen);
                }
                AscendC::MicroAPI::Duplicate(outReg, 0, pregLoop);
                AscendC::MicroAPI::Arange(orderReg, i * vfLen);
                AscendC::MicroAPI::Muls(orderReg, orderReg, rankSize, pregLoop);
                for (uint16_t dim = 0; dim < rankSizeLoops; dim++) {
                    U strideValue = strideLocal(dim);
                    indexReg = (IndexRegType&)orderReg;
                    AscendC::MicroAPI::DataCopyGather(inReg, indicesLocalPtr, indexReg, pregLoop);
                    AscendC::MicroAPI::Muls(inReg, inReg, strideValue, pregLoop);
                    AscendC::MicroAPI::Add(outReg, inReg, outReg, pregLoop);
                    AscendC::MicroAPI::Adds(orderReg, orderReg, (U)(1), pregLoop);
                }
                auto outOfstAddr = outOfstLocalPtr + i * vfLen;
                AscendC::MicroAPI::DataCopy(outOfstAddr, outReg, pregLoop);
            }
        }
        return;
    }

    __aicore__ uint32_t ComputeUniqueIdNum(
        LocalTensor<U> indicesLocal, LocalTensor<int32_t> uniqueIdCountLocal, int64_t dataLen)
    {
        __local_mem__ U* indicesAddr = (__local_mem__ U*)indicesLocal[shiftOffset_].GetPhyAddr();
        __local_mem__ int32_t* uniqueIdCountsAddr = (__local_mem__ int32_t*)uniqueIdCountLocal.GetPhyAddr();

        int64_t vfLen = Ops::Base::GetVRegSize() / sizeof(U);
        uint16_t loopCnt = Ops::Base::CeilDiv(dataLen + 1, vfLen);
        uint32_t counter = dataLen + 1;
        __VEC_SCOPE__
        {
            AscendC::MicroAPI::RegTensor<int32_t> orderReg;
            AscendC::MicroAPI::RegTensor<U> sortedIdxReg;
            AscendC::MicroAPI::RegTensor<U> sortedIdxShiftOneReg;
            AscendC::MicroAPI::RegTensor<int32_t> selReg;
            AscendC::MicroAPI::MaskReg cmpMask;
            AscendC::MicroAPI::MaskReg maskReg;
            AscendC::MicroAPI::UnalignReg u0;
            AscendC::MicroAPI::UnalignReg uOut;
            AscendC::MicroAPI::ClearSpr<AscendC::SpecialPurposeReg::AR>();

            for (uint16_t i = 0; i < loopCnt; ++i) {
                AscendC::MicroAPI::Arange(orderReg, i * vfLen);
                maskReg = AscendC::MicroAPI::UpdateMask<U>(counter);
                auto startAddr = indicesAddr + i * vfLen;
                DataCopy(sortedIdxReg, startAddr);
                AscendC::MicroAPI::DataCopyUnAlignPre(u0, startAddr - 1);
                AscendC::MicroAPI::DataCopyUnAlign<U>(sortedIdxShiftOneReg, u0, startAddr - 1);
                AscendC::MicroAPI::Compare<U, CMPMODE::NE>(cmpMask, sortedIdxReg, sortedIdxShiftOneReg, maskReg);
                if constexpr (std::is_same<int64_t, U>::value) {
                    AscendC::MicroAPI::MaskReg maskHalf;
                    AscendC::MicroAPI::MaskPack<AscendC::MicroAPI::HighLowPart::LOWEST>(maskHalf, cmpMask);
                    AscendC::MicroAPI::GatherMask<int32_t, AscendC::MicroAPI::GatherMaskMode::STORE_REG>(
                        selReg, orderReg, maskHalf);
                } else {
                    AscendC::MicroAPI::GatherMask<int32_t, AscendC::MicroAPI::GatherMaskMode::STORE_REG>(
                        selReg, orderReg, cmpMask);
                }
                AscendC::MicroAPI::DataCopyUnAlign<int32_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    uniqueIdCountsAddr, selReg, uOut);
            }
            AscendC::MicroAPI::DataCopyUnAlignPost(uniqueIdCountsAddr, uOut);
        }
        uint32_t uniqueIdNum = ((AscendC::MicroAPI::GetSpr<AscendC::SpecialPurposeReg::AR>()) / sizeof(int32_t)) - 1;
        return uniqueIdNum;
    }

    __aicore__ void SortAndComputeUniqueIdx(LocalTensor<U> outOfstLocal, int64_t rowLen)
    {
        LocalTensor<U> sortIndicesLocal = sortIndicesQue_.Get<U>();
        LocalTensor<int32_t> uniqueIdCountLocal = uniqueIdCountQue_.AllocTensor<int32_t>();
        LocalTensor<uint32_t> updatesOriginIdexLocal = updatesOriginIdexQue_.AllocTensor<uint32_t>();
        LocalTensor<U> shiftSortLocal = sortIndicesLocal[shiftOffset_];
        AscendC::Sort<U, true, sortConfig>(
            shiftSortLocal, updatesOriginIdexLocal, outOfstLocal, static_cast<uint32_t>(rowLen));
        Duplicate(sortIndicesLocal, (U)-1, shiftOffset_);
        shiftSortLocal(rowLen) = -1;
        
        event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventIdSToV);
        WaitFlag<HardEvent::S_V>(eventIdSToV);
        uniqueIdNum_ = ComputeUniqueIdNum(sortIndicesLocal, uniqueIdCountLocal, rowLen);

        uniqueIdCountQue_.EnQue(uniqueIdCountLocal);
        updatesOriginIdexQue_.EnQue(updatesOriginIdexLocal);
    }

    __aicore__ inline void CopyIndiceInSplitAfter(int64_t rowIdx, int64_t rowLen)
    {
        LocalTensor<U> indicesLocal = indicesBuf_.Get<U>();
        LocalTensor<U> outOfstLocal = outOfstBuf_.Get<U>();
        LocalTensor<float> dstLocal = maxScoreBuf_.Get<float>();
        
        int64_t rankSize = indexRankSize_;
        int64_t indicesOfset = rowIdx * indicesFactor_;
        CopyIn<U>(indicesLocal, indicesGm_[indicesOfset * rankSize], rowLen * rankSize);
        event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        ComputeOutOfset(indicesLocal, outOfstLocal, rowLen, rankSize);
        if constexpr (IsSameType<U, int32_t>::value) {
            IndexStatisticInt32(outOfstLocal, dstLocal, maxScore_, rowLen, afterAxis_);
        } else {
            IndexStatisticInt64(outOfstLocal, dstLocal, maxScore_, rowLen, afterAxis_);
        }
    }

    __aicore__ inline void CopyUpdatesInSplitAfter(
        int64_t rowIdx, int64_t colIdx, int64_t rowLen, int64_t colLen)
    {
        LocalTensor<T> updatesLocal = this->dataQueue_.template AllocTensor<T>();
        int64_t indicesOfset = rowIdx * indicesFactor_;
        DataCopyExtParams copyParams = {static_cast<uint16_t>(rowLen),
                                        static_cast<uint32_t>(colLen * sizeof(T)),
                                        static_cast<uint32_t>((afterAxis_ - colLen) * sizeof(T)),
                                        static_cast<uint32_t>(0),
                                        static_cast<uint32_t>(0)};
        DataCopyPadExtParams<T> updatePadParams = {false, 0, 0, 0};
        int64_t rowOfset = indicesOfset * afterAxis_;
        int64_t updatesOfset = rowOfset + GetBlockIdx() * eachCoreAfterAxisCount_ + colIdx * afterAxisFactor_;
        DataCopyPad(updatesLocal, updatesGm_[updatesOfset], copyParams, updatePadParams);
        this->dataQueue_.template EnQue(updatesLocal);
    }

    __aicore__ inline void CopyOutSplitAfter(
        int64_t rowLen, int64_t colLen, int64_t colIdx)
    {
        LocalTensor<T> dataLocal = dataQueue_.DeQue<T>();
        LocalTensor<U> outOfstLocal = outOfstBuf_.Get<U>();
        int64_t colLenAlignSize = Ops::Base::CeilAlign(colLen * sizeof(T), UB_AGLIN_VALUE) / sizeof(T);
        
        event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventIdVToS);
        WaitFlag<HardEvent::V_S>(eventIdVToS);
        for (int64_t i = 0; i < rowLen; i++) {
            int64_t rowOfset = outOfstLocal(i) * afterAxis_;
            int64_t outOfset = rowOfset + GetBlockIdx() * eachCoreAfterAxisCount_ + colIdx * afterAxisFactor_;
            CopyOut<T>(yGm_[outOfset], dataLocal[i * colLenAlignSize], colLen);
        }
        dataQueue_.FreeTensor(dataLocal);
    }
        
    __aicore__ inline void CopyOutSplitAfterWithSort(
        int64_t rowLen, int64_t colLen, int64_t colIdx)
    {
        LocalTensor<T> dataLocal = dataQueue_.DeQue<T>();
        LocalTensor<U> sortIndicesLocal = sortIndicesQue_.Get<U>();
        LocalTensor<U> shiftSortLocal = sortIndicesLocal[shiftOffset_];
        LocalTensor<uint32_t> updatesOriginIdexLocal = updatesOriginIdexQue_.DeQue<uint32_t>();
        LocalTensor<int32_t> uniqueIdCountLocal = uniqueIdCountQue_.DeQue<int32_t>();
        int64_t colLenAlignSize = Ops::Base::CeilAlign(colLen * sizeof(T), UB_AGLIN_VALUE) / sizeof(T);

        event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventIdVToS);
        WaitFlag<HardEvent::V_S>(eventIdVToS);
        for (int64_t i = 0; i < rowLen; i++) {
            int32_t uniqueIdx = uniqueIdCountLocal(i);
            int64_t inOfset = updatesOriginIdexLocal(uniqueIdx) * colLenAlignSize;
            int64_t outOfset = shiftSortLocal(uniqueIdx) * afterAxis_;
            outOfset += GetBlockIdx() * eachCoreAfterAxisCount_ + colIdx * afterAxisFactor_;
            CopyOut<T>(yGm_[outOfset], dataLocal[inOfset], colLen);
        }

        uniqueIdCountQue_.EnQue(uniqueIdCountLocal);
        updatesOriginIdexQue_.EnQue(updatesOriginIdexLocal);
        dataQueue_.FreeTensor(dataLocal);
    }
};


template <typename PARAMS_T, typename INDICES_T, typename TYPE_T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void SimtCalcMask(
    __gm__ INDICES_T* idxGmAddr, __gm__ TYPE_T* maskGmAddr, __gm__ PARAMS_T* outputGmAddr, 
    const __ubuf__ TYPE_T* strideListAddr, const __ubuf__ TYPE_T* outputShapeAddr, TYPE_T indiceBlockOffSet, 
    uint32_t rankSize, uint32_t currBlockHandleIdx, int64_t varInAxis, __gm__ TYPE_T* varIdxGmAddr)
{
    for (uint32_t index = Simt::GetThreadIdx(); index < currBlockHandleIdx; index += Simt::GetThreadNum()) {
        TYPE_T globalIndiceRowOffset = indiceBlockOffSet + index;
        INDICES_T idx = 0;
        bool outOfBound = false;
        for (TYPE_T dim = 0; dim < rankSize; ++dim) {
            INDICES_T indiceVal = idxGmAddr[globalIndiceRowOffset * rankSize + dim];
            outOfBound |= static_cast<TYPE_T>(indiceVal) > outputShapeAddr[dim];
            idx += indiceVal * strideListAddr[dim];
        }
        if (!outOfBound) {
            if (idx >= 0 && idx < varInAxis){
                Simt::AtomicMax(maskGmAddr + idx, globalIndiceRowOffset);
                varIdxGmAddr[globalIndiceRowOffset] = idx;
            }
        }
    }
}

template <typename PARAMS_T, typename INDICES_T, typename TYPE_T>
class ScatterNdUpdateDeterministicCommon
{
public:
    __aicore__ inline ScatterNdUpdateDeterministicCommon(const ScatterNdUpdateRegBaseTilingData& tilingData, TPipe& pipe) : pipe_(pipe), tiling_(tilingData){};
    __aicore__ inline void InitBase(GM_ADDR x, GM_ADDR indices, GM_ADDR updates, GM_ADDR y, GM_ADDR workspace);
    __aicore__ inline void CalcMask();
    __aicore__ inline void InitUpdateBuffer();

protected:
    TPipe& pipe_;
    const ScatterNdUpdateRegBaseTilingData& tiling_;
    GlobalTensor<INDICES_T> idxGm;
    GlobalTensor<PARAMS_T> updateGm;
    GlobalTensor<PARAMS_T> outputGm;
    GlobalTensor<TYPE_T> maskGm;
    GlobalTensor<TYPE_T> varIdxGm;
    GlobalTensor<TYPE_T> maskBlockGm;

    TQue<QuePosition::VECIN, DOUBLE_BUFFER> inQueX;
    TBuf<TPosition::VECCALC> strideListBuf;
    TBuf<TPosition::VECCALC> outputShapeBuf;

    LocalTensor<TYPE_T> strideList;
    LocalTensor<TYPE_T> outputShape;

    uint32_t blockIdx;
    TYPE_T currBlockHandleIdx = 0;
    TYPE_T indiceBlockOffSet = 0;    
};

template <typename PARAMS_T, typename INDICES_T, typename TYPE_T>
__aicore__ inline void ScatterNdUpdateDeterministicCommon<PARAMS_T, INDICES_T, TYPE_T>::InitBase(GM_ADDR x, GM_ADDR indices, GM_ADDR updates, GM_ADDR y ,GM_ADDR workspace)
{    
    blockIdx = GetBlockIdx();
    idxGm.SetGlobalBuffer((__gm__ INDICES_T*)indices);
    updateGm.SetGlobalBuffer((__gm__ PARAMS_T*)updates);
    outputGm.SetGlobalBuffer((__gm__ PARAMS_T*)y);
    maskGm.SetGlobalBuffer((__gm__ TYPE_T*)workspace);
    varIdxGm.SetGlobalBuffer((__gm__ TYPE_T*) workspace + (tiling_.varInAxis + 1));

    if (blockIdx >= tiling_.calcMaskUsedCoreNum) {
        return;
    }

    this->indiceBlockOffSet = static_cast<TYPE_T>(blockIdx * tiling_.normCoreHandleIdx);

    if (blockIdx == tiling_.calcMaskUsedCoreNum - 1) {
        this->currBlockHandleIdx = tiling_.tailCoreHandleIdx;
    } else {
        this->currBlockHandleIdx = tiling_.normCoreHandleIdx;
    }

    uint64_t maskBlockOffset = blockIdx * tiling_.maskNormBlockLen;
    uint64_t maskBlockLen = blockIdx == tiling_.calcMaskUsedCoreNum - 1 ? tiling_.maskTailBlockLen : tiling_.maskNormBlockLen;
    maskBlockGm.SetGlobalBuffer((__gm__ TYPE_T*) workspace + maskBlockOffset);

    pipe_.InitBuffer(strideListBuf, MAX_RANK_COUNT * sizeof(TYPE_T));
    pipe_.InitBuffer(outputShapeBuf, MAX_SHAPE_RANK * sizeof(TYPE_T));

    InitGlobalMemory(maskBlockGm, maskBlockLen, static_cast<TYPE_T>(MASK_DEFAULT));
}

template <typename PARAMS_T, typename INDICES_T, typename TYPE_T>
__aicore__ inline void ScatterNdUpdateDeterministicCommon<PARAMS_T, INDICES_T, TYPE_T>::CalcMask()
{
    if (blockIdx >= tiling_.calcMaskUsedCoreNum) {
        return;
    }
    uint32_t currBlockHandleIdx = this->currBlockHandleIdx;
    uint32_t rankSize = tiling_.rankSize;
    TYPE_T indiceBlockOffSet = this->indiceBlockOffSet;
    int64_t varInAxis = tiling_.varInAxis;

    strideList = strideListBuf.Get<TYPE_T>();
    outputShape = outputShapeBuf.Get<TYPE_T>();
    for (uint32_t i = 0; i < MAX_RANK_COUNT; i++) {
        strideList(i) = tiling_.strideList[i];
    }
    for (uint32_t i = 0; i < MAX_SHAPE_RANK; i++) {
        outputShape(i) = tiling_.outPutShape[i];
    }
    DataSyncBarrier<MemDsbT::UB>();
    AscendC::Simt::VF_CALL<SimtCalcMask<PARAMS_T, INDICES_T, TYPE_T>>(
        Simt::Dim3(THREAD_NUM), (__gm__ INDICES_T*)idxGm.GetPhyAddr(), (__gm__ TYPE_T*)maskGm.GetPhyAddr(), 
        (__gm__ PARAMS_T*)(outputGm.GetPhyAddr()), (__ubuf__ TYPE_T*)strideList.GetPhyAddr(),
        (__ubuf__ TYPE_T*)outputShape.GetPhyAddr(), indiceBlockOffSet, rankSize, currBlockHandleIdx, varInAxis, (__gm__ TYPE_T*) varIdxGm.GetPhyAddr());
}

template <typename PARAMS_T, typename INDICES_T, typename TYPE_T>
__aicore__ inline void ScatterNdUpdateDeterministicCommon<PARAMS_T, INDICES_T, TYPE_T>::InitUpdateBuffer() {
    pipe_.Reset();
    pipe_.InitBuffer(inQueX, DOUBLE_BUFFER, Ops::Base::CeilAlign(tiling_.afterAxisFactor * sizeof(PARAMS_T), UB_AGLIN_VALUE));
}

}  // namespace ScatterNdUpdate

#endif