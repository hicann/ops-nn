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
 * \file index_put_with_sort_v2_simd.h
 * \brief
 */

#ifndef INDEX_PUT_WITH_SORT_V2_SIMD_H
#define INDEX_PUT_WITH_SORT_V2_SIMD_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "../inc/platform.h"
#include "../inc/kernel_utils.h"
#include "index_put_with_sort_v2_common.h"
#include "index_put_with_sort_v2_struct.h"
namespace IndexPutWithSortV2 {
using namespace AscendC;

constexpr uint32_t DOUBLE_BUFFER = 2;
constexpr uint32_t TWO = 2;
constexpr uint32_t ONE = 1;
constexpr uint32_t HIGH_TYPE = 4;
constexpr uint64_t UB_BLOCK = 32;
constexpr uint64_t COL_ALIGN = 512;
constexpr uint32_t IDX_BLOCK = 128;

template<typename SELF_TYPE, typename IDX_TYPE, bool ACCUMULATE, bool IS_CAST, typename CAST_TYPE>
class IndexPutWithSortV2SIMDKernel {
public:
    __aicore__ inline IndexPutWithSortV2SIMDKernel(TPipe *pipe, const IndexPutWithSortV2SimdTilingData *tilingData)
        : pipe_(pipe), tilingData_(tilingData)
    {}

    __aicore__ inline void Init(GM_ADDR self, GM_ADDR sortIndices, GM_ADDR posIdx, GM_ADDR values,
        GM_ADDR output, GM_ADDR workspace);
    __aicore__ inline void Process(void);

private:
    __aicore__ inline void ProcessFirstStep(void);
    __aicore__ inline void ProcessSecondStep(void);
    __aicore__ inline void CopyInIdx(int64_t rowIdx, int64_t rowLen, bool isRowLoopEnd);
    __aicore__ inline void CopyInValue(const GlobalTensor<SELF_TYPE>& srcTensor, int64_t dataLen);
    __aicore__ inline void CopyOutSum(bool isRowLoopEnd, int64_t row, int64_t rowLen, int64_t repeatCount, IDX_TYPE curIndex, IDX_TYPE preIndex,
                                        IDX_TYPE nextIndex, int64_t colIdx, int64_t colLen, uint32_t vfLen, uint16_t loopCnt);
    __aicore__ inline void ComputeSumValue(int64_t colLen, uint32_t vfLen, uint16_t loopCnt);
    __aicore__ inline void ComputeAcc(int64_t rowIdx, int64_t colIdx, int64_t rowLen, int64_t colLen, bool isRowLoopEnd,  
                            LocalTensor<IDX_TYPE>& indexTotalLocal, LocalTensor<int32_t>& posidxLocal);
    __aicore__ inline void ComputeReplace(int64_t rowIdx, int64_t colIdx, int64_t rowLen, int64_t colLen,  
                            LocalTensor<IDX_TYPE>& indexTotalLocal, LocalTensor<int32_t>& posidxLocal);
    __aicore__ inline void GetComputeRangeIdx(IDX_TYPE &startIdx, IDX_TYPE &endIdx, IDX_TYPE &indexValue) ;
    __aicore__ inline void CopyInValueFromWs(int64_t offset, int64_t dataLen);
    __aicore__ inline void GetSumValueStep2(int64_t dataLen, LocalTensor<CAST_TYPE>& valueSumLocal);
    __aicore__ inline void CastSumValueStep2(int64_t dataLen);
    
private:
    TPipe *pipe_;
    const IndexPutWithSortV2SimdTilingData *tilingData_;
    AscendC::GlobalTensor<SELF_TYPE> outputGm_;
    AscendC::GlobalTensor<IDX_TYPE> indicesGm_;
    AscendC::GlobalTensor<int32_t> posIdxGm_;
    AscendC::GlobalTensor<SELF_TYPE> valuesGm_;

    AscendC::GlobalTensor<CAST_TYPE> valuesTempSumWsGm_;
    AscendC::GlobalTensor<IDX_TYPE> indexSumWsGm_;
    AscendC::GlobalTensor<IDX_TYPE> indexSumInitWsGm_;

    TQue<QuePosition::VECIN, DOUBLE_BUFFER> indexQue_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> posidxQue_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> valueQue_;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER> valueSumQue_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, DOUBLE_BUFFER> dataQue_;
    TQue<QuePosition::VECOUT, 1> valueCastQue_;

    constexpr static uint32_t V_REG_SIZE = platform::GetVRegSize();
    int64_t valueCols_{0};
    int64_t blockId_{0};
    int64_t blockNum_{0};
    int64_t rowUseCoreNum_{0};
    int64_t colUseCoreNum_{0};
    int64_t indexFactor_{0};
    int64_t rowBlockFactor_{0};
    int64_t lastLoopRepateCount_{0};
    int64_t lastLoopPreIndex_{-1};
    int64_t dupIdxStart_{0};
    int64_t valueTmpCols_{0};
    int64_t indexBlockNum_{0};
    int64_t lastLoopIdxCount_{0};
    int64_t coreLoopOfst_{0};
    int64_t blockRow_{0};
    int64_t blockCol_{0};
    int64_t colFactor_{0};
    int64_t curCoreProcessRowCount_{0};
    bool firstFlag_{false};
    bool isInit_{false};
};

template<typename SELF_TYPE, typename IDX_TYPE, bool ACCUMULATE, bool IS_CAST, typename CAST_TYPE>
__aicore__ inline void IndexPutWithSortV2SIMDKernel<SELF_TYPE, IDX_TYPE, ACCUMULATE, IS_CAST, CAST_TYPE>::Init(GM_ADDR self, GM_ADDR sortIndices, 
            GM_ADDR posIdx, GM_ADDR values, GM_ADDR output, GM_ADDR workspace)
{
    outputGm_.SetGlobalBuffer((__gm__ SELF_TYPE *)output);
    indicesGm_.SetGlobalBuffer((__gm__ IDX_TYPE *)sortIndices);
    posIdxGm_.SetGlobalBuffer((__gm__ int32_t *)posIdx);
    valuesGm_.SetGlobalBuffer((__gm__ SELF_TYPE *)values);
    blockId_ = GetBlockIdx();
    blockNum_ = GetBlockNum();
    rowUseCoreNum_ = tilingData_->rowUseCoreNum;
    colUseCoreNum_ = tilingData_->colUseCoreNum;
    rowBlockFactor_ = (blockId_ == blockNum_ - 1) ? tilingData_->rowTailBlockFactor : tilingData_->rowBlockFactor;
    coreLoopOfst_ = UB_BLOCK / sizeof(IDX_TYPE);
    blockRow_ = blockId_ / colUseCoreNum_;
    blockCol_ = blockId_ % colUseCoreNum_; 
    valueTmpCols_ = ops::CeilAlign(tilingData_->nonIndexedDimSize * sizeof(CAST_TYPE), COL_ALIGN) / sizeof(CAST_TYPE);
    indexBlockNum_ = IDX_BLOCK / sizeof(IDX_TYPE);

    valuesTempSumWsGm_.SetGlobalBuffer((__gm__ CAST_TYPE*)workspace, valueTmpCols_ * rowUseCoreNum_ * TWO);
    auto indexSumStartAddr = (__gm__ CAST_TYPE*)workspace + valueTmpCols_ * rowUseCoreNum_ * TWO;
    indexSumWsGm_.SetGlobalBuffer((__gm__ IDX_TYPE*)indexSumStartAddr, rowUseCoreNum_ * indexBlockNum_ * TWO);
    if (blockId_ < rowUseCoreNum_) {
        indexSumInitWsGm_.SetGlobalBuffer((__gm__ IDX_TYPE*)indexSumStartAddr + blockId_ * indexBlockNum_ * TWO, indexBlockNum_ * TWO);
        InitGlobalMemory(indexSumInitWsGm_, indexBlockNum_ * TWO, IDX_TYPE(-1));
    }
    AscendC::SyncAll();
}

template<typename SELF_TYPE, typename IDX_TYPE, bool ACCUMULATE, bool IS_CAST, typename CAST_TYPE>
__aicore__ inline void IndexPutWithSortV2SIMDKernel<SELF_TYPE, IDX_TYPE, ACCUMULATE, IS_CAST, CAST_TYPE>::CopyInValue(const GlobalTensor<SELF_TYPE>& srcTensor,
     int64_t dataLen)
{
    LocalTensor<SELF_TYPE> valueLocal;
    if (ACCUMULATE) {
        valueLocal = valueQue_.AllocTensor<SELF_TYPE>();
    } else {
        valueLocal = dataQue_.AllocTensor<SELF_TYPE>();
    }
    CopyIn<SELF_TYPE>(valueLocal, srcTensor, dataLen);
    if (ACCUMULATE) {
        valueQue_.EnQue(valueLocal);
    } else {
        dataQue_.EnQue(valueLocal);
    }
}

template<typename SELF_TYPE, typename IDX_TYPE, bool ACCUMULATE, bool IS_CAST, typename CAST_TYPE>
__aicore__ inline void IndexPutWithSortV2SIMDKernel<SELF_TYPE, IDX_TYPE, ACCUMULATE, IS_CAST, CAST_TYPE>::CopyInIdx(int64_t rowIdx, int64_t rowLen, 
        bool isRowLoopEnd)
{
    LocalTensor<IDX_TYPE> indexLocal = indexQue_.AllocTensor<IDX_TYPE>();
    LocalTensor<int32_t> posidxLocal = posidxQue_.AllocTensor<int32_t>(); 
  
    int64_t posidxGmOfst = blockRow_ * tilingData_->rowBlockFactor + rowIdx * indexFactor_;
    int64_t indexGmOfst = posidxGmOfst;
    int64_t indexLen = rowLen + TWO;
    if ((blockRow_ == 0) && (rowIdx == 0)) {
        indexLen -= 1;
        CopyIn<IDX_TYPE>(indexLocal[coreLoopOfst_], indicesGm_[indexGmOfst], indexLen);
        EventMsg<HardEvent::MTE2_S>();
        indexLocal.SetValue(coreLoopOfst_ - 1, (IDX_TYPE)-2);
    } else if ((blockRow_ == rowUseCoreNum_ - 1) && isRowLoopEnd) {
        indexLen -= 1;
        indexGmOfst -= 1;
        CopyIn<IDX_TYPE>(indexLocal, indicesGm_[indexGmOfst], indexLen);
        EventMsg<HardEvent::MTE2_S>();
        indexLocal.SetValue(indexLen, (IDX_TYPE)-2);
    } else {
        indexGmOfst -= 1;
        CopyIn<IDX_TYPE>(indexLocal, indicesGm_[indexGmOfst], indexLen);
    }
    CopyIn<int32_t>(posidxLocal, posIdxGm_[posidxGmOfst], rowLen);
    indexQue_.EnQue(indexLocal);
    posidxQue_.EnQue(posidxLocal);
}

template<typename SELF_TYPE, typename IDX_TYPE, bool ACCUMULATE, bool IS_CAST, typename CAST_TYPE>
__aicore__ inline void IndexPutWithSortV2SIMDKernel<SELF_TYPE, IDX_TYPE, ACCUMULATE, IS_CAST, CAST_TYPE>::ComputeSumValue(int64_t colLen, uint32_t vfLen, uint16_t loopCnt) 
{
    LocalTensor<SELF_TYPE> valueLocal = valueQue_.DeQue<SELF_TYPE>();
    LocalTensor<SELF_TYPE> valueSumLocal = isInit_ ? valueSumQue_.AllocTensor<SELF_TYPE>() : valueSumQue_.DeQue<SELF_TYPE>();
    LocalTensor<CAST_TYPE> castLocal;
    if constexpr (IS_CAST) {
        castLocal = isInit_ ? valueCastQue_.AllocTensor<CAST_TYPE>() : valueCastQue_.DeQue<CAST_TYPE>();
    }
    if (isInit_) {
        firstFlag_ = false;
        isInit_ = false;
        if constexpr (IS_CAST) {
            Duplicate(castLocal, (CAST_TYPE)0, colLen);
        } else {
            int64_t count = (colLen * sizeof(SELF_TYPE) + sizeof(uint16_t) - 1) / sizeof(uint16_t);
            LocalTensor<uint16_t> valueSumTempLocal = valueSumLocal.template ReinterpretCast<uint16_t>(); 
            Duplicate(valueSumTempLocal, (uint16_t)0, count);
        }
    }

    __local_mem__ SELF_TYPE* valueAddr = (__ubuf__ SELF_TYPE*)valueLocal.GetPhyAddr();
    __local_mem__ SELF_TYPE* valueSumAddr = (__ubuf__ SELF_TYPE*)valueSumLocal.GetPhyAddr();
    __local_mem__ CAST_TYPE* castSumAddr = (__ubuf__ CAST_TYPE*)castLocal.GetPhyAddr();
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<SELF_TYPE> valueReg;
        AscendC::MicroAPI::RegTensor<SELF_TYPE> valueSumReg;
        AscendC::MicroAPI::RegTensor<CAST_TYPE> valueCastReg;
        AscendC::MicroAPI::RegTensor<CAST_TYPE> castsumReg;
        AscendC::MicroAPI::MaskReg valueMaskReg;
        uint32_t maskLen = static_cast<uint32_t>(colLen);
        for (uint16_t i = 0; i < loopCnt; i++) {
            valueMaskReg = AscendC::MicroAPI::UpdateMask<CAST_TYPE>(maskLen);
            AscendC::MicroAPI::AddrReg valueAddrOfst = AscendC::MicroAPI::CreateAddrReg<SELF_TYPE>(i, vfLen);
            if constexpr (IS_CAST) {
                AscendC::MicroAPI::DataCopy<SELF_TYPE, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(valueReg, valueAddr, valueAddrOfst);
                AscendC::MicroAPI::AddrReg castSumAddrOfst = AscendC::MicroAPI::CreateAddrReg<CAST_TYPE>(i, vfLen);
                AscendC::MicroAPI::DataCopy(castsumReg, castSumAddr, castSumAddrOfst);
                AscendC::MicroAPI::Cast<CAST_TYPE, SELF_TYPE, castTrait16ToFloat>(valueCastReg, valueReg, valueMaskReg);
                AscendC::MicroAPI::Add(castsumReg, valueCastReg, castsumReg, valueMaskReg);
                AscendC::MicroAPI::DataCopy(castSumAddr, castsumReg, castSumAddrOfst, valueMaskReg);
            } else {
                AscendC::MicroAPI::DataCopy(valueReg, valueAddr, valueAddrOfst);
                AscendC::MicroAPI::AddrReg valueSumAddrOfst = AscendC::MicroAPI::CreateAddrReg<SELF_TYPE>(i, vfLen);
                AscendC::MicroAPI::DataCopy(valueSumReg, valueSumAddr, valueSumAddrOfst);
                if constexpr (IsSameType<SELF_TYPE, bool>::value) {
                    AscendC::MicroAPI::Or(valueSumReg, valueReg, valueSumReg, valueMaskReg);
                } else {
                    AscendC::MicroAPI::Add(valueSumReg, valueReg, valueSumReg, valueMaskReg);
                }
                AscendC::MicroAPI::DataCopy(valueSumAddr, valueSumReg, valueSumAddrOfst , valueMaskReg);   
            }
        }
    }
    if constexpr (IS_CAST) {
        valueCastQue_.EnQue(castLocal);
    }
    valueSumQue_.EnQue(valueSumLocal);
    valueQue_.FreeTensor(valueLocal);
}

template<typename SELF_TYPE, typename IDX_TYPE, bool ACCUMULATE, bool IS_CAST, typename CAST_TYPE>
__aicore__ inline void IndexPutWithSortV2SIMDKernel<SELF_TYPE, IDX_TYPE, ACCUMULATE, IS_CAST, CAST_TYPE>::ComputeAcc(int64_t rowIdx, int64_t colIdx, 
    int64_t rowLen, int64_t colLen, bool isRowLoopEnd, LocalTensor<IDX_TYPE>& indexTotalLocal, LocalTensor<int32_t>& posidxLocal)
{
    LocalTensor<IDX_TYPE> indexLocal = ((blockRow_ == 0) && (rowIdx == 0)) ? indexTotalLocal[coreLoopOfst_ - 1] : indexTotalLocal[0];
    uint32_t vfLen = V_REG_SIZE / sizeof(SELF_TYPE);
    if constexpr (IS_CAST) {
        vfLen = V_REG_SIZE / sizeof(CAST_TYPE);
    }

    uint16_t loopCnt = (colLen + vfLen - 1) / vfLen;
    int64_t repeatCount = 0;
    for (int64_t row = 0; row < rowLen; row++) {
        IDX_TYPE preIndex = -1;
        IDX_TYPE curIndex = indexLocal.GetValue(row + 1);
        IDX_TYPE nextIndex = indexLocal.GetValue(row + 2);
        repeatCount++;
        if (curIndex != nextIndex || row == rowLen - 1) {
            IDX_TYPE curPreIndex = indexLocal(row - repeatCount + 1);
            preIndex = (lastLoopRepateCount_ == 0) ? curPreIndex : lastLoopPreIndex_;
            for (int64_t k = 0; k < repeatCount; k++) {
                isInit_ = (rowIdx == 0 && firstFlag_) || (curIndex != curPreIndex && k == 0);
                if (preIndex != curIndex && k == 0 && lastLoopRepateCount_ == 0) {
                    if constexpr (IS_CAST || IsSameType<SELF_TYPE, int64_t>::value || IsSameType<SELF_TYPE, uint8_t>::value || 
                                IsSameType<SELF_TYPE, bool>::value) {
                        int64_t selfGmOfst = curIndex + blockCol_ * tilingData_->colBlockFactor + colIdx * colFactor_;
                        CopyInValue(outputGm_[selfGmOfst], colLen);
                        ComputeSumValue(colLen, vfLen, loopCnt);
                    }
                }
                int64_t valueGmOfst = posidxLocal(row - repeatCount + ONE + k) * tilingData_->nonIndexedDimSize + blockCol_ * tilingData_->colBlockFactor
                    + colIdx * colFactor_;
                CopyInValue(valuesGm_[valueGmOfst], colLen);
                ComputeSumValue(colLen, vfLen, loopCnt);
            }
            CopyOutSum(isRowLoopEnd, row, rowLen, repeatCount, curIndex, preIndex, nextIndex, colIdx, colLen, vfLen, loopCnt);
            repeatCount = 0;
        }
    }
}

template<typename SELF_TYPE, typename IDX_TYPE, bool ACCUMULATE, bool IS_CAST, typename CAST_TYPE>
__aicore__ inline void IndexPutWithSortV2SIMDKernel<SELF_TYPE, IDX_TYPE, ACCUMULATE, IS_CAST, CAST_TYPE>::CopyOutSum(bool isRowLoopEnd, int64_t row, int64_t rowLen, 
int64_t repeatCount, IDX_TYPE curIndex, IDX_TYPE preIndex, IDX_TYPE nextIndex, int64_t colIdx, int64_t colLen, 
uint32_t vfLen, uint16_t loopCnt)
{
    bool isFullUniqueIdxFront = (repeatCount + lastLoopRepateCount_) == curCoreProcessRowCount_ && curIndex == preIndex && curIndex != nextIndex;
    bool isFullUniqueIdxRear = (repeatCount + lastLoopRepateCount_) == curCoreProcessRowCount_ && curIndex == preIndex && curIndex == nextIndex;
    bool isFrontHalf = dupIdxStart_ == 0 && curIndex == preIndex && curIndex != nextIndex;
    bool isRearHalf = isRowLoopEnd && row == rowLen - 1 && curIndex == nextIndex;
    LocalTensor<SELF_TYPE> valueSumLocal = valueSumQue_.DeQue<SELF_TYPE>();
    LocalTensor<CAST_TYPE> castLocal;
    if constexpr (IS_CAST) {
        castLocal = valueCastQue_.DeQue<CAST_TYPE>();
    }

    if (isFullUniqueIdxFront || isFullUniqueIdxRear || isFrontHalf || isRearHalf) {
        int64_t sumWsOfset = 0;
        int64_t idxWsOfset = 0;
        auto colOfset = blockCol_ * tilingData_->colBlockFactor + colIdx * colFactor_;
        if (isFullUniqueIdxRear || isRearHalf) {
            sumWsOfset = (blockRow_ * TWO + ONE) * valueTmpCols_ + colOfset;
            idxWsOfset = (blockRow_ * TWO + ONE) * indexBlockNum_;
        } else {
            sumWsOfset = blockRow_ * TWO * valueTmpCols_ + colOfset;
            idxWsOfset = blockRow_ * TWO * indexBlockNum_;
        }
        indexSumWsGm_.SetValue(idxWsOfset, (IDX_TYPE)curIndex);
        DataCacheCleanAndInvalid<IDX_TYPE, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(indexSumWsGm_[idxWsOfset]);
        if constexpr (IS_CAST) {
            CopyOut<CAST_TYPE>(valuesTempSumWsGm_[sumWsOfset], castLocal, colLen);
        } else {
            CopyOut<CAST_TYPE>(valuesTempSumWsGm_[sumWsOfset], valueSumLocal, colLen);
        }
    } else if (curIndex != preIndex && curIndex != nextIndex) {
        int64_t gmOfset = curIndex + blockCol_ * tilingData_->colBlockFactor + colIdx * colFactor_;
        if constexpr (IsSameType<SELF_TYPE, int64_t>::value || IsSameType<SELF_TYPE, uint8_t>::value || IsSameType<SELF_TYPE, bool>::value) {        
            CopyOut<SELF_TYPE>(outputGm_[gmOfset], valueSumLocal, colLen);
        } else if (IS_CAST) {
            CastSumValue(valueSumLocal, castLocal, colLen, vfLen, loopCnt);
            EventMsg<HardEvent::V_MTE3>();
            CopyOut<SELF_TYPE>(outputGm_[gmOfset], valueSumLocal, colLen);
        } else {
            SetAtomicAdd<SELF_TYPE>();
            CopyOut<SELF_TYPE>(outputGm_[gmOfset], valueSumLocal, colLen);
            SetAtomicNone();
        }
    }

    if (curIndex == nextIndex && !(isRowLoopEnd && row == rowLen - 1)) {
        valueSumQue_.EnQue(valueSumLocal);
        if constexpr (IS_CAST) {
            valueCastQue_.EnQue(castLocal);
        }
        lastLoopRepateCount_ += repeatCount;
        lastLoopPreIndex_ = preIndex;
    } else {
        valueSumQue_.FreeTensor(valueSumLocal);
        if constexpr (IS_CAST) {
            valueCastQue_.FreeTensor(castLocal);
        }
        dupIdxStart_ += repeatCount + lastLoopRepateCount_;
        lastLoopPreIndex_ = -1;
        lastLoopRepateCount_ = 0;
    }
}

template<typename SELF_TYPE, typename IDX_TYPE, bool ACCUMULATE, bool IS_CAST, typename CAST_TYPE>
__aicore__ inline void IndexPutWithSortV2SIMDKernel<SELF_TYPE, IDX_TYPE, ACCUMULATE, IS_CAST, CAST_TYPE>::ComputeReplace(int64_t rowIdx, int64_t colIdx, 
int64_t rowLen, int64_t colLen, LocalTensor<IDX_TYPE>& indexTotalLocal, LocalTensor<int32_t>& posidxLocal)
{
    LocalTensor<IDX_TYPE> indexLocal = ((blockRow_ == 0) && (rowIdx == 0)) ? indexTotalLocal[coreLoopOfst_ - 1] : indexTotalLocal[0];
    for (int64_t row = 0; row < rowLen; row++) {
        IDX_TYPE preIndex = indexLocal.GetValue(row);
        IDX_TYPE curIndex = indexLocal.GetValue(row + 1);
        if (preIndex != curIndex) {
            int64_t gmOfset = curIndex + blockCol_ * tilingData_->colBlockFactor + colIdx * colFactor_;
            int64_t valueGmOfst = posidxLocal(row) * tilingData_->nonIndexedDimSize + blockCol_ * tilingData_->colBlockFactor
                + colIdx * colFactor_;
            CopyInValue(valuesGm_[valueGmOfst], colLen);
            LocalTensor<SELF_TYPE> valueLocal = dataQue_.DeQue<SELF_TYPE>();
            CopyOut<SELF_TYPE>(outputGm_[gmOfset], valueLocal, colLen);
            dataQue_.FreeTensor(valueLocal);
        }
    }  
}

template<typename SELF_TYPE, typename IDX_TYPE, bool ACCUMULATE, bool IS_CAST, typename CAST_TYPE>
__aicore__ inline void IndexPutWithSortV2SIMDKernel<SELF_TYPE, IDX_TYPE, ACCUMULATE, IS_CAST, CAST_TYPE>::ProcessFirstStep(void)
{
    if (blockId_ >= blockNum_) {
        return;
    }
    pipe_->InitBuffer(indexQue_, DOUBLE_BUFFER, tilingData_->indicesFactor * sizeof(IDX_TYPE));
    pipe_->InitBuffer(posidxQue_, DOUBLE_BUFFER, tilingData_->indicesFactor * sizeof(int32_t));
    if constexpr (ACCUMULATE) {
        pipe_->InitBuffer(valueQue_, DOUBLE_BUFFER, tilingData_->ubFactor * sizeof(SELF_TYPE));
        pipe_->InitBuffer(valueSumQue_, DOUBLE_BUFFER, tilingData_->ubFactor * sizeof(SELF_TYPE));       
    } else {
        pipe_->InitBuffer(dataQue_, DOUBLE_BUFFER, tilingData_->ubFactor * sizeof(SELF_TYPE));
    }
    if constexpr (IS_CAST) {
        pipe_->InitBuffer(valueCastQue_, 1, tilingData_->ubFactor * HIGH_TYPE);
    }

    int64_t rowLoopDataLen = 0;
    int64_t rowLoopNum = 0;
    int64_t rowTailDataLen = 0;
    int64_t colLoopDataLen = 0;
    int64_t curCoreProcessColCount = 0;
    int64_t colLoopNum = 0;
    int64_t colTailDataLen = 0;

    int64_t actualIndicesFactor = tilingData_->indicesFactor - coreLoopOfst_ - 1;
    curCoreProcessRowCount_ = (blockRow_ == rowUseCoreNum_ - 1) ? tilingData_->rowTailBlockFactor : tilingData_->rowBlockFactor;
    indexFactor_ = actualIndicesFactor >= curCoreProcessRowCount_ ? curCoreProcessRowCount_ : actualIndicesFactor;
    rowLoopDataLen = indexFactor_;
    rowLoopNum = ops::CeilDiv(curCoreProcessRowCount_, rowLoopDataLen);
    rowTailDataLen = curCoreProcessRowCount_ - rowLoopDataLen * (rowLoopNum - 1);
    if (colUseCoreNum_ == 1) {
        curCoreProcessColCount = tilingData_->colBlockFactor;
    } else {
        curCoreProcessColCount = (blockCol_ == colUseCoreNum_ - 1) ? tilingData_->colTailBlockFactor : tilingData_->colBlockFactor;
    }
    colFactor_ = curCoreProcessColCount >= tilingData_->ubFactor ? tilingData_->ubFactor : curCoreProcessColCount;
    colLoopDataLen = colFactor_;
    colLoopNum = ops::CeilDiv(curCoreProcessColCount, colLoopDataLen);
    colTailDataLen = curCoreProcessColCount - colLoopDataLen * (colLoopNum - 1); 

    for (uint64_t colIdx = 0; colIdx < colLoopNum; colIdx++) {
        auto colLen = colIdx == colLoopNum - 1 ? colTailDataLen : colLoopDataLen;
        lastLoopRepateCount_ = 0;
        lastLoopPreIndex_ = -1;
        firstFlag_ = true;
        for (uint64_t rowIdx = 0; rowIdx < rowLoopNum; rowIdx++) {
            auto rowLen = rowIdx == rowLoopNum - 1 ? rowTailDataLen : rowLoopDataLen;
            bool endFlag = rowIdx == rowLoopNum - 1 ? true : false;
            dupIdxStart_ = rowIdx * rowLoopDataLen;
            CopyInIdx(rowIdx, rowLen, endFlag);
            LocalTensor<IDX_TYPE> indexTotalLocal = indexQue_.DeQue<IDX_TYPE>();
            LocalTensor<int32_t> posidxLocal = posidxQue_.DeQue<int32_t>();
            EventMsg<HardEvent::MTE2_S>();
            if constexpr (ACCUMULATE) {
                ComputeAcc(rowIdx, colIdx, rowLen, colLen, endFlag, indexTotalLocal, posidxLocal);
            } else {
                ComputeReplace(rowIdx, colIdx, rowLen, colLen, indexTotalLocal, posidxLocal);
            }
            indexQue_.FreeTensor(indexTotalLocal);
            posidxQue_.FreeTensor(posidxLocal);
        }
    }
}

template<typename SELF_TYPE, typename IDX_TYPE, bool ACCUMULATE, bool IS_CAST, typename CAST_TYPE>
__aicore__ inline void IndexPutWithSortV2SIMDKernel<SELF_TYPE, IDX_TYPE, ACCUMULATE, IS_CAST, CAST_TYPE>::GetComputeRangeIdx(
    IDX_TYPE &startIdx, IDX_TYPE &endIdx, IDX_TYPE &indexValue) 
{
    IDX_TYPE afterIdx = (blockRow_ * TWO + ONE) * indexBlockNum_;
    if (indexSumWsGm_(afterIdx) == -1){
        return;
    }
    IDX_TYPE preCoreAfterIdx = afterIdx - indexBlockNum_ * TWO;
    if (preCoreAfterIdx  >= 0 && indexSumWsGm_(preCoreAfterIdx) == indexSumWsGm_(afterIdx) ){
        return;
    }
    startIdx = afterIdx;
    indexValue = indexSumWsGm_(afterIdx);
    for (IDX_TYPE idx = startIdx + indexBlockNum_; idx < indexBlockNum_ * TWO * tilingData_->rowUseCoreNum * tilingData_->colUseCoreNum; idx += indexBlockNum_ * TWO) {
        if (indexSumWsGm_(idx) == indexSumWsGm_(afterIdx)) {
            endIdx = idx;
            break;
        }
    }
    startIdx /= indexBlockNum_;
    endIdx /= indexBlockNum_;
}

template<typename SELF_TYPE, typename IDX_TYPE, bool ACCUMULATE, bool IS_CAST, typename CAST_TYPE>
__aicore__ inline void IndexPutWithSortV2SIMDKernel<SELF_TYPE, IDX_TYPE, ACCUMULATE, IS_CAST, CAST_TYPE>::CopyInValueFromWs(int64_t offset, int64_t dataLen)
{
    DataCopyExtParams inParams = {static_cast<uint16_t>(1), static_cast<uint32_t>(dataLen * sizeof(CAST_TYPE)), 0, 0, 0 };
    DataCopyPadExtParams<CAST_TYPE> padParams = { false, 0, 0, 0 };
    LocalTensor<CAST_TYPE> valueLocal = valueQue_.AllocTensor<CAST_TYPE>();
    DataCopyPad(valueLocal, valuesTempSumWsGm_[offset], inParams, padParams);
    valueQue_.EnQue(valueLocal);
}

template<typename SELF_TYPE, typename IDX_TYPE, bool ACCUMULATE, bool IS_CAST, typename CAST_TYPE>
__aicore__ inline void IndexPutWithSortV2SIMDKernel<SELF_TYPE, IDX_TYPE, ACCUMULATE, IS_CAST, CAST_TYPE>::GetSumValueStep2(int64_t dataLen,
    LocalTensor<CAST_TYPE>& valueSumLocal) {
    LocalTensor<CAST_TYPE> valueLocal = valueQue_.DeQue<CAST_TYPE>();
    __local_mem__ CAST_TYPE* valueAddr = (__ubuf__ CAST_TYPE*)valueLocal.GetPhyAddr();
    __local_mem__ CAST_TYPE* valueSumAddr = (__ubuf__ CAST_TYPE*)valueSumLocal.GetPhyAddr();
    
    uint32_t vfLen = V_REG_SIZE / sizeof(CAST_TYPE);
    uint16_t loopCnt = (dataLen + vfLen - 1) / vfLen;
    uint32_t maskCount = static_cast<uint32_t>(dataLen);
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<CAST_TYPE> valueReg;
        AscendC::MicroAPI::RegTensor<CAST_TYPE> sumReg;
        AscendC::MicroAPI::MaskReg valueMaskReg;
        for (uint16_t i = 0; i < loopCnt; i++) {
            valueMaskReg = AscendC::MicroAPI::UpdateMask<CAST_TYPE>(maskCount);
            AscendC::MicroAPI::AddrReg addrOfst = AscendC::MicroAPI::CreateAddrReg<CAST_TYPE>(i, vfLen);
            AscendC::MicroAPI::DataCopy(valueReg, valueAddr, addrOfst);
            AscendC::MicroAPI::DataCopy(sumReg, valueSumAddr, addrOfst);
            if constexpr(IsSameType<SELF_TYPE, bool>::value) {
                AscendC::MicroAPI::Or(sumReg, sumReg, valueReg, valueMaskReg);
            } else {
                AscendC::MicroAPI::Add(sumReg, sumReg, valueReg, valueMaskReg);
            }
            AscendC::MicroAPI::DataCopy(valueSumAddr, sumReg, addrOfst, valueMaskReg);
        }
    }
    valueQue_.FreeTensor(valueLocal);
}

template<typename SELF_TYPE, typename IDX_TYPE, bool ACCUMULATE, bool IS_CAST, typename CAST_TYPE>
__aicore__ inline void IndexPutWithSortV2SIMDKernel<SELF_TYPE, IDX_TYPE, ACCUMULATE, IS_CAST, CAST_TYPE>::CastSumValueStep2(int64_t dataLen) {
    LocalTensor<CAST_TYPE> valueSumLocal = valueCastQue_.DeQue<CAST_TYPE>();
    __local_mem__ CAST_TYPE* valueSumAddr = (__ubuf__ CAST_TYPE*)valueSumLocal.GetPhyAddr();
    LocalTensor<SELF_TYPE> valueResult = valueSumQue_.AllocTensor<SELF_TYPE>();
    __local_mem__ SELF_TYPE* valueResultAddr = (__ubuf__ SELF_TYPE*)valueResult.GetPhyAddr();
    
    uint32_t vfLen = V_REG_SIZE / sizeof(CAST_TYPE);
    uint16_t loopCnt = (dataLen + vfLen - 1) / vfLen;
    uint32_t maskCount = static_cast<uint32_t>(dataLen);
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<CAST_TYPE> valueSumReg;
        AscendC::MicroAPI::RegTensor<SELF_TYPE> valueResultReg;
        AscendC::MicroAPI::MaskReg valueMaskReg;
        for (uint16_t i = 0; i < loopCnt; i++) {
            valueMaskReg = AscendC::MicroAPI::UpdateMask<CAST_TYPE>(maskCount);
            AscendC::MicroAPI::AddrReg addrOfstCast = AscendC::MicroAPI::CreateAddrReg<CAST_TYPE>(i, vfLen);
            AscendC::MicroAPI::AddrReg addrOfstSelf = AscendC::MicroAPI::CreateAddrReg<SELF_TYPE>(i, vfLen);
            AscendC::MicroAPI::DataCopy(valueSumReg, valueSumAddr, addrOfstCast);
            AscendC::MicroAPI::Cast<SELF_TYPE, CAST_TYPE, castTraitFloatTo16>(valueResultReg, valueSumReg, valueMaskReg);
            AscendC::MicroAPI::DataCopy<SELF_TYPE, MicroAPI::StoreDist::DIST_PACK_B32>(valueResultAddr, valueResultReg, addrOfstSelf, valueMaskReg);
        }
    }
    valueSumQue_.EnQue(valueResult);
    valueCastQue_.FreeTensor(valueSumLocal);
}

template<typename SELF_TYPE, typename IDX_TYPE, bool ACCUMULATE, bool IS_CAST, typename CAST_TYPE>
__aicore__ inline void IndexPutWithSortV2SIMDKernel<SELF_TYPE, IDX_TYPE, ACCUMULATE, IS_CAST, CAST_TYPE>::ProcessSecondStep(void)
{
    if (GetBlockIdx() >= blockNum_) {
        return;
    }
    pipe_->Reset();
    pipe_->InitBuffer(valueQue_, DOUBLE_BUFFER, tilingData_->ubFactor * sizeof(CAST_TYPE));
    pipe_->InitBuffer(valueCastQue_, 1, tilingData_->ubFactor * HIGH_TYPE);

    if constexpr (IS_CAST) {
        pipe_->InitBuffer(valueSumQue_, DOUBLE_BUFFER, tilingData_->ubFactor * sizeof(SELF_TYPE));
    }
    IDX_TYPE indexValue = -1, startIdx = -1, endIdx = -1;
    GetComputeRangeIdx(startIdx, endIdx, indexValue);
    if (indexValue == -1) {
        return;
    }
    int64_t curCoreProcessColCount = (blockCol_ == tilingData_->colUseCoreNum - 1) ? 
        tilingData_->colTailBlockFactor : tilingData_->colBlockFactor;
    int64_t colFactor = (curCoreProcessColCount >= tilingData_->ubFactor) ? tilingData_->ubFactor: curCoreProcessColCount;
    int64_t colLoopNum = ops::CeilDiv(curCoreProcessColCount, colFactor);
    int64_t colTailFactor = curCoreProcessColCount - (colLoopNum - ONE) * colFactor;

    for (int32_t i = 0; i < colLoopNum; i++) {
        int64_t colOffset = blockCol_ * tilingData_->colBlockFactor + i * colFactor;
        int64_t curUbFactor = (i == colLoopNum -1) ? colTailFactor : colFactor;
        int64_t outLineOffset = indexValue  + colOffset;
        LocalTensor<CAST_TYPE> sumLocal;
        if constexpr (IS_CAST) {
            sumLocal = valueCastQue_.AllocTensor<CAST_TYPE>();
        } else {
            sumLocal = valueSumQue_.AllocTensor<CAST_TYPE>();
        }
        Duplicate(sumLocal, (CAST_TYPE)0, colFactor);
        for (IDX_TYPE idx = startIdx; idx >=0 && idx <= endIdx; idx++) {
            if (indexSumWsGm_(idx * indexBlockNum_) == -1) {
                continue;
            }
            CopyInValueFromWs(idx * valueTmpCols_ + colOffset, curUbFactor);
            GetSumValueStep2(curUbFactor, sumLocal);
        }
        if constexpr (IS_CAST) {
            valueCastQue_.EnQue(sumLocal);
        } else {
            valueSumQue_.EnQue(sumLocal);
        }

        if constexpr (IS_CAST) {
            CastSumValueStep2(curUbFactor);
        }
        LocalTensor<SELF_TYPE> valueLocal = valueSumQue_.DeQue<SELF_TYPE>();
        DataCopyExtParams outParams = {static_cast<uint16_t>(1), static_cast<uint32_t>(curUbFactor * sizeof(SELF_TYPE)), 0, 0, 0 };
        if constexpr (IS_CAST || IsSameType<SELF_TYPE, int64_t>::value || IsSameType<SELF_TYPE, uint8_t>::value || IsSameType<SELF_TYPE, bool>::value) {
            DataCopyPad(outputGm_[outLineOffset], valueLocal, outParams);
        } else {
            SetAtomicAdd<SELF_TYPE>();
            DataCopyPad(outputGm_[outLineOffset], valueLocal, outParams);
            SetAtomicNone();
        }
        valueSumQue_.FreeTensor(valueLocal);
    }
}

template<typename SELF_TYPE, typename IDX_TYPE, bool ACCUMULATE, bool IS_CAST, typename CAST_TYPE>
__aicore__ inline void IndexPutWithSortV2SIMDKernel<SELF_TYPE, IDX_TYPE, ACCUMULATE, IS_CAST, CAST_TYPE>::Process(void) 
{
    ProcessFirstStep();
    AscendC::SyncAll();
    ProcessSecondStep();
}
};


#endif