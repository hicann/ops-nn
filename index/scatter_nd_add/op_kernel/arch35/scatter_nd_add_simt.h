/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file scatter_nd_add.h
 * \brief
 */

#ifndef SCATTER_ND_ADD_SMIT_H
#define SCATTER_ND_ADD_SMIT_H

#include "kernel_operator.h"
#include "scatter_nd_add_common.h"

namespace ScatterNdAdd {
using namespace AscendC;

template <typename INDICES_T, typename PARAMS_T, typename TYPE_T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM_LAUNCH_BOUND) inline void SimtCompute(
    __ubuf__ INDICES_T* idxLocalAddr, __ubuf__ PARAMS_T* xLocalAddr, __gm__ PARAMS_T* outputGmAddr,
    const __ubuf__ TYPE_T* strideListAddr, const __ubuf__ TYPE_T* outputShapeAddr, const uint32_t currUbTilingSize,
    const TYPE_T xOffSet, const TYPE_T sliceSize, const uint32_t rankSize, const TYPE_T indiceOffSet, int64_t varInAxis,
    const TYPE_T magic, const TYPE_T shift)
{
    for (uint32_t index = Simt::GetThreadIdx(); index < currUbTilingSize; index += Simt::GetThreadNum()) {
        TYPE_T globalIdx = xOffSet + index;
        TYPE_T quotient = Simt::UintDiv(globalIdx, magic, shift);
        TYPE_T currIndiceIdx = quotient * rankSize;
        TYPE_T scatterAxisIdx = globalIdx - quotient * sliceSize;
        INDICES_T idx = 0;
        bool outOfBound = false;
        for (TYPE_T dim = 0; dim < rankSize; ++dim) {
            INDICES_T indiceVal = idxLocalAddr[currIndiceIdx + dim - indiceOffSet];
            outOfBound |= static_cast<TYPE_T>(indiceVal) > outputShapeAddr[dim];
            idx += indiceVal * strideListAddr[dim];
        }
        if (!outOfBound) {
            if (idx >= 0 && idx < varInAxis) {//索引越界
                uint64_t dstIndex = static_cast<uint64_t>(idx * sliceSize + scatterAxisIdx);
                if constexpr (IsSameType<PARAMS_T, bool>::value) {
                    PARAMS_T value = xLocalAddr[index];
                    if (value > 0) {
                        outputGmAddr[dstIndex] = value;
                    }
                } else {
                    Simt::AtomicAdd(outputGmAddr + dstIndex, xLocalAddr[index]);
                }
            }
        }
    }
}

template <typename INDICES_T, typename PARAMS_T, typename TYPE_T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM_LAUNCH_BOUND) inline void SimtComputeDimensionOne(
    __gm__ INDICES_T* idxGmAddr, __ubuf__ PARAMS_T* xLocalAddr, __gm__ PARAMS_T* outputGmAddr,
    uint32_t currUbTilingSize, TYPE_T xOffSet, TYPE_T sliceSize, uint32_t rankSize, TYPE_T indiceOffSet,
    int64_t varInAxis, TYPE_T magic, TYPE_T shift)
{
    for (uint32_t index = Simt::GetThreadIdx(); index < currUbTilingSize; index += Simt::GetThreadNum()) {
        TYPE_T globalIdx = xOffSet + index;
        TYPE_T quotient = Simt::UintDiv(globalIdx, magic, shift);
        TYPE_T currIndiceIdx = quotient * rankSize;
        TYPE_T scatterAxisIdx = globalIdx - quotient * sliceSize;
        INDICES_T idx = idxGmAddr[currIndiceIdx];

        if (idx >= 0 && idx < varInAxis) {
            uint64_t dstOffet = static_cast<uint64_t>(idx * sliceSize + scatterAxisIdx);
            if constexpr (IsSameType<PARAMS_T, bool>::value) {
                PARAMS_T value = xLocalAddr[index];
                if (value > 0) {
                    outputGmAddr[dstOffet] = value;
                }
            } else {
                Simt::AtomicAdd(outputGmAddr + dstOffet, xLocalAddr[index]);
            }
        }
    }
}

template <typename PARAMS_T, typename INDICES_T, typename TYPE_T>
class ScatterNdAddSimt
{
public:
    __aicore__ inline ScatterNdAddSimt(const ScatterNdAddRegBaseTilingData& tilingData, TPipe& pipe)
        : pipe_(pipe), tiling_(tilingData){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR indices, GM_ADDR updates, GM_ADDR y, GM_ADDR workspace);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ComputeData();
    __aicore__ inline void CopyIn(LocalTensor<INDICES_T>& idxLocal, LocalTensor<PARAMS_T>& xLocal);
    __aicore__ inline void SimdFree(LocalTensor<INDICES_T>& idxLocal, LocalTensor<PARAMS_T>& xLocal);
    __aicore__ inline void ComputeDimensionOther();
    __aicore__ inline void ComputeDimensionOne();
    __aicore__ inline void CopyInUpdate(LocalTensor<PARAMS_T>& xLocal);

private:
    TPipe& pipe_;
    const ScatterNdAddRegBaseTilingData& tiling_;
    GlobalTensor<INDICES_T> idxGm;
    GlobalTensor<PARAMS_T> xGm;
    GlobalTensor<PARAMS_T> outputGm;

    TQue<QuePosition::VECIN, DOUBLE_BUFFER> inQueIdx, inQueX;
    TBuf<TPosition::VECCALC> strideListBuf;
    TBuf<TPosition::VECCALC> outputShapeBuf;

    uint32_t blockIdx;
    TYPE_T blockTilingSize = 0;
    TYPE_T currBlockTilingSize = 0;

    uint32_t ubTilingSize = 0;
    uint32_t currUbTilingSize = 0;

    TYPE_T xBlockOffSet = 0;
    TYPE_T xOffSet = 0;
    TYPE_T indiceBlockOffSet = 0;
    TYPE_T indiceOffSet = 0;
    uint32_t currIdxTilingSize = 0;
    TYPE_T ubLoopCnt = 0;
};

template <typename PARAMS_T, typename INDICES_T, typename TYPE_T>
__aicore__ inline void ScatterNdAddSimt<PARAMS_T, INDICES_T, TYPE_T>::Init(
    GM_ADDR x, GM_ADDR indices, GM_ADDR updates, GM_ADDR y, GM_ADDR workspace)
{
    if (tiling_.sliceSize == 0) {
        return;
    }

    blockIdx = GetBlockIdx();

    this->xBlockOffSet = tiling_.blockTilingSize * blockIdx;
    // calculate indice offset size by x offset size
    this->indiceBlockOffSet = this->xBlockOffSet / tiling_.sliceSize * tiling_.rankSize;

    if (blockIdx == tiling_.blockNum - 1) {
        this->currBlockTilingSize = tiling_.tailBlockTilingSize;
    } else {
        this->currBlockTilingSize = tiling_.blockTilingSize;
    }

    this->ubTilingSize = tiling_.ubTilingSize;
    if (this->currBlockTilingSize <= tiling_.ubTilingSize) {
        this->ubTilingSize = this->currBlockTilingSize;
    }

    auto indiceUbTilingSize = (this->ubTilingSize + tiling_.sliceSize - 1) / tiling_.sliceSize * tiling_.rankSize;
    // add 2 indices  for the scenario where boundary values are not accessible.
    indiceUbTilingSize += 2;
    idxGm.SetGlobalBuffer((__gm__ INDICES_T*)indices);
    xGm.SetGlobalBuffer((__gm__ PARAMS_T*)updates);
    outputGm.SetGlobalBuffer((__gm__ PARAMS_T*)y);

    pipe_.InitBuffer(inQueX, DOUBLE_BUFFER, ROUND_UP32(this->ubTilingSize * sizeof(PARAMS_T)));
    if (tiling_.rankSize >= INDICE_RANK_TWO) {
        pipe_.InitBuffer(inQueIdx, DOUBLE_BUFFER, ROUND_UP32(indiceUbTilingSize * sizeof(INDICES_T)));
        pipe_.InitBuffer(strideListBuf, MAX_RANK_COUNT * sizeof(TYPE_T));
        pipe_.InitBuffer(outputShapeBuf, MAX_SHAPE_RANK * sizeof(TYPE_T));
    }
}

template <typename PARAMS_T, typename INDICES_T, typename TYPE_T>
__aicore__ inline void ScatterNdAddSimt<PARAMS_T, INDICES_T, TYPE_T>::Process()
{
    if (blockIdx < tiling_.blockNum) {
        this->ubLoopCnt = (this->currBlockTilingSize + this->ubTilingSize - 1) / this->ubTilingSize;
        for (TYPE_T idx = 0; idx < this->ubLoopCnt - 1; idx++) {
            this->currUbTilingSize = this->ubTilingSize;
            this->xOffSet = this->xBlockOffSet + idx * this->ubTilingSize;
            ComputeData();
        }
        this->xOffSet = this->xBlockOffSet + (this->ubLoopCnt - 1) * this->ubTilingSize;
        this->currUbTilingSize = this->currBlockTilingSize - this->ubTilingSize * (this->ubLoopCnt - 1);
        ComputeData();
    }
}

template <typename PARAMS_T, typename INDICES_T, typename TYPE_T>
__aicore__ inline void ScatterNdAddSimt<PARAMS_T, INDICES_T, TYPE_T>::ComputeData()
{
    auto currEnd = this->xOffSet + this->currUbTilingSize;
    auto indiceBegin = this->xOffSet / tiling_.sliceSize * tiling_.rankSize;
    auto indiceEnd = (currEnd + tiling_.sliceSize - 1) / tiling_.sliceSize * tiling_.rankSize;
    this->currIdxTilingSize = indiceEnd - indiceBegin;
    this->indiceOffSet = indiceBegin;
    if (tiling_.rankSize >= INDICE_RANK_TWO) {
        ComputeDimensionOther();
    } else {
        ComputeDimensionOne();
    }
}

template <typename PARAMS_T, typename INDICES_T, typename TYPE_T>
__aicore__ inline void ScatterNdAddSimt<PARAMS_T, INDICES_T, TYPE_T>::ComputeDimensionOther()
{
    LocalTensor<INDICES_T> idxLocal = inQueIdx.AllocTensor<INDICES_T>();
    LocalTensor<PARAMS_T> xLocal = inQueX.AllocTensor<PARAMS_T>();
    CopyIn(idxLocal, xLocal);
    uint32_t currUbTilingSize = this->currUbTilingSize;
    TYPE_T xOffSetCur = this->xOffSet;
    TYPE_T sliceSize = tiling_.sliceSize;
    uint32_t rankSize = tiling_.rankSize;
    TYPE_T indiceOffSetCur = this->indiceOffSet;
    int64_t varInAxis = tiling_.varInAxis;

    LocalTensor<TYPE_T> strideList = strideListBuf.Get<TYPE_T>();
    LocalTensor<TYPE_T> outputShape = outputShapeBuf.Get<TYPE_T>();
    for (uint32_t i = 0; i < MAX_RANK_COUNT; i++) {
        strideList(i) = tiling_.strideList[i];
    }
    for (uint32_t i = 0; i < MAX_SHAPE_RANK; i++) {
        outputShape(i) = tiling_.outPutShape[i];
    }
    DataSyncBarrier<MemDsbT::UB>();
    TYPE_T magic = 0;
    TYPE_T shift = 0;
    GetUintDivMagicAndShift(magic, shift, sliceSize);
    AscendC::Simt::VF_CALL<SimtCompute<INDICES_T, PARAMS_T, TYPE_T>>(
        Simt::Dim3(THREAD_NUM), (__ubuf__ INDICES_T*)idxLocal.GetPhyAddr(), (__ubuf__ PARAMS_T*)xLocal.GetPhyAddr(),
        (__gm__ PARAMS_T*)(outputGm.GetPhyAddr()), (__ubuf__ TYPE_T*)strideList.GetPhyAddr(),
        (__ubuf__ TYPE_T*)outputShape.GetPhyAddr(), currUbTilingSize, xOffSetCur, sliceSize, rankSize, indiceOffSetCur,
        varInAxis, magic, shift);
    SimdFree(idxLocal, xLocal);
}

template <typename PARAMS_T, typename INDICES_T, typename TYPE_T>
__aicore__ inline void ScatterNdAddSimt<PARAMS_T, INDICES_T, TYPE_T>::ComputeDimensionOne()
{
    LocalTensor<PARAMS_T> xLocal = inQueX.AllocTensor<PARAMS_T>();
    CopyInUpdate(xLocal);
    uint32_t currUbTilingSize = this->currUbTilingSize;
    TYPE_T xOffSetCur = this->xOffSet;
    TYPE_T sliceSize = tiling_.sliceSize;
    uint32_t rankSize = tiling_.rankSize;
    TYPE_T indiceOffSetCur = this->indiceOffSet;
    int64_t varInAxis = tiling_.varInAxis;
    
    TYPE_T magic = 0;
    TYPE_T shift = 0;
    GetUintDivMagicAndShift(magic, shift, sliceSize);
    AscendC::Simt::VF_CALL<SimtComputeDimensionOne<INDICES_T, PARAMS_T, TYPE_T>>(
        Simt::Dim3(THREAD_NUM), (__gm__ INDICES_T*)(idxGm.GetPhyAddr()), (__ubuf__ PARAMS_T*)xLocal.GetPhyAddr(),
        (__gm__ PARAMS_T*)(outputGm.GetPhyAddr()), currUbTilingSize, xOffSetCur, sliceSize, rankSize, indiceOffSetCur,
        varInAxis, magic, shift);
    inQueX.FreeTensor(xLocal);
}

template <typename PARAMS_T, typename INDICES_T, typename TYPE_T>
__aicore__ inline void ScatterNdAddSimt<PARAMS_T, INDICES_T, TYPE_T>::CopyIn(
    LocalTensor<INDICES_T>& idxLocal, LocalTensor<PARAMS_T>& xLocal)
{
    DataCopyExtParams idxCopyParams{1, static_cast<uint32_t>(this->currIdxTilingSize * sizeof(INDICES_T)), 0, 0, 0};
    DataCopyPadExtParams<INDICES_T> idxPadParams{false, 0, 0, 0};
    DataCopyPad(idxLocal, idxGm[this->indiceOffSet], idxCopyParams, idxPadParams);

    DataCopyExtParams xCopyParams{1, static_cast<uint32_t>(this->currUbTilingSize * sizeof(PARAMS_T)), 0, 0, 0};
    DataCopyPadExtParams<PARAMS_T> xPadParams{false, 0, 0, 0};
    DataCopyPad(xLocal, xGm[this->xOffSet], xCopyParams, xPadParams);

    inQueIdx.EnQue(idxLocal);
    inQueX.EnQue(xLocal);
    inQueIdx.DeQue<INDICES_T>();
    inQueX.DeQue<PARAMS_T>();
}

template <typename PARAMS_T, typename INDICES_T, typename TYPE_T>
__aicore__ inline void ScatterNdAddSimt<PARAMS_T, INDICES_T, TYPE_T>::SimdFree(
    LocalTensor<INDICES_T>& idxLocal, LocalTensor<PARAMS_T>& xLocal)
{
    inQueIdx.FreeTensor(idxLocal);
    inQueX.FreeTensor(xLocal);
}

template <typename PARAMS_T, typename INDICES_T, typename TYPE_T>
__aicore__ inline void ScatterNdAddSimt<PARAMS_T, INDICES_T, TYPE_T>::CopyInUpdate(LocalTensor<PARAMS_T>& xLocal)
{
    DataCopyExtParams xCopyParams{1, static_cast<uint32_t>(this->currUbTilingSize * sizeof(PARAMS_T)), 0, 0, 0};
    DataCopyPadExtParams<PARAMS_T> xPadParams{false, 0, 0, 0};
    DataCopyPad(xLocal, xGm[this->xOffSet], xCopyParams, xPadParams);

    inQueX.EnQue(xLocal);
    inQueX.DeQue<PARAMS_T>();
}

} // namespace ScatterNdAdd

#endif // SCATTER_ND_ADD_SMIT_H