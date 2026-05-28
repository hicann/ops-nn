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
 * \file scatter_nd_common_simt.h
 * \brief scatter_nd_common_simt
 */

#ifndef SCATTER_ND_COMMON_SIMT_H
#define SCATTER_ND_COMMON_SIMT_H

#include "scatter_nd_common_base.h"
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"

namespace ScatterNdCommon {
using namespace AscendC;

template <typename INDICES_T, typename PARAMS_T, typename TYPE_T, uint8_t Mode>
__simt_vf__ __aicore__ __launch_bounds__(THREAD_NUM_LAUNCH_BOUND) inline void SimtCompute(
    __local_mem__ INDICES_T* idxLocalAddr, __local_mem__ PARAMS_T* xLocalAddr, __gm__ PARAMS_T* outputGmAddr,
    const __local_mem__ TYPE_T* strideListAddr, const __local_mem__ TYPE_T* outputShapeAddr, const uint32_t currUbTilingSize,
    const TYPE_T xOffSet, const TYPE_T sliceSize, const uint32_t rankSize, const TYPE_T indiceOffSet,
    const TYPE_T magic, const TYPE_T shift)
{
    for (uint32_t index = threadIdx.x; index < currUbTilingSize; index += blockDim.x) {
        TYPE_T globalIdx = xOffSet + index;
        TYPE_T quotient = Simt::UintDiv(globalIdx, magic, shift); // GM第几行update
        TYPE_T currIndiceIdx = quotient * rankSize; // GM第几个索引
        TYPE_T scatterAxisIdx = globalIdx - quotient * sliceSize; // update尾轴第几个元素(globalIdx % sliceSize)
        TYPE_T idx = 0;
        bool outOfBound = false;
        for (TYPE_T dim = 0; dim < rankSize; ++dim) {
            INDICES_T indiceVal = idxLocalAddr[currIndiceIdx + dim - indiceOffSet];
            outOfBound |= (indiceVal < 0 || indiceVal >= outputShapeAddr[dim]);
            idx += indiceVal * strideListAddr[dim];
        }
        if (!outOfBound) {
            TYPE_T dstIndex = idx * sliceSize + scatterAxisIdx;
            if constexpr (IsSameType<PARAMS_T, bool>::value) {
                PARAMS_T value = xLocalAddr[index];
                if constexpr (Mode == MODE_MAX) {
                    if (value > 0) {
                        outputGmAddr[dstIndex] = value;
                    }
                } else {
                    if (value <= 0) {
                        outputGmAddr[dstIndex] = value;
                    }
                }
            } else {
                if constexpr (Mode == MODE_MAX) {
                    Simt::AtomicMax(outputGmAddr + dstIndex, xLocalAddr[index]);
                } else {
                    Simt::AtomicMin(outputGmAddr + dstIndex, xLocalAddr[index]);
                }
            }
        }
    }
}

template <typename INDICES_T, typename PARAMS_T, typename TYPE_T, uint8_t Mode>
__simt_vf__ __aicore__ __launch_bounds__(THREAD_NUM_LAUNCH_BOUND) inline void SimtComputeDimensionOne(
    __gm__ INDICES_T* idxGmAddr, __local_mem__ PARAMS_T* xLocalAddr, __gm__ PARAMS_T* outputGmAddr,
    uint32_t currUbTilingSize, TYPE_T xOffSet, TYPE_T sliceSize, TYPE_T indiceOffSet,
    TYPE_T varInAxis, TYPE_T magic, TYPE_T shift)
{
    for (uint32_t index = threadIdx.x; index < currUbTilingSize; index += blockDim.x) {
        TYPE_T globalIdx = xOffSet + index;
        TYPE_T currIndiceIdx = Simt::UintDiv(globalIdx, magic, shift);
        TYPE_T scatterAxisIdx = globalIdx - currIndiceIdx * sliceSize;
        INDICES_T idx = idxGmAddr[currIndiceIdx];

        if (idx >= 0 && idx < varInAxis) {
            TYPE_T dstOffet = idx * sliceSize + scatterAxisIdx;
            if constexpr (IsSameType<PARAMS_T, bool>::value) {
                PARAMS_T value = xLocalAddr[index];
                if constexpr (Mode == MODE_MAX) {
                    if (value > 0) {
                        outputGmAddr[dstOffet] = value;
                    }
                } else {
                    if (value <= 0) {
                        outputGmAddr[dstOffet] = value;
                    }
                }
            } else {
                if constexpr (Mode == MODE_MAX) {
                    Simt::AtomicMax(outputGmAddr + dstOffet, xLocalAddr[index]);
                } else {
                    Simt::AtomicMin(outputGmAddr + dstOffet, xLocalAddr[index]);
                }
            }
        }
    }
}

template <typename PARAMS_T, typename INDICES_T, typename TYPE_T, uint8_t Mode>
class ScatterNdCommonSimt
{
public:
    __aicore__ inline ScatterNdCommonSimt(const ScatterNdCommonSimtTilingData& tilingData, TPipe& pipe)
        : pipe_(pipe), tiling_(tilingData){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR indices, GM_ADDR updates, GM_ADDR y);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ComputeData();
    __aicore__ inline void CopyIn(LocalTensor<INDICES_T>& idxLocal, LocalTensor<PARAMS_T>& xLocal);
    __aicore__ inline void ComputeDimensionOther();
    __aicore__ inline void ComputeDimensionOne();
    __aicore__ inline void CopyInUpdate(LocalTensor<PARAMS_T>& xLocal);

private:
    TPipe& pipe_;
    const ScatterNdCommonSimtTilingData& tiling_;
    GlobalTensor<INDICES_T> idxGm;
    GlobalTensor<PARAMS_T> xGm;
    GlobalTensor<PARAMS_T> outputGm;

    TQue<QuePosition::VECIN, DOUBLE_BUFFER> inQueIdx, inQueX;
    TBuf<TPosition::VECCALC> strideListBuf;
    TBuf<TPosition::VECCALC> outputShapeBuf;

    uint32_t blockIdx_;
    TYPE_T currBlockTilingSize_ = 0; // 当前核中要更新的update元素个数

    uint32_t ubTilingSize_ = 0;
    uint32_t currUbTilingSize_ = 0; // 当前ub循环处理的update元素个数

    TYPE_T xBlockOffSet_ = 0;
    TYPE_T xOffSet_ = 0; // 当前ub循环update的GM偏移
    TYPE_T indiceBlockOffSet_ = 0;
    TYPE_T indiceOffSet_ = 0;
    uint32_t currIdxTilingSize_ = 0;
    TYPE_T ubLoopCnt_ = 0;
};

template <typename PARAMS_T, typename INDICES_T, typename TYPE_T, uint8_t Mode>
__aicore__ inline void ScatterNdCommonSimt<PARAMS_T, INDICES_T, TYPE_T, Mode>::Init(
    GM_ADDR x, GM_ADDR indices, GM_ADDR updates, GM_ADDR y)
{
    if (tiling_.sliceSize == 0) {
        return;
    }

    blockIdx_ = GetBlockIdx();

    this->xBlockOffSet_ = tiling_.blockTilingSize * blockIdx_; // update的GM偏移
    // calculate indice offset size by x offset size
    this->indiceBlockOffSet_ = this->xBlockOffSet_ / tiling_.sliceSize * tiling_.rankSize; // 索引的偏移

    if (blockIdx_ == tiling_.blockNum - 1) {
        this->currBlockTilingSize_ = tiling_.tailBlockTilingSize;
    } else {
        this->currBlockTilingSize_ = tiling_.blockTilingSize;
    }

    this->ubTilingSize_ = tiling_.ubTilingSize; // ub 中要处理的update元素个数
    if (this->currBlockTilingSize_ <= tiling_.ubTilingSize) {
        this->ubTilingSize_ = this->currBlockTilingSize_;
    }

    auto indiceUbTilingSize = (this->ubTilingSize_ + tiling_.sliceSize - 1) / tiling_.sliceSize * tiling_.rankSize; // ub 中indices元素个数
    idxGm.SetGlobalBuffer((__gm__ INDICES_T*)indices);
    xGm.SetGlobalBuffer((__gm__ PARAMS_T*)updates);
    outputGm.SetGlobalBuffer((__gm__ PARAMS_T*)y);

    pipe_.InitBuffer(inQueX, DOUBLE_BUFFER, ROUND_UP32(this->ubTilingSize_ * sizeof(PARAMS_T)));
    if (tiling_.rankSize >= INDICE_RANK_TWO) {
        pipe_.InitBuffer(inQueIdx, DOUBLE_BUFFER, ROUND_UP32(indiceUbTilingSize * sizeof(INDICES_T)));
        pipe_.InitBuffer(strideListBuf, MAX_RANK_COUNT * sizeof(TYPE_T));
        pipe_.InitBuffer(outputShapeBuf, MAX_SHAPE_RANK * sizeof(TYPE_T));
    }
}

template <typename PARAMS_T, typename INDICES_T, typename TYPE_T, uint8_t Mode>
__aicore__ inline void ScatterNdCommonSimt<PARAMS_T, INDICES_T, TYPE_T, Mode>::Process()
{
    if (blockIdx_ < tiling_.blockNum) {
        this->ubLoopCnt_ = (this->currBlockTilingSize_ + this->ubTilingSize_ - 1) / this->ubTilingSize_;
        for (TYPE_T idx = 0; idx < this->ubLoopCnt_ - 1; idx++) {
            this->currUbTilingSize_ = this->ubTilingSize_;
            this->xOffSet_ = this->xBlockOffSet_ + idx * this->ubTilingSize_;
            ComputeData();
        }
        this->xOffSet_ = this->xBlockOffSet_ + (this->ubLoopCnt_ - 1) * this->ubTilingSize_;
        this->currUbTilingSize_ = this->currBlockTilingSize_ - this->ubTilingSize_ * (this->ubLoopCnt_ - 1);
        ComputeData();
    }
}

template <typename PARAMS_T, typename INDICES_T, typename TYPE_T, uint8_t Mode>
__aicore__ inline void ScatterNdCommonSimt<PARAMS_T, INDICES_T, TYPE_T, Mode>::ComputeData()
{
    auto currEnd = this->xOffSet_ + this->currUbTilingSize_;
    auto indiceBegin = this->xOffSet_ / tiling_.sliceSize * tiling_.rankSize;
    auto indiceEnd = (currEnd + tiling_.sliceSize - 1) / tiling_.sliceSize * tiling_.rankSize;
    this->currIdxTilingSize_ = indiceEnd - indiceBegin; // 左闭右开
    this->indiceOffSet_ = indiceBegin;
    if (tiling_.rankSize >= INDICE_RANK_TWO) {
        ComputeDimensionOther();
    } else {
        ComputeDimensionOne();
    }
}

template <typename PARAMS_T, typename INDICES_T, typename TYPE_T, uint8_t Mode>
__aicore__ inline void ScatterNdCommonSimt<PARAMS_T, INDICES_T, TYPE_T, Mode>::ComputeDimensionOther()
{
    LocalTensor<INDICES_T> idxLocal = inQueIdx.AllocTensor<INDICES_T>();
    LocalTensor<PARAMS_T> xLocal = inQueX.AllocTensor<PARAMS_T>();
    CopyIn(idxLocal, xLocal);
    uint32_t currUbTilingSize = this->currUbTilingSize_;
    TYPE_T sliceSize = tiling_.sliceSize;
    uint32_t rankSize = tiling_.rankSize;

    LocalTensor<TYPE_T> strideList = strideListBuf.Get<TYPE_T>();
    LocalTensor<TYPE_T> outputShape = outputShapeBuf.Get<TYPE_T>();
    for (uint32_t i = 0; i < MAX_RANK_COUNT; i++) {
        strideList(i) = tiling_.strideList[i];
    }
    for (uint32_t i = 0; i < MAX_SHAPE_RANK; i++) {
        outputShape(i) = tiling_.outPutShape[i];
    }
    DataSyncBarrier<MemDsbT::UB>(); //
    TYPE_T magic = 0;
    TYPE_T shift = 0;
    GetUintDivMagicAndShift(magic, shift, sliceSize);
    asc_vf_call<SimtCompute<INDICES_T, PARAMS_T, TYPE_T, Mode>>(
        dim3(THREAD_NUM), (__local_mem__ INDICES_T*)idxLocal.GetPhyAddr(), (__local_mem__ PARAMS_T*)xLocal.GetPhyAddr(),
        (__gm__ PARAMS_T*)(outputGm.GetPhyAddr()), (__local_mem__ TYPE_T*)strideList.GetPhyAddr(),
        (__local_mem__ TYPE_T*)outputShape.GetPhyAddr(), currUbTilingSize, this->xOffSet_, sliceSize, rankSize, this->indiceOffSet_,
        magic, shift);

    inQueIdx.FreeTensor(idxLocal);
    inQueX.FreeTensor(xLocal);
}

template <typename PARAMS_T, typename INDICES_T, typename TYPE_T, uint8_t Mode>
__aicore__ inline void ScatterNdCommonSimt<PARAMS_T, INDICES_T, TYPE_T, Mode>::ComputeDimensionOne()
{
    LocalTensor<PARAMS_T> xLocal = inQueX.AllocTensor<PARAMS_T>();
    CopyInUpdate(xLocal);
    uint32_t currUbTilingSize = this->currUbTilingSize_;
    TYPE_T sliceSize = tiling_.sliceSize;
    TYPE_T varInAxis = tiling_.varInAxis;
    
    TYPE_T magic = 0;
    TYPE_T shift = 0;
    GetUintDivMagicAndShift(magic, shift, sliceSize);
    asc_vf_call<SimtComputeDimensionOne<INDICES_T, PARAMS_T, TYPE_T, Mode>>(
        dim3(THREAD_NUM), (__gm__ INDICES_T*)(idxGm.GetPhyAddr()), (__local_mem__ PARAMS_T*)xLocal.GetPhyAddr(),
        (__gm__ PARAMS_T*)(outputGm.GetPhyAddr()), currUbTilingSize, this->xOffSet_, sliceSize, this->indiceOffSet_,
        varInAxis, magic, shift);
    inQueX.FreeTensor(xLocal);
}

template <typename PARAMS_T, typename INDICES_T, typename TYPE_T, uint8_t Mode>
__aicore__ inline void ScatterNdCommonSimt<PARAMS_T, INDICES_T, TYPE_T, Mode>::CopyIn(
    LocalTensor<INDICES_T>& idxLocal, LocalTensor<PARAMS_T>& xLocal)
{
    DataCopyExtParams idxCopyParams{1, static_cast<uint32_t>(this->currIdxTilingSize_ * sizeof(INDICES_T)), 0, 0, 0};
    DataCopyPadExtParams<INDICES_T> idxPadParams{false, 0, 0, 0};
    DataCopyPad(idxLocal, idxGm[this->indiceOffSet_], idxCopyParams, idxPadParams);

    DataCopyExtParams xCopyParams{1, static_cast<uint32_t>(this->currUbTilingSize_ * sizeof(PARAMS_T)), 0, 0, 0};
    DataCopyPadExtParams<PARAMS_T> xPadParams{false, 0, 0, 0};
    DataCopyPad(xLocal, xGm[this->xOffSet_], xCopyParams, xPadParams);

    inQueIdx.EnQue(idxLocal);
    inQueX.EnQue(xLocal);
    inQueIdx.DeQue<INDICES_T>();
    inQueX.DeQue<PARAMS_T>();
}

template <typename PARAMS_T, typename INDICES_T, typename TYPE_T, uint8_t Mode>
__aicore__ inline void ScatterNdCommonSimt<PARAMS_T, INDICES_T, TYPE_T, Mode>::CopyInUpdate(LocalTensor<PARAMS_T>& xLocal)
{
    DataCopyExtParams xCopyParams{1, static_cast<uint32_t>(this->currUbTilingSize_ * sizeof(PARAMS_T)), 0, 0, 0};
    DataCopyPadExtParams<PARAMS_T> xPadParams{false, 0, 0, 0};
    DataCopyPad(xLocal, xGm[this->xOffSet_], xCopyParams, xPadParams);

    inQueX.EnQue(xLocal);
    inQueX.DeQue<PARAMS_T>();
}

} // namespace ScatterNdCommon
#endif // SCATTER_ND_COMMON_SIMT_H