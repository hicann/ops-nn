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
 * \file scatter_nd.h
 * \brief
 */

#ifndef SCATTER_ND_SMIT_H
#define SCATTER_ND_SMIT_H

#include "kernel_operator.h"

namespace ScatterNdSimt {
using namespace AscendC;

constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t BLOCK_SIZE = 32;
#ifdef __DAV_FPGA__
constexpr uint32_t THREAD_NUM = 128;
constexpr uint32_t THREAD_NUM_LAUNCH_BOUND = 512;
#else
constexpr uint32_t THREAD_NUM = 1024;
constexpr uint32_t THREAD_NUM_LAUNCH_BOUND = 1024;
#endif
constexpr uint16_t MAX_RANK_COUNT = 7;
constexpr uint16_t MAX_SHAPE_RANK = 8;

__aicore__ inline uint32_t ROUND_UP32(uint32_t x)
{
  if (x % BLOCK_SIZE != 0) {
    return (x / BLOCK_SIZE + 1) * BLOCK_SIZE;
  }
  return x;
}

template <typename INDICES_T, typename PARAMS_T, typename TYPE_T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM_LAUNCH_BOUND) inline void SimtCompute(const TYPE_T loopNum,
    __ubuf__ INDICES_T *idxLocalAddr, __ubuf__ PARAMS_T *xLocalAddr, __gm__ PARAMS_T *outputGmAddr,
    const __ubuf__ TYPE_T *strideListAddr, const __ubuf__ TYPE_T *outputShapeAddr,
    const uint32_t currUbTilingSize, const TYPE_T xOffSet,
    const TYPE_T sliceSize, const uint32_t rankSize, const TYPE_T indiceOffSet, const TYPE_T magic, const TYPE_T shift)
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
      uint64_t dst = static_cast<uint64_t>(idx * sliceSize + scatterAxisIdx);
      Simt::AtomicAdd(outputGmAddr + dst, xLocalAddr[index]);
    }
  }
}

template <typename INDICES_T, typename PARAMS_T, typename TYPE_T>
class ScatterNd {
public:
  __aicore__ inline ScatterNd() {};
  __aicore__ inline void Init(GM_ADDR indices, GM_ADDR x, GM_ADDR shape, GM_ADDR y,
                              const ScatterNdTilingData* __restrict tilingData, TPipe* pipeIn);
  __aicore__ inline void Process(const ScatterNdTilingData* __restrict tilingData);

private:
  __aicore__ inline void ParseTilingData(const ScatterNdTilingData* __restrict tilingData);
  __aicore__ inline void CopyIn(LocalTensor<INDICES_T> &idxLocal,
                                LocalTensor<PARAMS_T> &xLocal);
  __aicore__ inline void SimdFree(LocalTensor<INDICES_T> &idxLocal, LocalTensor<PARAMS_T> &xLocal);

private:
  TPipe* pipe;
  GlobalTensor<INDICES_T> idxGm;
  GlobalTensor<PARAMS_T> xGm;
  GlobalTensor<PARAMS_T> clrGm, outputGm;

  TQue<QuePosition::VECIN, BUFFER_NUM> inQueIdx, inQueX;
  TBuf<TPosition::VECCALC> strideListBuf;
  TBuf<TPosition::VECCALC> outputShapeBuf;

  TYPE_T clrBlockNum = 0;
  TYPE_T clrBlockTilingSize = 0;
  TYPE_T currClrBlockTilingSize = 0;
  TYPE_T clrTailBlockTilingSize = 0;
  TYPE_T clrBlockOffSet = 0;
  TYPE_T blockNum = 0;
  uint32_t blockIdx;
  TYPE_T blockTilingSize = 0;
  TYPE_T currBlockTilingSize = 0;
  TYPE_T tailBlockTilingSize = 0;
  uint32_t ubTilingSize = 0;
  uint32_t currUbTilingSize = 0;
  uint32_t rankSize = 0;
  TYPE_T sliceSize = 0;
  TYPE_T xBlockOffSet = 0;
  TYPE_T xOffSet = 0;
  TYPE_T indiceBlockOffSet = 0;
  TYPE_T indiceOffSet = 0;
  uint32_t currIdxTilingSize = 0;
  TYPE_T ubLoopCnt = 0;
};

template <typename INDICES_T, typename PARAMS_T, typename TYPE_T>
__aicore__ inline void ScatterNd<INDICES_T, PARAMS_T, TYPE_T>::Init(GM_ADDR indices, GM_ADDR x, GM_ADDR shape, GM_ADDR y,
                                const ScatterNdTilingData* __restrict tilingData, TPipe* pipeIn)
{
  ParseTilingData(tilingData);
  blockIdx = GetBlockIdx();

  this->clrBlockOffSet = this->clrBlockTilingSize * blockIdx;
  this->xBlockOffSet = this->blockTilingSize * blockIdx;
  // calculate indice offset size by x offset size
  this->indiceBlockOffSet = this->xBlockOffSet / this->sliceSize * this->rankSize;

  if (blockIdx == blockNum - 1) {
    this->currBlockTilingSize = this->tailBlockTilingSize;
  } else {
    this->currBlockTilingSize = this->blockTilingSize;
  }
  if (blockIdx == clrBlockNum - 1) {
    this->currClrBlockTilingSize = this->clrTailBlockTilingSize;
  } else {
    this->currClrBlockTilingSize = this->clrBlockTilingSize;
  }

  if (this->currBlockTilingSize <= this->ubTilingSize) {
    this->ubTilingSize = this->currBlockTilingSize;
  }

  auto indiceUbTilingSize = (this->ubTilingSize + this->sliceSize - 1) / this->sliceSize * this-> rankSize;
  // add 2 indices  for the scenario where boundary values are not accessible.
  indiceUbTilingSize += 2;
  idxGm.SetGlobalBuffer((__gm__ INDICES_T *)indices);
  xGm.SetGlobalBuffer((__gm__ PARAMS_T *)x);
  clrGm.SetGlobalBuffer((__gm__ PARAMS_T *)y);
  outputGm.SetGlobalBuffer((__gm__ PARAMS_T *)y);
  pipe = pipeIn;
  pipe->InitBuffer(inQueIdx, BUFFER_NUM, ROUND_UP32(indiceUbTilingSize *sizeof (INDICES_T)));
  pipe->InitBuffer(inQueX, BUFFER_NUM, ROUND_UP32(this->ubTilingSize * sizeof(PARAMS_T)));
  pipe->InitBuffer(strideListBuf, MAX_RANK_COUNT * sizeof(TYPE_T));
  pipe->InitBuffer(outputShapeBuf, MAX_SHAPE_RANK * sizeof(TYPE_T));
}

template <typename INDICES_T, typename PARAMS_T, typename TYPE_T>
__aicore__ inline void ScatterNd<INDICES_T, PARAMS_T, TYPE_T>::Process(const ScatterNdTilingData* __restrict tilingData)
{
  if (blockIdx < clrBlockNum) {
    InitOutput<PARAMS_T>(clrGm[this->clrBlockOffSet], this->currClrBlockTilingSize, 0);
  }
  SyncAll();
  if (blockIdx < blockNum) {
    this->ubLoopCnt = (this->currBlockTilingSize + this->ubTilingSize - 1) / this->ubTilingSize;
    for (auto idx = 0; idx < this->ubLoopCnt; idx++) {
      this->currUbTilingSize = this->ubTilingSize;
      if ((idx == this->ubLoopCnt - 1) && (this->currBlockTilingSize % this->ubTilingSize != 0)) {
        this->currUbTilingSize = this->currBlockTilingSize % this->ubTilingSize;
      }
      this->xOffSet = this->xBlockOffSet + idx * this->ubTilingSize;
      auto currEnd = this->xOffSet + this->currUbTilingSize;
      auto indiceBegin = this->xOffSet / this->sliceSize * this->rankSize;
      auto indiceEnd = (currEnd + this->sliceSize - 1) / this->sliceSize * this->rankSize;
      this->currIdxTilingSize = indiceEnd - indiceBegin;
      this->indiceOffSet = indiceBegin;
      LocalTensor<INDICES_T> idxLocal = inQueIdx.AllocTensor<INDICES_T>();
      LocalTensor<PARAMS_T> xLocal = inQueX.AllocTensor<PARAMS_T>();
      CopyIn(idxLocal, xLocal);

      uint32_t currUbTilingSize = this->currUbTilingSize;
      TYPE_T xOffSet = this->xOffSet;
      TYPE_T sliceSize = this->sliceSize;
      uint32_t rankSize = this->rankSize;
      TYPE_T indiceOffSet = this->indiceOffSet;

      TYPE_T magic = 0;
      TYPE_T shift = 0;
      GetUintDivMagicAndShift(magic, shift, this->sliceSize);

      LocalTensor<TYPE_T> strideList = strideListBuf.Get<TYPE_T>();
      LocalTensor<TYPE_T> outputShape = outputShapeBuf.Get<TYPE_T>();
      for (uint32_t i = 0; i < MAX_RANK_COUNT; i++) {
        strideList.SetValue(i, tilingData->strideList[i]);
      }
      for (uint32_t i = 0; i < MAX_SHAPE_RANK; i++) {
        outputShape.SetValue(i, tilingData->outPutShape[i]);
      }
      DataSyncBarrier<MemDsbT::UB>();
      
      AscendC::Simt::VF_CALL<SimtCompute<INDICES_T, PARAMS_T, TYPE_T>>(
          Simt::Dim3(THREAD_NUM), idx, (__ubuf__ INDICES_T*)idxLocal.GetPhyAddr(),
          (__ubuf__ PARAMS_T*)xLocal.GetPhyAddr(), (__gm__ PARAMS_T*)(outputGm.GetPhyAddr()),
          (__ubuf__ TYPE_T *)strideList.GetPhyAddr(),
          (__ubuf__ TYPE_T *)outputShape.GetPhyAddr(),
          currUbTilingSize, xOffSet, sliceSize, rankSize, indiceOffSet, magic, shift);
      SimdFree(idxLocal, xLocal);
    }
  }
}

template <typename INDICES_T, typename PARAMS_T, typename TYPE_T>
__aicore__ inline void ScatterNd<INDICES_T, PARAMS_T, TYPE_T>::ParseTilingData(
                                  const ScatterNdTilingData* __restrict tilingData)
{
  this->clrBlockNum = tilingData->clrBlockNum;
  this->clrBlockTilingSize = tilingData->clrBlockTilingSize;
  this->clrTailBlockTilingSize = tilingData->clrTailBlockTilingSize;
  this->blockNum = tilingData->blockNum;
  this->blockTilingSize = tilingData->blockTilingSize;
  this->tailBlockTilingSize = tilingData->tailBlockTilingSize;
  this->ubTilingSize = tilingData->ubTilingSize;
  this->rankSize = tilingData->rankSize;
  this->sliceSize = tilingData->sliceSize;
}

template <typename INDICES_T, typename PARAMS_T, typename TYPE_T>
__aicore__ inline void ScatterNd<INDICES_T, PARAMS_T, TYPE_T>::CopyIn(LocalTensor<INDICES_T> &idxLocal,
                                                              LocalTensor<PARAMS_T> &xLocal)
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

template <typename INDICES_T, typename PARAMS_T, typename TYPE_T>
__aicore__ inline void ScatterNd<INDICES_T, PARAMS_T, TYPE_T>::SimdFree(LocalTensor<INDICES_T> &idxLocal,
                        LocalTensor<PARAMS_T> &xLocal)
{
  inQueIdx.FreeTensor(idxLocal);
  inQueX.FreeTensor(xLocal);
}

} //namespace ScatterNdSimt

#endif //SCATTER_ND_SMIT_H