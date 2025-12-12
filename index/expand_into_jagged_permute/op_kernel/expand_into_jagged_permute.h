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
 * \file expand_into_jagged_permute.h
 * \brief
 */

#ifndef EXPAND_INTO_JAGGED_PERMUTE_H
#define EXPAND_INTO_JAGGED_PERMUTE_H

#include "kernel_operator.h"
using namespace AscendC;

template <typename T, bool T0> class ExpandIntoJaggedPermute{
public:
    __aicore__ inline ExpandIntoJaggedPermute(){};
    __aicore__ inline void Init(GM_ADDR permute, GM_ADDR input_offsets, GM_ADDR output_offsets, GM_ADDR output_permute,
        const ExpandIntoJaggedPermuteTilingData *__restrict tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void initRange();
    __aicore__ inline void CopyInPermute(int64_t copyOffsetLen, int64_t nowoutputOffsets);
    __aicore__ inline void CopyInOffset(int64_t copyOffsetLen, int64_t nowoutputOffsets);
    __aicore__ inline void CopyOut(const int64_t outoffsetStart, const int64_t nowlen);

    __aicore__ inline int64_t Ceil(int64_t a, int64_t b) {
        if (b == 0) {
          return 0;
        }
        return (a + b - 1) / b;
      }

    static constexpr uint64_t bufferNum = 1;
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> permuteQueue, inputOffsetsQueue, outputOffsetQueue, outputOffsetNextQueue;
    TQue<QuePosition::VECOUT, 1> outputPermuteQue;

    TBuf<TPosition::VECCALC> rangeQueue;
    
    GlobalTensor<T> permuteGm, inputOffsetGm, outputOffsetGm, outputPermuteGm;
    LocalTensor<T> inputOffsetLocal, rangeLocal;

    int64_t blockIdx{0};
    int64_t realCoreNum{0};
    int64_t frontCoreNum{0};
    int64_t blockFactor{0};
    int64_t tailCoreBlockFactor{0};
    int64_t lastTaskLen{0};
    int64_t oneTaskLen{0};
    int64_t oneTaskOffsetLen{0};
    int64_t inputLen{0};
    int64_t outputSize{0};
    int64_t offsetLen{0};
    int64_t coreTaskNum{0};
    int64_t coreLen{0};
    int64_t sCoreOffset{0};
    int64_t outoffsetStart{0};
    int64_t ubOffset{0};
    int64_t nowlen{0};
};


template <typename T, bool T0>
__aicore__ inline void ExpandIntoJaggedPermute<T, T0>::Init(GM_ADDR permute, GM_ADDR input_offsets, GM_ADDR output_offsets, GM_ADDR output_permute,
                                            const ExpandIntoJaggedPermuteTilingData *__restrict tilingData) {
  int64_t blockNum = GetBlockNum();
  this->blockIdx = GetBlockIdx();
  this->realCoreNum = GetBlockNum();
  this->frontCoreNum = tilingData->frontCoreNum;
  this->blockFactor = tilingData->blockFactor;
  this->tailCoreBlockFactor = tilingData->tailCoreBlockFactor;
  this->lastTaskLen = tilingData->lastTaskLen;
  this->oneTaskLen = tilingData->oneTaskLen;
  this->oneTaskOffsetLen = tilingData->oneTaskOffsetLen;
  this->inputLen = tilingData->inputLen;
  this->outputSize = tilingData->outputSize;
  this->offsetLen = tilingData->offsetLen;
  this->coreTaskNum = this->blockFactor;

  coreLen = (blockIdx == this->realCoreNum - 1) ?
            (coreTaskNum - 1) * oneTaskLen + lastTaskLen : coreTaskNum * oneTaskLen;
  outoffsetStart = sCoreOffset;
  ubOffset = sCoreOffset;
  this->pipe.InitBuffer(rangeQueue, this->oneTaskLen * sizeof(T));
  this->pipe.InitBuffer(permuteQueue, bufferNum, this->oneTaskOffsetLen * sizeof(T));
  this->pipe.InitBuffer(inputOffsetsQueue, bufferNum, this->offsetLen * sizeof(T));
  this->pipe.InitBuffer(outputOffsetQueue, bufferNum, this->oneTaskOffsetLen * sizeof(T));
  this->pipe.InitBuffer(outputOffsetNextQueue, bufferNum, this->oneTaskOffsetLen * sizeof(T));
  this->pipe.InitBuffer(outputPermuteQue, bufferNum, this->oneTaskLen * sizeof(T));

  this->permuteGm.SetGlobalBuffer((__gm__ T *)permute);
  this->inputOffsetGm.SetGlobalBuffer((__gm__ T *)input_offsets);
  this->outputOffsetGm.SetGlobalBuffer((__gm__ T *)output_offsets);
  this->outputPermuteGm.SetGlobalBuffer((__gm__ T *)output_permute);
} 

template <typename T, bool T0>
__aicore__ inline void ExpandIntoJaggedPermute<T, T0>::CopyInPermute(int64_t copyOffsetLen, int64_t nowoutputOffsets) {
  DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(copyOffsetLen * sizeof(T)), 0, 0, 0};
  DataCopyPadExtParams<T> dataCopyPadParams{false, 0, 0, 0};
  inputOffsetLocal = inputOffsetsQueue.AllocTensor<T>();
  DataCopyPad(inputOffsetLocal, inputOffsetGm[nowoutputOffsets], dataCopyParams, dataCopyPadParams);
  inputOffsetsQueue.EnQue(inputOffsetLocal);
}

template <typename T, bool T0>
__aicore__ inline void ExpandIntoJaggedPermute<T, T0>::CopyInOffset(int64_t copyOffsetLen, int64_t nowoutputOffsets) {
  auto outputOffsetLocal = outputOffsetQueue.AllocTensor<T>();
  DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(copyOffsetLen * sizeof(T)), 0, 0, 0};
  DataCopyPadExtParams<T> dataCopyPadParams{false, 0, 0, 0};
  DataCopyPad(outputOffsetLocal, outputOffsetGm[nowoutputOffsets], dataCopyParams, dataCopyPadParams);
  outputOffsetQueue.EnQue(outputOffsetLocal);
  DataCopyExtParams dataCopyNextParams{1, static_cast<uint32_t>(copyOffsetLen * sizeof(T)), 0, 0, 0}; 
  auto outputOffsetNextLocal = outputOffsetNextQueue.AllocTensor<T>();
  DataCopyPad(outputOffsetNextLocal, outputOffsetGm[nowoutputOffsets + 1], dataCopyNextParams, dataCopyPadParams);
  outputOffsetNextQueue.EnQue(outputOffsetNextLocal);
  auto permuteLocal = permuteQueue.AllocTensor<T>();
  DataCopyPad(permuteLocal, permuteGm[nowoutputOffsets], dataCopyParams, dataCopyPadParams);
  permuteQueue.EnQue(permuteLocal);
}

template <typename T, bool T0>
__aicore__ inline void ExpandIntoJaggedPermute<T, T0>::initRange() {  
  rangeLocal = rangeQueue.Get<T>();
  CreateVecIndex(rangeLocal,T(0),oneTaskLen);
}

template <typename T, bool T0>
__aicore__ inline void ExpandIntoJaggedPermute<T, T0>::CopyOut(const int64_t outoffsetStart,
                                                                           const int64_t nowlen)
{
  auto outputPermuteLocal = outputPermuteQue.DeQue<T>();  
  DataCopyExtParams copyParams{1, static_cast<uint32_t>(this->nowlen * sizeof(T)), 0, 0, 0}; 
  DataCopyPad(outputPermuteGm[outoffsetStart], outputPermuteLocal, copyParams); 

  outputPermuteQue.FreeTensor<T>(outputPermuteLocal); 
}

template <typename T, bool T0>
__aicore__ inline void ExpandIntoJaggedPermute<T, T0>::Process() {
  int64_t outoffset = sCoreOffset;
  int64_t outoffsetEnd = 0;
  int64_t nowoutputOffsets = 0;
  int64_t nowoutputOffsete = 0;
  int64_t taskidA = 0;
  int64_t blockid = 0;
  int64_t UBoffsetId = 0;

  initRange();
  if constexpr (T0 == false){
    CopyInPermute(offsetLen, 0); 
    inputOffsetLocal = inputOffsetsQueue.DeQue<T>();
  }

  for (int32_t tokensId = 0; tokensId < this->coreTaskNum; tokensId++) {
    int64_t copyOffsetLen = (inputLen - nowoutputOffsets) < oneTaskOffsetLen ? 
                            (inputLen - nowoutputOffsets) :
                            oneTaskOffsetLen;                   
    CopyInOffset(copyOffsetLen, nowoutputOffsets);
    nowoutputOffsets +=  copyOffsetLen; 
  
    auto outputOffsetLocal = outputOffsetQueue.DeQue<T>();
    auto outputOffsetNextLocal = outputOffsetNextQueue.DeQue<T>();
    Sub(outputOffsetNextLocal, outputOffsetNextLocal, outputOffsetLocal, copyOffsetLen);
    auto permuteLocal = permuteQueue.DeQue<T>();
    
    UBoffsetId = 0;
    outoffsetStart = outputOffsetLocal.GetValue(UBoffsetId);   
    for(UBoffsetId = 0; UBoffsetId < copyOffsetLen; UBoffsetId++){
      int64_t outLen = outputOffsetNextLocal.GetValue(UBoffsetId);
      int64_t permuteId = permuteLocal.GetValue(UBoffsetId);
      int64_t inputoffset;
      if constexpr (T0 == false){
        inputoffset = inputOffsetLocal.GetValue(permuteId);
      }else{
        inputoffset = inputOffsetGm.GetValue(permuteId);
      }
      
      if(UBoffsetId == copyOffsetLen - 1){
        outputOffsetQueue.FreeTensor<T>(outputOffsetLocal);
        outputOffsetNextQueue.FreeTensor<T>(outputOffsetNextLocal);
        permuteQueue.FreeTensor<T>(permuteLocal);
      }
      int64_t tasklen = Ceil(outLen, this->oneTaskLen);  

      for(int32_t taskid = 0; taskid < tasklen; taskid++){
          if(taskid == tasklen-1){
            this->nowlen = outLen - this->oneTaskLen*(tasklen-1);
          }else{
            this->nowlen = this->oneTaskLen;
          }
        if (taskidA % this->realCoreNum == this->blockIdx){
            auto outputPermuteLocal = outputPermuteQue.AllocTensor<T>();
            Adds(outputPermuteLocal, rangeLocal, static_cast<int32_t>(inputoffset), oneTaskLen);
            outputPermuteQue.EnQue<T>(outputPermuteLocal);
            CopyOut(outoffsetStart, nowlen);
            taskidA += 1;
            inputoffset += this->nowlen;
            outoffsetStart += this->nowlen;
        }else{
          taskidA += 1;
          inputoffset += this->nowlen;
          outoffsetStart += this->nowlen;
        }
      }
    }
  }
  if constexpr (T0 == false){
    inputOffsetsQueue.FreeTensor<T>(inputOffsetLocal);
  }
}
#endif 
// EXPEND_INTO_JAGGED_PERMUTE_H