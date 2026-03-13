/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ADD_LAYER_NORM_GRAD_DETERMINSTIC_COMPUTE
#define ADD_LAYER_NORM_GRAD_DETERMINSTIC_COMPUTE

#include "kernel_operator.h"
#include "add_layer_norm_grad_utils.h"

namespace AddLayerNormGrad {
using namespace AscendC;
constexpr int64_t DOUBLE_BUFFER = 2;
constexpr int64_t ROW_TEMPLATE = 180;
constexpr int64_t COL_TEMPLATE = 64;
constexpr int64_t UB_SIZE = 180 * 1024 + 2 * DOUBLE_BUFFER * COL_TEMPLATE * sizeof(float);
constexpr int64_t FLOAT_ALIGN = 8;
constexpr int64_t MAX_REPEAT_TIMES = 255;
constexpr uint64_t B32_REPEAT_STRIDE = 8;
constexpr uint64_t UINT16_MAX_VALUE = 65535;

class AddLayerNormGradDeterminsticCompute {
public:
  __aicore__ inline AddLayerNormGradDeterminsticCompute(){};
  __aicore__ inline void initBuffer(TPipe& pipe,GlobalTensor<float>& pdGammaOutTensorGM,GlobalTensor<float>& pdBetaOutTensorGM,GlobalTensor<float>& workspaceGM,int64_t workspaceNum) {
    pipe_ = pipe;
    pipe_.InitBuffer(queueGammaOut_, DOUBLE_BUFFER, COL_TEMPLATE * sizeof(float));
    pipe_.InitBuffer(queueGammaIn_, DOUBLE_BUFFER, ROW_TEMPLATE * COL_TEMPLATE * sizeof(float));
    pipe_.InitBuffer(queueBetaOut_, DOUBLE_BUFFER, COL_TEMPLATE * sizeof(float));
    pipe_.InitBuffer(queueBetaIn_, DOUBLE_BUFFER, ROW_TEMPLATE * COL_TEMPLATE * sizeof(float));
    pdBetaOutTensorGM_ = pdBetaOutTensorGM;
    pdGammaOutTensorGM_ = pdGammaOutTensorGM;
    workspaceNum_ = workspaceNum;
    workspaceGM_ = workspaceGM;
  }

  __aicore__ inline void FinalProcessDeterministic(int64_t tcolAlignV, int64_t tblockNum, int64_t tcol) {
    colAlignV_ = tcolAlignV;
    row_ = tblockNum;
    col_ = tcol;
    buffer1_ = queueGammaIn_.AllocTensor<float>();
    buffer2_ = queueGammaOut_.AllocTensor<float>();
    buffer3_ = queueBetaIn_.AllocTensor<float>();
    buffer4_ = queueBetaOut_.AllocTensor<float>();
    int64_t colCycleCount = (colAlignV_ + COL_TEMPLATE - 1) / COL_TEMPLATE;
    int64_t colCyclePerBlockCount = (colCycleCount + GetBlockNum() - 1) / GetBlockNum();
    int64_t rowCycleCount = (row_ + ROW_TEMPLATE - 1) / ROW_TEMPLATE;
    int64_t colSize = COL_TEMPLATE;
    int64_t rowSize = ROW_TEMPLATE;
    int64_t taskId = 0;
    for (int64_t blocktaskId = 0; blocktaskId < colCyclePerBlockCount; blocktaskId++) {
        taskId = blocktaskId * GetBlockNum() + GetBlockIdx();
        if (taskId < colCycleCount) {
            if (taskId == colCycleCount - 1) {
                colSize = col_ - COL_TEMPLATE * taskId;
            }
            for(int64_t i = 0; i < rowCycleCount; i++) {
                if (i == rowCycleCount - 1) {
                    rowSize = row_ - ROW_TEMPLATE * i;
                }
                copyIn(taskId,i,colSize,rowSize);
                compute(taskId,i,colSize,rowSize);
            }
            copyOut(taskId,colSize);
            rowSize = ROW_TEMPLATE;
        } else {
            break;
        }
    }
    queueGammaOut_.FreeTensor(buffer2_);
    queueGammaIn_.FreeTensor(buffer1_);
    queueBetaOut_.FreeTensor(buffer4_);
    queueBetaIn_.FreeTensor(buffer3_);
  }

  __aicore__ inline void copyIn(int64_t colIndex, int64_t rowIndex, int64_t colSize, int64_t rowSize) {
    uint8_t rightPad = 0;
    int64_t colSizeMod = colSize % FLOAT_ALIGN;
    if (colSizeMod != 0) {
        rightPad = FLOAT_ALIGN - colSizeMod;
    }
    
    int64_t srcStrideValue = (colAlignV_ * workspaceNum_ - colSize) / FLOAT_ALIGN;
    int64_t dstStrideValue = (COL_TEMPLATE - (colSize + rightPad)) / FLOAT_ALIGN;
    int64_t offset = (colIndex * COL_TEMPLATE) + rowIndex * colAlignV_ * workspaceNum_ * ROW_TEMPLATE;

#if __CCE_AICORE__ == 220 || __CCE_AICORE__ == 310
    if ASCEND_IS_AIV {
      copyInWithPad(offset, colSize, rowSize, rightPad, dstStrideValue);
    }
#else
    if (static_cast<uint64_t>(srcStrideValue) <= UINT16_MAX_VALUE && 
        static_cast<uint64_t>(dstStrideValue) <= UINT16_MAX_VALUE) {
      copyInWithParams(offset, colSize, rowSize, srcStrideValue, dstStrideValue);
    } else {
      copyInByRow(offset, colSize, rowSize, rightPad);
    }
#endif
  }

  __aicore__ inline void copyInWithPad(int64_t offset, int64_t colSize, int64_t rowSize, uint8_t rightPad, int64_t dstStrideValue) {
    TEventID eventID = GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>();
    SetFlag<HardEvent::V_MTE2>(eventID);
    WaitFlag<HardEvent::V_MTE2>(eventID);
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(eventID);
    
    DataCopyExtParams intriParams;
    intriParams.blockCount = static_cast<uint16_t>(rowSize);
    intriParams.blockLen = static_cast<uint32_t>(colSize * sizeof(float));
    intriParams.srcStride = static_cast<uint32_t>((colAlignV_ * workspaceNum_ - colSize) * sizeof(float));
    intriParams.dstStride = static_cast<uint32_t>(dstStrideValue);
    intriParams.rsv = 0;
    DataCopyPadExtParams<float> padParams;
    padParams.isPad = false;
    padParams.leftPadding = 0;
    padParams.rightPadding = rightPad;
    padParams.paddingValue = 0.0f;
    DataCopyPad(buffer1_, workspaceGM_[offset], intriParams, padParams);
    DataCopyPad(buffer3_, workspaceGM_[offset + colAlignV_], intriParams, padParams);
  }

  __aicore__ inline void copyInWithParams(int64_t offset, int64_t colSize, int64_t rowSize, int64_t srcStrideValue, int64_t dstStrideValue) {
    DataCopyParams intriParams;
    intriParams.blockCount = static_cast<uint16_t>(rowSize);
    intriParams.blockLen = static_cast<uint16_t>((colSize + FLOAT_ALIGN - 1) / FLOAT_ALIGN);
    intriParams.srcStride = static_cast<uint16_t>(srcStrideValue);
    intriParams.dstStride = static_cast<uint16_t>(dstStrideValue);
    TEventID eventID = GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>();
    SetFlag<HardEvent::V_MTE2>(eventID);
    WaitFlag<HardEvent::V_MTE2>(eventID);
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(eventID);
    DataCopy(buffer1_, workspaceGM_[offset], intriParams);
    DataCopy(buffer3_, workspaceGM_[offset + colAlignV_], intriParams);
  }

  __aicore__ inline void copyInByRow(int64_t baseOffset, int64_t colSize, int64_t rowSize, uint8_t rightPad) {
    int64_t rowStride = colAlignV_ * workspaceNum_;
    TEventID eventID = GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>();
    SetFlag<HardEvent::V_MTE2>(eventID);
    WaitFlag<HardEvent::V_MTE2>(eventID);
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(eventID);
    for (int64_t row = 0; row < rowSize; row++) {
      int64_t srcOffset = baseOffset + row * rowStride;
      int64_t dstOffset = row * COL_TEMPLATE;
      
      if (colSize % FLOAT_ALIGN == 0 && rightPad == 0) {
        DataCopy(buffer1_[dstOffset], workspaceGM_[srcOffset], colSize);
        DataCopy(buffer3_[dstOffset], workspaceGM_[srcOffset + colAlignV_], colSize);
      } else {
        DataCopyParams rowParams;
        rowParams.blockCount = 1;
        rowParams.blockLen = static_cast<uint16_t>((colSize + FLOAT_ALIGN - 1) / FLOAT_ALIGN);
        rowParams.srcStride = 0;
        rowParams.dstStride = static_cast<uint16_t>((COL_TEMPLATE - colSize - rightPad) / FLOAT_ALIGN);
        DataCopy(buffer1_[dstOffset], workspaceGM_[srcOffset], rowParams);
        DataCopy(buffer3_[dstOffset], workspaceGM_[srcOffset + colAlignV_], rowParams);
      }
    }
  }

  __aicore__ inline void compute(int64_t colIndex,int64_t rowIndex, int64_t colSize, int64_t rowSize) {
    int64_t colSizeMod = colSize % FLOAT_ALIGN;
    int64_t colSizeAlign = colSize;
    // 尾核补齐对齐
    if (colSizeMod != 0) {
        colSizeAlign += FLOAT_ALIGN - colSizeMod;
    }
    TEventID eventId = GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>();
    SetFlag<HardEvent::MTE2_V>(eventId);
    WaitFlag<HardEvent::MTE2_V>(eventId);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(eventId);
    TEventID eventId1 = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
    SetFlag<HardEvent::MTE3_V>(eventId1);
    WaitFlag<HardEvent::MTE3_V>(eventId1);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(eventId1);
    Duplicate(buffer2_, static_cast<float>(0.0), COL_TEMPLATE);
    Duplicate(buffer4_, static_cast<float>(0.0), COL_TEMPLATE);
    PipeBarrier<PIPE_V>();
    uint64_t mask = colSize;
    uint8_t repeatTimes = MAX_REPEAT_TIMES;
    BinaryRepeatParams binaryRepeatParams;
    binaryRepeatParams.dstBlkStride = 1;
    binaryRepeatParams.src0BlkStride = 1;
    binaryRepeatParams.src1BlkStride = 1;
    binaryRepeatParams.dstRepStride = 0;
    binaryRepeatParams.src0RepStride = B32_REPEAT_STRIDE;
    binaryRepeatParams.src1RepStride = 0;
    int64_t rowRepeatTimes = (rowSize + MAX_REPEAT_TIMES - 1) / MAX_REPEAT_TIMES;
    for (int64_t i = 0; i < rowRepeatTimes; i++) {
        if (i == rowRepeatTimes - 1) {
            repeatTimes = rowSize - (rowRepeatTimes - 1) * MAX_REPEAT_TIMES;
        }
        Add(buffer2_,buffer1_[i * MAX_REPEAT_TIMES],buffer2_,mask,repeatTimes,binaryRepeatParams);
        Add(buffer4_,buffer3_[i * MAX_REPEAT_TIMES],buffer4_,mask,repeatTimes,binaryRepeatParams);
        PipeBarrier<PIPE_V>();
    }
  }

  __aicore__ inline void copyOut(int64_t colIndex, int64_t colSize) {
#if __CCE_AICORE__ == 220 || __CCE_AICORE__ == 310
    if ASCEND_IS_AIV {
#endif
      int64_t offset = colIndex * COL_TEMPLATE;
      if(colSize == COL_TEMPLATE) {
        DataCopyParams intriParams;
        intriParams.blockCount = 1;
        intriParams.blockLen   = colSize / FLOAT_ALIGN;
        intriParams.srcStride  = 0;
        intriParams.dstStride  = 0;
        TEventID eventID = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
        SetFlag<HardEvent::V_MTE3>(eventID);
        WaitFlag<HardEvent::V_MTE3>(eventID);
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(eventID);
        DataCopy(pdGammaOutTensorGM_[offset], buffer2_, intriParams);
        DataCopy(pdBetaOutTensorGM_[offset], buffer4_, intriParams);
      } else {
        TEventID eventID = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
        SetFlag<HardEvent::V_MTE3>(eventID);
        WaitFlag<HardEvent::V_MTE3>(eventID);
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(eventID);
        SafeDataCopy(pdGammaOutTensorGM_[offset], buffer2_, colSize);
        SafeDataCopy(pdBetaOutTensorGM_[offset], buffer4_, colSize);
      }
#if __CCE_AICORE__ == 220 || __CCE_AICORE__ == 310
    }
#endif
  }

private:
  TPipe pipe_;
  TQue<QuePosition::VECOUT, DOUBLE_BUFFER> queueGammaOut_;
  TQue<QuePosition::VECOUT, DOUBLE_BUFFER> queueBetaOut_;
  TQue<QuePosition::VECIN, DOUBLE_BUFFER> queueBetaIn_;
  TQue<QuePosition::VECIN, DOUBLE_BUFFER> queueGammaIn_;
  LocalTensor<float> buffer1_;
  LocalTensor<float> buffer2_;
  LocalTensor<float> buffer3_;
  LocalTensor<float> buffer4_;

  GlobalTensor<float> pdGammaOutTensorGM_;
  GlobalTensor<float> pdBetaOutTensorGM_;
  GlobalTensor<float> workspaceGM_;

  int64_t workspaceNum_;
  int64_t colAlignV_;
  int64_t row_;
  int64_t col_;
};
}
#endif

