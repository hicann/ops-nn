/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file avg_pool3_d_ndhwc_split_w.h
 * \brief
 */

#ifndef AVG_POOL3D_NDHWC_SPLIT_W_H_
#define AVG_POOL3D_NDHWC_SPLIT_W_H_

#include "kernel_operator.h"
#include "avg_pool3d_common.h"

namespace AvgPool3d {
template <typename T, int32_t QUEUE_DEPTH>
class KernelAvgPool3dSplitW {
public:
    __aicore__ inline KernelAvgPool3dSplitW() {}
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const AvgPool3DTilingData* tiling, TPipe* pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitTiling(const AvgPool3DTilingData* tiling);
    __aicore__ inline void CopyIn(int64_t offset, uint16_t blockCount, uint32_t blockLen, uint8_t rightPadding);
    __aicore__ inline void CopyOut(int64_t offset, int64_t len);
    __aicore__ inline void DataCopyOutNonPad(
        LocalTensor<T>& outputLocal, int64_t outputPointIdx, uint32_t validDataLen);
    __aicore__ inline void ReduceMeanWindow(int64_t outputPointIdx);
    __aicore__ inline void ReduceSumWindow(const Index& index, LocalTensor<float>& sumBufLocal, int64_t nOffset);

    TPipe* pipe;
    TQue<QuePosition::VECIN, QUEUE_DEPTH> inputQueue;
    TQue<QuePosition::VECOUT, QUEUE_DEPTH> outputQueue;

    TBuf<QuePosition::VECCALC> tmpPattern;
    TBuf<TPosition::VECCALC> sumBuf;
    LocalTensor<float> sumBufLocal;

    GlobalTensor<T> inputGlobal;
    GlobalTensor<T> outputGlobal;

    int64_t inC;
    int64_t alignC;
    int64_t outputPointNum;
    int64_t outputPointOffset;
    int64_t lastPointOffset;
    int64_t tileInput;
    int64_t atomicAddNum;

    PoolShape inputShape;
    PoolShape outputShape;

    int64_t indexBufLen;
    IndexBuffer indexBuf;
    PoolParameter poolParam;

    uint32_t numPerBlock;
    uint32_t inputBufLen;
    int32_t validTailLen;

    TQue<QuePosition::VECIN, QUEUE_DEPTH> syncWorkQueue;
    GlobalTensor<int32_t> syncTensorsGM;
    TBuf<TPosition::VECCALC> clearTensorBuff;
    uint32_t usedCoreNum;
};

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAvgPool3dSplitW<T, QUEUE_DEPTH>::InitTiling(const AvgPool3DTilingData* tiling) {
    inputShape = PoolShape(tiling->inN, tiling->inC, tiling->inD, tiling->inH, tiling->inW);
    outputShape = PoolShape(tiling->inN, tiling->inC, tiling->outD, tiling->outH, tiling->outW);

    poolParam = PoolParameter(tiling->kD, tiling->kH, tiling->kW, tiling->dD, tiling->dH, tiling->dW,
                              tiling->pD, tiling->pH, tiling->pW, tiling->divisorOverride, tiling->countIncludePad);

    indexBuf.SetComputeParameter(outputShape, inputShape, poolParam);

    numPerBlock = GetDataBlockSizeInBytes() / sizeof(T);
    inC = tiling->inC;
    alignC = AlignUp(inC, numPerBlock);
    tileInput = tiling->tileInput;

    outputPointNum = GetBlockIdx() < tiling->formerNum ? tiling->formerLength : tiling->tailLength;
    outputPointOffset = GetBlockIdx() < tiling->formerNum
        ? tiling->formerLength * GetBlockIdx()
        : tiling->formerNum * tiling->formerLength + tiling->tailLength * (GetBlockIdx() - tiling->formerNum);
    lastPointOffset = outputPointNum + outputPointOffset - 1;
    atomicAddNum = outputPointNum < tiling->atomicAddNum ? outputPointNum : tiling->atomicAddNum;
    indexBufLen = tiling->indexBufLen;
    validTailLen = inC % numPerBlock;
    usedCoreNum = tiling->usedCoreNum;
}

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAvgPool3dSplitW<T, QUEUE_DEPTH>::CopyIn(
    int64_t offset, uint16_t blockCount, uint32_t blockLen, uint8_t rightPadding) {
    LocalTensor<T> inputLocal = inputQueue.template AllocTensor<T>();
#if __CCE_AICORE__ < 220
    if constexpr (std::is_same_v<T, float>) {
        if (blockLen == alignC) {
            DataCopyParams copyParams{blockCount, static_cast<uint16_t>(blockLen / numPerBlock), 0, 0};
            DataCopy(inputLocal, inputGlobal[offset], copyParams);
        } else {
            for (int i = 0; i < blockCount; i++) {
                DataCopy(inputLocal[i * alignC], inputGlobal[offset + i * blockLen], alignC);
            }
        }
    } else {
        if (blockLen == alignC) {
            DataCopyParams copyParams{blockCount, static_cast<uint16_t>(blockLen / numPerBlock), 0, 0};
            DataCopy(inputLocal[inputBufLen], inputGlobal[offset], copyParams);
        } else {
            for (int i = 0; i < blockCount; i++) {
                DataCopy(inputLocal[inputBufLen + i * alignC], inputGlobal[offset + i * blockLen], alignC);
            }
        }
    }
#else
    DataCopyExtParams copyParams{blockCount, static_cast<uint32_t>(blockLen * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{true, 0, rightPadding, 0};
    if constexpr (std::is_same_v<T, float>) {
        DataCopyPad(inputLocal, inputGlobal[offset], copyParams, padParams);
    } else {
        DataCopyPad(inputLocal[inputBufLen], inputGlobal[offset], copyParams, padParams);
    }
#endif
    inputQueue.EnQue(inputLocal);
}

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAvgPool3dSplitW<T, QUEUE_DEPTH>::DataCopyOutNonPad(
    LocalTensor<T>& outputLocal, int64_t outputPointIdx, uint32_t validDataLen) {
    if ((validDataLen < numPerBlock) && (outputPointIdx >= lastPointOffset - atomicAddNum)) {
      uint64_t mask0 = (1ul << numPerBlock) - (1ul << validDataLen);
      uint64_t mask[2] = {mask0, 0};
      Duplicate<T>(outputLocal, 0, mask, 1, 1, 1);
      VToMTE3Sync();
      if (outputPointIdx > lastPointOffset - atomicAddNum) {
          SetAtomicAdd<T>();
          DataCopy(outputGlobal[outputPointIdx * validDataLen], outputLocal, alignC);
          SetAtomicNone();
          AscendC::PipeBarrier<PIPE_MTE3>();
      } else {
          DataCopy(outputGlobal[outputPointIdx * validDataLen], outputLocal, alignC);
      }
    } else if (outputPointIdx == lastPointOffset) {
        DataCopy(outputGlobal[outputPointIdx * validDataLen], outputLocal, alignC - numPerBlock);
        int32_t lastLeftShift = validTailLen;
        uint32_t mask = numPerBlock * 2;
        uint64_t rsvdCnt = 0;
        uint64_t gatherOffset = alignC - mask;
        MTE3ToVSync();
        if constexpr (std::is_same_v<T, float>) {
            LocalTensor<uint32_t> bufPattern = tmpPattern.Get<uint32_t>();
            int32_t preLeftShift = numPerBlock + lastLeftShift;

            bufPattern.SetValue(0, (1u << preLeftShift) - (1u << lastLeftShift));
            SToVSync();
            GatherMask(outputLocal[gatherOffset], outputLocal[gatherOffset], bufPattern, true, mask, {1, 1, 8, 8}, rsvdCnt);
        } else {
            LocalTensor<uint16_t> bufPattern = tmpPattern.Get<uint16_t>();
            int32_t preLeftShift = numPerBlock - lastLeftShift;

            bufPattern.SetValue(0, ((1u << preLeftShift) - 1u) << lastLeftShift);
            bufPattern.SetValue(1, (1u << lastLeftShift) - 1u);
            SToVSync();
            GatherMask(outputLocal[gatherOffset], outputLocal[gatherOffset], bufPattern, true, mask, {1, 1, 8, 8}, rsvdCnt);
        }
        VToMTE3Sync();
        DataCopy(outputGlobal[(outputPointIdx + 1) * validDataLen - numPerBlock], outputLocal[gatherOffset], numPerBlock);
    } else {
        DataCopy(outputGlobal[outputPointIdx * validDataLen], outputLocal, alignC);
    }
}

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAvgPool3dSplitW<T, QUEUE_DEPTH>::CopyOut(int64_t offset, int64_t len) {
    LocalTensor<T> outputLocal = outputQueue.template DeQue<T>();
#if __CCE_AICORE__ < 220
    if (len == alignC) {
        DataCopyParams copyParams{1, static_cast<uint16_t>(len / numPerBlock), 0, 0};
        DataCopy(outputGlobal[offset * len], outputLocal, copyParams);
    } else {
        DataCopyOutNonPad(outputLocal, offset, len);
    }
#else
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(len * sizeof(T)), 0, 0, 0};
    DataCopyPad(outputGlobal[offset * len], outputLocal, copyParams);
#endif
    outputQueue.FreeTensor(outputLocal);
}

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAvgPool3dSplitW<T, QUEUE_DEPTH>::ReduceSumWindow(
    const Index& index, LocalTensor<float>& sumBufLocal, int64_t nOffset) {
    int64_t dstart = index.D.start;
    int64_t dend = index.D.end;
    int64_t hstart = index.H.start;
    int64_t hend = index.H.end;
    int64_t wstart = index.W.start;
    int64_t wend = index.W.end;

    int64_t kW = (wend - wstart + tileInput - 1) / tileInput;
    uint8_t rightPadding = static_cast<uint8_t>(alignC - inC);

    for (int64_t id = dstart; id < dend; ++id) {
        int64_t dOffset = id * inputShape.strideD * inC;
        for (int64_t ih = hstart; ih < hend; ++ih) {
            int64_t hOffset = ih * inputShape.strideH * inC;
            for (int64_t j = 0, iw = wstart; j < kW; ++j) {
                int64_t tileNum = j < kW - 1 ? tileInput : wend - iw;

                CopyIn(nOffset * inputShape.strideN + dOffset + hOffset + iw * inC,
                      static_cast<uint16_t>(tileNum), static_cast<uint32_t>(inC), rightPadding);
                LocalTensor<T> inputLocal = inputQueue.template DeQue<T>();

                if constexpr (!std::is_same_v<T, float>) {
                    Cast(inputLocal.template ReinterpretCast<float>(), inputLocal[inputBufLen],
                         RoundMode::CAST_NONE, inputBufLen);
                }

                for (int64_t i = 0; i < tileNum; ++i) {
                    if constexpr (std::is_same_v<T, float>) {
                        Add(sumBufLocal, sumBufLocal, inputLocal[i * alignC], alignC);
                    } else {
                        Add(sumBufLocal, sumBufLocal, inputLocal.template ReinterpretCast<float>()[i * alignC], alignC);
                    }
                }

                iw += tileNum;

                inputQueue.FreeTensor(inputLocal);
            }
        }
    }
}

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAvgPool3dSplitW<T, QUEUE_DEPTH>::ReduceMeanWindow(int64_t outputPointIdx) {
    Index index;
    indexBuf.GetIndex(outputPointIdx, index);

    int64_t poolSize = poolParam.divisorOverride ?
                       poolParam.divisorOverride : index.D.poolSize * index.H.poolSize * index.W.poolSize;
    float factor = 1.0f / static_cast<float>(poolSize);

    SToVSync();

    Duplicate(sumBufLocal, 0.0f, alignC);

    ReduceSumWindow(index, sumBufLocal, outputPointIdx / outputShape.strideC);
    Muls(sumBufLocal, sumBufLocal, factor, alignC);

    LocalTensor<T> outputLocal = outputQueue.template AllocTensor<T>();
    if constexpr (std::is_same_v<T, float>) {
#if __CCE_AICORE__ < 220
        Adds(outputLocal, sumBufLocal, 0.0f, alignC);
#else
        DataCopy(outputLocal, sumBufLocal, alignC);
#endif
    } else if constexpr (std::is_same_v<T, half>) {
        Cast(outputLocal, sumBufLocal, RoundMode::CAST_NONE, alignC);
    } else {
        Cast(outputLocal, sumBufLocal, RoundMode::CAST_RINT, alignC);
    }
    outputQueue.EnQue(outputLocal);

    CopyOut(outputPointIdx, inC);
}

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAvgPool3dSplitW<T, QUEUE_DEPTH>::Init(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const AvgPool3DTilingData* tiling, TPipe* pipe) {
    InitTiling(tiling);

    inputGlobal.SetGlobalBuffer((__gm__ T*)x);
    outputGlobal.SetGlobalBuffer((__gm__ T*)y);

    inputBufLen = tileInput * alignC;
    pipe->InitBuffer(inputQueue, QUEUE_DEPTH, inputBufLen * sizeof(float));
    pipe->InitBuffer(outputQueue, QUEUE_DEPTH, alignC * sizeof(T));

    pipe->InitBuffer(sumBuf, alignC * sizeof(float));
    sumBufLocal = sumBuf.Get<float>();

    indexBuf.Init(pipe, indexBufLen);
#if __CCE_AICORE__ < 220
    if (atomicAddNum) {
        pipe->InitBuffer(tmpPattern, numPerBlock * sizeof(T));

        pipe->InitBuffer(syncWorkQueue, QUEUE_DEPTH, 8 * 32 * sizeof(int32_t));
        syncTensorsGM.SetGlobalBuffer((__gm__ int32_t *)workspace, usedCoreNum * 8 * 32);
        pipe->InitBuffer(clearTensorBuff, DEFAULT_CLEAR_UB_SIZE * sizeof(T));
    } else if (validTailLen != 0) {
        pipe->InitBuffer(tmpPattern, numPerBlock * sizeof(T));
    }
#endif
}

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void KernelAvgPool3dSplitW<T, QUEUE_DEPTH>::Process() {
#if __CCE_AICORE__ < 220
    if (atomicAddNum) {
        LocalTensor<T> clearUb = clearTensorBuff.Get<T>();
        Duplicate(clearUb, (T)0, DEFAULT_CLEAR_UB_SIZE);

        VToMTE3Sync();
        int64_t curOutputPointIdx = lastPointOffset;
        for (int i = 0; i < atomicAddNum; i++, curOutputPointIdx--) {
            DataCopy<T>(outputGlobal[curOutputPointIdx * inC], clearUb, numPerBlock);
        }

        DataCopy(syncTensorsGM[0], clearUb.template ReinterpretCast<int32_t>(), usedCoreNum * 8 * 32);
        LocalTensor<int32_t> syncLocalTensor = syncWorkQueue.template AllocTensor<int32_t>();
        AscendC::SyncAll(syncTensorsGM, syncLocalTensor, int32_t(usedCoreNum));
        syncWorkQueue.FreeTensor(syncLocalTensor);
    }
#endif
    for (int64_t outputPointIdx = outputPointOffset;
        outputPointIdx < outputPointOffset + outputPointNum; ++outputPointIdx) {
        ReduceMeanWindow(outputPointIdx);
    }
}

} // namespace AvgPool3d

#endif // AVG_POOL3D_NDHWC_SPLIT_W_H_
