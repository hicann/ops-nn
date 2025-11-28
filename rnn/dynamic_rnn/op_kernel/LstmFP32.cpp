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
 * \file LstmFP32.cpp
 * \brief
 */
#include "LstmFP32.h"

using namespace AscendC;

template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP32<T>::ProcessInputMM()
{
    if (GetBlockIdx() < this->inputMMTiling.usedCoreNum) {
        this->inputMM.SetTensorA(this->inputGm.xGm[this->inputOffsets.AOffset]);
        this->inputMM.SetTensorB(this->inputGm.weightInputGm[this->inputOffsets.BOffset]);
        if (this->tiling->isBias == 1) {
            this->inputMM.SetBias(this->inputGm.biasGm[this->inputOffsets.BOffset]);
        }

        if (this->inputTail.nCoreIndx == this->inputTail.notTailNCoreCount &&
            this->inputTail.mCoreIndx == this->inputTail.notTailMCoreCount) {
            this->inputMM.SetTail(this->inputTail.tailSingleCoreM, this->inputTail.tailSingleCoreN);
        } else if (this->inputTail.nCoreIndx == this->inputTail.notTailNCoreCount) {
            this->inputMM.SetTail(this->inputMMTiling.singleCoreM, this->inputTail.tailSingleCoreN);
        } else if (this->inputTail.mCoreIndx == this->inputTail.notTailMCoreCount) {
            this->inputMM.SetTail(this->inputTail.tailSingleCoreM, this->inputMMTiling.singleCoreN);
        }
        this->inputMM.IterateAll(this->outputGm.workspace[this->inputOffsets.COffset], false);
    }
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP32<T>::ProcessHiddenMM(int64_t tIdx)
{
    if (GetBlockIdx() < this->hiddenMMTiling.usedCoreNum) {
        if (this->tiling->direction == 1) {
            this->hiddenOffsets.COffset =
                this->oriHiddenOffsets.COffset + (this->tiling->timeStep - 1 - tIdx) * this->allCellSize;
        } else {
            this->hiddenOffsets.COffset = this->oriHiddenOffsets.COffset + tIdx * this->allCellSize;
        }
        hiddenMM.SetTensorA(this->inputGm.initHGm[this->hiddenOffsets.AOffset]);
        hiddenMM.SetTensorB(this->inputGm.weightHiddenGm[this->hiddenOffsets.BOffset]);
        if (this->hiddenTail.nCoreIndx == this->hiddenTail.notTailNCoreCount &&
            this->hiddenTail.mCoreIndx == this->hiddenTail.notTailMCoreCount) {
            hiddenMM.SetTail(this->hiddenTail.tailSingleCoreM, this->hiddenTail.tailSingleCoreN);
        } else if (this->hiddenTail.nCoreIndx == this->hiddenTail.notTailNCoreCount) {
            hiddenMM.SetTail(this->hiddenMMTiling.singleCoreM, this->hiddenTail.tailSingleCoreN);
        } else if (this->hiddenTail.mCoreIndx == this->hiddenTail.notTailMCoreCount) {
            hiddenMM.SetTail(this->hiddenTail.tailSingleCoreM, this->hiddenMMTiling.singleCoreN);
        }
        hiddenMM.IterateAll(this->outputGm.workspace[this->hiddenOffsets.COffset], true);
    }
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP32<T>::CopyGate(
    LocalTensor<T> &ub, GlobalTensor<T> &gm, int64_t mIdx, int64_t nIdx, int64_t gateOffset)
{
    DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = this->calcM;
    dataCopyParams.blockLen = this->calcN * sizeof(T);
    dataCopyParams.srcStride = (4 * this->tiling->hiddenSize - this->calcN) * sizeof(T);
    dataCopyParams.dstStride = 0;

    DataCopyPadParams padParams;
    padParams.isPad = false;
    padParams.leftPadding = 0;
    padParams.rightPadding = this->Ceil(this->calcN, this->blockSize) * this->blockSize - this->calcN;
    padParams.paddingValue = 0;

    DataCopyPad(ub,
        gm[gateOffset + this->blockIdx * this->vectorCoreM * this->tiling->hiddenSize * 4 +
            mIdx * this->vectorBaseM * this->tiling->hiddenSize * 4 + nIdx * this->vectorBaseN],
        dataCopyParams,
        padParams);
    this->qidVecIn.EnQue(ub);
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP32<T>::CopyWithSigmoid(
    LocalTensor<T> &dstUb, GlobalTensor<T> &mixGm, int64_t mIdx, int64_t nIdx, int64_t gateOffset)
{
    LocalTensor<T> ubLocalIn = this->qidVecIn.template AllocTensor<T>();
    this->CopyGate(ubLocalIn, mixGm, mIdx, nIdx, gateOffset);
    ubLocalIn = this->qidVecIn.template DeQue<T>();
    Sigmoid(dstUb, ubLocalIn, this->calcSizeAlign);
    this->qidVecIn.FreeTensor(ubLocalIn);
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP32<T>::CopyWithSigmoidAddBias(
    LocalTensor<float> &dstUb, GlobalTensor<float> &mixGm, int64_t mIdx, int64_t nIdx, int64_t gateOffset)
{
    LocalTensor<float> ubLocalIn = this->qidVecIn.template AllocTensor<float>();
    this->CopyGate(ubLocalIn, mixGm, mIdx, nIdx, gateOffset);
    ubLocalIn = this->qidVecIn.template DeQue<float>();
    Adds(ubLocalIn, ubLocalIn, (float)this->tiling->forgetBias, this->calcSizeAlign);
    PipeBarrier<PIPE_V>();
    Sigmoid(dstUb, ubLocalIn, this->calcSizeAlign);
    this->qidVecIn.FreeTensor(ubLocalIn);
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP32<T>::CopyWithTanh(
    LocalTensor<T> &dstUb, GlobalTensor<T> &mixGm, int64_t mIdx, int64_t nIdx, int64_t gateOffset)
{
    LocalTensor<T> ubLocalIn = this->qidVecIn.template AllocTensor<T>();
    this->CopyGate(ubLocalIn, mixGm, mIdx, nIdx, gateOffset);
    ubLocalIn = this->qidVecIn.template DeQue<T>();
    Tanh(dstUb, ubLocalIn, this->calcSizeAlign);
    this->qidVecIn.FreeTensor(ubLocalIn);
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP32<T>::CopyWithMul(
    LocalTensor<T> &dstUb, LocalTensor<T> &other, GlobalTensor<T> &mixGm, int64_t mIdx, int64_t nIdx)
{
    LocalTensor<T> ubLocalIn = this->qidVecIn.template AllocTensor<T>();
    DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = this->calcM;
    dataCopyParams.blockLen = this->calcN * sizeof(T);
    dataCopyParams.srcStride = (this->tiling->hiddenSize - this->calcN) * sizeof(T);
    dataCopyParams.dstStride = 0;

    DataCopyPadParams padParams;
    padParams.isPad = false;
    padParams.leftPadding = 0;
    padParams.rightPadding = this->Ceil(this->calcN, this->blockSize) * this->blockSize - this->calcN;
    padParams.paddingValue = 0;

    DataCopyPad(ubLocalIn,
        mixGm[this->blockIdx * this->vectorCoreM * this->tiling->hiddenSize +
              mIdx * this->vectorBaseM * this->tiling->hiddenSize + nIdx * this->vectorBaseN],
        dataCopyParams,
        padParams);
    this->qidVecIn.EnQue(ubLocalIn);
    ubLocalIn = this->qidVecIn.template DeQue<T>();
    Mul(dstUb, ubLocalIn, other, this->calcSizeAlign);
    this->qidVecIn.FreeTensor(ubLocalIn);
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP32<T>::CopyInHC(
    LocalTensor<T> &dstUb, GlobalTensor<T> &mixGm, int64_t tIdx, int64_t mIdx, int64_t nIdx)
{
    LocalTensor<T> ubLocalIn = this->qidVecIn.template AllocTensor<T>();
    DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = this->calcM;
    dataCopyParams.blockLen = this->calcN * sizeof(T);
    dataCopyParams.srcStride = (this->tiling->hiddenSize - this->calcN) * sizeof(T);
    dataCopyParams.dstStride = 0;

    DataCopyPadParams padParams;
    padParams.isPad = false;
    padParams.leftPadding = 0;
    padParams.rightPadding = this->Ceil(this->calcN, this->blockSize) * this->blockSize - this->calcN;
    padParams.paddingValue = 0;

    DataCopyPad(ubLocalIn,
        mixGm[this->blockIdx * this->vectorCoreM * this->tiling->hiddenSize +
              mIdx * this->vectorBaseM * this->tiling->hiddenSize + nIdx * this->vectorBaseN],
        dataCopyParams,
        padParams);
    this->qidVecIn.EnQue(ubLocalIn);
    ubLocalIn = this->qidVecIn.template DeQue<T>();
    PipeBarrier<PIPE_V>();
    Adds(dstUb, ubLocalIn, (float)0.0, this->calcSizeAlign);
    PipeBarrier<PIPE_V>();
    this->qidVecIn.FreeTensor(ubLocalIn);
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP32<T>::CopyInSeq(
    LocalTensor<T> &dstUb, GlobalTensor<T> &mixGm, int64_t tIdx, int64_t mIdx, int64_t nIdx)
{
    LocalTensor<T> ubLocalIn = this->qidVecIn.template AllocTensor<T>();
    DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = this->calcM;
    dataCopyParams.blockLen = this->calcN * sizeof(T);
    dataCopyParams.srcStride = (this->tiling->hiddenSize - this->calcN) * sizeof(T);
    dataCopyParams.dstStride = 0;

    DataCopyPadParams padParams;
    padParams.isPad = false;
    padParams.leftPadding = 0;
    padParams.rightPadding = this->Ceil(this->calcN, this->blockSize) * this->blockSize - this->calcN;
    padParams.paddingValue = 0;

    int64_t tOffset = tIdx * this->tiling->batch * this->tiling->hiddenSize;

    if (this->tiling->direction == 1) {
        tOffset = (this->tiling->timeStep - 1 - tIdx) * this->tiling->batch * this->tiling->hiddenSize;
    }

    DataCopyPad(ubLocalIn,
        mixGm[tOffset + this->blockIdx * this->vectorCoreM * this->tiling->hiddenSize +
              mIdx * this->vectorBaseM * this->tiling->hiddenSize + nIdx * this->vectorBaseN],
        dataCopyParams,
        padParams);
    this->qidVecIn.EnQue(ubLocalIn);
    ubLocalIn = this->qidVecIn.template DeQue<T>();
    PipeBarrier<PIPE_V>();
    Adds(dstUb, ubLocalIn, (float)0.0, this->calcSizeAlign);
    PipeBarrier<PIPE_V>();
    this->qidVecIn.FreeTensor(ubLocalIn);
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP32<T>::CopyOutput(
    GlobalTensor<T> &gm, LocalTensor<T> &ub, int64_t tIdx, int64_t mIdx, int64_t nIdx)
{
    LocalTensor<T> outLocal = this->qidVecOut.template AllocTensor<T>();
    PipeBarrier<PIPE_V>();
    Muls(outLocal, ub, (float)1.0, this->calcSizeAlign);
    this->qidVecOut.EnQue(outLocal);
    outLocal = this->qidVecOut.template DeQue<T>();
    int64_t offset;
    if (this->tiling->direction == 1) {
        offset = (this->tiling->timeStep - 1 - tIdx) * this->tiling->batch * this->tiling->hiddenSize +
                 this->blockIdx * this->vectorCoreM * this->tiling->hiddenSize +
                 mIdx * this->vectorBaseM * this->tiling->hiddenSize + nIdx * this->vectorBaseN;
    } else {
        offset = tIdx * this->tiling->batch * this->tiling->hiddenSize +
                 this->blockIdx * this->vectorCoreM * this->tiling->hiddenSize +
                 mIdx * this->vectorBaseM * this->tiling->hiddenSize + nIdx * this->vectorBaseN;
    }
    DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = this->calcM;
    dataCopyParams.blockLen = this->calcN * sizeof(T);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = (this->tiling->hiddenSize - this->calcN) * sizeof(T);

    DataCopyPadParams padParams;
    padParams.isPad = false;
    padParams.leftPadding = 0;
    padParams.rightPadding = 0;
    padParams.paddingValue = 0;

    DataCopyPad(gm[offset], outLocal, dataCopyParams);
    this->qidVecOut.FreeTensor(outLocal);
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP32<T>::ProcessVectorOnce(
    int64_t tIdx, int64_t mIdx, int64_t nIdx, GlobalTensor<T> &mixGm)
{
    this->blockIdx = GetBlockIdx();
    // get n size
    if ((this->vectorTailN > 0) && (nIdx == this->vectorSplitN - 1)) {
        this->calcN = this->vectorTailN;
    } else {
        this->calcN = this->vectorBaseN;
    }
    // get m size
    this->calcM = this->vectorBaseM;
    if ((this->blockIdx < this->vectorCoreNum - 1) && (this->vectorBaseTailM > 0) && (mIdx == this->vectorSplitM - 1)) {
        // Calc block's m_size in the base core last block.
        this->calcM = this->vectorBaseTailM;
    }
    if ((this->blockIdx == this->vectorCoreNum - 1) && (this->vectorTailTailM > 0) &&
        (mIdx == this->vectorTailSplitM - 1)) {
        // Calc block's m_size in the last core last block.
        this->calcM = this->vectorTailTailM;
    }

    // get calc once block size
    this->calcSize = this->calcM * this->calcN;
    this->calcSizeAlign = this->calcM * this->Ceil(this->calcN, this->calBlockSize) * this->calBlockSize;

    PipeBarrier<PIPE_V>();

    // f 1 2 3 4 -> [1] 2 3 4
    auto fSigmoid = this->ubLocal1;
    LocalTensor<float> ubLocalIn = this->qidVecIn.template AllocTensor<float>();
    CopyGate(ubLocalIn, mixGm, mIdx, nIdx, this->fOffset);
    ubLocalIn = this->qidVecIn.template DeQue<float>();
    Adds(ubLocalIn, ubLocalIn, (float)this->tiling->forgetBias, this->calcSizeAlign);
    PipeBarrier<PIPE_V>();
    Sigmoid(fSigmoid, ubLocalIn, this->calcSizeAlign);
    this->qidVecIn.FreeTensor(ubLocalIn);
    if (this->tiling->isTraining == 1) {
        CopyOutput(this->outputGm.outFGm, fSigmoid, tIdx, mIdx, nIdx);
    }
    PipeBarrier<PIPE_V>();

    // [1] 2 3 4 -> 1 [2] 3 4
    auto cTmp1 = this->ubLocal2;
    CopyWithMul(cTmp1, fSigmoid, this->inputGm.initCGm, mIdx, nIdx);
    PipeBarrier<PIPE_V>();
    // i 1 [2] 3 4 -> [1] [2] 3 4
    auto iSigmoid = this->ubLocal1;
    CopyWithSigmoid(iSigmoid, mixGm, mIdx, nIdx, this->iOffset);
    if (this->tiling->isTraining == 1) {
        CopyOutput(this->outputGm.outIGm, iSigmoid, tIdx, mIdx, nIdx);
    }
    PipeBarrier<PIPE_V>();
    // j [1] [2] 3 4 -> [1] [2] [3] 4
    auto jTanh = this->ubLocal3;
    CopyWithTanh(jTanh, mixGm, mIdx, nIdx, this->jOffset);
    if (this->tiling->isTraining == 1) {
        CopyOutput(this->outputGm.outJGm, jTanh, tIdx, mIdx, nIdx);
    }
    PipeBarrier<PIPE_V>();
    // i * j [1] [2] [3] 4 -> 1 [2] 3 [4]
    auto cTmp2 = this->ubLocal4;
    Mul(cTmp2, jTanh, iSigmoid, this->calcSizeAlign);
    PipeBarrier<PIPE_V>();

    // i * j + f * c 1 [2] 3 [4] -> [1] 2 3 4
    auto updateC = this->ubLocal1;
    Add(updateC, cTmp1, cTmp2, this->calcSizeAlign);

    if (this->tiling->isSeqLength == 1) {
        auto initC = this->ubLocal2;
        auto seqLength = this->ubLocal4;
        CopyInHC(initC, this->inputGm.initCGm, 0, mIdx, nIdx);
        Sub(updateC, updateC, initC, this->calcSizeAlign);
        PipeBarrier<PIPE_V>();
        CopyInSeq(seqLength, this->inputGm.seqLengthGm, tIdx, mIdx, nIdx);
        Mul(updateC, updateC, seqLength, this->calcSizeAlign);
        PipeBarrier<PIPE_V>();
        Add(updateC, updateC, initC, this->calcSizeAlign);
        PipeBarrier<PIPE_V>();
    }

    if (this->tiling->cellClip > 0) {
        PipeBarrier<PIPE_V>();
        Mins(updateC, updateC, static_cast<float>(this->tiling->cellClip), this->calcSizeAlign);
    }

    CopyOutput(this->outputGm.outCGm, updateC, tIdx, mIdx, nIdx);
    PipeBarrier<PIPE_V>();

    // tanh(c) 1 [2] 3 4 -> 1 [2] 3 4
    auto cTanh = this->ubLocal2;
    Tanh(cTanh, updateC, this->calcSizeAlign);
    if (this->tiling->isTraining == 1) {
        CopyOutput(this->outputGm.outTanhCGm, cTanh, tIdx, mIdx, nIdx);
    }
    PipeBarrier<PIPE_V>();
    // o 1 [2] 3 4 -> [1] [2] 3 4
    auto oSigmoid = this->ubLocal1;
    CopyWithSigmoid(oSigmoid, mixGm, mIdx, nIdx, this->oOffset);
    if (this->tiling->isTraining == 1) {
        CopyOutput(this->outputGm.outOGm, oSigmoid, tIdx, mIdx, nIdx);
    }
    PipeBarrier<PIPE_V>();

    // o * Tanh(c) [1] [2] 3 4 -> 1 2 [3] 4
    auto updateH = this->ubLocal3;
    Mul(updateH, oSigmoid, cTanh, this->calcSizeAlign);

    if (this->tiling->isSeqLength == 1) {
        PipeBarrier<PIPE_V>();
        auto updateY = this->ubLocal1;
        auto initH = this->ubLocal2;
        auto seqLength = this->ubLocal4;
        Mul(updateY, updateH, seqLength, this->calcSizeAlign);
        PipeBarrier<PIPE_V>();
        CopyInHC(initH, this->inputGm.initHGm, 0, mIdx, nIdx);
        Sub(updateH, updateH, initH, this->calcSizeAlign);
        PipeBarrier<PIPE_V>();
        Mul(updateH, updateH, seqLength, this->calcSizeAlign);
        PipeBarrier<PIPE_V>();
        Add(updateH, updateH, initH, this->calcSizeAlign);
        CopyOutput(this->outputGm.outYGm, updateY, tIdx, mIdx, nIdx);
    } else {
        CopyOutput(this->outputGm.outYGm, updateH, tIdx, mIdx, nIdx);
    }

    CopyOutput(this->outputGm.outHGm, updateH, tIdx, mIdx, nIdx);
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP32<T>::ProcessVectorInitHC(int64_t mIdx, int64_t nIdx, GlobalTensor<T> &mixGm)
{
    this->blockIdx = GetBlockIdx();
    // get n size
    if ((this->vectorTailN > 0) && (nIdx == this->vectorSplitN - 1)) {
        this->calcN = this->vectorTailN;
    } else {
        this->calcN = this->vectorBaseN;
    }

    // get m size
    this->calcM = this->vectorBaseM;
    if ((this->blockIdx < this->vectorCoreNum - 1) && (this->vectorBaseTailM > 0) && (mIdx == this->vectorSplitM - 1)) {
        // Calc block's m_size in the base core last block.
        this->calcM = this->vectorBaseTailM;
    }
    if ((this->blockIdx == this->vectorCoreNum - 1) && (this->vectorTailTailM > 0) &&
        (mIdx == this->vectorTailSplitM - 1)) {
        // Calc block's m_size in the last core last block.
        this->calcM = this->vectorTailTailM;
    }

    // get calc once block size
    this->calcSize = this->calcM * this->calcN;
    this->calcSizeAlign = this->calcM * this->Ceil(this->calcN, this->calBlockSize) * this->calBlockSize;

    PipeBarrier<PIPE_V>();

    // f 1 2 3 4 -> [1] 2 3 4
    auto fSigmoid = this->ubLocal1;
    this->CopyWithSigmoidAddBias(fSigmoid, mixGm, mIdx, nIdx, this->iOffset);
    if (this->tiling->isTraining == 1) {
        this->CopyOutput(this->outputGm.outFGm, fSigmoid, 0, mIdx, nIdx);
    }

    PipeBarrier<PIPE_V>();

    // i 1 [2] 3 4 -> [1] [2] 3 4
    auto iSigmoid = this->ubLocal1;
    this->CopyWithSigmoid(iSigmoid, mixGm, mIdx, nIdx, this->iOffset);

    if (this->tiling->isTraining == 1) {
        this->CopyOutput(this->outputGm.outIGm, iSigmoid, 0, mIdx, nIdx);
    }
    PipeBarrier<PIPE_V>();

    // j [1] [2] 3 4 -> [1] [2] [3] 4
    auto jTanh = this->ubLocal3;
    this->CopyWithTanh(jTanh, mixGm, mIdx, nIdx, this->jOffset);
    if (this->tiling->isTraining == 1) {
        this->CopyOutput(this->outputGm.outJGm, jTanh, 0, mIdx, nIdx);
    }
    PipeBarrier<PIPE_V>();

    // i * j [1] [2] [3] 4 -> 1 [2] 3 [4]
    auto cTmp2 = this->ubLocal4;
    Mul(cTmp2, jTanh, iSigmoid, this->calcSizeAlign);
    PipeBarrier<PIPE_V>();

    // i * j + f * c 1 [2] 3 [4] -> [1] 2 3 4
    auto updateC = cTmp2;

    if (this->tiling->isSeqLength == 1) {
        auto seqLength = this->ubLocal3;
        this->CopyInSeq(seqLength, this->inputGm.seqLengthGm, 0, mIdx, nIdx);
        Mul(updateC, updateC, seqLength, this->calcSizeAlign);
        PipeBarrier<PIPE_V>();
    }

    if (this->tiling->cellClip > 0) {
        PipeBarrier<PIPE_V>();
        Mins(updateC, updateC, static_cast<float>(this->tiling->cellClip), this->calcSizeAlign);
    }

    this->CopyOutput(this->outputGm.outCGm, updateC, 0, mIdx, nIdx);
    PipeBarrier<PIPE_V>();

    // tanh(c) 1 [2] 3 4 -> 1 [2] 3 4
    auto cTanh = this->ubLocal2;
    Tanh(cTanh, updateC, this->calcSizeAlign);
    if (this->tiling->isTraining == 1) {
        this->CopyOutput(this->outputGm.outTanhCGm, cTanh, 0, mIdx, nIdx);
    }
    PipeBarrier<PIPE_V>();
    // o 1 [2] 3 4 -> [1] [2] 3 4
    auto oSigmoid = this->ubLocal1;
    this->CopyWithSigmoid(oSigmoid, mixGm, mIdx, nIdx, this->oOffset);
    if (this->tiling->isTraining == 1) {
        this->CopyOutput(this->outputGm.outOGm, oSigmoid, 0, mIdx, nIdx);
    }
    PipeBarrier<PIPE_V>();

    // o * Tanh(c) [1] [2] 3 4 -> 1 2 [3] 4
    auto updateH = this->ubLocal4;
    Mul(updateH, oSigmoid, cTanh, this->calcSizeAlign);

    if (this->tiling->isSeqLength == 1) {
        auto seqLength = this->ubLocal3;
        PipeBarrier<PIPE_V>();
        Mul(updateH, updateH, seqLength, this->calcSizeAlign);
    }

    this->CopyOutput(this->outputGm.outHGm, updateH, 0, mIdx, nIdx);
    this->CopyOutput(this->outputGm.outYGm, updateH, 0, mIdx, nIdx);
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP32<T>::ProcessVector(int64_t tIdx)
{
    auto mCoreIndex = GetBlockIdx();
    if (mCoreIndex < this->vectorCoreNum) {
        auto coreLoopM = this->vectorSplitM;
        if (mCoreIndex == this->vectorCoreNum - 1) {
            // Calc the last core.
            coreLoopM = this->vectorTailSplitM;
        }
        int64_t offset;
        if (this->tiling->direction == 1) {
            offset = (this->tiling->timeStep - 1 - tIdx) * this->allCellSize;
        } else {
            offset = tIdx * this->allCellSize;
        }
        for (int64_t j = 0; j < coreLoopM; ++j) {
            for (int64_t k = 0; k < this->vectorSplitN; ++k) {
                auto mixGm = this->outputGm.workspace[offset];
                ProcessVectorOnce(tIdx, j, k, mixGm);
            }
        }
    }
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP32<T>::ProcessInitalT()
{
    auto mCoreIndex = GetBlockIdx();
    if (mCoreIndex < this->vectorCoreNum) {
        auto coreLoopM = this->vectorSplitM;
        if (mCoreIndex == this->vectorCoreNum - 1) {
            // Calc the last core.
            coreLoopM = this->vectorTailSplitM;
        }
        int64_t offset;
        if (this->tiling->direction == 1) {
            offset = (this->tiling->timeStep - 1) * this->allCellSize;
        } else {
            offset = 0;
        }
        for (int64_t j = 0; j < coreLoopM; ++j) {
            for (int64_t k = 0; k < this->vectorSplitN; ++k) {
                auto mixGm = this->outputGm.workspace[offset];
                this->ProcessVectorInitHC(j, k, mixGm);
            }
        }
    }
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP32<T>::Process()
{
    this->ProcessInputMM();
    if (this->tiling->isInithc == 0) {
        SyncAll();
        this->ProcessInitalT();
        if (this->tiling->direction == 1) {
            this->inputGm.initCGm =
                this->outputGm.outCGm[(this->tiling->timeStep - 1) * this->tiling->batch * this->tiling->hiddenSize];
            this->inputGm.initHGm =
                this->outputGm.outHGm[(this->tiling->timeStep - 1) * this->tiling->batch * this->tiling->hiddenSize];
        } else {
            this->inputGm.initCGm = this->outputGm.outCGm;
            this->inputGm.initHGm = this->outputGm.outHGm;
        }
    }

    int64_t tIdx = this->tiling->isInithc == 0 ? 1 : 0;

    for (tIdx; tIdx < this->tiling->timeStep; tIdx++) {
        SyncAll();

        this->ProcessHiddenMM(tIdx);

        SyncAll();

        this->ProcessVector(tIdx);

        SyncAll();

        if (this->tiling->direction == 1) {
            this->inputGm.initCGm =
                this->outputGm
                    .outCGm[(this->tiling->timeStep - 1 - tIdx) * this->tiling->batch * this->tiling->hiddenSize];
            this->inputGm.initHGm =
                this->outputGm
                    .outHGm[(this->tiling->timeStep - 1 - tIdx) * this->tiling->batch * this->tiling->hiddenSize];
        } else {
            this->inputGm.initCGm = this->outputGm.outCGm[tIdx * this->tiling->batch * this->tiling->hiddenSize];
            this->inputGm.initHGm = this->outputGm.outHGm[tIdx * this->tiling->batch * this->tiling->hiddenSize];
        }
    }
}
