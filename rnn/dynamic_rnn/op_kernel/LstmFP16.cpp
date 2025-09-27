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
 * \file LstmFP16.cpp
 * \brief
 */
#include "LstmFP16.h"

using namespace AscendC;
constexpr static const uint32_t FLOAT_BLOCK_NUM = 8;
constexpr static const uint32_t IJFO_GATE_NUM = 4;

template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP16<T>::ProcessInputMM()
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
__aicore__ inline void LstmMmSplitNDNDFP16<T>::ProcessHiddenMM(int64_t tIdx)
{
    if (GetBlockIdx() < this->hiddenMMTiling.usedCoreNum) {
        if (this->tiling->direction == 1) {
            this->hiddenOffsets.COffset =
                this->oriHiddenOffsets.COffset + (this->tiling->timeStep - 1 - tIdx) * this->allCellSize;
        } else {
            this->hiddenOffsets.COffset = this->oriHiddenOffsets.COffset + tIdx * this->allCellSize;
        }
        this->hiddenMM.SetTensorA(this->inputGm.initHGm[this->hiddenOffsets.AOffset]);
        this->hiddenMM.IterateAll(this->outputGm.workspace[this->hiddenOffsets.COffset], true);
    }
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP16<T>::CopyInHCSeq(
    LocalTensor<float> &dstUb, GlobalTensor<T> &mixGm, int64_t offset)
{
    LocalTensor<T> ubLocalIn = this->qidCIn.template AllocTensor<T>();
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

    DataCopyPad(ubLocalIn, mixGm[offset], dataCopyParams, padParams);
    this->qidCIn.EnQue(ubLocalIn);
    ubLocalIn = this->qidCIn.template DeQue<T>();
    Cast(dstUb, ubLocalIn, RoundMode::CAST_NONE, this->calcSizeAlign);
    this->qidCIn.FreeTensor(ubLocalIn);
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP16<T>::CopyOutput(GlobalTensor<T> &gm, LocalTensor<float> &ub, int64_t offset)
{
    auto outLocal = this->qidVecOut.template AllocTensor<T>();
    PipeBarrier<PIPE_V>();
    Cast(outLocal, ub, RoundMode::CAST_ROUND, this->calcSizeAlign);
    this->qidVecOut.EnQue(outLocal);

    outLocal = this->qidVecOut.template DeQue<T>();

    DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = this->calcM;
    dataCopyParams.blockLen = this->calcN * sizeof(T);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = (this->tiling->hiddenSize - this->calcN) * sizeof(T);

    DataCopyPad(gm[offset], outLocal, dataCopyParams);

    this->qidVecOut.FreeTensor(outLocal);
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP16<T>::CalcVecScaler(
    int64_t tIdx, int64_t mIdx, int64_t nIdx, int64_t &ijfoBaseOffset, int64_t &initcOffset, int64_t &offset)
{
    this->blockIdx = GetBlockIdx();

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
    this->calcSizeAlign = this->calcM * this->Ceil(this->calcN, this->blockSize) * this->blockSize;

    ijfoBaseOffset = this->blockIdx * this->vectorCoreM * this->tiling->hiddenSize * IJFO_GATE_NUM +
                     mIdx * this->vectorBaseM * this->tiling->hiddenSize * IJFO_GATE_NUM + nIdx * this->vectorBaseN;
    initcOffset = this->blockIdx * this->vectorCoreM * this->tiling->hiddenSize +
                  mIdx * this->vectorBaseM * this->tiling->hiddenSize + nIdx * this->vectorBaseN;
    if (this->tiling->direction == 1) {
        offset = (this->tiling->timeStep - 1 - tIdx) * this->tiling->batch * this->tiling->hiddenSize + initcOffset;
    } else {
        offset = tIdx * this->tiling->batch * this->tiling->hiddenSize + initcOffset;
    }
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP16<T>::CopyInFJ(
    LocalTensor<float> &dstUb, GlobalTensor<float> &mixGm, int64_t GmOffset)
{
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = this->calcM;
    dataCopyParams.blockLen = this->calcN * sizeof(float);
    dataCopyParams.srcStride = (IJFO_GATE_NUM * this->tiling->hiddenSize - this->calcN) * sizeof(float);
    dataCopyParams.dstStride =
        (this->Ceil(this->calcN, this->blockSize) * this->blockSize - this->calcN) / FLOAT_BLOCK_NUM;

    DataCopyPadExtParams<float> padParams{
        false, 0, (uint8_t)(this->Ceil(this->calcN, this->calBlockSize) * this->calBlockSize - this->calcN), 0};

    dstUb = this->qidVecIn.template AllocTensor<float>();
    DataCopyPad(dstUb, mixGm[GmOffset], dataCopyParams, padParams);
    this->qidVecIn.EnQue(dstUb);
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP16<T>::CopyInC(
    LocalTensor<T> &dstUb, GlobalTensor<T> &mixGm, const int64_t initcOffset)
{
    DataCopyExtParams dataCopyParams1;
    dataCopyParams1.blockCount = this->calcM;
    dataCopyParams1.blockLen = this->calcN * sizeof(T);
    dataCopyParams1.srcStride = (this->tiling->hiddenSize - this->calcN) * sizeof(T);
    dataCopyParams1.dstStride = 0;

    DataCopyPadExtParams<T> padParams1{
        false, 0, (uint8_t)(this->Ceil(this->calcN, this->blockSize) * this->blockSize - this->calcN), 0};

    dstUb = this->qidCIn.template AllocTensor<T>();
    DataCopyPad(dstUb, mixGm[initcOffset], dataCopyParams1, padParams1);
    this->qidCIn.EnQue(dstUb);
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP16<T>::CopyInIO(
    LocalTensor<float> &dstUb, GlobalTensor<float> &mixGm, int64_t GmOffset)
{
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = this->calcM;
    dataCopyParams.blockLen = this->calcN * sizeof(float);
    dataCopyParams.srcStride = (IJFO_GATE_NUM * this->tiling->hiddenSize - this->calcN) * sizeof(float);
    dataCopyParams.dstStride =
        (this->Ceil(this->calcN, this->blockSize) * this->blockSize - this->calcN) / FLOAT_BLOCK_NUM;

    DataCopyPadExtParams<float> padParams{
        false, 0, (uint8_t)(this->Ceil(this->calcN, this->calBlockSize) * this->calBlockSize - this->calcN), 0};

    dstUb = this->qidVecIn2.template AllocTensor<float>();
    DataCopyPad(dstUb, mixGm[GmOffset], dataCopyParams, padParams);
    this->qidVecIn2.EnQue(dstUb);
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP16<T>::AddfSigmoid(
    LocalTensor<float> &fSigmoid, LocalTensor<float> &ubLocalIn, int64_t offset)
{
    ubLocalIn = this->qidVecIn.template DeQue<float>();
    Adds(ubLocalIn, ubLocalIn, (float)this->tiling->forgetBias, this->calcSizeAlign);
    PipeBarrier<PIPE_V>();
    Sigmoid(fSigmoid, ubLocalIn, this->calcSizeAlign);
    this->qidVecIn.FreeTensor(ubLocalIn);

    if (this->tiling->isTraining == 1) {
        this->CopyOutput(this->outputGm.outFGm, fSigmoid, offset);
    }
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP16<T>::InitCMulfSigmoid(
    LocalTensor<float> &cTmp1, LocalTensor<T> &ubLocalIn, LocalTensor<float> &fSigmoid)
{
    ubLocalIn = this->qidCIn.template DeQue<T>();
    Cast(this->ubLocal3, ubLocalIn, RoundMode::CAST_NONE, this->calcSizeAlign);
    this->qidCIn.FreeTensor(ubLocalIn);

    PipeBarrier<PIPE_V>();
    Mul(cTmp1, this->ubLocal3, fSigmoid, this->calcSizeAlign);
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP16<T>::CaliSigmoid(
    LocalTensor<float> &iSigmoid, LocalTensor<float> &ubLocalIn, int64_t offset)
{
    ubLocalIn = this->qidVecIn2.template DeQue<float>();
    PipeBarrier<PIPE_V>();
    Sigmoid(iSigmoid, ubLocalIn, this->calcSizeAlign);
    this->qidVecIn2.FreeTensor(ubLocalIn);

    if (this->tiling->isTraining == 1) {
        this->CopyOutput(this->outputGm.outIGm, iSigmoid, offset);
    }
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP16<T>::CaljTanh(
    LocalTensor<float> &jTanh, LocalTensor<float> &ubLocalIn, int64_t offset)
{
    ubLocalIn = this->qidVecIn.template DeQue<float>();
    PipeBarrier<PIPE_V>();
    Tanh(jTanh, ubLocalIn, this->calcSizeAlign);
    this->qidVecIn.FreeTensor(ubLocalIn);

    if (this->tiling->isTraining == 1) {
        this->CopyOutput(this->outputGm.outJGm, jTanh, offset);
    }
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP16<T>::CaloSigmoid(
    LocalTensor<float> &oSigmoid, LocalTensor<float> &ubLocalIn, int64_t offset)
{
    ubLocalIn = this->qidVecIn2.template DeQue<float>();
    PipeBarrier<PIPE_V>();
    Sigmoid(oSigmoid, ubLocalIn, this->calcSizeAlign);
    this->qidVecIn2.FreeTensor(ubLocalIn);

    if (this->tiling->isTraining == 1) {
        this->CopyOutput(this->outputGm.outOGm, oSigmoid, offset);
    }
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP16<T>::CopyOutYH(
    LocalTensor<float> &updateH, int64_t offset, int64_t initcOffset)
{
    DataCopyExtParams dataCopyParams1;
    dataCopyParams1.blockCount = this->calcM;
    dataCopyParams1.blockLen = this->calcN * sizeof(T);
    dataCopyParams1.srcStride = (this->tiling->hiddenSize - this->calcN) * sizeof(T);
    dataCopyParams1.dstStride = 0;

    DataCopyPadExtParams<T> padParams1{
        false, 0, (uint8_t)(this->Ceil(this->calcN, this->blockSize) * this->blockSize - this->calcN), 0};

    if (this->tiling->isSeqLength == 1) {
        PipeBarrier<PIPE_V>();
        auto updateY = this->ubLocal1;
        auto initH = this->ubLocal2;
        auto seqLength = this->ubLocal4;
        Mul(updateY, updateH, seqLength, this->calcSizeAlign);
        PipeBarrier<PIPE_V>();
        CopyInHCSeq(initH, this->inputGm.initHGm, initcOffset);
        PipeBarrier<PIPE_V>();
        Sub(updateH, updateH, initH, this->calcSizeAlign);
        PipeBarrier<PIPE_V>();
        Mul(updateH, updateH, seqLength, this->calcSizeAlign);
        PipeBarrier<PIPE_V>();
        Add(updateH, updateH, initH, this->calcSizeAlign);
        this->CopyOutput(this->outputGm.outYGm, updateY, offset);
        this->CopyOutput(this->outputGm.outHGm, updateH, offset);
    } else {
        LocalTensor<T> outLocal = this->qidVecOut.template AllocTensor<T>();
        PipeBarrier<PIPE_V>();
        Cast(outLocal, updateH, RoundMode::CAST_ROUND, this->calcSizeAlign);
        this->qidVecOut.EnQue(outLocal);
        outLocal = this->qidVecOut.template DeQue<T>();

        DataCopyPad(this->outputGm.outYGm[offset], outLocal, dataCopyParams1);
        DataCopyPad(this->outputGm.outHGm[offset], outLocal, dataCopyParams1);

        this->qidVecOut.FreeTensor(outLocal);
    }
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP16<T>::CopyOutYHt0(LocalTensor<float> &updateH, int64_t offset)
{
    DataCopyExtParams dataCopyParams1;
    dataCopyParams1.blockCount = this->calcM;
    dataCopyParams1.blockLen = this->calcN * sizeof(T);
    dataCopyParams1.srcStride = (this->tiling->hiddenSize - this->calcN) * sizeof(T);
    dataCopyParams1.dstStride = 0;

    DataCopyPadExtParams<T> padParams1{
        false, 0, (uint8_t)(this->Ceil(this->calcN, this->blockSize) * this->blockSize - this->calcN), 0};

    if (this->tiling->isSeqLength == 1) {
        auto seqLength = this->ubLocal3;
        PipeBarrier<PIPE_V>();
        Mul(updateH, updateH, seqLength, this->calcSizeAlign);
    }

    LocalTensor<T> outLocal = this->qidVecOut.template AllocTensor<T>();
    PipeBarrier<PIPE_V>();
    Cast(outLocal, updateH, RoundMode::CAST_ROUND, this->calcSizeAlign);
    this->qidVecOut.EnQue(outLocal);
    outLocal = this->qidVecOut.template DeQue<T>();

    DataCopyPad(this->outputGm.outYGm[offset], outLocal, dataCopyParams1);
    DataCopyPad(this->outputGm.outHGm[offset], outLocal, dataCopyParams1);

    this->qidVecOut.FreeTensor(outLocal);
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP16<T>::CalAddTanh(LocalTensor<float> &cTanh, LocalTensor<float> &ubLocal2,
    LocalTensor<float> &cTmp2, int64_t offset, int64_t initcOffset)
{
    PipeBarrier<PIPE_V>();

    auto updateC = this->ubLocal1;
    Add(updateC, this->ubLocal2, cTmp2, this->calcSizeAlign);

    if (this->tiling->isSeqLength == 1) {
        PipeBarrier<PIPE_V>();
        auto initC = this->ubLocal2;
        auto seqLength = this->ubLocal4;
        this->CopyInHCSeq(initC, this->inputGm.initCGm, initcOffset);
        PipeBarrier<PIPE_V>();
        Sub(updateC, updateC, initC, this->calcSizeAlign);
        PipeBarrier<PIPE_V>();
        this->CopyInHCSeq(seqLength, this->inputGm.seqLengthGm, offset);
        PipeBarrier<PIPE_V>();
        Mul(updateC, updateC, seqLength, this->calcSizeAlign);
        PipeBarrier<PIPE_V>();
        Add(updateC, updateC, initC, this->calcSizeAlign);
        PipeBarrier<PIPE_V>();
    }

    if (this->tiling->cellClip > 0) {
        PipeBarrier<PIPE_V>();
        Mins(updateC, updateC, static_cast<float>(this->tiling->cellClip), this->calcSizeAlign);
    }
    this->CopyOutput(this->outputGm.outCGm, updateC, offset);
    PipeBarrier<PIPE_V>();
    Tanh(cTanh, updateC, this->calcSizeAlign);
    if (this->tiling->isTraining == 1) {
        this->CopyOutput(this->outputGm.outTanhCGm, cTanh, offset);
    }
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP16<T>::CalAddTanht0(LocalTensor<float> &cTanh, LocalTensor<float> &ubLocal2,
    LocalTensor<float> &cTmp2, int64_t offset, int64_t initcOffset)
{
    auto updateC = this->ubLocal4;

    if (this->tiling->isSeqLength == 1) {
        auto seqLength = this->ubLocal3;
        this->CopyInHCSeq(seqLength, this->inputGm.seqLengthGm, offset);
        PipeBarrier<PIPE_V>();
        Mul(updateC, updateC, seqLength, this->calcSizeAlign);
        PipeBarrier<PIPE_V>();
    }

    if (this->tiling->cellClip > 0) {
        PipeBarrier<PIPE_V>();
        Mins(updateC, updateC, static_cast<float>(this->tiling->cellClip), this->calcSizeAlign);
    }

    this->CopyOutput(this->outputGm.outCGm, updateC, offset);

    PipeBarrier<PIPE_V>();
    Tanh(cTanh, updateC, this->calcSizeAlign);
    if (this->tiling->isTraining == 1) {
        this->CopyOutput(this->outputGm.outTanhCGm, cTanh, offset);
    }
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP16<T>::ProcessVectorOnce(
    int64_t tIdx, int64_t mIdx, int64_t nIdx, GlobalTensor<float> &mixGm)
{
    int64_t ijfoBaseOffset, initcOffset, offset = 0;
    this->CalcVecScaler(tIdx, mIdx, nIdx, ijfoBaseOffset, initcOffset, offset);

    LocalTensor<T> initCVec;
    LocalTensor<float> fInput, jInput, iInput, oInput;
    auto fSigmoid = this->ubLocal1;
    auto iSigmoid = this->ubLocal1;
    auto oSigmoid = this->ubLocal1;
    auto jTanh = this->ubLocal3;
    auto cTanh = this->ubLocal2;
    auto updateH = this->ubLocal3;
    /*
    initc          f           i           j            o
      |            |           |           |            |
      |         1.sigmoid   3.sigmoid   4.tanh      7.sigmoid
      |            |           |           |            |
       ----2.mul---             ---5.mul---             |
             |                         |                |
             ----------6.add------------                |
                          |                             |
                       7.tanh                           |
                          |                             |
                          -------------8.mul------------
    */
    this->CopyInFJ(fInput, mixGm, this->fOffset + ijfoBaseOffset);

    this->CopyInC(initCVec, this->inputGm.initCGm, initcOffset);

    this->CopyInIO(iInput, mixGm, this->iOffset + ijfoBaseOffset);

    this->AddfSigmoid(fSigmoid, fInput, offset);

    this->CopyInFJ(jInput, mixGm, this->jOffset + ijfoBaseOffset);

    this->InitCMulfSigmoid(this->ubLocal2, initCVec, fSigmoid);

    this->CaliSigmoid(iSigmoid, iInput, offset);

    this->CopyInIO(oInput, mixGm, this->oOffset + ijfoBaseOffset);

    this->CaljTanh(jTanh, jInput, offset);

    PipeBarrier<PIPE_V>();
    Mul(this->ubLocal4, jTanh, iSigmoid, this->calcSizeAlign);

    this->CalAddTanh(cTanh, this->ubLocal2, this->ubLocal4, offset, initcOffset);

    this->CaloSigmoid(oSigmoid, oInput, offset);

    PipeBarrier<PIPE_V>();
    Mul(updateH, oSigmoid, cTanh, this->calcSizeAlign);

    this->CopyOutYH(updateH, offset, initcOffset);
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP16<T>::ProcessVectorInitHC(
    int64_t mIdx, int64_t nIdx, GlobalTensor<float> &mixGm)
{
    int64_t ijfoBaseOffset, initcOffset, offset = 0;
    this->CalcVecScaler(0, mIdx, nIdx, ijfoBaseOffset, initcOffset, offset);

    LocalTensor<float> fInput, jInput, iInput, oInput;
    auto fSigmoid = this->ubLocal1;
    auto iSigmoid = this->ubLocal1;
    auto jTanh = this->ubLocal3;
    auto cTanh = this->ubLocal2;
    auto oSigmoid = this->ubLocal1;
    auto updateH = this->ubLocal4;
    /*
     null          f           i           j            o
      |            |           |           |            |
      |         1.sigmoid   3.sigmoid   4.tanh      7.sigmoid
      |                        |           |            |
                                ---5.mul---             |
                                       |                |
                           -------------                |
                          |                             |
                       7.tanh                           |
                          |                             |
                          -------------8.mul------------
    */
    this->CopyInFJ(fInput, mixGm, this->fOffset + ijfoBaseOffset);

    this->CopyInIO(iInput, mixGm, this->iOffset + ijfoBaseOffset);

    this->AddfSigmoid(fSigmoid, fInput, offset);

    this->CopyInFJ(jInput, mixGm, this->jOffset + ijfoBaseOffset);

    this->CaliSigmoid(iSigmoid, iInput, offset);

    this->CopyInIO(oInput, mixGm, this->oOffset + ijfoBaseOffset);

    this->CaljTanh(jTanh, jInput, offset);

    PipeBarrier<PIPE_V>();
    Mul(this->ubLocal4, jTanh, iSigmoid, this->calcSizeAlign);

    this->CalAddTanht0(cTanh, this->ubLocal2, this->ubLocal4, offset, initcOffset);

    this->CaloSigmoid(oSigmoid, oInput, offset);

    PipeBarrier<PIPE_V>();
    Mul(updateH, oSigmoid, cTanh, this->calcSizeAlign);

    this->CopyOutYHt0(updateH, offset);
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP16<T>::ProcessVector(int64_t tIdx)
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
        auto mixGm = this->outputGm.workspace[offset];
        for (int64_t j = 0; j < coreLoopM; ++j) {
            for (int64_t k = 0; k < this->vectorSplitN; ++k) {
                this->ProcessVectorOnce(tIdx, j, k, mixGm);
            }
        }
    }
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP16<T>::ProcessInitalT()
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
        auto mixGm = this->outputGm.workspace[offset];
        for (int64_t j = 0; j < coreLoopM; ++j) {
            for (int64_t k = 0; k < this->vectorSplitN; ++k) {
                this->ProcessVectorInitHC(j, k, mixGm);
            }
        }
    }
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP16<T>::Process()
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

    this->hiddenMM.SetTensorB(this->inputGm.weightHiddenGm[this->hiddenOffsets.BOffset]);
    if (this->hiddenTail.nCoreIndx == this->hiddenTail.notTailNCoreCount &&
        this->hiddenTail.mCoreIndx == this->hiddenTail.notTailMCoreCount) {
        this->hiddenMM.SetTail(this->hiddenTail.tailSingleCoreM, this->hiddenTail.tailSingleCoreN);
    } else if (this->hiddenTail.nCoreIndx == this->hiddenTail.notTailNCoreCount) {
        this->hiddenMM.SetTail(this->hiddenMMTiling.singleCoreM, this->hiddenTail.tailSingleCoreN);
    } else if (this->hiddenTail.mCoreIndx == this->hiddenTail.notTailMCoreCount) {
        this->hiddenMM.SetTail(this->hiddenTail.tailSingleCoreM, this->hiddenMMTiling.singleCoreN);
    }
    SyncAll();
    for (tIdx; tIdx < this->tiling->timeStep; tIdx++) {
        this->ProcessHiddenMM(tIdx);

        SyncAll();

        this->ProcessVector(tIdx);

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
        SyncAll();
    }
}
