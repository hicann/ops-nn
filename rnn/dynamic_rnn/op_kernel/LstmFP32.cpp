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
#if DYNAMIC_RNN_HIACC_GEMM_INPUT
    if constexpr (std::is_same<T, float>::value) {
        // A5 fp32 输入 GEMM 用向量补偿累加：K 链由向量单元完成。
        this->ProcessInputMMHighAcc();
        return;
    }
#endif
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

#if DYNAMIC_RNN_HIACC_GEMM_INPUT
template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP32<T>::ProcessInputMMHighAcc()
{
    const int64_t coreIdx = GetBlockIdx();
    const int64_t batch = this->tiling->batch;
    const int64_t timeStep = this->tiling->timeStep;
    const int64_t inputSize = this->tiling->inputSize;
    const int64_t hidden4 = this->tiling->hiddenSize * LSTM_GATE_SIZE;
    const int64_t cch = this->hiaccCch;
    const int64_t calAlign = this->calBlockSize;
    const int64_t kEnd = inputSize;
    const int64_t kAl = this->hiaccKAl;
    const int64_t kBlk = this->hiaccKB;
    const int64_t rowBatch = this->hiaccRows;
    // 输入 GEMM 覆盖全部 (s,b) 行并按全部核(GetBlockNum())切分；写 workspace 后各阶段由 SyncAll 隔离。
    const int64_t totalRows = timeStep * batch;
    const int64_t coresHi = (GetBlockNum() > 0) ? GetBlockNum() : 1;
    const int64_t rowsPer = this->Ceil(totalRows, coresHi);
    const int64_t rowBeg = coreIdx * rowsPer;
    const int64_t rowCntAll = ((totalRows - rowBeg) < rowsPer) ? (totalRows - rowBeg) : rowsPer;
    if (rowCntAll <= 0 || kBlk < 1 || rowBatch < 1 || totalRows <= 0) {
        return;
    }
    const int64_t nKBlk = (kEnd + kBlk - 1) / kBlk;

    LocalTensor<float> hiaccLocal = this->hiaccBuf.template Get<float>(0);
    LocalTensor<float> accA = hiaccLocal;
    LocalTensor<float> accB = hiaccLocal[rowBatch * cch];
    LocalTensor<float> corrBuf = hiaccLocal[2 * rowBatch * cch];
    LocalTensor<float> xBuf = hiaccLocal[3 * rowBatch * cch];
    LocalTensor<float> pBuf = xBuf[rowBatch * kAl];
    LocalTensor<float> blkBuf = pBuf[rowBatch * cch];
    LocalTensor<float> biasBuf = blkBuf[rowBatch * cch];

    DataCopyParams cpCol;
    cpCol.blockCount = 1;
    cpCol.blockLen = 0; // 每列分块按实际 cnt 设置
    cpCol.srcStride = 0;
    cpCol.dstStride = 0;
    DataCopyPadParams ppCol;
    ppCol.isPad = false;
    ppCol.leftPadding = 0;
    ppCol.rightPadding = 0;
    ppCol.paddingValue = 0.0f;

    DataCopyParams cpX;
    cpX.blockCount = 1;
    cpX.blockLen = inputSize * sizeof(float);
    cpX.srcStride = 0;
    cpX.dstStride = 0;
    DataCopyPadParams ppX;
    ppX.isPad = false;
    ppX.leftPadding = 0;
    ppX.rightPadding = kAl - inputSize;
    ppX.paddingValue = 0.0f;

    constexpr int32_t revK = DYNAMIC_RNN_HIACC_REVK_INPUT;
    for (int64_t c0 = 0; c0 < hidden4; c0 += cch) {
        const int64_t cnt = (hidden4 - c0 < cch) ? (hidden4 - c0) : cch;
        const int64_t cntAl = this->Ceil(cnt, calAlign) * calAlign;
        const int64_t padRight = cntAl - cnt;
        cpCol.blockLen = cnt * sizeof(float);
        ppCol.rightPadding = padRight;

        for (int64_t g0 = 0; g0 < rowCntAll; g0 += rowBatch) {
            const int64_t gRows = (rowCntAll - g0 < rowBatch) ? (rowCntAll - g0) : rowBatch;
            // 1) x 行（及偏置）载入；xBuf 覆写前已由上一批的收尾 barrier 保证向量读结束
            if (inputSize > 0) {
                for (int64_t i = 0; i < gRows; ++i) {
                    const int64_t gi = rowBeg + g0 + i;
                    const int64_t s = gi / batch;
                    const int64_t b = gi % batch;
                    DataCopyPad(xBuf[i * kAl], this->inputGm.xGm[gi * inputSize], cpX, ppX);
                }
            }
            if (this->tiling->isBias == 1) {
                DataCopyPad(biasBuf, this->inputGm.biasGm[c0], cpCol, ppCol);
            }
            PipeBarrier<PIPE_ALL>(); // x/偏置数据就绪

            // 2) A/B/corr 清零（行步长 cntAl）
            for (int64_t i = 0; i < gRows; ++i) {
                Duplicate(accA[i * cntAl], (float)0.0f, cntAl);
                Duplicate(accB[i * cntAl], (float)0.0f, cntAl);
                Duplicate(corrBuf[i * cntAl], (float)0.0f, cntAl);
            }

            // 3) K 方向分块流式累加：双缓冲预取 W 行块
            if (kEnd > 0) {
                // W 双缓冲同步靠 hiaccWQue(深度=2) 的 EnQue/DeQue 事件与 PIPE_V/PIPE_ALL barrier。
                {
                    LocalTensor<float> w0 = this->hiaccWQue.template AllocTensor<float>();
                    const int64_t kb0 = (kBlk < kEnd) ? kBlk : kEnd;
                    for (int64_t jj = 0; jj < kb0; ++jj) {
                        const int64_t k = revK ? (kEnd - 1 - jj) : jj;
                        DataCopyPad(w0[jj * cntAl], this->inputGm.weightInputGm[k * hidden4 + c0], cpCol, ppCol);
                    }
                    this->hiaccWQue.EnQue(w0);
                }
                bool useB = false;
                for (int64_t bi = 0; bi < nKBlk; ++bi) {
                    if (bi + 1 < nKBlk) {
                        // 覆写 (bi-1)%2 槽位前，先确保上一块向量计算已结束
                        PipeBarrier<PIPE_V>();
                        LocalTensor<float> wn = this->hiaccWQue.template AllocTensor<float>();
                        const int64_t kbN = (kEnd - (bi + 1) * kBlk < kBlk) ? (kEnd - (bi + 1) * kBlk) : kBlk;
                        for (int64_t jj = 0; jj < kbN; ++jj) {
                            const int64_t kk = (bi + 1) * kBlk + jj;
                            const int64_t k = revK ? (kEnd - 1 - kk) : kk;
                            DataCopyPad(wn[jj * cntAl], this->inputGm.weightInputGm[k * hidden4 + c0], cpCol, ppCol);
                        }
                        this->hiaccWQue.EnQue(wn);
                    }
                    LocalTensor<float> wRow = this->hiaccWQue.template DeQue<float>();
                    const int64_t kb = (kEnd - bi * kBlk < kBlk) ? (kEnd - bi * kBlk) : kBlk;
                    // 块内：p=x*w 后普通累加进 blkBuf
                    for (int64_t jj = 0; jj < kb; ++jj) {
                        const int64_t kk = bi * kBlk + jj;
                        const int64_t k = revK ? (kEnd - 1 - kk) : kk;
                        LocalTensor<float> wCur = wRow[jj * cntAl];
                        for (int64_t i = 0; i < gRows; ++i) {
                            const float xVal = xBuf.GetValue(i * kAl + k);
                            LocalTensor<float> kRow = blkBuf[i * cntAl];
                            if (jj == 0) {
                                Muls(kRow, wCur, xVal, cntAl);
                            } else {
                                LocalTensor<float> pRow = pBuf[i * cntAl];
                                Muls(pRow, wCur, xVal, cntAl);
                                Add(kRow, kRow, pRow, cntAl);
                            }
                        }
                    }
                    // 块间：对 blkBuf 做一次 Kahan 补偿合并；s1/s2 ping-pong，块内值整体只舍入一次
                    for (int64_t i = 0; i < gRows; ++i) {
                        LocalTensor<float> aRow = accA[i * cntAl];
                        LocalTensor<float> bRow = accB[i * cntAl];
                        LocalTensor<float> cRow = corrBuf[i * cntAl];
                        LocalTensor<float> kRow = blkBuf[i * cntAl];
#if DYNAMIC_RNN_HIACC_PLAIN
                        if (!useB) {
                            Add(bRow, aRow, kRow, cntAl);
                        } else {
                            Add(aRow, bRow, kRow, cntAl);
                        }
#else
                        // Kahan 补偿：y=blk-corr; s2=s1+y; corr=(s2-s1)-y
                        Sub(kRow, kRow, cRow, cntAl);
                        if (!useB) {
                            Add(bRow, aRow, kRow, cntAl);
                            Sub(cRow, bRow, aRow, cntAl);
                        } else {
                            Add(aRow, bRow, kRow, cntAl);
                            Sub(cRow, aRow, bRow, cntAl);
                        }
                        Sub(cRow, cRow, kRow, cntAl);
#endif
                    }
                    useB = !useB;
                    this->hiaccWQue.FreeTensor(wRow);
                }
            }

            // 4) 加偏置并写回 workspace 行（与 cube C 输出布局一致），收尾 barrier 供下批覆写 xBuf
            // 每完成一个 K 块翻转一次 useB，故最终落点由 K 块数的奇偶决定。
            const bool useBFinal = ((nKBlk & 1) != 0);
            if (this->tiling->isBias == 1) {
                for (int64_t i = 0; i < gRows; ++i) {
                    LocalTensor<float> fRow = useBFinal ? accB[i * cntAl] : accA[i * cntAl];
                    Add(fRow, fRow, biasBuf, cntAl);
                }
            }
            PipeBarrier<PIPE_ALL>();
            for (int64_t i = 0; i < gRows; ++i) {
                const int64_t gi = rowBeg + g0 + i;
                LocalTensor<float> fRow = useBFinal ? accB[i * cntAl] : accA[i * cntAl];
                DataCopyPad(this->outputGm.workspace[gi * hidden4 + c0], fRow, cpCol);
            }
            PipeBarrier<PIPE_ALL>();
        }
    }
}
#endif

template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP32<T>::ProcessHiddenMM(int64_t tIdx)
{
    if (GetBlockIdx() < this->hiddenMMTiling.usedCoreNum) {
        if (this->tiling->direction == 1) {
            this->hiddenOffsets.COffset = this->oriHiddenOffsets.COffset +
                                          (this->tiling->timeStep - 1 - tIdx) * this->allCellSize;
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
__aicore__ inline void LstmMmSplitNDNDFP32<T>::CopyGate(LocalTensor<T>& ub, GlobalTensor<T>& gm, int64_t mIdx,
                                                        int64_t nIdx, int64_t gateOffset)
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
                dataCopyParams, padParams);
    this->qidVecIn.EnQue(ub);
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP32<T>::CopyWithSigmoid(LocalTensor<T>& dstUb, GlobalTensor<T>& mixGm,
                                                               int64_t mIdx, int64_t nIdx, int64_t gateOffset)
{
    LocalTensor<T> ubLocalIn = this->qidVecIn.template AllocTensor<T>();
    this->CopyGate(ubLocalIn, mixGm, mIdx, nIdx, gateOffset);
    ubLocalIn = this->qidVecIn.template DeQue<T>();
    Sigmoid(dstUb, ubLocalIn, this->calcSizeAlign);
    this->qidVecIn.FreeTensor(ubLocalIn);
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP32<T>::CopyWithSigmoidAddBias(LocalTensor<float>& dstUb,
                                                                      GlobalTensor<float>& mixGm, int64_t mIdx,
                                                                      int64_t nIdx, int64_t gateOffset)
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
__aicore__ inline void LstmMmSplitNDNDFP32<T>::CopyWithTanh(LocalTensor<T>& dstUb, GlobalTensor<T>& mixGm, int64_t mIdx,
                                                            int64_t nIdx, int64_t gateOffset)
{
    LocalTensor<T> ubLocalIn = this->qidVecIn.template AllocTensor<T>();
    this->CopyGate(ubLocalIn, mixGm, mIdx, nIdx, gateOffset);
    ubLocalIn = this->qidVecIn.template DeQue<T>();
    Tanh(dstUb, ubLocalIn, this->calcSizeAlign);
    this->qidVecIn.FreeTensor(ubLocalIn);
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP32<T>::CopyWithTanhHighPrecision(LocalTensor<T>& dstUb, GlobalTensor<T>& mixGm,
                                                                         int64_t mIdx, int64_t nIdx, int64_t gateOffset,
                                                                         LocalTensor<T>& temp1, LocalTensor<T>& temp2,
                                                                         int64_t calcSizeAlign)
{
    LocalTensor<T> ubLocalIn = this->qidVecIn.template AllocTensor<T>();
    this->CopyGate(ubLocalIn, mixGm, mIdx, nIdx, gateOffset);
    ubLocalIn = this->qidVecIn.template DeQue<T>();
    Tanh(dstUb, ubLocalIn, this->calcSizeAlign);
    PipeBarrier<PIPE_V>();
    this->TanhPartialHighPrecision(ubLocalIn, dstUb, temp1, temp2, calcSizeAlign);
    PipeBarrier<PIPE_V>();
    this->qidVecIn.FreeTensor(ubLocalIn);
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP32<T>::CopyWithMul(LocalTensor<T>& dstUb, LocalTensor<T>& other,
                                                           GlobalTensor<T>& mixGm, int64_t mIdx, int64_t nIdx)
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
                dataCopyParams, padParams);
    this->qidVecIn.EnQue(ubLocalIn);
    ubLocalIn = this->qidVecIn.template DeQue<T>();
    Mul(dstUb, ubLocalIn, other, this->calcSizeAlign);
    this->qidVecIn.FreeTensor(ubLocalIn);
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP32<T>::CopyInHC(LocalTensor<T>& dstUb, GlobalTensor<T>& mixGm, int64_t tIdx,
                                                        int64_t mIdx, int64_t nIdx)
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
                dataCopyParams, padParams);
    this->qidVecIn.EnQue(ubLocalIn);
    ubLocalIn = this->qidVecIn.template DeQue<T>();
    PipeBarrier<PIPE_V>();
    Adds(dstUb, ubLocalIn, (float)0.0, this->calcSizeAlign);
    PipeBarrier<PIPE_V>();
    this->qidVecIn.FreeTensor(ubLocalIn);
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP32<T>::CopyInSeq(LocalTensor<T>& dstUb, GlobalTensor<T>& mixGm, int64_t tIdx,
                                                         int64_t mIdx, int64_t nIdx)
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
                dataCopyParams, padParams);
    this->qidVecIn.EnQue(ubLocalIn);
    ubLocalIn = this->qidVecIn.template DeQue<T>();
    PipeBarrier<PIPE_V>();
    Adds(dstUb, ubLocalIn, (float)0.0, this->calcSizeAlign);
    PipeBarrier<PIPE_V>();
    this->qidVecIn.FreeTensor(ubLocalIn);
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP32<T>::CopyOutput(GlobalTensor<T>& gm, LocalTensor<T>& ub, int64_t tIdx,
                                                          int64_t mIdx, int64_t nIdx)
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
__aicore__ inline void LstmMmSplitNDNDFP32<T>::CalcVectorBlockSize(int64_t mIdx, int64_t nIdx)
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
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP32<T>::ProcessSeqLengthUpdateC(LocalTensor<T>& updateC, int64_t tIdx,
                                                                       int64_t mIdx, int64_t nIdx)
{
    if (this->tiling->isSeqLength == 1) {
        auto initC = this->ubLocal2;
        auto seqLength = this->ubLocal4;
        CopyInHC(initC, this->inputGm.initCGm, 0, mIdx, nIdx);
        CopyInSeq(seqLength, this->inputGm.seqLengthGm, tIdx, mIdx, nIdx);
        Mul(updateC, updateC, seqLength, this->calcSizeAlign);
        PipeBarrier<PIPE_V>();
        Muls(seqLength, seqLength, (T)-1.0f, this->calcSizeAlign);
        PipeBarrier<PIPE_V>();
        Adds(seqLength, seqLength, (T)1.0f, this->calcSizeAlign);
        PipeBarrier<PIPE_V>();
        Mul(initC, initC, seqLength, this->calcSizeAlign);
        PipeBarrier<PIPE_V>();
        Add(updateC, updateC, initC, this->calcSizeAlign);
        PipeBarrier<PIPE_V>();
    }
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP32<T>::ProcessSeqLengthUpdateH(LocalTensor<T>& updateH, int64_t tIdx,
                                                                       int64_t mIdx, int64_t nIdx)
{
    if (this->tiling->isSeqLength == 1) {
        PipeBarrier<PIPE_V>();
        auto updateY = this->ubLocal1;
        auto initH = this->ubLocal2;
        auto seqLength = this->ubLocal4;
        // 现在是反转的mask，先拿到增量initH
        CopyInHC(initH, this->inputGm.initHGm, 0, mIdx, nIdx);
        Mul(initH, initH, seqLength, this->calcSizeAlign);
        PipeBarrier<PIPE_V>();
        // 反转恢复
        Muls(seqLength, seqLength, (T)-1.0f, this->calcSizeAlign);
        PipeBarrier<PIPE_V>();
        Adds(seqLength, seqLength, (T)1.0f, this->calcSizeAlign);
        PipeBarrier<PIPE_V>();
        Mul(updateY, updateH, seqLength, this->calcSizeAlign);
        PipeBarrier<PIPE_V>();
        Add(updateH, updateY, initH, this->calcSizeAlign);
        CopyOutput(this->outputGm.outYGm, updateY, tIdx, mIdx, nIdx);
    } else {
        CopyOutput(this->outputGm.outYGm, updateH, tIdx, mIdx, nIdx);
    }
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP32<T>::ProcessTanhC(LocalTensor<T>& updateC, LocalTensor<T>& temp1,
                                                            int64_t tIdx, int64_t mIdx, int64_t nIdx)
{
    // tanh(c) 1 [2] 3 4 -> 1 [2] 3 4
    auto cTanh = this->ubLocal2;
    Tanh(cTanh, updateC, this->calcSizeAlign); // 这里只有u3可用，还差1块UB，暂且从qidVecIn中取
    LocalTensor<T> temp2Tensor = this->qidVecIn.template AllocTensor<T>();
    this->TanhPartialHighPrecision(updateC, cTanh, temp1, temp2Tensor, this->calcSizeAlign);
    this->qidVecIn.FreeTensor(temp2Tensor);
    if (this->tiling->isTraining == 1) {
        CopyOutput(this->outputGm.outTanhCGm, cTanh, tIdx, mIdx, nIdx);
    }
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP32<T>::ProcessVectorOnce(int64_t tIdx, int64_t mIdx, int64_t nIdx,
                                                                 GlobalTensor<T>& mixGm)
{
    this->CalcVectorBlockSize(mIdx, nIdx);

    PipeBarrier<PIPE_V>();

    // f 1 2 3 4 -> [1] 2 3 4
    auto fSigmoid = this->ubLocal1;
    this->CopyWithSigmoidAddBias(fSigmoid, mixGm, mIdx, nIdx, this->fOffset);
    if (this->tiling->isTraining == 1) {
        CopyOutput(this->outputGm.outFGm, fSigmoid, tIdx, mIdx, nIdx);
    }
    PipeBarrier<PIPE_V>();

    // [1] 2 3 4 -> 1 [2] 3 4
    auto cTmp1 = this->ubLocal2;
    CopyWithMul(cTmp1, fSigmoid, this->inputGm.initCGm, mIdx, nIdx);
    PipeBarrier<PIPE_V>();

    // j [1] [2] 3 4 -> [1] [2] [3] 4
    auto jTanh = this->ubLocal3;
    CopyWithTanhHighPrecision(jTanh, mixGm, mIdx, nIdx, this->jOffset, this->ubLocal1, this->ubLocal4,
                              this->calcSizeAlign);
    if (this->tiling->isTraining == 1) {
        CopyOutput(this->outputGm.outJGm, jTanh, tIdx, mIdx, nIdx);
    }
    PipeBarrier<PIPE_V>();

    // i 1 [2] 3 4 -> [1] [2] 3 4
    auto iSigmoid = this->ubLocal1;
    CopyWithSigmoid(iSigmoid, mixGm, mIdx, nIdx, this->iOffset);
    if (this->tiling->isTraining == 1) {
        CopyOutput(this->outputGm.outIGm, iSigmoid, tIdx, mIdx, nIdx);
    }
    PipeBarrier<PIPE_V>();

    // i * j [1] [2] [3] 4 -> 1 [2] 3 [4]
    auto cTmp2 = this->ubLocal4;
    Mul(cTmp2, jTanh, iSigmoid, this->calcSizeAlign);
    PipeBarrier<PIPE_V>();

    // i * j + f * c 1 [2] 3 [4] -> [1] 2 3 4
    auto updateC = this->ubLocal1;
    Add(updateC, cTmp1, cTmp2, this->calcSizeAlign);

    this->ProcessSeqLengthUpdateC(updateC, tIdx, mIdx, nIdx);

    if (this->tiling->cellClip > 0) {
        PipeBarrier<PIPE_V>();
        Mins(updateC, updateC, static_cast<float>(this->tiling->cellClip), this->calcSizeAlign);
    }

    CopyOutput(this->outputGm.outCGm, updateC, tIdx, mIdx, nIdx);
    PipeBarrier<PIPE_V>();

    this->ProcessTanhC(updateC, this->ubLocal3, tIdx, mIdx, nIdx);
    auto cTanh = this->ubLocal2;

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

    this->ProcessSeqLengthUpdateH(updateH, tIdx, mIdx, nIdx);

    CopyOutput(this->outputGm.outHGm, updateH, tIdx, mIdx, nIdx);
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP32<T>::ProcessInitSeqLengthC(LocalTensor<T>& updateC, int64_t mIdx,
                                                                     int64_t nIdx)
{
    if (this->tiling->isSeqLength == 1) {
        auto seqLength = this->ubLocal3;
        this->CopyInSeq(seqLength, this->inputGm.seqLengthGm, 0, mIdx, nIdx);
        Mul(updateC, updateC, seqLength, this->calcSizeAlign);
        PipeBarrier<PIPE_V>();
    }
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP32<T>::ProcessInitSeqLengthH(LocalTensor<T>& updateH, int64_t mIdx,
                                                                     int64_t nIdx)
{
    if (this->tiling->isSeqLength == 1) {
        auto seqLength = this->ubLocal3;
        PipeBarrier<PIPE_V>();
        Mul(updateH, updateH, seqLength, this->calcSizeAlign);
    }
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP32<T>::ProcessVectorInitHC(int64_t mIdx, int64_t nIdx, GlobalTensor<T>& mixGm)
{
    this->CalcVectorBlockSize(mIdx, nIdx);

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
    CopyWithTanhHighPrecision(jTanh, mixGm, mIdx, nIdx, this->jOffset, this->ubLocal2, this->ubLocal4,
                              this->calcSizeAlign);
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

    this->ProcessInitSeqLengthC(updateC, mIdx, nIdx);

    if (this->tiling->cellClip > 0) {
        PipeBarrier<PIPE_V>();
        Mins(updateC, updateC, static_cast<float>(this->tiling->cellClip), this->calcSizeAlign);
    }

    this->CopyOutput(this->outputGm.outCGm, updateC, 0, mIdx, nIdx);
    PipeBarrier<PIPE_V>();

    this->ProcessTanhC(updateC, this->ubLocal1, 0, mIdx, nIdx);
    auto cTanh = this->ubLocal2;

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

    this->ProcessInitSeqLengthH(updateH, mIdx, nIdx);

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
            this->inputGm.initCGm = this->outputGm.outCGm[(this->tiling->timeStep - 1) * this->tiling->batch *
                                                          this->tiling->hiddenSize];
            this->inputGm.initHGm = this->outputGm.outHGm[(this->tiling->timeStep - 1) * this->tiling->batch *
                                                          this->tiling->hiddenSize];
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
            this->inputGm.initCGm = this->outputGm.outCGm[(this->tiling->timeStep - 1 - tIdx) * this->tiling->batch *
                                                          this->tiling->hiddenSize];
            this->inputGm.initHGm = this->outputGm.outHGm[(this->tiling->timeStep - 1 - tIdx) * this->tiling->batch *
                                                          this->tiling->hiddenSize];
        } else {
            this->inputGm.initCGm = this->outputGm.outCGm[tIdx * this->tiling->batch * this->tiling->hiddenSize];
            this->inputGm.initHGm = this->outputGm.outHGm[tIdx * this->tiling->batch * this->tiling->hiddenSize];
        }
    }
}
