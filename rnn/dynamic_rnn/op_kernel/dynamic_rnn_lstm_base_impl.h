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
 * \file dynamic_rnn_lstm_base_impl.h
 * \brief LstmMmSplitNDNDBase 模板类的成员函数实现。
 *        本头文件由 dynamic_rnn_common.h 在类声明结束之后引入，不可单独被外部引用。
 */
#ifndef _DYNAMIC_RNN_LSTM_BASE_IMPL_H_
#define _DYNAMIC_RNN_LSTM_BASE_IMPL_H_

template <typename T>
__aicore__ inline void LstmMmSplitNDNDBase<T>::GetCoreIndex(TCubeTiling& param, int32_t& subKIndx, tailSize& mmTail,
                                                            int32_t kSize)
{
    auto temp0 = this->Ceil(param.M, param.singleCoreM);
    auto temp1 = this->Ceil(param.N, param.singleCoreN);
    auto temp2 = this->Ceil(kSize, param.singleCoreK); // 不切K, 应该=1
    if (temp0 == 0) {
        temp0 = 1;
    }
    if (temp2 == 0) {
        temp2 = 1;
    }
    auto divideKcoreNum = param.usedCoreNum / temp2;
    mmTail.mCoreIndx = (GetBlockIdx() % divideKcoreNum) % temp0;
    mmTail.nCoreIndx = (GetBlockIdx() % divideKcoreNum) / temp0;
    subKIndx = GetBlockIdx() / divideKcoreNum; // 缺省为0
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDBase<T>::CalcGMOffset(TCubeTiling& param, TRnnOffsets& offset, tailSize& mmTail,
                                                            int32_t kSize)
{
    int32_t subKIndx;
    this->GetCoreIndex(param, subKIndx, mmTail, kSize);
    offset.AOffset = mmTail.mCoreIndx * kSize * param.singleCoreM;
    offset.BOffset = mmTail.nCoreIndx * param.singleCoreN;
    offset.BiasOffset = mmTail.nCoreIndx * param.singleCoreN;

    mmTail.nCoreLoop = this->Ceil(param.N, param.singleCoreN);
    mmTail.tailSingleCoreN = param.N - (mmTail.nCoreLoop - 1) * param.singleCoreN;
    mmTail.notTailNCoreCount = mmTail.nCoreLoop - 1;
    mmTail.mCoreLoop = this->Ceil(param.M, param.singleCoreM);
    mmTail.tailSingleCoreM = param.M - (mmTail.mCoreLoop - 1) * param.singleCoreM;
    mmTail.notTailMCoreCount = mmTail.mCoreLoop - 1;
    offset.COffset = mmTail.mCoreIndx * param.N * param.singleCoreM + mmTail.nCoreIndx * param.singleCoreN;
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDBase<T>::InitBuffersOffsets()
{
    this->CalcGMOffset(this->hiddenMMTiling, this->hiddenOffsets, this->hiddenTail,
                       static_cast<int32_t>(this->tiling->hiddenSize));
    this->CalcGMOffset(this->inputMMTiling, this->inputOffsets, this->inputTail,
                       static_cast<int32_t>(this->tiling->inputSize));
    this->oneCellSize = this->tiling->batch * this->tiling->hiddenSize;
    this->allCellSize = this->oneCellSize * LSTM_GATE_SIZE;
    this->oriHiddenOffsets = this->hiddenOffsets;
    this->oriInputOffsets = this->inputOffsets;
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDBase<T>::SetInputGmBuffers(GM_ADDR inputX, GM_ADDR weight, GM_ADDR bias,
                                                                 GM_ADDR seqLength, GM_ADDR initH, GM_ADDR initC)
{
    this->inputGm.xGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(inputX),
                                      this->tiling->timeStep * this->tiling->batch * this->tiling->inputSize);
    this->inputGm.weightInputGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(weight),
                                                this->tiling->inputSize * LSTM_GATE_SIZE * this->tiling->hiddenSize);
    this->inputGm.weightHiddenGm.SetGlobalBuffer(
        reinterpret_cast<__gm__ T*>(weight +
                                    this->tiling->inputSize * LSTM_GATE_SIZE * this->tiling->hiddenSize * sizeof(T)),
        this->tiling->hiddenSize * LSTM_GATE_SIZE * this->tiling->hiddenSize);

    this->inputGm.biasGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(bias), LSTM_GATE_SIZE * this->tiling->hiddenSize);

    if (this->tiling->isSeqLength != 0) {
        this->inputGm.seqLengthGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ T*>(seqLength),
            this->tiling->timeStep * this->tiling->batch * this->tiling->hiddenSize);
    }

    if (this->tiling->isInithc != 0) {
        this->inputGm.initHGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(initH),
                                              this->tiling->batch * this->tiling->hiddenSize);
        this->inputGm.initCGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(initC),
                                              this->tiling->batch * this->tiling->hiddenSize);
    }
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDBase<T>::SetOutputGmBuffers(GM_ADDR outputY, GM_ADDR outputH, GM_ADDR outputC,
                                                                  GM_ADDR outputI, GM_ADDR outputJ, GM_ADDR outputF,
                                                                  GM_ADDR outputO, GM_ADDR outputTanhC,
                                                                  GM_ADDR workspace)
{
    this->outputGm.outYGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(outputY),
                                          this->tiling->timeStep * this->tiling->batch * this->tiling->hiddenSize);
    this->outputGm.outHGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(outputH),
                                          this->tiling->timeStep * this->tiling->batch * this->tiling->hiddenSize);
    this->outputGm.outCGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(outputC),
                                          this->tiling->timeStep * this->tiling->batch * this->tiling->hiddenSize);
    if (this->tiling->isTraining == 1) {
        this->outputGm.outIGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(outputI),
                                              this->tiling->timeStep * this->tiling->batch * this->tiling->hiddenSize);
        this->outputGm.outJGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(outputJ),
                                              this->tiling->timeStep * this->tiling->batch * this->tiling->hiddenSize);
        this->outputGm.outFGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(outputF),
                                              this->tiling->timeStep * this->tiling->batch * this->tiling->hiddenSize);
        this->outputGm.outOGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(outputO),
                                              this->tiling->timeStep * this->tiling->batch * this->tiling->hiddenSize);
        this->outputGm.outTanhCGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ T*>(outputTanhC),
            this->tiling->timeStep * this->tiling->batch * this->tiling->hiddenSize);
    }
    this->outputGm.workspace.SetGlobalBuffer(
        reinterpret_cast<__gm__ float*>(workspace),
        this->tiling->timeStep * this->tiling->batch * LSTM_GATE_SIZE * this->tiling->hiddenSize);
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDBase<T>::InitBuffers(GM_ADDR inputX, GM_ADDR weight, GM_ADDR bias,
                                                           GM_ADDR seqLength, GM_ADDR initH, GM_ADDR initC, GM_ADDR wCi,
                                                           GM_ADDR wCf, GM_ADDR wCo, GM_ADDR mask, GM_ADDR outputY,
                                                           GM_ADDR outputH, GM_ADDR outputC, GM_ADDR outputI,
                                                           GM_ADDR outputJ, GM_ADDR outputF, GM_ADDR outputO,
                                                           GM_ADDR outputTanhC, GM_ADDR workspace)
{
    this->InitBuffersOffsets();
    this->SetInputGmBuffers(inputX, weight, bias, seqLength, initH, initC);
    this->SetOutputGmBuffers(outputY, outputH, outputC, outputI, outputJ, outputF, outputO, outputTanhC, workspace);
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDBase<T>::InitVars()
{
    int64_t ubSize = 21504; // after div the max num of exiting node at the same time,include 4 gate
    if constexpr (std::is_same<T, float>::value) {
        ubSize = 16384;
    }
    int64_t calcMaxSize = ubSize / sizeof(float);
    int64_t blockN = this->tiling->usedCoreNum;
    this->blockSize = 32 / sizeof(T);
    this->calBlockSize = 32 / sizeof(float);
    int64_t calcSize = this->blockSize;
    if constexpr (std::is_same<T, float>::value) {
        calcSize = this->calBlockSize;
    }
    this->vectorCoreM = this->Ceil(this->tiling->batch, blockN);
    this->vectorCoreNum = this->Ceil(this->tiling->batch, this->vectorCoreM);
    this->vectorTailM = this->tiling->batch % this->vectorCoreM ? this->tiling->batch % this->vectorCoreM :
                                                                  this->vectorCoreM;

    this->vectorSplitN = this->Ceil(this->tiling->hiddenSize, calcMaxSize);
    if (this->vectorSplitN == 1) {
        this->vectorBaseN = this->tiling->hiddenSize;
        this->vectorTailN = 0;
    } else {
        this->vectorBaseN = this->Ceil(this->Ceil(this->tiling->hiddenSize, this->vectorSplitN), calcSize) * calcSize;
        this->vectorTailN = this->tiling->hiddenSize - this->vectorBaseN * (this->vectorSplitN - 1);
    }

    this->vectorBaseM = ((calcMaxSize / this->vectorBaseN) > this->vectorCoreM) ? this->vectorCoreM :
                                                                                  (calcMaxSize / this->vectorBaseN);

    this->vectorBaseTailM = this->vectorCoreM % this->vectorBaseM;
    this->vectorTailTailM = this->vectorTailM % this->vectorBaseM;

    this->vectorSplitM = this->Ceil(this->vectorCoreM, this->vectorBaseM);
    this->vectorTailSplitM = this->Ceil(this->vectorTailM, this->vectorBaseM);

    this->baseVector = this->vectorBaseM * this->Ceil(this->vectorBaseN, calcSize) * calcSize;

    this->iOffset = 0;
    this->jOffset = this->tiling->gateOrder == 0 ? this->tiling->hiddenSize : 2 * this->tiling->hiddenSize;
    this->fOffset = this->tiling->gateOrder == 0 ? 2 * this->tiling->hiddenSize : this->tiling->hiddenSize;
    this->oOffset = 3 * this->tiling->hiddenSize;
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDBase<T>::InitQue()
{
    this->pipe.InitBuffer(this->qidVecIn, 1, this->baseVector * sizeof(float));
    this->pipe.InitBuffer(this->qidVecOut, 1, this->baseVector * sizeof(T));
    this->pipe.InitBuffer(this->calcBuf, 4 * this->baseVector * sizeof(float));
    if constexpr (std::is_same<T, float>::value) {
#if DYNAMIC_RNN_HIACC_GEMM_INPUT
        int64_t hiaccCchTmp = this->Ceil(this->tiling->hiddenSize * LSTM_GATE_SIZE, this->calBlockSize) *
                              this->calBlockSize;
        if (hiaccCchTmp > HIACC_MM_CHUNK) {
            hiaccCchTmp = HIACC_MM_CHUNK;
        }
        if (hiaccCchTmp < this->calBlockSize) {
            hiaccCchTmp = this->calBlockSize;
        }
        this->hiaccCch = hiaccCchTmp;
        int64_t hiaccKAlTmp = this->Ceil(this->tiling->inputSize, this->calBlockSize) * this->calBlockSize;
        if (hiaccKAlTmp < this->calBlockSize) {
            hiaccKAlTmp = this->calBlockSize;
        }
        this->hiaccKAl = hiaccKAlTmp;
        // 新路径预算(fp32 元素数)= HIACC_UB_BUDGET 扣除既有队列 24*baseVector 字节。
        const int64_t hiaccBudgetF = (HIACC_UB_BUDGET - 24 * this->baseVector) / sizeof(float);
        int64_t hiaccKBTmp = HIACC_WBLK_MAX;
        const int64_t hiaccKEnd = this->tiling->inputSize;
        if (hiaccKBTmp > hiaccKEnd) {
            hiaccKBTmp = hiaccKEnd;
        }
        if (hiaccKBTmp < 1) {
            hiaccKBTmp = 1;
        }
        // W 双缓冲与 scratch 若超出预算，先收缩 K 分块
        while (hiaccKBTmp > 1 && 2 * hiaccKBTmp * hiaccCchTmp + 2 * hiaccCchTmp > hiaccBudgetF) {
            hiaccKBTmp = hiaccKBTmp / 2;
        }
        // 每行 UB 需求 = accA/accB/corr(3*cch) + x(kAl) + pBuf(cch) + blkBuf(cch)，另加 bias scratch(cch)。
        const int64_t hiaccPerRowF = 5 * hiaccCchTmp + hiaccKAlTmp;
        int64_t hiaccRowsTmp = (hiaccBudgetF - 2 * hiaccKBTmp * hiaccCchTmp - hiaccCchTmp) / hiaccPerRowF;
        if (hiaccRowsTmp < 1) {
            hiaccRowsTmp = 1;
        }
        if (hiaccRowsTmp > HIACC_ROWS_MAX) {
            hiaccRowsTmp = HIACC_ROWS_MAX;
        }
        // 输入 GEMM 行空间覆盖全部 (s,b) 行，按启动核数切分。
        const int64_t hiaccCores = (GetBlockNum() > 0) ? GetBlockNum() : 1;
        const int64_t hiaccMaxRowsCore = this->Ceil(this->tiling->timeStep * this->tiling->batch, hiaccCores);
        if (hiaccRowsTmp > hiaccMaxRowsCore) {
            hiaccRowsTmp = hiaccMaxRowsCore;
        }
        this->hiaccKB = hiaccKBTmp;
        this->hiaccRows = hiaccRowsTmp;
        const int64_t hiaccBufFloats = hiaccRowsTmp * hiaccPerRowF + hiaccCchTmp;
        this->pipe.InitBuffer(this->hiaccWQue, 2, hiaccKBTmp * hiaccCchTmp * sizeof(float));
        this->pipe.InitBuffer(this->hiaccBuf, hiaccBufFloats * sizeof(float));
#endif
    }
    if constexpr (!std::is_same<T, float>::value) {
        this->pipe.InitBuffer(this->qidCIn, 1, this->baseVector * sizeof(T));
        this->pipe.InitBuffer(this->qidVecIn2, 1, this->baseVector * sizeof(float));
    }
    // Init Local Tensors
    this->ubLocal1 = this->calcBuf.template Get<float>(4 * this->baseVector);
    this->ubLocal2 = this->ubLocal1[this->baseVector];
    this->ubLocal3 = this->ubLocal2[this->baseVector];
    this->ubLocal4 = this->ubLocal3[this->baseVector];
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDBase<T>::Init(GM_ADDR inputX, GM_ADDR weight, GM_ADDR bias, GM_ADDR seqLength,
                                                    GM_ADDR initH, GM_ADDR initC, GM_ADDR wCi, GM_ADDR wCf, GM_ADDR wCo,
                                                    GM_ADDR mask, GM_ADDR outputY, GM_ADDR outputH, GM_ADDR outputC,
                                                    GM_ADDR outputI, GM_ADDR outputJ, GM_ADDR outputF, GM_ADDR outputO,
                                                    GM_ADDR outputTanhC,
                                                    const DynamicRNNTilingData* __restrict rnnTiling, GM_ADDR workspace)
{
    this->tiling = rnnTiling;
    this->inputMMTiling = this->tiling->inputMMParam;
    this->hiddenMMTiling = this->tiling->hiddenMMParam;
    this->InitBuffers(inputX, weight, bias, seqLength, initH, initC, wCi, wCf, wCo, mask, outputY, outputH, outputC,
                      outputI, outputJ, outputF, outputO, outputTanhC, workspace);
    this->InitVars();
    this->InitQue();
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDBase<T>::SetInputGmBuffersV2(GM_ADDR inputX, GM_ADDR weightInput,
                                                                   GM_ADDR weightHidden, GM_ADDR bias,
                                                                   GM_ADDR seqLength, GM_ADDR initH, GM_ADDR initC)
{
    this->inputGm.xGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(inputX),
                                      this->tiling->timeStep * this->tiling->batch * this->tiling->inputSize);
    this->inputGm.weightInputGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(weightInput),
                                                this->tiling->inputSize * LSTM_GATE_SIZE * this->tiling->hiddenSize);
    this->inputGm.weightHiddenGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(weightHidden),
                                                 this->tiling->hiddenSize * LSTM_GATE_SIZE * this->tiling->hiddenSize);

    if (this->tiling->isBias == 1) {
        this->inputGm.biasGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(bias),
                                             LSTM_GATE_SIZE * this->tiling->hiddenSize);
    }
    if (this->tiling->isSeqLength != 0) {
        this->inputGm.seqLengthGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ T*>(seqLength),
            this->tiling->timeStep * this->tiling->batch * this->tiling->hiddenSize);
    }
    if (this->tiling->isInithc != 0) {
        this->inputGm.initHGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(initH),
                                              this->tiling->batch * this->tiling->hiddenSize);
        this->inputGm.initCGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(initC),
                                              this->tiling->batch * this->tiling->hiddenSize);
    }
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDBase<T>::InitBuffersV2(GM_ADDR inputX, GM_ADDR weightInput, GM_ADDR weightHidden,
                                                             GM_ADDR bias, GM_ADDR seqLength, GM_ADDR initH,
                                                             GM_ADDR initC, GM_ADDR wCi, GM_ADDR wCf, GM_ADDR wCo,
                                                             GM_ADDR mask, GM_ADDR outputY, GM_ADDR outputH,
                                                             GM_ADDR outputC, GM_ADDR outputI, GM_ADDR outputJ,
                                                             GM_ADDR outputF, GM_ADDR outputO, GM_ADDR outputTanhC,
                                                             GM_ADDR workspace)
{
    this->InitBuffersOffsets();
    this->SetInputGmBuffersV2(inputX, weightInput, weightHidden, bias, seqLength, initH, initC);
    this->SetOutputGmBuffers(outputY, outputH, outputC, outputI, outputJ, outputF, outputO, outputTanhC, workspace);
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDBase<T>::InitV2(
    GM_ADDR inputX, GM_ADDR weightInput, GM_ADDR weightHidden, GM_ADDR bias, GM_ADDR seqLength, GM_ADDR initH,
    GM_ADDR initC, GM_ADDR wCi, GM_ADDR wCf, GM_ADDR wCo, GM_ADDR mask, GM_ADDR outputY, GM_ADDR outputH,
    GM_ADDR outputC, GM_ADDR outputI, GM_ADDR outputJ, GM_ADDR outputF, GM_ADDR outputO, GM_ADDR outputTanhC,
    const DynamicRNNTilingData* __restrict rnnTiling, GM_ADDR workspace)
{
    this->tiling = rnnTiling;
    this->inputMMTiling = this->tiling->inputMMParam;
    this->hiddenMMTiling = this->tiling->hiddenMMParam;
    this->InitBuffersV2(inputX, weightInput, weightHidden, bias, seqLength, initH, initC, wCi, wCf, wCo, mask, outputY,
                        outputH, outputC, outputI, outputJ, outputF, outputO, outputTanhC, workspace);
    this->InitVars();
    this->InitQue();
}

template <typename T>
__aicore__ inline int64_t LstmMmSplitNDNDBase<T>::Ceil(int64_t x, int64_t y)
{
    if (y == 0) {
        return x;
    }
    return (x + y - 1) / y;
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDBase<T>::TanhPartialHighPrecision(LocalTensor<float>& inputTensor,
                                                                        LocalTensor<float>& tanhLowTensor,
                                                                        LocalTensor<float>& temp1Tensor,
                                                                        LocalTensor<float>& temp2Tensor,
                                                                        int64_t calcSizeAlign)
{
    // temp1Tensor: x^2, temp2Tensor: tanh(x)
    Mul(temp1Tensor, inputTensor, inputTensor, calcSizeAlign);
    PipeBarrier<PIPE_V>();
    Muls(temp2Tensor, temp1Tensor, 0.016090461204f, calcSizeAlign); // Tanh多项式系数
    PipeBarrier<PIPE_V>();
    Adds(temp2Tensor, temp2Tensor, -0.052421370438f, calcSizeAlign); // Tanh多项式系数
    PipeBarrier<PIPE_V>();
    Mul(temp2Tensor, temp2Tensor, temp1Tensor, calcSizeAlign);
    PipeBarrier<PIPE_V>();
    Adds(temp2Tensor, temp2Tensor, 0.133147126779f, calcSizeAlign); // Tanh多项式系数
    PipeBarrier<PIPE_V>();
    Mul(temp2Tensor, temp2Tensor, temp1Tensor, calcSizeAlign);
    PipeBarrier<PIPE_V>();
    Adds(temp2Tensor, temp2Tensor, -0.333324134737f, calcSizeAlign); // Tanh多项式系数
    PipeBarrier<PIPE_V>();
    Mul(temp2Tensor, temp2Tensor, temp1Tensor, calcSizeAlign);
    PipeBarrier<PIPE_V>();
    Adds(temp2Tensor, temp2Tensor, 0.999999873294f, calcSizeAlign); // Tanh多项式系数
    PipeBarrier<PIPE_V>();
    Mul(temp2Tensor, temp2Tensor, inputTensor, calcSizeAlign);
    PipeBarrier<PIPE_V>();
    // temp1Tensor: abs(x)
    Abs(temp1Tensor, inputTensor, calcSizeAlign);
    PipeBarrier<PIPE_V>();
    // tanhMaskTensor: mask for abs(x) <= 0.55, inplace
    auto tanhMaskTensor = temp1Tensor.template ReinterpretCast<uint8_t>();
    int64_t cnt256B = SIZE_256 / sizeof(float);
    CompareScalar(tanhMaskTensor, temp1Tensor, 0.55f, CMPMODE::LE,
                  Ceil(calcSizeAlign, cnt256B) * cnt256B); // Tanh多项式实现范围
    PipeBarrier<PIPE_V>();
    // pick higher result into tanhLowTensor
    Select(tanhLowTensor, tanhMaskTensor, temp2Tensor, tanhLowTensor, SELMODE::VSEL_TENSOR_TENSOR_MODE, calcSizeAlign);
}
#endif // _DYNAMIC_RNN_LSTM_BASE_IMPL_H_
