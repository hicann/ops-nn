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
 * \file LstmFP32.h
 * \brief
 */
#ifndef _ASCENDC_LSTMFP32_H_
#define _ASCENDC_LSTMFP32_H_

#include "dynamic_rnn_common.h"

template <typename T>
class LstmMmSplitNDNDFP32 : public LstmMmSplitNDNDBase<T> {
public:
    __aicore__ inline LstmMmSplitNDNDFP32() = default;
    __aicore__ inline void Process();

protected:
    __aicore__ inline void ProcessInputMM();
    __aicore__ inline void ProcessHiddenMM(int64_t tIdx);
    __aicore__ inline void ProcessVectorOnce(int64_t tIdx, int64_t mIdx, int64_t nIdx, AscendC::GlobalTensor<T> &mixGm);
    __aicore__ inline void ProcessVectorInitHC(int64_t mIdx, int64_t nIdx, AscendC::GlobalTensor<T> &mixGm);
    __aicore__ inline void ProcessVector(int64_t tIdx);
    __aicore__ inline void ProcessInitalT();
    __aicore__ inline void CopyGate(
        AscendC::LocalTensor<T> &ub, AscendC::GlobalTensor<T> &gm, int64_t mIdx, int64_t nIdx, int64_t gateOffset);
    __aicore__ inline void CopyWithSigmoid(AscendC::LocalTensor<T> &dstUb, AscendC::GlobalTensor<T> &mixGm,
        int64_t mIdx, int64_t nIdx, int64_t gateOffset);
    __aicore__ inline void CopyWithSigmoidAddBias(AscendC::LocalTensor<float> &dstUb,
        AscendC::GlobalTensor<float> &mixGm, int64_t mIdx, int64_t nIdx, int64_t gateOffset);
    __aicore__ inline void CopyWithTanh(AscendC::LocalTensor<T> &dstUb, AscendC::GlobalTensor<T> &mixGm, int64_t mIdx,
        int64_t nIdx, int64_t gateOffset);
    __aicore__ inline void CopyWithMul(AscendC::LocalTensor<T> &dstUb, AscendC::LocalTensor<T> &other,
        AscendC::GlobalTensor<T> &mixGm, int64_t mIdx, int64_t nIdx);
    __aicore__ inline void CopyInHC(
        AscendC::LocalTensor<T> &dstUb, AscendC::GlobalTensor<T> &mixGm, int64_t tIdx, int64_t mIdx, int64_t nIdx);
    __aicore__ inline void CopyInSeq(
        AscendC::LocalTensor<T> &dstUb, AscendC::GlobalTensor<T> &mixGm, int64_t tIdx, int64_t mIdx, int64_t nIdx);
    __aicore__ inline void CopyOutput(
        AscendC::GlobalTensor<T> &gm, AscendC::LocalTensor<T> &ub, int64_t tIdx, int64_t mIdx, int64_t nIdx);

public:
    // describe Matmul input/output dtype&format
    matmul::Matmul<matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T>,
        matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T>,
        matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>,
        matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T>>
        inputMM;

    matmul::Matmul<matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T>,
        matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T>,
        matmul::MatmulType<AscendC::TPosition::VECCALC, CubeFormat::ND, float>>
        hiddenMM;
};

#endif
