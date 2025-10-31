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
 * \file LstmFP16.h
 * \brief
 */
#ifndef _ASCENDC_LSTMFP16_H_
#define _ASCENDC_LSTMFP16_H_

#include "dynamic_rnn_common.h"
__aicore__ inline constexpr auto GetRnnMmConfig()
{
    auto cfg = GetNormalConfig();
    cfg.enableSetOrgShape = false;
    cfg.enableEnd = false;
    cfg.enableGetTensorC = false;
    cfg.enableQuantVector = false;
    cfg.enableSetDefineData = false;
    return cfg;
}
constexpr auto RNN_MM_CFG = GetRnnMmConfig();
template <typename T>
class LstmMmSplitNDNDFP16 : public LstmMmSplitNDNDBase<T> {
public:
    __aicore__ inline LstmMmSplitNDNDFP16() = default;
    __aicore__ inline void Process();

public:
    __aicore__ inline void ProcessInputMM();
    __aicore__ inline void ProcessHiddenMM(int64_t tIdx);
    __aicore__ inline void ProcessVectorOnce(
        int64_t tIdx, int64_t mIdx, int64_t nIdx, AscendC::GlobalTensor<float> &mixGm);
    __aicore__ inline void ProcessVectorInitHC(int64_t mIdx, int64_t nIdx, AscendC::GlobalTensor<float> &mixGm);
    __aicore__ inline void ProcessVector(int64_t tIdx);
    __aicore__ inline void ProcessInitalT();
    __aicore__ inline void CopyInHCSeq(
        AscendC::LocalTensor<float> &dstUb, AscendC::GlobalTensor<T> &mixGm, int64_t off);
    __aicore__ inline void CopyOutput(AscendC::GlobalTensor<T> &gm, AscendC::LocalTensor<float> &ub, int64_t off);
    __aicore__ inline void CalcVecScaler(
        int64_t tIdx, int64_t mIdx, int64_t nIdx, int64_t &off1, int64_t &off2, int64_t &off3);
    __aicore__ inline void CopyInFJ(AscendC::LocalTensor<float> &dst, AscendC::GlobalTensor<float> &mixGm, int64_t off);
    __aicore__ inline void CopyInIO(AscendC::LocalTensor<float> &dst, AscendC::GlobalTensor<float> &mixGm, int64_t off);
    __aicore__ inline void CopyInC(AscendC::LocalTensor<T> &dst, AscendC::GlobalTensor<T> &mixGm, const int64_t off);
    __aicore__ inline void AddfSigmoid(AscendC::LocalTensor<float> &dst, AscendC::LocalTensor<float> &src, int64_t off);
    __aicore__ inline void CaliSigmoid(AscendC::LocalTensor<float> &dst, AscendC::LocalTensor<float> &src, int64_t off);
    __aicore__ inline void CaljTanh(AscendC::LocalTensor<float> &dst, AscendC::LocalTensor<float> &src, int64_t off);
    __aicore__ inline void CaloSigmoid(AscendC::LocalTensor<float> &dst, AscendC::LocalTensor<float> &src, int64_t off);
    __aicore__ inline void InitCMulfSigmoid(
        AscendC::LocalTensor<float> &dst, AscendC::LocalTensor<T> &src1, AscendC::LocalTensor<float> &src2);
    __aicore__ inline void CalAddTanh(AscendC::LocalTensor<float> &dst, AscendC::LocalTensor<float> &src1,
        AscendC::LocalTensor<float> &src2, int64_t off1, int64_t off2);
    __aicore__ inline void CalAddTanht0(AscendC::LocalTensor<float> &dst, AscendC::LocalTensor<float> &src1,
        AscendC::LocalTensor<float> &src2, int64_t off1, int64_t off2);
    __aicore__ inline void CopyOutYH(AscendC::LocalTensor<float> &src, int64_t off1, int64_t off2);
    __aicore__ inline void CopyOutYHt0(AscendC::LocalTensor<float> &src, int64_t off);

public:
    // describe Matmul input/output dtype&format
    matmul::Matmul<matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T>,
        matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T>,
        matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>,
        matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T>, RNN_MM_CFG>
        inputMM;

    matmul::Matmul<matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T>,
        matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T>,
        matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>,
        matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>, RNN_MM_CFG>
        hiddenMM;
};

#endif