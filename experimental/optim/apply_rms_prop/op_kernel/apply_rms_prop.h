/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/*!
 * \file apply_rms_prop.h
 * \brief ApplyRmsProp 算子 Kernel 类定义（AscendC 标准编程）
 */

#ifndef APPLY_RMS_PROP_H
#define APPLY_RMS_PROP_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "apply_rms_prop_tiling_data.h"
#include "apply_rms_prop_tiling_key.h"

namespace NsApplyRmsProp {

using AscendC::TPipe;
using AscendC::TQue;
using AscendC::QuePosition;
using AscendC::TBuf;
using AscendC::GlobalTensor;
using AscendC::LocalTensor;
using AscendC::DataCopy;
using AscendC::DataCopyPad;
using AscendC::DataCopyParams;
using AscendC::DataCopyExtParams;
using AscendC::DataCopyPadExtParams;

constexpr int32_t BUFFER_NUM = 1;

// =============================================================================
// Base：抽取 fp32 / Cast 两条分支共用的成员、Init 公共部分、CopyIn/CopyOut、Process 模板
// =============================================================================
template <typename T>
class ApplyRmsPropBase {
public:
    __aicore__ inline ApplyRmsPropBase() {}

protected:
    __aicore__ inline void InitCommon(GM_ADDR var, GM_ADDR ms, GM_ADDR mom, GM_ADDR grad,
                                      GM_ADDR varOut, GM_ADDR msOut, GM_ADDR momOut,
                                      const ApplyRmsPropTilingData* td)
    {
        blockIdx_  = AscendC::GetBlockIdx();
        usedCores_ = td->usedCoreNum;

        if (blockIdx_ >= usedCores_) {
            isValid_ = false;
            return;
        }
        isValid_ = true;

        bool isLastCore = (blockIdx_ == usedCores_ - 1);
        coreLength_     = isLastCore ? td->lastBlockLength : td->blockLength;
        tileNum_        = isLastCore ? td->lastBlockTileNum : td->tileNum;
        lastTileLength_ = isLastCore ? td->lastBlockLastTileLength : td->lastTileLength;
        tileLength_     = td->tileLength;

        lr_    = td->lr;
        rho1m_ = td->rho1m;
        mom_   = td->mom;
        eps_   = td->eps;

        int64_t gmOffset = static_cast<int64_t>(blockIdx_) * td->blockLength;
        varGm_.SetGlobalBuffer((__gm__ T*)var  + gmOffset, coreLength_);
        msGm_.SetGlobalBuffer((__gm__ T*)ms   + gmOffset, coreLength_);
        momGm_.SetGlobalBuffer((__gm__ T*)mom  + gmOffset, coreLength_);
        gradGm_.SetGlobalBuffer((__gm__ T*)grad + gmOffset, coreLength_);
        varOutGm_.SetGlobalBuffer((__gm__ T*)varOut + gmOffset, coreLength_);
        msOutGm_.SetGlobalBuffer((__gm__ T*)msOut  + gmOffset, coreLength_);
        momOutGm_.SetGlobalBuffer((__gm__ T*)momOut + gmOffset, coreLength_);

        InitIoBuffers();
    }

    __aicore__ inline void InitIoBuffers()
    {
        pipe_.InitBuffer(qVar_,    BUFFER_NUM, tileLength_ * sizeof(T));
        pipe_.InitBuffer(qMs_,     BUFFER_NUM, tileLength_ * sizeof(T));
        pipe_.InitBuffer(qMom_,    BUFFER_NUM, tileLength_ * sizeof(T));
        pipe_.InitBuffer(qGrad_,   BUFFER_NUM, tileLength_ * sizeof(T));
        pipe_.InitBuffer(qVarOut_, BUFFER_NUM, tileLength_ * sizeof(T));
        pipe_.InitBuffer(qMsOut_,  BUFFER_NUM, tileLength_ * sizeof(T));
        pipe_.InitBuffer(qMomOut_, BUFFER_NUM, tileLength_ * sizeof(T));
    }

    template <typename Derived>
    __aicore__ inline void ProcessImpl(Derived* self)
    {
        if (!isValid_) return;
        for (int64_t i = 0; i < tileNum_; ++i) {
            int64_t curLen = (i == tileNum_ - 1) ? lastTileLength_ : tileLength_;
            CopyIn(i, curLen);
            self->Compute(curLen);
            CopyOut(i, curLen);
        }
    }

    __aicore__ inline void CopyIn(int64_t progress, int64_t curLen)
    {
        LocalTensor<T> varL  = qVar_.template AllocTensor<T>();
        LocalTensor<T> msL   = qMs_.template AllocTensor<T>();
        LocalTensor<T> momL  = qMom_.template AllocTensor<T>();
        LocalTensor<T> gradL = qGrad_.template AllocTensor<T>();

        DataCopyExtParams params{1, static_cast<uint32_t>(curLen * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        int64_t off = progress * tileLength_;
        DataCopyPad(varL,  varGm_[off],  params, padParams);
        DataCopyPad(msL,   msGm_[off],   params, padParams);
        DataCopyPad(momL,  momGm_[off],  params, padParams);
        DataCopyPad(gradL, gradGm_[off], params, padParams);

        qVar_.EnQue(varL);
        qMs_.EnQue(msL);
        qMom_.EnQue(momL);
        qGrad_.EnQue(gradL);
    }

    __aicore__ inline void CopyOut(int64_t progress, int64_t curLen)
    {
        LocalTensor<T> varNewL = qVarOut_.template DeQue<T>();
        LocalTensor<T> msNewL  = qMsOut_.template DeQue<T>();
        LocalTensor<T> momNewL = qMomOut_.template DeQue<T>();

        DataCopyExtParams params{1, static_cast<uint32_t>(curLen * sizeof(T)), 0, 0, 0};
        int64_t off = progress * tileLength_;
        DataCopyPad(varOutGm_[off], varNewL, params);
        DataCopyPad(msOutGm_[off],  msNewL,  params);
        DataCopyPad(momOutGm_[off], momNewL, params);

        qVarOut_.FreeTensor(varNewL);
        qMsOut_.FreeTensor(msNewL);
        qMomOut_.FreeTensor(momNewL);
    }

protected:
    TPipe pipe_;
    TQue<QuePosition::VECIN,  BUFFER_NUM> qVar_;
    TQue<QuePosition::VECIN,  BUFFER_NUM> qMs_;
    TQue<QuePosition::VECIN,  BUFFER_NUM> qMom_;
    TQue<QuePosition::VECIN,  BUFFER_NUM> qGrad_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> qVarOut_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> qMsOut_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> qMomOut_;

    GlobalTensor<T> varGm_, msGm_, momGm_, gradGm_;
    GlobalTensor<T> varOutGm_, msOutGm_, momOutGm_;

    int32_t blockIdx_  = 0;
    int32_t usedCores_ = 0;
    bool    isValid_   = false;
    int64_t coreLength_     = 0;
    int64_t tileLength_     = 0;
    int64_t tileNum_        = 0;
    int64_t lastTileLength_ = 0;

    float lr_    = 0.0f;
    float rho1m_ = 0.0f;
    float mom_   = 0.0f;
    float eps_   = 0.0f;
};

// =============================================================================
// fp32 直算分支
// =============================================================================
template <typename T>
class ApplyRmsPropFp32 : public ApplyRmsPropBase<T> {
    using Base = ApplyRmsPropBase<T>;
public:
    __aicore__ inline ApplyRmsPropFp32() {}

    __aicore__ inline void Init(GM_ADDR var, GM_ADDR ms, GM_ADDR mom, GM_ADDR grad,
                                GM_ADDR varOut, GM_ADDR msOut, GM_ADDR momOut,
                                const ApplyRmsPropTilingData* td)
    {
        Base::InitCommon(var, ms, mom, grad, varOut, msOut, momOut, td);
        if (!Base::isValid_) return;
        Base::pipe_.InitBuffer(tmpG2_,   Base::tileLength_ * sizeof(T));
        Base::pipe_.InitBuffer(tmpDiff_, Base::tileLength_ * sizeof(T));
        Base::pipe_.InitBuffer(tmpEps_,  Base::tileLength_ * sizeof(T));
        Base::pipe_.InitBuffer(tmpSq_,   Base::tileLength_ * sizeof(T));
    }

    __aicore__ inline void Process() { Base::template ProcessImpl(this); }

    __aicore__ inline void Compute(int64_t curLen)
    {
        LocalTensor<T> varL  = Base::qVar_.template DeQue<T>();
        LocalTensor<T> msL   = Base::qMs_.template DeQue<T>();
        LocalTensor<T> momL  = Base::qMom_.template DeQue<T>();
        LocalTensor<T> gradL = Base::qGrad_.template DeQue<T>();

        LocalTensor<T> g2   = tmpG2_.template Get<T>();
        LocalTensor<T> diff = tmpDiff_.template Get<T>();
        LocalTensor<T> eps  = tmpEps_.template Get<T>();
        LocalTensor<T> sq   = tmpSq_.template Get<T>();

        LocalTensor<T> msNewL  = Base::qMsOut_.template AllocTensor<T>();
        LocalTensor<T> momNewL = Base::qMomOut_.template AllocTensor<T>();
        LocalTensor<T> varNewL = Base::qVarOut_.template AllocTensor<T>();

        int32_t n = static_cast<int32_t>(curLen);

        AscendC::Mul(g2,    gradL, gradL, n);
        AscendC::Sub(diff,  g2,    msL,   n);
        AscendC::Muls(diff, diff,  Base::rho1m_, n);
        AscendC::Add(msNewL, msL,  diff,  n);

        AscendC::Adds(eps,  msNewL, Base::eps_, n);
        AscendC::Sqrt(sq,   eps,         n);
        AscendC::Reciprocal(eps, sq,     n);
        AscendC::Mul(diff,  eps,    gradL, n);
        AscendC::Muls(diff, diff,   Base::lr_, n);
        AscendC::Muls(momNewL, momL, Base::mom_, n);
        AscendC::Add(momNewL, momNewL, diff, n);

        AscendC::Sub(varNewL, varL, momNewL, n);

        Base::qVarOut_.template EnQue<T>(varNewL);
        Base::qMsOut_.template EnQue<T>(msNewL);
        Base::qMomOut_.template EnQue<T>(momNewL);

        Base::qVar_.FreeTensor(varL);
        Base::qMs_.FreeTensor(msL);
        Base::qMom_.FreeTensor(momL);
        Base::qGrad_.FreeTensor(gradL);
    }

private:
    TBuf<AscendC::TPosition::VECCALC> tmpG2_;
    TBuf<AscendC::TPosition::VECCALC> tmpDiff_;
    TBuf<AscendC::TPosition::VECCALC> tmpEps_;
    TBuf<AscendC::TPosition::VECCALC> tmpSq_;
};

// =============================================================================
// fp16 / bf16 分支：T -> fp32 -> compute -> T
// =============================================================================
template <typename T>
class ApplyRmsPropCast : public ApplyRmsPropBase<T> {
    using Base = ApplyRmsPropBase<T>;
public:
    __aicore__ inline ApplyRmsPropCast() {}

    __aicore__ inline void Init(GM_ADDR var, GM_ADDR ms, GM_ADDR mom, GM_ADDR grad,
                                GM_ADDR varOut, GM_ADDR msOut, GM_ADDR momOut,
                                const ApplyRmsPropTilingData* td)
    {
        Base::InitCommon(var, ms, mom, grad, varOut, msOut, momOut, td);
        if (!Base::isValid_) return;
        Base::pipe_.InitBuffer(fpVar_,    Base::tileLength_ * sizeof(float));
        Base::pipe_.InitBuffer(fpMs_,     Base::tileLength_ * sizeof(float));
        Base::pipe_.InitBuffer(fpMom_,    Base::tileLength_ * sizeof(float));
        Base::pipe_.InitBuffer(fpGrad_,   Base::tileLength_ * sizeof(float));
        Base::pipe_.InitBuffer(fpMsNew_,  Base::tileLength_ * sizeof(float));
        Base::pipe_.InitBuffer(fpMomNew_, Base::tileLength_ * sizeof(float));
        Base::pipe_.InitBuffer(fpVarNew_, Base::tileLength_ * sizeof(float));
        Base::pipe_.InitBuffer(fpTmp_,    Base::tileLength_ * sizeof(float));
    }

    __aicore__ inline void Process() { Base::template ProcessImpl(this); }

    __aicore__ inline void Compute(int64_t curLen)
    {
        LocalTensor<T> varL  = Base::qVar_.template DeQue<T>();
        LocalTensor<T> msL   = Base::qMs_.template DeQue<T>();
        LocalTensor<T> momL  = Base::qMom_.template DeQue<T>();
        LocalTensor<T> gradL = Base::qGrad_.template DeQue<T>();

        LocalTensor<float> varF  = fpVar_.template Get<float>();
        LocalTensor<float> msF   = fpMs_.template Get<float>();
        LocalTensor<float> momF  = fpMom_.template Get<float>();
        LocalTensor<float> gradF = fpGrad_.template Get<float>();
        LocalTensor<float> msNewF  = fpMsNew_.template Get<float>();
        LocalTensor<float> momNewF = fpMomNew_.template Get<float>();
        LocalTensor<float> varNewF = fpVarNew_.template Get<float>();
        LocalTensor<float> tmpF    = fpTmp_.template Get<float>();

        int32_t n = static_cast<int32_t>(curLen);

        AscendC::Cast(varF,  varL,  AscendC::RoundMode::CAST_NONE, n);
        AscendC::Cast(msF,   msL,   AscendC::RoundMode::CAST_NONE, n);
        AscendC::Cast(momF,  momL,  AscendC::RoundMode::CAST_NONE, n);
        AscendC::Cast(gradF, gradL, AscendC::RoundMode::CAST_NONE, n);

        Base::qVar_.FreeTensor(varL);
        Base::qMs_.FreeTensor(msL);
        Base::qMom_.FreeTensor(momL);
        Base::qGrad_.FreeTensor(gradL);

        AscendC::Mul(tmpF,    gradF, gradF, n);
        AscendC::Sub(tmpF,    tmpF,  msF,   n);
        AscendC::Muls(tmpF,   tmpF,  Base::rho1m_, n);
        AscendC::Add(msNewF,  msF,   tmpF,  n);

        AscendC::Adds(tmpF,   msNewF, Base::eps_, n);
        AscendC::Sqrt(tmpF,   tmpF,   n);
        AscendC::Reciprocal(tmpF, tmpF, n);
        AscendC::Mul(tmpF,    tmpF,   gradF, n);
        AscendC::Muls(tmpF,   tmpF,   Base::lr_, n);
        AscendC::Muls(momNewF, momF,  Base::mom_, n);
        AscendC::Add(momNewF, momNewF, tmpF, n);

        AscendC::Sub(varNewF, varF, momNewF, n);

        LocalTensor<T> msNewL  = Base::qMsOut_.template AllocTensor<T>();
        LocalTensor<T> momNewL = Base::qMomOut_.template AllocTensor<T>();
        LocalTensor<T> varNewL = Base::qVarOut_.template AllocTensor<T>();
        AscendC::Cast(msNewL,  msNewF,  AscendC::RoundMode::CAST_RINT, n);
        AscendC::Cast(momNewL, momNewF, AscendC::RoundMode::CAST_RINT, n);
        AscendC::Cast(varNewL, varNewF, AscendC::RoundMode::CAST_RINT, n);

        Base::qMsOut_.template EnQue<T>(msNewL);
        Base::qMomOut_.template EnQue<T>(momNewL);
        Base::qVarOut_.template EnQue<T>(varNewL);
    }

private:
    TBuf<AscendC::TPosition::VECCALC> fpVar_;
    TBuf<AscendC::TPosition::VECCALC> fpMs_;
    TBuf<AscendC::TPosition::VECCALC> fpMom_;
    TBuf<AscendC::TPosition::VECCALC> fpGrad_;
    TBuf<AscendC::TPosition::VECCALC> fpMsNew_;
    TBuf<AscendC::TPosition::VECCALC> fpMomNew_;
    TBuf<AscendC::TPosition::VECCALC> fpVarNew_;
    TBuf<AscendC::TPosition::VECCALC> fpTmp_;
};

}  // namespace NsApplyRmsProp

#endif  // APPLY_RMS_PROP_H
