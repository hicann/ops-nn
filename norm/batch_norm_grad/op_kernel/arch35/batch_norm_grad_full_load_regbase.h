/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file batch_norm_grad_full_load_regbase.h
 * \brief
 */

#ifndef __BATCH_NORM_GRAD_FULL_LOAD_REGBASE_H__
#define __BATCH_NORM_GRAD_FULL_LOAD_REGBASE_H__

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "batch_norm_grad_common.h"

namespace BatchNormGrad {
using namespace AscendC;
using AscendC::Reg::LoadAlign;
using AscendC::Reg::Reduce;
using AscendC::Reg::StoreAlign;

template <typename DY_TYPE, typename WEIGHT_TYPE, int BUFFER_NUM = 1>
class BatchNormGradRARFullLoad {
public:
    __aicore__ inline BatchNormGradRARFullLoad(TPipe* pipe) { pipe_ = pipe; }

    __aicore__ inline void Init(__gm__ uint8_t* dy, __gm__ uint8_t* x, __gm__ uint8_t* mean, __gm__ uint8_t* rstd,
                                __gm__ uint8_t* gamma, __gm__ uint8_t* dx, __gm__ uint8_t* dgamma,
                                __gm__ uint8_t* dbeta, __gm__ uint8_t* workspace,
                                const BatchNormGradRARFullLoadTilingData* tilingData)
    {
        const BatchNormGradBaseTilingData& baseTilingData = tilingData->baseTilingData;
        const BatchNormGradBinaryAddTilingData& binaryAddTilingData = tilingData->binaryAddTilingData;
        r1Dim_ = baseTilingData.r1Dim;
        aDim_ = baseTilingData.aDim;
        r0Dim_ = baseTilingData.r0Dim;
        rAlign_ = baseTilingData.rAlign;
        binaryAddQuotient_ = binaryAddTilingData.binaryAddQuotient;
        binaryAddK_ = binaryAddTilingData.binaryAddk;
        binaryAddLastNum_ = binaryAddTilingData.binaryAddLastNum;
        aTailCoreNum_ = baseTilingData.tailBlockNum;
        aDimPerCore_ = GetBlockIdx() < baseTilingData.tailBlockNum ? baseTilingData.tailBlockDim :
                                                                     baseTilingData.formerBlockDim;
        gmOffset_ = GetBlockIdx() < baseTilingData.tailBlockNum ?
                        GetBlockIdx() * baseTilingData.tailBlockDim :
                        baseTilingData.tailBlockNum * baseTilingData.tailBlockDim +
                            (GetBlockIdx() - baseTilingData.tailBlockNum) * baseTilingData.formerBlockDim;
        aDimPerLoop_ = tilingData->formerUbDim;
        aDimLoopNum_ = GetBlockIdx() < baseTilingData.tailBlockNum ? tilingData->ubLoopOfTailBlock :
                                                                     tilingData->ubLoopOfFormerBlock;
        aDimLoopTail_ = GetBlockIdx() < baseTilingData.tailBlockNum ? tilingData->ubTailOfTailBlock :
                                                                      tilingData->ubTailOfFormerBlock;

        dyGm_.SetGlobalBuffer((__gm__ DY_TYPE*)(dy) + gmOffset_ * r0Dim_);
        xGm_.SetGlobalBuffer((__gm__ DY_TYPE*)(x) + gmOffset_ * r0Dim_);
        meanGm_.SetGlobalBuffer((__gm__ float*)(mean) + gmOffset_);
        rstdGm_.SetGlobalBuffer((__gm__ float*)(rstd) + gmOffset_);
        gammaGm_.SetGlobalBuffer((__gm__ WEIGHT_TYPE*)(gamma) + gmOffset_);
        dxGm_.SetGlobalBuffer((__gm__ DY_TYPE*)(dx) + gmOffset_ * r0Dim_);
        dgammaGm_.SetGlobalBuffer((__gm__ WEIGHT_TYPE*)(dgamma) + gmOffset_);
        dbetaGm_.SetGlobalBuffer((__gm__ WEIGHT_TYPE*)(dbeta) + gmOffset_);

        dyBufSize_ = RoundUpOneBlock(rAlign_ * sizeof(DY_TYPE)) * aDimPerLoop_;
        dyBufElemNum_ = dyBufSize_ / sizeof(DY_TYPE);
        xBufSize_ = RoundUpTwoBlock(rAlign_ * sizeof(float)) * aDimPerLoop_;
        xBufElemNum_ = xBufSize_ / sizeof(DY_TYPE);
        meanBufSize_ = RoundUpOneBlock(aDimPerLoop_ * sizeof(float));
        meanBufElemNum_ = meanBufSize_ / sizeof(float);
        gammaBufSize_ = RoundUpOneBlock(aDimPerLoop_ * sizeof(WEIGHT_TYPE));
        gammaBufElemNum_ = gammaBufSize_ / sizeof(WEIGHT_TYPE);
        binaryAddBufSize_ = RoundUpOneBlock(binaryAddQuotient_ / VL_FP32 * sizeof(float));
        binaryAddBufElemNum_ = binaryAddBufSize_ / sizeof(float);
        pipe_->InitBuffer(dyInQue_, BUFFER_NUM, dyBufSize_);
        pipe_->InitBuffer(xInQue_, BUFFER_NUM, xBufSize_);
        pipe_->InitBuffer(meanInQue_, BUFFER_NUM, meanBufSize_);
        pipe_->InitBuffer(rstdInQue_, BUFFER_NUM, meanBufSize_);
        pipe_->InitBuffer(gammaInQue_, BUFFER_NUM, gammaBufSize_);
        pipe_->InitBuffer(dbetaOutQue_, BUFFER_NUM, meanBufSize_);
        pipe_->InitBuffer(dgammaOutQue_, BUFFER_NUM, meanBufSize_);
        pipe_->InitBuffer(binaryAddBuf_, binaryAddBufSize_ * BUFFER_NUM);

        V_MTE3_EVENT = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        MTE3_MTE2_EVENT = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    }

    __aicore__ inline void Process()
    {
        LocalTensor<float> binaryAddTensor = binaryAddBuf_.Get<float>();
        for (uint32_t i = 0; i < aDimLoopNum_; i++) {
            uint32_t a = i == aDimLoopNum_ - 1 ? aDimLoopTail_ : aDimPerLoop_;
            uint64_t offset = i * aDimPerLoop_ * r0Dim_;
            uint64_t meanOffset = i * aDimPerLoop_;
            uint32_t bufferIdx = i % BUFFER_NUM;

            LocalTensor<DY_TYPE> dy = dyInQue_.template AllocTensor<DY_TYPE>();
            PrepareInDy(dy, offset, a);
            dyInQue_.EnQue(dy);
            dy = dyInQue_.template DeQue<DY_TYPE>();
            LocalTensor<float> dbeta = dbetaOutQue_.template AllocTensor<float>();
            CalcDbeta(dy, dbeta, binaryAddTensor, a, r1Dim_ * r0Dim_, rAlign_);
            dbetaOutQue_.EnQue(dbeta);

            LocalTensor<float> mean = meanInQue_.template AllocTensor<float>();
            PrepareMean(mean, meanOffset, a);
            meanInQue_.EnQue(mean);

            LocalTensor<float> rstd = rstdInQue_.template AllocTensor<float>();
            PrepareRstd(rstd, meanOffset, a);
            rstdInQue_.EnQue(rstd);

            LocalTensor<WEIGHT_TYPE> gamma = gammaInQue_.template AllocTensor<WEIGHT_TYPE>();
            PrepareInGamma(gamma, meanOffset, a);
            gammaInQue_.EnQue(gamma);

            if (i >= BUFFER_NUM) {
                WaitFlag<HardEvent::MTE3_MTE2>(MTE3_MTE2_EVENT + bufferIdx);
            }
            LocalTensor<DY_TYPE> x = xInQue_.template AllocTensor<DY_TYPE>();
            PrepareInX(x, offset, a);
            xInQue_.EnQue(x);

            x = xInQue_.template DeQue<DY_TYPE>();
            mean = meanInQue_.template DeQue<float>();

            CalcXhat(x, mean, a, r0Dim_ * r1Dim_, rAlign_);
            meanInQue_.FreeTensor(mean);

            rstd = rstdInQue_.template DeQue<float>();
            LocalTensor<float> dgamma = dgammaOutQue_.template AllocTensor<float>();
            CalcDgamma(dy, x, rstd, dgamma, binaryAddTensor, a, r0Dim_ * r1Dim_, rAlign_);
            dgammaOutQue_.EnQue(dgamma);

            gamma = gammaInQue_.template DeQue<WEIGHT_TYPE>();
            CalcDx(x, dy, rstd, gamma, dgamma, dbeta, a, r1Dim_ * r0Dim_, rAlign_);
            SetFlag<HardEvent::V_MTE3>(V_MTE3_EVENT + bufferIdx);
            WaitFlag<HardEvent::V_MTE3>(V_MTE3_EVENT + bufferIdx);

            dyInQue_.FreeTensor(dy);
            xInQue_.FreeTensor(x);
            rstdInQue_.FreeTensor(rstd);
            gammaInQue_.FreeTensor(gamma);

            LocalTensor<WEIGHT_TYPE> dbetaOut = dbetaOutQue_.template DeQue<WEIGHT_TYPE>();
            CopyOutDbeta(dbetaOut, meanOffset, a);
            dbetaOutQue_.FreeTensor(dbetaOut);

            LocalTensor<WEIGHT_TYPE> dgammaOut = dgammaOutQue_.template DeQue<WEIGHT_TYPE>();
            CopyOutDgamma(dgammaOut, meanOffset, a);
            dgammaOutQue_.FreeTensor(dgammaOut);

            CopyOutDx(x, offset, a);
            SetFlag<HardEvent::MTE3_MTE2>(MTE3_MTE2_EVENT + bufferIdx);
        }
    }

    __aicore__ inline void PrepareInDy(LocalTensor<DY_TYPE>& dy, uint64_t offset, uint64_t a)
    {
        CopyInRAR(dy, dyGm_, 0, offset, a);
    }

    __aicore__ inline void PrepareInX(LocalTensor<DY_TYPE>& x, int64_t offset, uint64_t a)
    {
        if constexpr (IsSameType<DY_TYPE, float>::value) {
            CopyInRAR(x, xGm_, 0, offset, a);
        } else {
            CopyInRAR(x, xGm_, xBufElemNum_ / 2, offset, a);
        }
    }

    __aicore__ inline void PrepareMean(LocalTensor<float>& mean, uint64_t offset, uint64_t a)
    {
        CopyInA(mean, meanGm_, offset, a);
    }

    __aicore__ inline void PrepareRstd(LocalTensor<float>& rstd, uint64_t offset, uint64_t a)
    {
        CopyInA(rstd, rstdGm_, offset, a);
    }

    __aicore__ inline void PrepareInGamma(LocalTensor<WEIGHT_TYPE>& gamma, uint64_t offset, uint64_t a)
    {
        CopyInA(gamma, gammaGm_, offset, a);
    }

    __aicore__ inline void CalcDbetaVF(const __ubuf__ DY_TYPE* data, const __ubuf__ float* out,
                                       const __ubuf__ float* binaryAdd, uint32_t a, uint32_t r, uint32_t rAlign)
    {
        uint32_t binaryAddQuotientOffset = binaryAddQuotient_;
        int64_t binaryAddRemainder = r - binaryAddQuotient_;
        uint16_t binaryAddRemainderLoop = CeilDiv(binaryAddRemainder, VL_FP32);
        uint16_t remainderGeneral = binaryAddRemainderLoop == 0 ? 0 : binaryAddRemainderLoop - 1;
        uint16_t binaryAddQuotientLoop = CeilDiv(binaryAddQuotient_, VL_FP32);

        uint16_t binaryAddKLoop = binaryAddK_;
        uint16_t binaryAddLoop = ((binaryAddQuotient_ / VL_FP32) / VL_FP32);

        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<float> x;
            MicroAPI::RegTensor<float> sum;
            MicroAPI::RegTensor<float> dbeta;

            MicroAPI::RegTensor<float> binaryAddQ;
            MicroAPI::RegTensor<float> binaryAddR;
            MicroAPI::RegTensor<float> binaryAddRes;
            MicroAPI::RegTensor<float> vlSum;

            MicroAPI::MaskReg pregMain = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
            MicroAPI::MaskReg pregMerge = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::VL1>();
            for (uint16_t k = 0; k < static_cast<uint16_t>(a); k++) {
                uint32_t aOffset = k * rAlign;
                uint32_t sreg0 = binaryAddRemainder;
                for (uint16_t i = 0; i < remainderGeneral; i++) {
                    MicroAPI::MaskReg pregLoop = MicroAPI::UpdateMask<float>(sreg0);
                    LoadOneTensor<DY_TYPE>(data, binaryAddQ, pregMain, aOffset + i * VL_FP32);
                    LoadOneTensor<DY_TYPE>(data, binaryAddR, pregLoop, aOffset + i * VL_FP32 + binaryAddQuotientOffset);
                    Add(binaryAddQ, binaryAddQ, binaryAddR, pregLoop);
                    Reduce<AscendC::Reg::ReduceType::SUM>(vlSum, binaryAddQ, pregMain);
                    StoreAlign<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>((__ubuf__ float*)binaryAdd + i,
                                                                                   vlSum, pregMerge);
                }

                MicroAPI::MaskReg pregLoop = MicroAPI::UpdateMask<float>(sreg0);
                LoadOneTensor<DY_TYPE>(data, binaryAddQ, pregMain, aOffset + remainderGeneral * VL_FP32);
                LoadOneTensor<DY_TYPE>(data, binaryAddR, pregLoop,
                                       aOffset + remainderGeneral * VL_FP32 + binaryAddQuotientOffset);
                Add(binaryAddRes, binaryAddQ, binaryAddR, pregLoop);
                Select(binaryAddRes, binaryAddRes, binaryAddQ, pregLoop);
                Reduce<AscendC::Reg::ReduceType::SUM>(vlSum, binaryAddRes, pregMain);
                StoreAlign<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                    (__ubuf__ float*)binaryAdd + remainderGeneral, vlSum, pregMerge);

                for (uint16_t i = 0; i < static_cast<uint16_t>(binaryAddQuotientLoop - binaryAddRemainderLoop); i++) {
                    LoadOneTensor<DY_TYPE>(data, x, pregMain, aOffset + (i + binaryAddRemainderLoop) * VL_FP32);
                    Reduce<AscendC::Reg::ReduceType::SUM>(vlSum, x, pregMain);
                    StoreAlign<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                        (__ubuf__ float*)binaryAdd + binaryAddRemainderLoop + i, vlSum, pregMerge);
                }
                MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
                uint16_t curBinaryAddLoop = binaryAddLoop;
                for (uint16_t i = 0; i < binaryAddKLoop; i++) {
                    curBinaryAddLoop = curBinaryAddLoop / 2;
                    for (uint16_t j = 0; j < curBinaryAddLoop; j++) {
                        LoadAlign(binaryAddQ, ((__ubuf__ float*)binaryAdd) + j * VL_FP32);
                        LoadAlign(binaryAddR, ((__ubuf__ float*)binaryAdd) + (j + curBinaryAddLoop) * VL_FP32);
                        Add(binaryAddQ, binaryAddQ, binaryAddR, pregMain);
                        StoreAlign(((__ubuf__ float*)binaryAdd) + j * VL_FP32, binaryAddQ, pregMain);
                    }
                    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
                }
                uint32_t sreg2 = binaryAddLastNum_;
                MicroAPI::MaskReg pregLast = MicroAPI::UpdateMask<float>(sreg2);
                LoadAlign(sum, ((__ubuf__ float*)binaryAdd));
                Reduce<AscendC::Reg::ReduceType::SUM>(dbeta, sum, pregLast);
                StoreAlign<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(((__ubuf__ float*)out) + k, dbeta,
                                                                               pregMerge);
                MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_LOAD, MicroAPI::MemType::VEC_STORE>();
            }
        }
    }

    __aicore__ inline void CalcDbetaLessThanVL64VF(const __ubuf__ DY_TYPE* data, const __ubuf__ float* out, uint32_t a,
                                                   uint32_t r, uint32_t rAlign)
    {
        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<float> x;
            MicroAPI::RegTensor<float> sum;

            MicroAPI::MaskReg pregMerge = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::VL1>();
            for (uint16_t k = 0; k < static_cast<uint16_t>(a); k++) {
                uint32_t aOffset = k * rAlign;
                uint32_t sreg0 = r;
                MicroAPI::MaskReg pregLoop = MicroAPI::UpdateMask<float>(sreg0);
                LoadOneTensor<DY_TYPE>(data, x, pregLoop, aOffset);
                Reduce<AscendC::Reg::ReduceType::SUM>(sum, x, pregLoop);
                StoreAlign<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(((__ubuf__ float*)out) + k, sum,
                                                                               pregMerge);
            }
        }
    }

    __aicore__ inline void CalcDbeta(LocalTensor<DY_TYPE>& dy, LocalTensor<float>& dbeta, LocalTensor<float>& binaryAdd,
                                     uint32_t a, uint32_t r, uint32_t rAlign)
    {
        if (r <= VL_FP32) {
            CalcDbetaLessThanVL64VF((__ubuf__ DY_TYPE*)dy.GetPhyAddr(), (__ubuf__ float*)dbeta.GetPhyAddr(), a, r,
                                    rAlign);
        } else {
            CalcDbetaVF((__ubuf__ DY_TYPE*)dy.GetPhyAddr(), (__ubuf__ float*)dbeta.GetPhyAddr(),
                        (__ubuf__ float*)binaryAdd.GetPhyAddr(), a, r, rAlign);
        }
    }

    __aicore__ inline void CalcXhat(LocalTensor<DY_TYPE>& x, LocalTensor<float>& mean, uint32_t a, uint32_t r,
                                    uint32_t rAlign)
    {
        uint16_t VL = VECTOR_LENGTH / sizeof(float);
        uint16_t loopTimes = (r + VL - 1) / VL;
        const __ubuf__ DY_TYPE* xSrcAddr = (__ubuf__ DY_TYPE*)x.GetPhyAddr();
        const __ubuf__ float* xDstAddr = (const __ubuf__ float*)x.GetPhyAddr();
        if constexpr (IsSameType<DY_TYPE, half>::value || IsSameType<DY_TYPE, bfloat16_t>::value) {
            xSrcAddr += xBufElemNum_ / 2;
        }
        const __ubuf__ float* meanAddr = (__ubuf__ float*)mean.GetPhyAddr();
        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<float> meanValue;
            MicroAPI::MaskReg pregR;
            MicroAPI::RegTensor<float> xValue;
            MicroAPI::RegTensor<float> xHat;
            uint32_t sregMask;

            for (uint16_t i = 0; i < (uint16_t)a; i++) {
                LoadAlign<float, MicroAPI::LoadDist::DIST_BRC_B32>(meanValue, (__ubuf__ float*)(meanAddr) + i);
                sregMask = (uint32_t)r;
                for (uint16_t j = 0; j < loopTimes; j++) {
                    pregR = MicroAPI::UpdateMask<float>(sregMask);
                    LoadOneTensor<DY_TYPE>(xSrcAddr, xValue, pregR, i * rAlign + VL * j);
                    Sub(xHat, xValue, meanValue, pregR);
                    StoreAlign(((__ubuf__ float*)xDstAddr) + i * rAlign + VL * j, xHat, pregR);
                }
            }
        }
    }

    __aicore__ inline void CalcDgammaVF(const __ubuf__ DY_TYPE* dyAddr, const __ubuf__ float* xhatAddr,
                                        const __ubuf__ float* rstdAddr, const __ubuf__ float* dgammaAddr,
                                        const __ubuf__ float* binaryAdd, uint32_t a, uint32_t r, uint32_t rAlign)
    {
        uint32_t binaryAddQuotientOffset = binaryAddQuotient_;
        int64_t binaryAddRemainder = r - binaryAddQuotient_;
        uint16_t binaryAddRemainderLoop = CeilDiv(binaryAddRemainder, VL_FP32);
        uint16_t remainderGeneral = binaryAddRemainderLoop == 0 ? 0 : binaryAddRemainderLoop - 1;
        uint16_t binaryAddQuotientLoop = CeilDiv(binaryAddQuotient_, VL_FP32);

        uint16_t binaryAddKLoop = binaryAddK_;
        uint16_t binaryAddLoop = ((binaryAddQuotient_ / VL_FP32) / VL_FP32);

        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<float> sum;
            MicroAPI::RegTensor<float> dgamma;
            MicroAPI::RegTensor<float> rstdValue;

            MicroAPI::RegTensor<float> binaryAddQ1;
            MicroAPI::RegTensor<float> binaryAddR1;
            MicroAPI::RegTensor<float> binaryAddQ2;
            MicroAPI::RegTensor<float> binaryAddR2;
            MicroAPI::RegTensor<float> binaryAddRes;
            MicroAPI::RegTensor<float> vlSum;

            MicroAPI::MaskReg pregMain = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
            MicroAPI::MaskReg pregMerge = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::VL1>();
            for (uint16_t k = 0; k < static_cast<uint16_t>(a); k++) {
                uint32_t aOffset = k * rAlign;
                uint32_t sreg0 = binaryAddRemainder;
                LoadAlign<float, MicroAPI::LoadDist::DIST_BRC_B32>(rstdValue, (__ubuf__ float*)(rstdAddr) + k);
                for (uint16_t i = 0; i < remainderGeneral; i++) {
                    MicroAPI::MaskReg pregLoop = MicroAPI::UpdateMask<float>(sreg0);
                    LoadOneTensor<DY_TYPE>(dyAddr, binaryAddQ1, pregMain, aOffset + i * VL_FP32);
                    LoadOneTensor<DY_TYPE>(dyAddr, binaryAddR1, pregLoop,
                                           aOffset + i * VL_FP32 + binaryAddQuotientOffset);

                    LoadAlign(binaryAddQ2, (__ubuf__ float*)(xhatAddr) + aOffset + i * VL_FP32);
                    LoadAlign(binaryAddR2,
                              (__ubuf__ float*)(xhatAddr) + aOffset + i * VL_FP32 + binaryAddQuotientOffset);
                    Mul(binaryAddQ1, binaryAddQ1, binaryAddQ2, pregMain);
                    Mul(binaryAddQ1, binaryAddQ1, rstdValue, pregMain);
                    Mul(binaryAddR1, binaryAddR1, binaryAddR2, pregLoop);
                    Mul(binaryAddR1, binaryAddR1, rstdValue, pregLoop);
                    Add(binaryAddQ1, binaryAddQ1, binaryAddR1, pregLoop);
                    Reduce<AscendC::Reg::ReduceType::SUM>(vlSum, binaryAddQ1, pregMain);
                    StoreAlign<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>((__ubuf__ float*)binaryAdd + i,
                                                                                   vlSum, pregMerge);
                }

                MicroAPI::MaskReg pregLoop = MicroAPI::UpdateMask<float>(sreg0);
                LoadOneTensor<DY_TYPE>(dyAddr, binaryAddQ1, pregMain, aOffset + remainderGeneral * VL_FP32);
                LoadOneTensor<DY_TYPE>(dyAddr, binaryAddR1, pregLoop,
                                       aOffset + remainderGeneral * VL_FP32 + binaryAddQuotientOffset);

                LoadAlign(binaryAddQ2, (__ubuf__ float*)(xhatAddr) + aOffset + remainderGeneral * VL_FP32);
                LoadAlign(binaryAddR2,
                          (__ubuf__ float*)(xhatAddr) + aOffset + remainderGeneral * VL_FP32 + binaryAddQuotientOffset);
                Mul(binaryAddQ1, binaryAddQ1, binaryAddQ2, pregMain);
                Mul(binaryAddQ1, binaryAddQ1, rstdValue, pregMain);
                Mul(binaryAddR1, binaryAddR1, binaryAddR2, pregLoop);
                Mul(binaryAddR1, binaryAddR1, rstdValue, pregLoop);
                Add(binaryAddRes, binaryAddQ1, binaryAddR1, pregLoop);
                Select(binaryAddRes, binaryAddRes, binaryAddQ1, pregLoop);
                Reduce<AscendC::Reg::ReduceType::SUM>(vlSum, binaryAddRes, pregMain);
                StoreAlign<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                    (__ubuf__ float*)binaryAdd + remainderGeneral, vlSum, pregMerge);

                for (uint16_t i = 0; i < static_cast<uint16_t>(binaryAddQuotientLoop - binaryAddRemainderLoop); i++) {
                    LoadOneTensor<DY_TYPE>(dyAddr, binaryAddQ1, pregMain,
                                           aOffset + (i + binaryAddRemainderLoop) * VL_FP32);
                    LoadAlign(binaryAddQ2,
                              (__ubuf__ float*)(xhatAddr) + aOffset + (i + binaryAddRemainderLoop) * VL_FP32);
                    Mul(binaryAddQ1, binaryAddQ1, binaryAddQ2, pregMain);
                    Mul(binaryAddQ1, binaryAddQ1, rstdValue, pregMain);
                    Reduce<AscendC::Reg::ReduceType::SUM>(vlSum, binaryAddQ1, pregMain);
                    StoreAlign<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                        (__ubuf__ float*)binaryAdd + binaryAddRemainderLoop + i, vlSum, pregMerge);
                }
                MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
                uint16_t curBinaryAddLoop = binaryAddLoop;
                for (uint16_t i = 0; i < binaryAddKLoop; i++) {
                    curBinaryAddLoop = curBinaryAddLoop / 2;
                    for (uint16_t j = 0; j < curBinaryAddLoop; j++) {
                        LoadAlign(binaryAddQ1, (__ubuf__ float*)(binaryAdd) + j * VL_FP32);
                        LoadAlign(binaryAddR1, (__ubuf__ float*)(binaryAdd) + (j + curBinaryAddLoop) * VL_FP32);
                        Add(binaryAddQ1, binaryAddQ1, binaryAddR1, pregMain);
                        StoreAlign(((__ubuf__ float*)binaryAdd) + j * VL_FP32, binaryAddQ1, pregMain);
                    }
                    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
                }
                uint32_t sreg2 = binaryAddLastNum_;
                MicroAPI::MaskReg pregLast = MicroAPI::UpdateMask<float>(sreg2);
                LoadAlign(sum, (__ubuf__ float*)(binaryAdd));
                Reduce<AscendC::Reg::ReduceType::SUM>(dgamma, sum, pregLast);
                StoreAlign<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(((__ubuf__ float*)dgammaAddr) + k,
                                                                               dgamma, pregMerge);
                MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_LOAD, MicroAPI::MemType::VEC_STORE>();
            }
        }
    }

    __aicore__ inline void CalcDgammaLessThanVL64VF(const __ubuf__ DY_TYPE* dyAddr, const __ubuf__ float* xhatAddr,
                                                    const __ubuf__ float* rstdAddr, const __ubuf__ float* dgammaAddr,
                                                    uint32_t a, uint32_t r, uint32_t rAlign)
    {
        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<float> x;
            MicroAPI::RegTensor<float> y;
            MicroAPI::RegTensor<float> sum;
            MicroAPI::RegTensor<float> rstdValue;

            MicroAPI::MaskReg pregMerge = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::VL1>();
            for (uint16_t k = 0; k < static_cast<uint16_t>(a); k++) {
                uint32_t aOffset = k * rAlign;
                uint32_t sreg0 = r;
                MicroAPI::MaskReg pregLoop = MicroAPI::UpdateMask<float>(sreg0);
                LoadOneTensor<DY_TYPE>(dyAddr, y, pregLoop, aOffset);
                LoadAlign(x, (__ubuf__ float*)(xhatAddr) + aOffset);
                LoadAlign<float, MicroAPI::LoadDist::DIST_BRC_B32>(rstdValue, (__ubuf__ float*)(rstdAddr) + k);
                Mul(x, y, x, pregLoop);
                Mul(x, x, rstdValue, pregLoop);
                Reduce<AscendC::Reg::ReduceType::SUM>(sum, x, pregLoop);
                StoreAlign<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(((__ubuf__ float*)dgammaAddr) + k, sum,
                                                                               pregMerge);
            }
        }
    }

    __aicore__ inline void CalcDgamma(LocalTensor<DY_TYPE>& dy, LocalTensor<DY_TYPE>& xhat, LocalTensor<float>& rstd,
                                      LocalTensor<float>& dgamma, LocalTensor<float>& binaryAdd, uint32_t a, uint32_t r,
                                      uint32_t rAlign)
    {
        if (r <= VL_FP32) {
            CalcDgammaLessThanVL64VF((__ubuf__ DY_TYPE*)dy.GetPhyAddr(), (__ubuf__ float*)xhat.GetPhyAddr(),
                                     (__ubuf__ float*)rstd.GetPhyAddr(), (__ubuf__ float*)dgamma.GetPhyAddr(), a, r,
                                     rAlign);
        } else {
            CalcDgammaVF((__ubuf__ DY_TYPE*)dy.GetPhyAddr(), (__ubuf__ float*)xhat.GetPhyAddr(),
                         (__ubuf__ float*)rstd.GetPhyAddr(), (__ubuf__ float*)dgamma.GetPhyAddr(),
                         (__ubuf__ float*)binaryAdd.GetPhyAddr(), a, r, rAlign);
        }
    }

    __aicore__ inline void CalcDx(LocalTensor<DY_TYPE>& xhat, LocalTensor<DY_TYPE>& dy, LocalTensor<float> rstd,
                                  LocalTensor<WEIGHT_TYPE>& gamma, LocalTensor<float>& dgamma,
                                  LocalTensor<float>& dbeta, uint32_t a, uint32_t r, uint32_t rAlign)
    {
        __ubuf__ DY_TYPE* xhatAddr = (__ubuf__ DY_TYPE*)xhat.GetPhyAddr();
        __ubuf__ DY_TYPE* dyAddr = (__ubuf__ DY_TYPE*)dy.GetPhyAddr();
        __ubuf__ float* rstdAddr = (__ubuf__ float*)rstd.GetPhyAddr();
        __ubuf__ WEIGHT_TYPE* gammaAddr = (__ubuf__ WEIGHT_TYPE*)gamma.GetPhyAddr();
        __ubuf__ float* dgammaAddr = (__ubuf__ float*)dgamma.GetPhyAddr();
        __ubuf__ float* dbetaAddr = (__ubuf__ float*)dbeta.GetPhyAddr();

        uint16_t VL = VECTOR_LENGTH / sizeof(float);
        uint16_t loopTimes = (r + VL - 1) / VL;
        float hRecipValue = 1.0f / (float)r;
        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<float> rstdValue;
            MicroAPI::RegTensor<float> gammaValue;
            MicroAPI::RegTensor<float> dgammaValue;
            MicroAPI::RegTensor<float> dbetaValue;
            MicroAPI::RegTensor<float> rstdMulSubDy;
            MicroAPI::MaskReg pregR;
            MicroAPI::RegTensor<float> dy;
            MicroAPI::RegTensor<float> xhat;
            MicroAPI::RegTensor<float> mulDgamma;
            MicroAPI::RegTensor<float> addDbeta;
            MicroAPI::RegTensor<float> divH;
            MicroAPI::RegTensor<float> subDy;
            MicroAPI::RegTensor<float> dx;
            MicroAPI::MaskReg preg = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
            MicroAPI::MaskReg pregMerge = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::VL1>();
            for (uint16_t i = 0; i < (uint16_t)a; i++) {
                LoadAlign<float, MicroAPI::LoadDist::DIST_BRC_B32>(rstdValue, (__ubuf__ float*)(rstdAddr) + i);
                LoadAlign<float, MicroAPI::LoadDist::DIST_BRC_B32>(dgammaValue, (__ubuf__ float*)(dgammaAddr) + i);
                LoadAlign<float, MicroAPI::LoadDist::DIST_BRC_B32>(dbetaValue, (__ubuf__ float*)(dbetaAddr) + i);
                LoadOneElement<WEIGHT_TYPE>(gammaAddr, gammaValue, preg, i);
                uint32_t sregMask = (uint32_t)r;
                for (uint16_t j = 0; j < loopTimes; j++) {
                    pregR = MicroAPI::UpdateMask<float>(sregMask);
                    LoadAlign(xhat, (__ubuf__ float*)(xhatAddr) + i * rAlign + VL * j);
                    LoadOneTensor<DY_TYPE>(dyAddr, dy, pregR, i * rAlign + VL * j);
                    Mul(mulDgamma, xhat, rstdValue, pregR);
                    Mul(mulDgamma, mulDgamma, dgammaValue, pregR);
                    Add(addDbeta, mulDgamma, dbetaValue, preg);
                    Muls(divH, addDbeta, hRecipValue, pregR);
                    Sub(subDy, dy, divH, pregR);
                    Mul(rstdMulSubDy, rstdValue, subDy, pregR);
                    Mul(dx, gammaValue, rstdMulSubDy, pregR);
                    StoreOneTensor<DY_TYPE>(xhatAddr, dx, pregR, i * rAlign + VL * j);
                }
                if constexpr (IsSameType<WEIGHT_TYPE, half>::value || IsSameType<WEIGHT_TYPE, bfloat16_t>::value) {
                    StoreOneElement<WEIGHT_TYPE>(dbetaAddr, dbetaValue, pregMerge, i);
                    StoreOneElement<WEIGHT_TYPE>(dgammaAddr, dgammaValue, pregMerge, i);
                }
            }
        }
    }

    __aicore__ inline void CopyOutDx(LocalTensor<DY_TYPE>& dx, uint64_t offset, uint32_t a)
    {
        CopyOutRAR(dx, dxGm_, offset, a);
    }

    __aicore__ inline void CopyOutDgamma(LocalTensor<WEIGHT_TYPE>& dgamma, uint64_t meanOffset, uint32_t a)
    {
        CopyOutA(dgamma, dgammaGm_, meanOffset, a);
    }

    __aicore__ inline void CopyOutDbeta(LocalTensor<WEIGHT_TYPE>& dbeta, uint64_t meanOffset, uint32_t a)
    {
        CopyOutA(dbeta, dbetaGm_, meanOffset, a);
    }

    __aicore__ inline void CopyInRAR(LocalTensor<DY_TYPE>& localTensor, GlobalTensor<DY_TYPE>& globalTensor,
                                     uint64_t localOffset, uint64_t globalOffset, uint64_t a)
    {
        // RAR -> AR，R轴通过Compact Mode补pad到block对齐。
        LoopModeParams loopParams;
        loopParams.loop2Size = 1;
        loopParams.loop1Size = a;
        loopParams.loop2SrcStride = 0;
        loopParams.loop2DstStride = 0;
        loopParams.loop1SrcStride = r0Dim_ * sizeof(DY_TYPE);
        loopParams.loop1DstStride = rAlign_ * sizeof(DY_TYPE);
        SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);

        DataCopyPadExtParams<DY_TYPE> dataCopyPadExtParams;
        dataCopyPadExtParams.isPad = false;
        dataCopyPadExtParams.leftPadding = 0;
        dataCopyPadExtParams.rightPadding = 0;
        dataCopyPadExtParams.paddingValue = 0;
        DataCopyExtParams copyInParams;
        copyInParams.blockCount = r1Dim_;
        copyInParams.blockLen = r0Dim_ * sizeof(DY_TYPE);
        copyInParams.srcStride = (aDim_ - 1) * r0Dim_ * sizeof(DY_TYPE);
        copyInParams.dstStride = 0;
        DataCopyPad<DY_TYPE, PaddingMode::Compact>(localTensor[localOffset], globalTensor[globalOffset], copyInParams,
                                                   dataCopyPadExtParams);
        ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
    }

    template <typename U>
    __aicore__ inline void CopyInA(LocalTensor<U>& localTensor, GlobalTensor<U>& globalTensor, uint64_t offset,
                                   uint64_t a)
    {
        DataCopyPadExtParams<U> dataCopyPadExtParams;
        dataCopyPadExtParams.isPad = false;
        dataCopyPadExtParams.leftPadding = 0;
        dataCopyPadExtParams.rightPadding = 0;
        dataCopyPadExtParams.paddingValue = 0;
        DataCopyExtParams copyInParams;
        copyInParams.blockCount = 1;
        copyInParams.blockLen = a * sizeof(U);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;
        DataCopyPad<U, PaddingMode::Compact>(localTensor, globalTensor[offset], copyInParams, dataCopyPadExtParams);
    }

    __aicore__ inline void CopyOutRAR(LocalTensor<DY_TYPE>& localTensor, GlobalTensor<DY_TYPE>& globalTensor,
                                      uint64_t offset, uint64_t a)
    {
        // AR -> RAR
        LoopModeParams loopParams;
        loopParams.loop2Size = 1;
        loopParams.loop1Size = a;
        loopParams.loop2SrcStride = 0;
        loopParams.loop2DstStride = 0;
        loopParams.loop1SrcStride = rAlign_ * sizeof(DY_TYPE);
        loopParams.loop1DstStride = r0Dim_ * sizeof(DY_TYPE);
        SetLoopModePara(loopParams, DataCopyMVType::UB_TO_OUT);

        DataCopyExtParams copyOutParams;
        copyOutParams.blockCount = r1Dim_;
        copyOutParams.blockLen = r0Dim_ * sizeof(DY_TYPE);
        copyOutParams.srcStride = 0;
        copyOutParams.dstStride = (aDim_ - 1) * r0Dim_ * sizeof(DY_TYPE);
        DataCopyPad<DY_TYPE, PaddingMode::Compact>(globalTensor[offset], localTensor, copyOutParams);
        ResetLoopModePara(DataCopyMVType::UB_TO_OUT);
    }

    __aicore__ inline void CopyOutA(LocalTensor<WEIGHT_TYPE>& localTensor, GlobalTensor<WEIGHT_TYPE>& globalTensor,
                                    uint64_t offset, uint64_t a)
    {
        DataCopyExtParams copyOutParams;
        copyOutParams.blockCount = 1;
        copyOutParams.blockLen = a * sizeof(WEIGHT_TYPE);
        copyOutParams.srcStride = 0;
        copyOutParams.dstStride = 0;
        DataCopyPad<WEIGHT_TYPE, PaddingMode::Compact>(globalTensor[offset], localTensor, copyOutParams);
    }

private:
    TPipe* pipe_ = nullptr;
    TBuf<TPosition::VECCALC> binaryAddBuf_;

    TQue<QuePosition::VECIN, BUFFER_NUM> dyInQue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> xInQue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> meanInQue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> rstdInQue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> gammaInQue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> dbetaOutQue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> dgammaOutQue_;

    GlobalTensor<DY_TYPE> dyGm_, xGm_, dxGm_;
    GlobalTensor<WEIGHT_TYPE> gammaGm_, dgammaGm_, dbetaGm_;
    GlobalTensor<float> meanGm_, rstdGm_, workspaceGm_;

    event_t V_MTE3_EVENT;
    event_t MTE3_MTE2_EVENT;

    uint32_t r1Dim_;
    uint32_t aDim_;
    uint32_t r0Dim_;
    uint32_t rAlign_;
    uint32_t binaryAddQuotient_;
    uint32_t binaryAddK_;
    uint32_t binaryAddLastNum_;
    uint32_t aDimPerCore_;
    uint32_t aTailCoreNum_;
    uint32_t aDimTail_;
    uint32_t gmOffset_;
    uint32_t aDimPerLoop_;
    uint32_t aDimLoopNum_;
    uint32_t aDimLoopTail_;
    uint32_t dyBufSize_;
    uint32_t dyBufElemNum_;
    uint32_t xBufSize_;
    uint32_t xBufElemNum_;
    uint32_t meanBufSize_;
    uint32_t meanBufElemNum_;
    uint32_t gammaBufSize_;
    uint32_t gammaBufElemNum_;
    uint32_t binaryAddBufSize_;
    uint32_t binaryAddBufElemNum_;
};
} // namespace BatchNormGrad
#endif // __BATCH_NORM_GRAD_FULL_LOAD_REGBASE_H__
