/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file layer_norm_grad_v3_transpose_backward_impl.h
 * \brief
 */

#ifndef LAYER_NORM_GRAD_V3_TRANSPOSE_BACKWARD_IMPL_
#define LAYER_NORM_GRAD_V3_TRANSPOSE_BACKWARD_IMPL_
#include "layer_norm_grad_v3_api.h"
#include "layer_norm_grad_v3_base.h"

namespace LayerNormGradV3 {
using namespace AscendC;

template <typename T, typename U>
class LayerNormGradV3TransposeRegBaseBackward : public LayerNormGradV3Base {
public:
    __aicore__ inline LayerNormGradV3TransposeRegBaseBackward() : LayerNormGradV3Base(){};
    __aicore__ inline void Init(GM_ADDR dy, GM_ADDR x, GM_ADDR rstd, GM_ADDR mean, GM_ADDR gamma, GM_ADDR pdX,
                                GM_ADDR workspace, const LayerNormGradV3TilingDataTransposeRegBase* tilingData,
                                TPipe* pipeIn);
    __aicore__ inline void Process();

private:
    __aicore__ inline void LoadGamma();
    __aicore__ inline void CopyInData(const int64_t mi, const int64_t mfactor);
    __aicore__ inline void Compute(const int64_t mfactor);
    __aicore__ inline void CopyOutData(const int64_t mi, const int64_t mfactor);
    __aicore__ inline void ComputeDyMulGammaXNorm(const LocalTensor<float>& xNormTensor,
                                                  const LocalTensor<float>& dyTensor, const LocalTensor<float>& xTensor,
                                                  const LocalTensor<float>& gammaTensor,
                                                  const LocalTensor<float>& meanTensor,
                                                  const LocalTensor<float>& rstdTensor, const int64_t rowSize,
                                                  const int64_t colSize, const int64_t stride);
    __aicore__ inline static void ComputeDx(const LocalTensor<T>& dstTensor, const LocalTensor<float>& dyGammaTensor,
                                            const LocalTensor<float>& xNormTensor, const LocalTensor<float>& sum1Tensor,
                                            const LocalTensor<float>& sum2Tensor, const LocalTensor<float>& rstdTensor,
                                            const int64_t rowSize, const int64_t colSize, const int64_t stride,
                                            const int64_t fullColSize);
    __aicore__ inline void DoPostTranspose(LocalTensor<T>& dstTensor, LocalTensor<T>& srcTensor, const int64_t mfactor);

private:
    const LayerNormGradV3TilingDataTransposeRegBase* __restrict td_;

    constexpr static int64_t TRANSPOSE_C0_SIZE = 16;
    constexpr static int64_t B32_ALIGN = 8;
    constexpr static int64_t B16_ALIGN = 16;
    constexpr static int64_t ONE_BUFFER = 1;
    constexpr static int64_t DOUBLE_BUFFER = 2;
    constexpr static int64_t NUM_TWO = 2;

    int64_t N = 0;
    int64_t NAlign = 0;
    int64_t MfactorAlign = 0;
    int64_t currentMLoop = 0;
    int64_t currentMTail = 0;

    GlobalTensor<T> dyInTensorGM;
    GlobalTensor<T> xInTensorGM;
    GlobalTensor<float> rstdInTensorGM;
    GlobalTensor<float> meanInTensorGM;
    GlobalTensor<U> gammaInTensorGM;
    GlobalTensor<T> pdXOutTensorGM;

    LocalTensor<float> gammaMain_;
    LocalTensor<float> xNorm_;

    TPipe* pipe_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> inQueueDy;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> inQueueX;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> inQueueMean;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> inQueueRstd;
    TQue<QuePosition::VECIN, ONE_BUFFER> inQueueGamma;
    TQue<QuePosition::VECOUT, ONE_BUFFER> outQueueDx;
    TBuf<> sumBuffer;
    TBuf<> xNormBuffer;
};

template <typename T, typename U>
__aicore__ inline void LayerNormGradV3TransposeRegBaseBackward<T, U>::Init(
    GM_ADDR dy, GM_ADDR x, GM_ADDR rstd, GM_ADDR mean, GM_ADDR gamma, GM_ADDR pdX, GM_ADDR workspace,
    const LayerNormGradV3TilingDataTransposeRegBase* tilingData, TPipe* pipeIn)
{
    td_ = tilingData;
    N = td_->col;
    NAlign = td_->backwardNAlign;
    MfactorAlign = td_->backwardMAlign;

    int64_t mPerCore = td_->backwardMPerCore;
    int64_t mTailCore = td_->backwardMTailCore;
    int64_t blockDim = td_->backwardUsedCoreNum;

    int64_t blockIdx = GetBlockIdx();
    if (blockIdx >= blockDim) {
        return;
    }

    // 主核每核mPerCore行，尾核mTailCore行
    int64_t isTail = (blockIdx == blockDim - 1) ? 1 : 0;
    int64_t currentBlockFactor = isTail ? mTailCore : mPerCore;
    currentMLoop = currentBlockFactor / MfactorAlign; // 整块迭代次数
    currentMTail = currentBlockFactor % MfactorAlign; // 不足一块的尾块大小

    int64_t gmOffset = blockIdx * mPerCore * N;
    int64_t dyShape = currentBlockFactor * N;
    dyInTensorGM.SetGlobalBuffer((__gm__ T*)dy + gmOffset, dyShape);
    xInTensorGM.SetGlobalBuffer((__gm__ T*)x + gmOffset, dyShape);
    pdXOutTensorGM.SetGlobalBuffer((__gm__ T*)pdX + gmOffset, dyShape);

    // rstd/mean每核一个标量, gamma全局共享不切分
    int64_t paramOffset = blockIdx * mPerCore;
    rstdInTensorGM.SetGlobalBuffer((__gm__ float*)rstd + paramOffset, currentBlockFactor);
    meanInTensorGM.SetGlobalBuffer((__gm__ float*)mean + paramOffset, currentBlockFactor);
    gammaInTensorGM.SetGlobalBuffer((__gm__ U*)gamma, N);

    pipe_ = pipeIn;

    int64_t dyBufSize = NAlign * MfactorAlign * sizeof(float);
    int64_t paramBufSize = MfactorAlign * sizeof(float);
    pipe_->InitBuffer(inQueueDy, DOUBLE_BUFFER, dyBufSize);
    pipe_->InitBuffer(inQueueX, DOUBLE_BUFFER, dyBufSize);
    pipe_->InitBuffer(inQueueMean, DOUBLE_BUFFER, paramBufSize);
    pipe_->InitBuffer(inQueueRstd, DOUBLE_BUFFER, paramBufSize);
    pipe_->InitBuffer(outQueueDx, DOUBLE_BUFFER, dyBufSize);
    pipe_->InitBuffer(inQueueGamma, ONE_BUFFER, NAlign * sizeof(float));
    pipe_->InitBuffer(sumBuffer, DOUBLE_BUFFER * paramBufSize);
    pipe_->InitBuffer(xNormBuffer, dyBufSize);
}

template <typename T, typename U>
__aicore__ inline void LayerNormGradV3TransposeRegBaseBackward<T, U>::Process()
{
    if (GetBlockIdx() >= td_->backwardUsedCoreNum) {
        return;
    }

    // gamma全局共享, 仅加载一次, 全程复用
    LoadGamma();
    // 搬运第一组数据块
    CopyInData(0, currentMLoop > 0 ? MfactorAlign : currentMTail);
    xNorm_ = xNormBuffer.Get<float>();

    int64_t mi = 0;
    int64_t totalTiles = currentMLoop + (currentMTail > 0 ? 1 : 0);
    for (mi = 0; mi < totalTiles - 1; ++mi) {
        int64_t nextMfactor = (mi + 1 < currentMLoop) ? MfactorAlign : currentMTail;
        int64_t curMfactor = (mi < currentMLoop) ? MfactorAlign : currentMTail;
        CopyInData(mi + 1, nextMfactor); // 预取下一块
        Compute(curMfactor);             // 计算当前块
        CopyOutData(mi, curMfactor);     // 搬出当前块
    }
    // 处理最后一个数据块
    int64_t lastMfactor = (mi < currentMLoop && totalTiles != 0) ? MfactorAlign : currentMTail;
    Compute(lastMfactor);
    CopyOutData(mi, lastMfactor);

    inQueueGamma.FreeTensor(gammaMain_);
}

template <typename T, typename U>
__aicore__ inline void LayerNormGradV3TransposeRegBaseBackward<T, U>::LoadGamma()
{
    gammaMain_ = inQueueGamma.template AllocTensor<float>();
    if constexpr (IsSameType<U, float>::value) {
        CopyIn(gammaMain_.ReinterpretCast<U>(), gammaInTensorGM, N);
        inQueueGamma.EnQue(gammaMain_);
        gammaMain_ = inQueueGamma.template DeQue<float>();
    } else if constexpr (IsSameType<U, bfloat16_t>::value || IsSameType<U, half>::value) {
        LocalTensor<U> gammaCastTemp = gammaMain_.ReinterpretCast<U>();
        CopyIn(gammaCastTemp, gammaInTensorGM, N);
        inQueueGamma.EnQue(gammaMain_);
        gammaMain_ = inQueueGamma.template DeQue<float>();
        CastToFp32From<U>(gammaMain_, gammaCastTemp, N);
    }
}

template <typename T, typename U>
__aicore__ inline void LayerNormGradV3TransposeRegBaseBackward<T, U>::CopyInData(const int64_t mi,
                                                                                 const int64_t mfactor)
{
    int64_t offset = mi * MfactorAlign; // 第mi块的M方向偏移

    // mean/rstd按M轴一维搬运
    LocalTensor<float> mean = inQueueMean.template AllocTensor<float>();
    CopyIn(mean, meanInTensorGM[offset], mfactor);
    inQueueMean.EnQue(mean);

    LocalTensor<float> rstd = inQueueRstd.template AllocTensor<float>();
    CopyIn(rstd, rstdInTensorGM[offset], mfactor);
    inQueueRstd.EnQue(rstd);

    int64_t gmOffset = offset * N; // GM中dy/x的偏移(行主序, 每行N)

    LocalTensor<float> dyMain = inQueueDy.template AllocTensor<float>();
    if constexpr (IsSameType<T, float>::value) {
        CopyInTranspose<T>(dyMain.ReinterpretCast<T>(), dyInTensorGM[gmOffset], mfactor, N, MfactorAlign);
    } else {
        // 非float: T数据落在[Mfactor]偏移处, cast融合到ComputeDyMulGammaXNorm中用
        LocalTensor<T> dyCastTemp = dyMain.template ReinterpretCast<T>()[MfactorAlign];
        CopyInTranspose<T>(dyCastTemp, dyInTensorGM[gmOffset], mfactor, N, 2 * MfactorAlign);
    }
    inQueueDy.EnQue(dyMain);

    LocalTensor<float> xMain = inQueueX.template AllocTensor<float>();
    if constexpr (IsSameType<T, float>::value) {
        CopyInTranspose<T>(xMain.ReinterpretCast<T>(), xInTensorGM[gmOffset], mfactor, N, MfactorAlign);
    } else {
        LocalTensor<T> xCastTemp = xMain.template ReinterpretCast<T>()[MfactorAlign];
        CopyInTranspose<T>(xCastTemp, xInTensorGM[gmOffset], mfactor, N, 2 * MfactorAlign);
    }
    inQueueX.EnQue(xMain);
}

template <typename T, typename U>
__aicore__ inline void LayerNormGradV3TransposeRegBaseBackward<T, U>::Compute(const int64_t mfactor)
{
    LocalTensor<float> dyMain = inQueueDy.template DeQue<float>();
    LocalTensor<float> xMain = inQueueX.template DeQue<float>();
    LocalTensor<float> mean = inQueueMean.template DeQue<float>();
    LocalTensor<float> rstd = inQueueRstd.template DeQue<float>();

    // 步骤1: VF计算 dy_mul_gamma 与 x_norm, 结果写回dyMain/xMain/xNorm
    ComputeDyMulGammaXNorm(xNorm_, dyMain, xMain, gammaMain_, mean, rstd, N, mfactor, MfactorAlign);
    inQueueMean.FreeTensor(mean);

    LocalTensor<T> dxOut = outQueueDx.template AllocTensor<T>();
    LocalTensor<float> sumBufferTensor = sumBuffer.Get<float>();
    LocalTensor<float> sum1 = sumBufferTensor;               // sum(dy*gamma)
    LocalTensor<float> sum2 = sumBufferTensor[MfactorAlign]; // sum(x_norm*dy*gamma)

    // ReduceSum RA模式 tmp复用dxOut空间
    uint32_t srcShape[2] = {static_cast<uint32_t>(N), static_cast<uint32_t>(MfactorAlign)};

    AscendC::ReduceSum<float, AscendC::Pattern::Reduce::RA, false>(
        sum2, xMain, dxOut.template ReinterpretCast<uint8_t>(), srcShape, false);

    inQueueX.FreeTensor(xMain);

    AscendC::ReduceSum<float, AscendC::Pattern::Reduce::RA, false>(
        sum1, dyMain, dxOut.template ReinterpretCast<uint8_t>(), srcShape, false);

    // 步骤2: VF计算dx, 结果按dtype T写入xNorm_(仍为转置布局)
    ComputeDx(xNorm_.template ReinterpretCast<T>(), dyMain, xNorm_, sum1, sum2, rstd, N, mfactor, MfactorAlign, N);

    inQueueDy.FreeTensor(dyMain);
    inQueueRstd.FreeTensor(rstd);

    // 步骤3: 将xNorm_从[N][M]转置布局转回[M][N]写入dxOut
    LocalTensor<T> xNormT = xNorm_.template ReinterpretCast<T>();
    DoPostTranspose(dxOut, xNormT, mfactor);
    outQueueDx.EnQue(dxOut);
}

template <typename T, typename U>
__aicore__ inline void LayerNormGradV3TransposeRegBaseBackward<T, U>::ComputeDyMulGammaXNorm(
    const LocalTensor<float>& xNormTensor, const LocalTensor<float>& dyTensor, const LocalTensor<float>& xTensor,
    const LocalTensor<float>& gammaTensor, const LocalTensor<float>& meanTensor, const LocalTensor<float>& rstdTensor,
    const int64_t rowSize, const int64_t colSize, const int64_t stride)
{
    if (rowSize <= 0 || colSize <= 0) {
        return;
    }
    uint16_t outerLoopTimes = static_cast<uint16_t>(rowSize);
    uint16_t innerLoopTimes = static_cast<uint16_t>(
        CeilDiv(static_cast<int64_t>(colSize * sizeof(float)), static_cast<int64_t>(GetVRegSize())));
    uint32_t outerLoopStride = static_cast<uint32_t>(stride);
    // stride字节数不变，dtype变小需要让stride翻倍
    uint32_t outerLoopStrideDtypeT = (IsSameType<T, float>::value) ? outerLoopStride : outerLoopStride * NUM_TWO;
    uint32_t inputDataOffset = (IsSameType<T, float>::value) ? 0 : static_cast<uint32_t>(MfactorAlign);
    uint32_t innerLoopStride = VL_FP32;
    if (innerLoopTimes == 1) {
        __VEC_SCOPE__
        {
            __ubuf__ float* dyg = (__ubuf__ float*)dyTensor.GetPhyAddr();
            __ubuf__ float* xNorm = (__ubuf__ float*)xNormTensor.GetPhyAddr();
            __ubuf__ float* xMain = (__ubuf__ float*)xTensor.GetPhyAddr();
            __ubuf__ T* dy = (__ubuf__ T*)dyTensor.GetPhyAddr() + inputDataOffset;
            __ubuf__ T* x = (__ubuf__ T*)xTensor.GetPhyAddr() + inputDataOffset;
            __ubuf__ float* gamma = (__ubuf__ float*)gammaTensor.GetPhyAddr();
            __ubuf__ float* mean = (__ubuf__ float*)meanTensor.GetPhyAddr();
            __ubuf__ float* rstd = (__ubuf__ float*)rstdTensor.GetPhyAddr();
            uint32_t count = static_cast<uint32_t>(colSize);
            AscendC::Reg::MaskReg pMask = AscendC::Reg::UpdateMask<float>(count);
            AscendC::Reg::RegTensor<float> dyReg, xReg, gammaReg;
            AscendC::Reg::RegTensor<float> dygReg, xNormReg, xMainReg;
            AscendC::Reg::RegTensor<float> meanReg, rstdReg, tmpReg;
            LoadAlign(meanReg, mean);
            LoadAlign(rstdReg, rstd);
            for (uint16_t i = 0; i < outerLoopTimes; ++i) {
                LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(gammaReg, gamma + i);
                LoadTensorForDtypeT<T>(dy, dyReg, pMask, i * outerLoopStrideDtypeT);
                Mul<float, AscendC::Reg::MaskMergeMode::ZEROING>(dygReg, dyReg, gammaReg, pMask);
                StoreAlign(dyg + i * outerLoopStride, dygReg, pMask);
                LoadTensorForDtypeT<T>(x, xReg, pMask, i * outerLoopStrideDtypeT);
                Sub<float, AscendC::Reg::MaskMergeMode::ZEROING>(tmpReg, xReg, meanReg, pMask);
                Mul<float, AscendC::Reg::MaskMergeMode::ZEROING>(xNormReg, tmpReg, rstdReg, pMask);
                StoreAlign(xNorm + i * outerLoopStride, xNormReg, pMask);
                Mul<float, AscendC::Reg::MaskMergeMode::ZEROING>(xMainReg, xNormReg, dygReg, pMask);
                StoreAlign(xMain + i * outerLoopStride, xMainReg, pMask);
            }
        }
    } else {
        __VEC_SCOPE__
        {
            __ubuf__ float* dyg = (__ubuf__ float*)dyTensor.GetPhyAddr();
            __ubuf__ float* xNorm = (__ubuf__ float*)xNormTensor.GetPhyAddr();
            __ubuf__ float* xMain = (__ubuf__ float*)xTensor.GetPhyAddr();
            __ubuf__ T* dy = (__ubuf__ T*)dyTensor.GetPhyAddr() + inputDataOffset;
            __ubuf__ T* x = (__ubuf__ T*)xTensor.GetPhyAddr() + inputDataOffset;
            __ubuf__ float* gamma = (__ubuf__ float*)gammaTensor.GetPhyAddr();
            __ubuf__ float* mean = (__ubuf__ float*)meanTensor.GetPhyAddr();
            __ubuf__ float* rstd = (__ubuf__ float*)rstdTensor.GetPhyAddr();
            AscendC::Reg::RegTensor<float> dyReg, xReg, gammaReg;
            AscendC::Reg::RegTensor<float> dygReg, xNormReg, xMainReg;
            AscendC::Reg::RegTensor<float> meanReg, rstdReg, tmpReg;
            AscendC::Reg::MaskReg pMask;
            for (uint16_t i = 0; i < outerLoopTimes; ++i) {
                LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(gammaReg, gamma + i);
                uint32_t count = static_cast<uint32_t>(colSize);
                for (uint16_t j = 0; j < innerLoopTimes; ++j) {
                    pMask = AscendC::Reg::UpdateMask<float>(count);
                    LoadAlign(meanReg, mean + j * innerLoopStride);
                    LoadAlign(rstdReg, rstd + j * innerLoopStride);
                    LoadTensorForDtypeT<T>(dy, dyReg, pMask, i * outerLoopStrideDtypeT + j * innerLoopStride);
                    Mul<float, AscendC::Reg::MaskMergeMode::ZEROING>(dygReg, dyReg, gammaReg, pMask);
                    StoreAlign(dyg + i * outerLoopStride + j * innerLoopStride, dygReg, pMask);
                    LoadTensorForDtypeT<T>(x, xReg, pMask, i * outerLoopStrideDtypeT + j * innerLoopStride);
                    Sub<float, AscendC::Reg::MaskMergeMode::ZEROING>(tmpReg, xReg, meanReg, pMask);
                    Mul<float, AscendC::Reg::MaskMergeMode::ZEROING>(xNormReg, tmpReg, rstdReg, pMask);
                    StoreAlign(xNorm + i * outerLoopStride + j * innerLoopStride, xNormReg, pMask);
                    Mul<float, AscendC::Reg::MaskMergeMode::ZEROING>(xMainReg, xNormReg, dygReg, pMask);
                    StoreAlign(xMain + i * outerLoopStride + j * innerLoopStride, xMainReg, pMask);
                }
            }
        }
    }
}

template <typename T, typename U>
__aicore__ inline void LayerNormGradV3TransposeRegBaseBackward<T, U>::ComputeDx(
    const LocalTensor<T>& dstTensor, const LocalTensor<float>& dyGammaTensor, const LocalTensor<float>& xNormTensor,
    const LocalTensor<float>& sum1Tensor, const LocalTensor<float>& sum2Tensor, const LocalTensor<float>& rstdTensor,
    const int64_t rowSize, const int64_t colSize, const int64_t stride, const int64_t fullColSize)
{
    constexpr static uint32_t VL = GetVRegSize() / sizeof(float);
    uint16_t outerLoopTimes = static_cast<uint16_t>(rowSize);
    uint16_t innerLoopTimes = static_cast<uint16_t>(
        CeilDiv(static_cast<int64_t>(colSize * sizeof(float)), static_cast<int64_t>(GetVRegSize())));
    uint32_t outerLoopStride = static_cast<uint32_t>(stride);
    uint32_t innerLoopStride = VL;
    float floatN = static_cast<float>(fullColSize);
    float reciprocalN = (floatN != 0.0f) ? static_cast<float>(1) / floatN : 0.0f;

    if (innerLoopTimes == 1) {
        __VEC_SCOPE__
        {
            __ubuf__ T* dst = (__ubuf__ T*)dstTensor.GetPhyAddr();
            __ubuf__ float* dyg = (__ubuf__ float*)dyGammaTensor.GetPhyAddr();
            __ubuf__ float* xn = (__ubuf__ float*)xNormTensor.GetPhyAddr();
            __ubuf__ float* sum1 = (__ubuf__ float*)sum1Tensor.GetPhyAddr();
            __ubuf__ float* sum2 = (__ubuf__ float*)sum2Tensor.GetPhyAddr();
            __ubuf__ float* rstd = (__ubuf__ float*)rstdTensor.GetPhyAddr();
            uint32_t count = static_cast<uint32_t>(colSize);
            AscendC::Reg::MaskReg pMask = AscendC::Reg::UpdateMask<float>(count);

            AscendC::Reg::RegTensor<float> xReg, dygReg, dxReg;
            AscendC::Reg::RegTensor<float> sum1Reg, sum2Reg, rstdReg;
            AscendC::Reg::RegTensor<float> Reg1, Reg2, Reg3, Reg4, Reg5;
            for (uint16_t i = 0; i < outerLoopTimes; ++i) {
                LoadAlign(dygReg, dyg + i * outerLoopStride);
                LoadAlign(sum1Reg, sum1);
                LoadAlign(sum2Reg, sum2);
                LoadAlign(rstdReg, rstd);
                LoadAlign(xReg, xn + i * outerLoopStride);
                Muls<float, float, AscendC::Reg::MaskMergeMode::ZEROING>(Reg1, dygReg, floatN, pMask);
                Sub<float, AscendC::Reg::MaskMergeMode::ZEROING>(Reg2, Reg1, sum1Reg, pMask);
                Mul<float, AscendC::Reg::MaskMergeMode::ZEROING>(Reg3, xReg, sum2Reg, pMask);
                Sub<float, AscendC::Reg::MaskMergeMode::ZEROING>(Reg4, Reg2, Reg3, pMask);
                Muls<float, float, AscendC::Reg::MaskMergeMode::ZEROING>(Reg5, Reg4, reciprocalN, pMask);
                Mul<float, AscendC::Reg::MaskMergeMode::ZEROING>(dxReg, Reg5, rstdReg, pMask);
                StoreTensorForDtypeT<T>(dst, dxReg, pMask, i * outerLoopStride);
            }
        }
    } else {
        __VEC_SCOPE__
        {
            __ubuf__ T* dst = (__ubuf__ T*)dstTensor.GetPhyAddr();
            __ubuf__ float* dyg = (__ubuf__ float*)dyGammaTensor.GetPhyAddr();
            __ubuf__ float* xn = (__ubuf__ float*)xNormTensor.GetPhyAddr();
            __ubuf__ float* sum1 = (__ubuf__ float*)sum1Tensor.GetPhyAddr();
            __ubuf__ float* sum2 = (__ubuf__ float*)sum2Tensor.GetPhyAddr();
            __ubuf__ float* rstd = (__ubuf__ float*)rstdTensor.GetPhyAddr();

            AscendC::Reg::RegTensor<float> xReg, dygReg, dxReg;
            AscendC::Reg::RegTensor<float> sum1Reg, sum2Reg, rstdReg;
            AscendC::Reg::RegTensor<float> Reg1, Reg2, Reg3, Reg4, Reg5;
            AscendC::Reg::MaskReg pMask;
            for (uint16_t i = 0; i < outerLoopTimes; ++i) {
                uint32_t count = static_cast<uint32_t>(colSize);
                for (uint16_t j = 0; j < innerLoopTimes; ++j) {
                    pMask = AscendC::Reg::UpdateMask<float>(count);
                    LoadAlign(dygReg, dyg + i * outerLoopStride + j * innerLoopStride);
                    LoadAlign(sum1Reg, sum1 + j * innerLoopStride);
                    LoadAlign(sum2Reg, sum2 + j * innerLoopStride);
                    LoadAlign(rstdReg, rstd + j * innerLoopStride);
                    LoadAlign(xReg, xn + i * outerLoopStride + j * innerLoopStride);
                    Muls<float, float, AscendC::Reg::MaskMergeMode::ZEROING>(Reg1, dygReg, floatN, pMask);
                    Sub<float, AscendC::Reg::MaskMergeMode::ZEROING>(Reg2, Reg1, sum1Reg, pMask);
                    Mul<float, AscendC::Reg::MaskMergeMode::ZEROING>(Reg3, xReg, sum2Reg, pMask);
                    Sub<float, AscendC::Reg::MaskMergeMode::ZEROING>(Reg4, Reg2, Reg3, pMask);
                    Muls<float, float, AscendC::Reg::MaskMergeMode::ZEROING>(Reg5, Reg4, reciprocalN, pMask);
                    Mul<float, AscendC::Reg::MaskMergeMode::ZEROING>(dxReg, Reg5, rstdReg, pMask);
                    StoreTensorForDtypeT<T>(dst, dxReg, pMask, i * outerLoopStride + j * innerLoopStride);
                }
            }
        }
    }
}

template <typename T, typename U>
__aicore__ inline void LayerNormGradV3TransposeRegBaseBackward<T, U>::DoPostTranspose(LocalTensor<T>& dstTensor,
                                                                                      LocalTensor<T>& srcTensor,
                                                                                      const int64_t mfactor)
{
    constexpr static int64_t DATA_BLOCK_COUNT = 16; // TransDataTo5HD单次16个分形
    int64_t rTileBase;
    if constexpr (IsSameType<T, float>::value) {
        rTileBase = B32_ALIGN;
    } else {
        rTileBase = B16_ALIGN;
    }

    int64_t rRepeartTimes = Arith::CeilDiv(NAlign, rTileBase);
    int64_t aRepeartTimes = Arith::CeilDiv(MfactorAlign, static_cast<int64_t>(TRANSPOSE_C0_SIZE));

    for (int64_t i = 0; i < rRepeartTimes; i++) {
        TransDataTo5HDParams params;
        params.repeatTimes = aRepeartTimes;
        params.srcRepStride = (aRepeartTimes == 1) ? 0 : 1;
        if constexpr (IsSameType<T, float>::value) {
            params.srcRepStride = (aRepeartTimes == 1) ? 0 : 2;
            params.dstRepStride = (aRepeartTimes == 1) ? 0 : (DATA_BLOCK_COUNT * rRepeartTimes);
            LocalTensor<T> srcLocalList[DATA_BLOCK_COUNT];
            LocalTensor<T> dstLocalList[DATA_BLOCK_COUNT];
            int64_t halfBlock = rTileBase;
            for (int64_t j = 0; j < halfBlock; j++) {
                int64_t srcOffset = rTileBase * MfactorAlign * i + MfactorAlign * j;
                srcLocalList[j] = srcTensor[srcOffset];
                srcLocalList[j + halfBlock] = srcTensor[srcOffset + halfBlock];
            }
            for (int64_t j = 0; j < halfBlock; j++) {
                int64_t dstOffset = rTileBase * i + halfBlock * rRepeartTimes * j;
                dstLocalList[j * 2] = dstTensor[dstOffset];
                dstLocalList[j * 2 + 1] = dstTensor[dstOffset + halfBlock * halfBlock * rRepeartTimes];
            }
            AscendC::TransDataTo5HD(dstLocalList, srcLocalList, params);
        } else {
            params.dstRepStride = (aRepeartTimes == 1) ? 0 : (rTileBase * rRepeartTimes);
            LocalTensor<T> srcLocalList[DATA_BLOCK_COUNT];
            LocalTensor<T> dstLocalList[DATA_BLOCK_COUNT];
            for (int64_t j = 0; j < DATA_BLOCK_COUNT; j++) {
                int64_t srcOffset = rTileBase * MfactorAlign * i + MfactorAlign * j;
                srcLocalList[j] = srcTensor[srcOffset];
            }
            for (int64_t j = 0; j < DATA_BLOCK_COUNT; j++) {
                int64_t dstOffset = rTileBase * i + NAlign * j;
                dstLocalList[j] = dstTensor[dstOffset];
            }
            AscendC::TransDataTo5HD<T>(dstLocalList, srcLocalList, params);
        }
    }
}

template <typename T, typename U>
__aicore__ inline void LayerNormGradV3TransposeRegBaseBackward<T, U>::CopyOutData(const int64_t mi,
                                                                                  const int64_t mfactor)
{
    int64_t gmOffset = mi * MfactorAlign * N;
    LocalTensor<T> dxOut = outQueueDx.template DeQue<T>();
    CopyOut(pdXOutTensorGM[gmOffset], dxOut, mfactor, N, N, NAlign);
    outQueueDx.FreeTensor(dxOut);
}

} // namespace LayerNormGradV3
#endif // LAYER_NORM_GRAD_V3_TRANSPOSE_BACKWARD_IMPL_
