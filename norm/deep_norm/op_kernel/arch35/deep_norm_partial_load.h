/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file deep_norm_partial_load.h
 * \brief DeepNorm regbase partial-load kernel for arch35 (Ascend950).
 */

#ifndef DEEP_NORM_PARTIAL_LOAD_ARCH35_H
#define DEEP_NORM_PARTIAL_LOAD_ARCH35_H

#include "deep_norm.h"

namespace NsDeepNorm {

using AscendC::MicroAPI::LocalMemBar;
using AscendC::MicroAPI::MemType;

constexpr uint32_t DEEP_NORM_REDUCE_TMP_ELEMS = 2 * DEEP_NORM_VL_FP32;

template <typename U>
__aicore__ inline U DeepNormMin(U lhs, U rhs)
{
    return lhs < rhs ? lhs : rhs;
}

template <typename T>
class DeepNormPartialLoad {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gx, GM_ADDR beta, GM_ADDR gamma, GM_ADDR mean, GM_ADDR rstd,
                                GM_ADDR y, const DeepNormTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline LocalTensor<T> CopyInTensor(TQue<QuePosition::VECIN, 1>& queue, GlobalTensor<T>& gm,
                                                  uint64_t offset, uint32_t count);
    __aicore__ inline void CopyOutTensor(GlobalTensor<T>& gm, LocalTensor<T>& local, uint64_t offset, uint32_t count);
    __aicore__ inline void CopyOutScalar(GlobalTensor<float>& gm, LocalTensor<float>& local, uint64_t offset);
    __aicore__ inline void InitScalar(LocalTensor<float>& scalar);
    __aicore__ inline void AccumulateScalar(LocalTensor<float>& dst, LocalTensor<float>& src);
    __aicore__ inline void ScaleScalar(LocalTensor<float>& scalar, float scale);
    __aicore__ inline uint32_t GetPowerSplit(uint32_t count) const;
    __aicore__ inline void ComputeH(LocalTensor<T>& x, LocalTensor<T>& gx, LocalTensor<float>& h, uint32_t count);
    __aicore__ inline void ComputeCenteredSquare(LocalTensor<T>& x, LocalTensor<T>& gx, LocalTensor<float>& mean,
                                                 LocalTensor<float>& square, uint32_t count);
    __aicore__ inline void ComputeY(LocalTensor<T>& x, LocalTensor<T>& gx, LocalTensor<T>& gamma, LocalTensor<T>& beta,
                                    LocalTensor<float>& mean, LocalTensor<float>& rstd, LocalTensor<T>& y,
                                    uint32_t count);
    __aicore__ inline void ProcessRow(uint64_t row);

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, 1> xQue_;
    TQue<QuePosition::VECIN, 1> gxQue_;
    TQue<QuePosition::VECIN, 1> gammaQue_;
    TQue<QuePosition::VECIN, 1> betaQue_;
    TQue<QuePosition::VECOUT, 1> yQue_;
    TQue<QuePosition::VECOUT, 1> meanQue_;
    TQue<QuePosition::VECOUT, 1> rstdQue_;
    TBuf<TPosition::VECCALC> calcBuf_;
    TBuf<TPosition::VECCALC> tileSumBuf_;
    TBuf<TPosition::VECCALC> reduceTmpBuf_;

    GlobalTensor<T> xGm_;
    GlobalTensor<T> gxGm_;
    GlobalTensor<T> betaGm_;
    GlobalTensor<T> gammaGm_;
    GlobalTensor<T> yGm_;
    GlobalTensor<float> meanGm_;
    GlobalTensor<float> rstdGm_;

    uint32_t numCol_ = 0;
    uint32_t rowStart_ = 0;
    uint32_t rowWork_ = 0;
    uint32_t tileLength_ = 0;
    float eps_ = 0.0f;
    float alpha_ = 0.0f;
    float avgFactor_ = 0.0f;
};

template <typename T>
__aicore__ inline void DeepNormPartialLoad<T>::Init(GM_ADDR x, GM_ADDR gx, GM_ADDR beta, GM_ADDR gamma, GM_ADDR mean,
                                                    GM_ADDR rstd, GM_ADDR y, const DeepNormTilingData* tilingData)
{
    numCol_ = tilingData->numCol;
    tileLength_ = tilingData->tileLength;
    eps_ = tilingData->eps;
    alpha_ = tilingData->alpha;
    avgFactor_ = tilingData->avgFactor;

    uint32_t rowPerCore = tilingData->rowPerCore;
    uint32_t numRow = tilingData->numRow;
    uint64_t rowStart = static_cast<uint64_t>(GetBlockIdx()) * rowPerCore;
    if (rowStart >= numRow) {
        return;
    }
    rowStart_ = static_cast<uint32_t>(rowStart);
    uint32_t remain = numRow - rowStart_;
    rowWork_ = remain > rowPerCore ? rowPerCore : remain;

    xGm_.SetGlobalBuffer((__gm__ T*)x);
    gxGm_.SetGlobalBuffer((__gm__ T*)gx);
    betaGm_.SetGlobalBuffer((__gm__ T*)beta);
    gammaGm_.SetGlobalBuffer((__gm__ T*)gamma);
    yGm_.SetGlobalBuffer((__gm__ T*)y);
    meanGm_.SetGlobalBuffer((__gm__ float*)mean);
    rstdGm_.SetGlobalBuffer((__gm__ float*)rstd);

    uint32_t dtypeBytes = tileLength_ * sizeof(T);
    uint32_t fp32Bytes = tileLength_ * sizeof(float);
    pipe_.InitBuffer(xQue_, 1, dtypeBytes);
    pipe_.InitBuffer(gxQue_, 1, dtypeBytes);
    pipe_.InitBuffer(gammaQue_, 1, dtypeBytes);
    pipe_.InitBuffer(betaQue_, 1, dtypeBytes);
    pipe_.InitBuffer(yQue_, 1, dtypeBytes);
    pipe_.InitBuffer(meanQue_, 1, DEEP_NORM_BLOCK_SIZE);
    pipe_.InitBuffer(rstdQue_, 1, DEEP_NORM_BLOCK_SIZE);
    pipe_.InitBuffer(calcBuf_, fp32Bytes);
    pipe_.InitBuffer(tileSumBuf_, DEEP_NORM_BLOCK_SIZE);
    pipe_.InitBuffer(reduceTmpBuf_, DEEP_NORM_REDUCE_TMP_ELEMS * sizeof(float));
}

template <typename T>
__aicore__ inline LocalTensor<T> DeepNormPartialLoad<T>::CopyInTensor(TQue<QuePosition::VECIN, 1>& queue,
                                                                      GlobalTensor<T>& gm, uint64_t offset,
                                                                      uint32_t count)
{
    LocalTensor<T> local = queue.template AllocTensor<T>();
    uint32_t blockElems = DEEP_NORM_BLOCK_SIZE / static_cast<uint32_t>(sizeof(T));
    uint32_t blockAligned = (count + blockElems - 1) / blockElems * blockElems;
    DataCopyExtParams params{1, count * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> pad{true, 0, static_cast<uint8_t>(blockAligned - count), static_cast<T>(0)};
    DataCopyPad(local, gm[offset], params, pad);
    queue.EnQue(local);
    return queue.template DeQue<T>();
}

template <typename T>
__aicore__ inline void DeepNormPartialLoad<T>::CopyOutTensor(GlobalTensor<T>& gm, LocalTensor<T>& local,
                                                             uint64_t offset, uint32_t count)
{
    DataCopyExtParams params{1, count * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
    DataCopyPad(gm[offset], local, params);
}

template <typename T>
__aicore__ inline void DeepNormPartialLoad<T>::CopyOutScalar(GlobalTensor<float>& gm, LocalTensor<float>& local,
                                                             uint64_t offset)
{
    DataCopyExtParams params{1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
    DataCopyPad(gm[offset], local, params);
}

template <typename T>
__aicore__ inline void DeepNormPartialLoad<T>::InitScalar(LocalTensor<float>& scalar)
{
    __local_mem__ float* ptr = (__local_mem__ float*)scalar.GetPhyAddr();
    __VEC_SCOPE__
    {
        RegTensor<float> value;
        MaskReg mask = CreateMask<float, MaskPattern::VL1>();
        Duplicate(value, 0.0f, mask);
        DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(ptr, value, mask);
    }
}

template <typename T>
__aicore__ inline void DeepNormPartialLoad<T>::AccumulateScalar(LocalTensor<float>& dst, LocalTensor<float>& src)
{
    __local_mem__ float* dstPtr = (__local_mem__ float*)dst.GetPhyAddr();
    __local_mem__ float* srcPtr = (__local_mem__ float*)src.GetPhyAddr();
    __VEC_SCOPE__
    {
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        RegTensor<float> dstReg;
        RegTensor<float> srcReg;
        RegTensor<float> result;
        MaskReg mask = CreateMask<float, MaskPattern::VL1>();
        DataCopy<float, LoadDist::DIST_BRC_B32>(dstReg, dstPtr);
        DataCopy<float, LoadDist::DIST_BRC_B32>(srcReg, srcPtr);
        Add(result, dstReg, srcReg, mask);
        DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dstPtr, result, mask);
    }
}

template <typename T>
__aicore__ inline void DeepNormPartialLoad<T>::ScaleScalar(LocalTensor<float>& scalar, float scale)
{
    __local_mem__ float* ptr = (__local_mem__ float*)scalar.GetPhyAddr();
    __VEC_SCOPE__
    {
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        RegTensor<float> value;
        RegTensor<float> result;
        MaskReg mask = CreateMask<float, MaskPattern::VL1>();
        DataCopy<float, LoadDist::DIST_BRC_B32>(value, ptr);
        Muls(result, value, scale, mask);
        DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(ptr, result, mask);
    }
}

template <typename T>
__aicore__ inline uint32_t DeepNormPartialLoad<T>::GetPowerSplit(uint32_t count) const
{
    uint32_t power = DEEP_NORM_VL_FP32;
    while (power <= count / 2) {
        power *= 2;
    }
    return power;
}

template <typename T>
__aicore__ inline void DeepNormPartialLoad<T>::ComputeH(LocalTensor<T>& x, LocalTensor<T>& gx, LocalTensor<float>& h,
                                                        uint32_t count)
{
    __local_mem__ T* xPtr = (__local_mem__ T*)x.GetPhyAddr();
    __local_mem__ T* gxPtr = (__local_mem__ T*)gx.GetPhyAddr();
    __local_mem__ float* hPtr = (__local_mem__ float*)h.GetPhyAddr();
    uint16_t loops = static_cast<uint16_t>((count + DEEP_NORM_VL_FP32 - 1) / DEEP_NORM_VL_FP32);
    uint32_t remaining = count;
    float alpha = alpha_;
    __VEC_SCOPE__
    {
        RegTensor<float> xReg;
        RegTensor<float> gxReg;
        RegTensor<float> hReg;
        for (uint16_t i = 0; i < loops; ++i) {
            uint32_t valid = DeepNormMin(remaining, DEEP_NORM_VL_FP32);
            MaskReg mask = UpdateMask<float>(valid);
            uint32_t offset = static_cast<uint32_t>(i) * DEEP_NORM_VL_FP32;
            NormCommon::NormCommonRegbase::LoadRegForDtype<T>(xPtr, xReg, mask, offset);
            NormCommon::NormCommonRegbase::LoadRegForDtype<T>(gxPtr, gxReg, mask, offset);
            Muls(xReg, xReg, alpha, mask);
            Add(hReg, xReg, gxReg, mask);
            DataCopy<float, StoreDist::DIST_NORM_B32>(hPtr + offset, hReg, mask);
            remaining -= valid;
        }
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    }
}

template <typename T>
__aicore__ inline void DeepNormPartialLoad<T>::ComputeCenteredSquare(LocalTensor<T>& x, LocalTensor<T>& gx,
                                                                     LocalTensor<float>& mean,
                                                                     LocalTensor<float>& square, uint32_t count)
{
    __local_mem__ T* xPtr = (__local_mem__ T*)x.GetPhyAddr();
    __local_mem__ T* gxPtr = (__local_mem__ T*)gx.GetPhyAddr();
    __local_mem__ float* meanPtr = (__local_mem__ float*)mean.GetPhyAddr();
    __local_mem__ float* squarePtr = (__local_mem__ float*)square.GetPhyAddr();
    uint16_t loops = static_cast<uint16_t>((count + DEEP_NORM_VL_FP32 - 1) / DEEP_NORM_VL_FP32);
    uint32_t remaining = count;
    float alpha = alpha_;
    __VEC_SCOPE__
    {
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        RegTensor<float> xReg;
        RegTensor<float> gxReg;
        RegTensor<float> meanReg;
        RegTensor<float> centeredReg;
        RegTensor<float> squareReg;
        DataCopy<float, LoadDist::DIST_BRC_B32>(meanReg, meanPtr);
        for (uint16_t i = 0; i < loops; ++i) {
            uint32_t valid = DeepNormMin(remaining, DEEP_NORM_VL_FP32);
            MaskReg mask = UpdateMask<float>(valid);
            uint32_t offset = static_cast<uint32_t>(i) * DEEP_NORM_VL_FP32;
            NormCommon::NormCommonRegbase::LoadRegForDtype<T>(xPtr, xReg, mask, offset);
            NormCommon::NormCommonRegbase::LoadRegForDtype<T>(gxPtr, gxReg, mask, offset);
            Muls(xReg, xReg, alpha, mask);
            Add(centeredReg, xReg, gxReg, mask);
            Sub(centeredReg, centeredReg, meanReg, mask);
            Mul(squareReg, centeredReg, centeredReg, mask);
            DataCopy<float, StoreDist::DIST_NORM_B32>(squarePtr + offset, squareReg, mask);
            remaining -= valid;
        }
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    }
}

template <typename T>
__aicore__ inline void DeepNormPartialLoad<T>::ComputeY(LocalTensor<T>& x, LocalTensor<T>& gx, LocalTensor<T>& gamma,
                                                        LocalTensor<T>& beta, LocalTensor<float>& mean,
                                                        LocalTensor<float>& rstd, LocalTensor<T>& y, uint32_t count)
{
    __local_mem__ T* xPtr = (__local_mem__ T*)x.GetPhyAddr();
    __local_mem__ T* gxPtr = (__local_mem__ T*)gx.GetPhyAddr();
    __local_mem__ T* gammaPtr = (__local_mem__ T*)gamma.GetPhyAddr();
    __local_mem__ T* betaPtr = (__local_mem__ T*)beta.GetPhyAddr();
    __local_mem__ float* meanPtr = (__local_mem__ float*)mean.GetPhyAddr();
    __local_mem__ float* rstdPtr = (__local_mem__ float*)rstd.GetPhyAddr();
    __local_mem__ T* yPtr = (__local_mem__ T*)y.GetPhyAddr();
    uint16_t loops = static_cast<uint16_t>((count + DEEP_NORM_VL_FP32 - 1) / DEEP_NORM_VL_FP32);
    uint32_t remaining = count;
    float alpha = alpha_;
    __VEC_SCOPE__
    {
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        RegTensor<float> xReg;
        RegTensor<float> gxReg;
        RegTensor<float> gammaReg;
        RegTensor<float> betaReg;
        RegTensor<float> meanReg;
        RegTensor<float> rstdReg;
        RegTensor<float> yReg;
        DataCopy<float, LoadDist::DIST_BRC_B32>(meanReg, meanPtr);
        DataCopy<float, LoadDist::DIST_BRC_B32>(rstdReg, rstdPtr);
        for (uint16_t i = 0; i < loops; ++i) {
            uint32_t valid = DeepNormMin(remaining, DEEP_NORM_VL_FP32);
            MaskReg mask = UpdateMask<float>(valid);
            uint32_t offset = static_cast<uint32_t>(i) * DEEP_NORM_VL_FP32;
            NormCommon::NormCommonRegbase::LoadRegForDtype<T>(xPtr, xReg, mask, offset);
            NormCommon::NormCommonRegbase::LoadRegForDtype<T>(gxPtr, gxReg, mask, offset);
            NormCommon::NormCommonRegbase::LoadRegForDtype<T>(gammaPtr, gammaReg, mask, offset);
            NormCommon::NormCommonRegbase::LoadRegForDtype<T>(betaPtr, betaReg, mask, offset);
            Muls(xReg, xReg, alpha, mask);
            Add(yReg, xReg, gxReg, mask);
            Sub(yReg, yReg, meanReg, mask);
            Mul(yReg, yReg, rstdReg, mask);
            Mul(yReg, yReg, gammaReg, mask);
            Add(yReg, yReg, betaReg, mask);
            NormCommon::NormCommonRegbase::StoreRegForDtype<T>(yPtr, yReg, mask, offset);
            remaining -= valid;
        }
    }
}

template <typename T>
__aicore__ inline void DeepNormPartialLoad<T>::ProcessRow(uint64_t row)
{
    uint64_t rowOffset = row * numCol_;
    LocalTensor<float> tileSum = tileSumBuf_.Get<float>();
    LocalTensor<float> calc = calcBuf_.Get<float>();

    LocalTensor<float> mean = meanQue_.template AllocTensor<float>();
    InitScalar(mean);
    for (uint64_t col = 0; col < numCol_; col += tileLength_) {
        uint32_t count = static_cast<uint32_t>(DeepNormMin(static_cast<uint64_t>(tileLength_), numCol_ - col));
        LocalTensor<T> x = CopyInTensor(xQue_, xGm_, rowOffset + col, count);
        LocalTensor<T> gx = CopyInTensor(gxQue_, gxGm_, rowOffset + col, count);
        ComputeH(x, gx, calc, count);
        NormCommon::NormCommonRegbase::CalculateReduceSum(calc, tileSum, reduceTmpBuf_, count, GetPowerSplit(count));
        AccumulateScalar(mean, tileSum);
        xQue_.FreeTensor(x);
        gxQue_.FreeTensor(gx);
    }
    ScaleScalar(mean, avgFactor_);
    meanQue_.EnQue(mean);
    mean = meanQue_.template DeQue<float>();
    CopyOutScalar(meanGm_, mean, row);

    LocalTensor<float> rstd = rstdQue_.template AllocTensor<float>();
    InitScalar(rstd);
    for (uint64_t col = 0; col < numCol_; col += tileLength_) {
        uint32_t count = static_cast<uint32_t>(DeepNormMin(static_cast<uint64_t>(tileLength_), numCol_ - col));
        LocalTensor<T> x = CopyInTensor(xQue_, xGm_, rowOffset + col, count);
        LocalTensor<T> gx = CopyInTensor(gxQue_, gxGm_, rowOffset + col, count);
        ComputeCenteredSquare(x, gx, mean, calc, count);
        NormCommon::NormCommonRegbase::CalculateReduceSum(calc, tileSum, reduceTmpBuf_, count, GetPowerSplit(count));
        AccumulateScalar(rstd, tileSum);
        xQue_.FreeTensor(x);
        gxQue_.FreeTensor(gx);
    }
    __VEC_SCOPE__ { LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>(); }
    NormCommon::ComputeRstdNewtonRaphson<true, true>(rstd, rstd, 1, eps_, avgFactor_, DEEP_NORM_VL_FP32);
    rstdQue_.EnQue(rstd);
    rstd = rstdQue_.template DeQue<float>();
    CopyOutScalar(rstdGm_, rstd, row);

    for (uint64_t col = 0; col < numCol_; col += tileLength_) {
        uint32_t count = static_cast<uint32_t>(DeepNormMin(static_cast<uint64_t>(tileLength_), numCol_ - col));
        LocalTensor<T> x = CopyInTensor(xQue_, xGm_, rowOffset + col, count);
        LocalTensor<T> gx = CopyInTensor(gxQue_, gxGm_, rowOffset + col, count);
        LocalTensor<T> gamma = CopyInTensor(gammaQue_, gammaGm_, col, count);
        LocalTensor<T> beta = CopyInTensor(betaQue_, betaGm_, col, count);
        LocalTensor<T> y = yQue_.template AllocTensor<T>();
        ComputeY(x, gx, gamma, beta, mean, rstd, y, count);
        yQue_.EnQue(y);
        xQue_.FreeTensor(x);
        gxQue_.FreeTensor(gx);
        gammaQue_.FreeTensor(gamma);
        betaQue_.FreeTensor(beta);
        y = yQue_.template DeQue<T>();
        CopyOutTensor(yGm_, y, rowOffset + col, count);
        yQue_.FreeTensor(y);
    }

    meanQue_.FreeTensor(mean);
    rstdQue_.FreeTensor(rstd);
}

template <typename T>
__aicore__ inline void DeepNormPartialLoad<T>::Process()
{
    for (uint32_t r = 0; r < rowWork_; ++r) {
        ProcessRow(static_cast<uint64_t>(rowStart_) + r);
    }
}

} // namespace NsDeepNorm

#endif // DEEP_NORM_PARTIAL_LOAD_ARCH35_H
