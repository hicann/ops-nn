/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DEEP_NORM_GRAD_ARCH35_H
#define DEEP_NORM_GRAD_ARCH35_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "deep_norm_grad_tiling_data.h"
#include "../../norm_common/reduce_common_regbase.h"

namespace DeepNormGradArch35 {

using namespace AscendC;
using AscendC::MicroAPI::CreateMask;
using AscendC::MicroAPI::LoadDist;
using AscendC::MicroAPI::LocalMemBar;
using AscendC::MicroAPI::MaskPattern;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::MemType;
using AscendC::MicroAPI::RegTensor;
using AscendC::MicroAPI::StoreDist;
using AscendC::MicroAPI::UpdateMask;

constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t VL_FP32 = 256 / sizeof(float);
constexpr uint32_t SCALAR_BLOCK_ELEMS = BLOCK_SIZE / sizeof(float);
constexpr uint32_t REDUCE_TMP_ELEMS = 2 * VL_FP32;
constexpr AscendC::MicroAPI::CastTrait CAST_TRAIT_B32_TO_F16 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};
constexpr AscendC::MicroAPI::CastTrait CAST_TRAIT_B32_TO_BF16 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

template <typename U>
__aicore__ inline U Min(U lhs, U rhs)
{
    return lhs < rhs ? lhs : rhs;
}

template <typename T>
class DeepNormGrad {
public:
    __aicore__ inline void Init(GM_ADDR dy, GM_ADDR x, GM_ADDR gx, GM_ADDR gamma, GM_ADDR mean, GM_ADDR rstd,
                                GM_ADDR dx, GM_ADDR dgx, GM_ADDR dbeta, GM_ADDR dgamma, GM_ADDR workspace,
                                const DeepNormGradTilingDataArch35* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline LocalTensor<T> CopyInTensor(TQue<QuePosition::VECIN, 1>& queue, GlobalTensor<T>& gm,
                                                  uint64_t offset, uint32_t count);
    __aicore__ inline LocalTensor<float> CopyInScalar(TQue<QuePosition::VECIN, 1>& queue, GlobalTensor<float>& gm,
                                                      uint64_t offset);
    __aicore__ inline LocalTensor<T> CopyInTensorBatch(TQue<QuePosition::VECIN, 1>& queue, GlobalTensor<T>& gm,
                                                       uint64_t offset, uint32_t rows);
    __aicore__ inline LocalTensor<float> CopyInScalars(TQue<QuePosition::VECIN, 1>& queue, GlobalTensor<float>& gm,
                                                       uint64_t offset, uint32_t count);
    __aicore__ inline LocalTensor<float> CopyInFloatTensor(TQue<QuePosition::VECIN, 1>& queue, GlobalTensor<float>& gm,
                                                           uint64_t offset, uint32_t count);
    __aicore__ inline void CopyOutTensor(TQue<QuePosition::VECOUT, 1>& queue, GlobalTensor<T>& gm, uint64_t offset,
                                         uint32_t count);
    __aicore__ inline void CopyOutTensorBatch(TQue<QuePosition::VECOUT, 1>& queue, GlobalTensor<T>& gm, uint64_t offset,
                                              uint32_t rows);
    __aicore__ inline void CopyOutFloat(TQue<QuePosition::VECOUT, 1>& queue, GlobalTensor<float>& gm, uint64_t offset,
                                        uint32_t count);
    __aicore__ inline void StoreOutputForDtype(__local_mem__ T* dst, RegTensor<float>& src, MaskReg& mask,
                                               uint32_t offset);
    __aicore__ inline void InitScalar(LocalTensor<float>& scalar);
    __aicore__ inline void AccumulateScalar(LocalTensor<float>& dst, LocalTensor<float>& src);
    __aicore__ inline void ScaleScalar(LocalTensor<float>& scalar, float scale);
    __aicore__ inline uint32_t GetPowerSplit(uint32_t count) const;
    __aicore__ inline void ComputeFirstPass(LocalTensor<T>& dy, LocalTensor<T>& x, LocalTensor<T>& gx,
                                            LocalTensor<T>& gamma, LocalTensor<float>& mean, LocalTensor<float>& rstd,
                                            uint32_t count);
    __aicore__ inline void ComputeSecondPass(LocalTensor<T>& dy, LocalTensor<T>& x, LocalTensor<T>& gx,
                                             LocalTensor<T>& gamma, LocalTensor<float>& mean, LocalTensor<float>& rstd,
                                             LocalTensor<float>& avgTmp, LocalTensor<float>& avgTmpNorm,
                                             uint32_t count);
    __aicore__ inline void ComputeGammaBeta(LocalTensor<T>& dy, LocalTensor<T>& x, LocalTensor<T>& gx,
                                            LocalTensor<float>& mean, LocalTensor<float>& rstd,
                                            LocalTensor<float>& dgamma, LocalTensor<float>& dbeta, uint32_t count);
    __aicore__ inline void ComputeSmallDSecondPass(LocalTensor<T>& dy, LocalTensor<T>& x, LocalTensor<T>& gx,
                                                   LocalTensor<T>& gamma, LocalTensor<float>& mean,
                                                   LocalTensor<float>& rstd, LocalTensor<float>& avgTmp,
                                                   LocalTensor<float>& avgTmpNorm, LocalTensor<T>& dx,
                                                   LocalTensor<T>& dgx, LocalTensor<float>& dgamma,
                                                   LocalTensor<float>& dbeta, LocalTensor<float>& dgammaComp,
                                                   LocalTensor<float>& dbetaComp, uint32_t count);
    template <uint32_t COLS>
    __aicore__ inline void ComputeTinyDBatch(LocalTensor<T>& dy, LocalTensor<T>& x, LocalTensor<T>& gx,
                                             LocalTensor<T>& gamma, LocalTensor<float>& mean, LocalTensor<float>& rstd,
                                             LocalTensor<T>& dx, LocalTensor<T>& dgx, LocalTensor<float>& dgamma,
                                             LocalTensor<float>& dbeta, LocalTensor<float>& dgammaComp,
                                             LocalTensor<float>& dbetaComp, uint32_t rows);
    __aicore__ inline void AccumulateKahan(LocalTensor<float>& sum, LocalTensor<float>& compensation,
                                           LocalTensor<float>& value, uint32_t count);
    __aicore__ inline void ProcessBackwardRow(uint64_t row);
    __aicore__ inline void ProcessBackward();
    __aicore__ inline void ProcessGammaBeta();
    __aicore__ inline void ProcessSmallD();
    __aicore__ inline void ReduceSmallDPartials();

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, 1> dyQueue_;
    TQue<QuePosition::VECIN, 1> xQueue_;
    TQue<QuePosition::VECIN, 1> gxQueue_;
    TQue<QuePosition::VECIN, 1> gammaQueue_;
    TQue<QuePosition::VECIN, 1> meanQueue_;
    TQue<QuePosition::VECIN, 1> rstdQueue_;
    TQue<QuePosition::VECOUT, 1> dxQueue_;
    TQue<QuePosition::VECOUT, 1> dgxQueue_;
    TQue<QuePosition::VECOUT, 1> dbetaQueue_;
    TQue<QuePosition::VECOUT, 1> dgammaQueue_;
    TBuf<TPosition::VECCALC> calc0Buf_;
    TBuf<TPosition::VECCALC> calc1Buf_;
    TBuf<TPosition::VECCALC> sumTmpBuf_;
    TBuf<TPosition::VECCALC> sumTmpNormBuf_;
    TBuf<TPosition::VECCALC> tileSumBuf_;
    TBuf<TPosition::VECCALC> reduceTmpBuf_;
    TBuf<TPosition::VECCALC> dgammaCompBuf_;
    TBuf<TPosition::VECCALC> dbetaCompBuf_;

    GlobalTensor<T> dyGm_;
    GlobalTensor<T> xGm_;
    GlobalTensor<T> gxGm_;
    GlobalTensor<T> gammaGm_;
    GlobalTensor<T> dxGm_;
    GlobalTensor<T> dgxGm_;
    GlobalTensor<float> meanGm_;
    GlobalTensor<float> rstdGm_;
    GlobalTensor<float> dbetaGm_;
    GlobalTensor<float> dgammaGm_;
    GlobalTensor<float> workspaceGm_;

    const DeepNormGradTilingDataArch35* tiling_ = nullptr;
    uint32_t tileLength_ = 0;
    uint32_t tileLengthAlign_ = 0;
    float alpha_ = 0.0f;
    float invCols_ = 0.0f;
};

template <typename T>
__aicore__ inline void DeepNormGrad<T>::Init(GM_ADDR dy, GM_ADDR x, GM_ADDR gx, GM_ADDR gamma, GM_ADDR mean,
                                             GM_ADDR rstd, GM_ADDR dx, GM_ADDR dgx, GM_ADDR dbeta, GM_ADDR dgamma,
                                             GM_ADDR workspace, const DeepNormGradTilingDataArch35* tilingData)
{
    tiling_ = tilingData;
    tileLength_ = tiling_->tileLength;
    tileLengthAlign_ = tiling_->tileLengthAlign;
    alpha_ = tiling_->alpha;
    invCols_ = tiling_->invCols;

    dyGm_.SetGlobalBuffer((__gm__ T*)dy);
    xGm_.SetGlobalBuffer((__gm__ T*)x);
    gxGm_.SetGlobalBuffer((__gm__ T*)gx);
    gammaGm_.SetGlobalBuffer((__gm__ T*)gamma);
    meanGm_.SetGlobalBuffer((__gm__ float*)mean);
    rstdGm_.SetGlobalBuffer((__gm__ float*)rstd);
    dxGm_.SetGlobalBuffer((__gm__ T*)dx);
    dgxGm_.SetGlobalBuffer((__gm__ T*)dgx);
    dbetaGm_.SetGlobalBuffer((__gm__ float*)dbeta);
    dgammaGm_.SetGlobalBuffer((__gm__ float*)dgamma);

    if (tiling_->gammaBetaRowSplit != 0) {
        workspaceGm_.SetGlobalBuffer((__gm__ float*)workspace);
        uint32_t tensorElements = tiling_->smallRowsPerTile * tiling_->smallRowStride + VL_FP32;
        uint32_t tensorBytes = tensorElements * sizeof(T);
        uint32_t gammaBytes = tiling_->smallColsAlign * sizeof(T);
        uint32_t paramElements = (tiling_->smallRowsPerTile + SCALAR_BLOCK_ELEMS - 1) / SCALAR_BLOCK_ELEMS *
                                 SCALAR_BLOCK_ELEMS;
        uint32_t partialBytes = tiling_->smallColsAlign * sizeof(float);
        pipe_.InitBuffer(dyQueue_, 1, tensorBytes);
        pipe_.InitBuffer(xQueue_, 1, tensorBytes);
        pipe_.InitBuffer(gxQueue_, 1, tensorBytes);
        pipe_.InitBuffer(gammaQueue_, 1, gammaBytes);
        pipe_.InitBuffer(meanQueue_, 1, paramElements * sizeof(float));
        pipe_.InitBuffer(rstdQueue_, 1, paramElements * sizeof(float));
        pipe_.InitBuffer(dxQueue_, 1, tensorBytes);
        pipe_.InitBuffer(dgxQueue_, 1, tensorBytes);
        pipe_.InitBuffer(dbetaQueue_, 1, partialBytes);
        pipe_.InitBuffer(dgammaQueue_, 1, partialBytes);
        pipe_.InitBuffer(calc0Buf_, partialBytes);
        pipe_.InitBuffer(calc1Buf_, partialBytes);
        pipe_.InitBuffer(dgammaCompBuf_, partialBytes);
        pipe_.InitBuffer(dbetaCompBuf_, partialBytes);
    } else {
        uint32_t dtypeBufferBytes = tileLengthAlign_ * sizeof(T);
        uint32_t fp32BufferBytes = tileLengthAlign_ * sizeof(float);
        pipe_.InitBuffer(dyQueue_, 1, dtypeBufferBytes);
        pipe_.InitBuffer(xQueue_, 1, dtypeBufferBytes);
        pipe_.InitBuffer(gxQueue_, 1, dtypeBufferBytes);
        pipe_.InitBuffer(gammaQueue_, 1, dtypeBufferBytes);
        pipe_.InitBuffer(meanQueue_, 1, BLOCK_SIZE);
        pipe_.InitBuffer(rstdQueue_, 1, BLOCK_SIZE);
        pipe_.InitBuffer(dxQueue_, 1, dtypeBufferBytes);
        pipe_.InitBuffer(dgxQueue_, 1, dtypeBufferBytes);
        pipe_.InitBuffer(dbetaQueue_, 1, fp32BufferBytes);
        pipe_.InitBuffer(dgammaQueue_, 1, fp32BufferBytes);
        pipe_.InitBuffer(calc0Buf_, fp32BufferBytes);
        pipe_.InitBuffer(calc1Buf_, fp32BufferBytes);
    }
    pipe_.InitBuffer(sumTmpBuf_, BLOCK_SIZE);
    pipe_.InitBuffer(sumTmpNormBuf_, BLOCK_SIZE);
    pipe_.InitBuffer(tileSumBuf_, 2 * BLOCK_SIZE);
    pipe_.InitBuffer(reduceTmpBuf_, REDUCE_TMP_ELEMS * sizeof(float));
}

template <typename T>
__aicore__ inline LocalTensor<T> DeepNormGrad<T>::CopyInTensor(TQue<QuePosition::VECIN, 1>& queue, GlobalTensor<T>& gm,
                                                               uint64_t offset, uint32_t count)
{
    LocalTensor<T> local = queue.template AllocTensor<T>();
    uint32_t blockElements = BLOCK_SIZE / sizeof(T);
    uint32_t blockAligned = (count + blockElements - 1) / blockElements * blockElements;
    DataCopyExtParams params{1, count * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> pad{true, 0, static_cast<uint8_t>(blockAligned - count), static_cast<T>(0)};
    DataCopyPad(local, gm[offset], params, pad);
    queue.EnQue(local);
    return queue.template DeQue<T>();
}

template <typename T>
__aicore__ inline LocalTensor<float> DeepNormGrad<T>::CopyInScalar(TQue<QuePosition::VECIN, 1>& queue,
                                                                   GlobalTensor<float>& gm, uint64_t offset)
{
    LocalTensor<float> local = queue.template AllocTensor<float>();
    DataCopyExtParams params{1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
    DataCopyPad(local, gm[offset], params, {true, 0, 0, 0});
    queue.EnQue(local);
    return queue.template DeQue<float>();
}

template <typename T>
__aicore__ inline LocalTensor<T> DeepNormGrad<T>::CopyInTensorBatch(TQue<QuePosition::VECIN, 1>& queue,
                                                                    GlobalTensor<T>& gm, uint64_t offset, uint32_t rows)
{
    LocalTensor<T> local = queue.template AllocTensor<T>();
    uint32_t rightPadding = tiling_->smallRowStride - static_cast<uint32_t>(tiling_->numCols);
    DataCopyExtParams params{static_cast<uint16_t>(rows),
                             static_cast<uint32_t>(tiling_->numCols * static_cast<uint64_t>(sizeof(T))), 0, 0, 0};
    DataCopyPadExtParams<T> pad{true, 0, static_cast<uint8_t>(rightPadding), static_cast<T>(0)};
    DataCopyPad(local, gm[offset], params, pad);
    queue.EnQue(local);
    return queue.template DeQue<T>();
}

template <typename T>
__aicore__ inline LocalTensor<float> DeepNormGrad<T>::CopyInScalars(TQue<QuePosition::VECIN, 1>& queue,
                                                                    GlobalTensor<float>& gm, uint64_t offset,
                                                                    uint32_t count)
{
    LocalTensor<float> local = queue.template AllocTensor<float>();
    uint32_t aligned = (count + SCALAR_BLOCK_ELEMS - 1) / SCALAR_BLOCK_ELEMS * SCALAR_BLOCK_ELEMS;
    DataCopyExtParams params{1, count * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
    DataCopyPadExtParams<float> pad{true, 0, static_cast<uint8_t>(aligned - count), 0.0f};
    DataCopyPad(local, gm[offset], params, pad);
    queue.EnQue(local);
    return queue.template DeQue<float>();
}

template <typename T>
__aicore__ inline LocalTensor<float> DeepNormGrad<T>::CopyInFloatTensor(TQue<QuePosition::VECIN, 1>& queue,
                                                                        GlobalTensor<float>& gm, uint64_t offset,
                                                                        uint32_t count)
{
    LocalTensor<float> local = queue.template AllocTensor<float>();
    uint32_t aligned = (count + SCALAR_BLOCK_ELEMS - 1) / SCALAR_BLOCK_ELEMS * SCALAR_BLOCK_ELEMS;
    DataCopyExtParams params{1, count * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
    DataCopyPadExtParams<float> pad{true, 0, static_cast<uint8_t>(aligned - count), 0.0f};
    DataCopyPad(local, gm[offset], params, pad);
    queue.EnQue(local);
    return queue.template DeQue<float>();
}

template <typename T>
__aicore__ inline void DeepNormGrad<T>::CopyOutTensor(TQue<QuePosition::VECOUT, 1>& queue, GlobalTensor<T>& gm,
                                                      uint64_t offset, uint32_t count)
{
    LocalTensor<T> local = queue.template DeQue<T>();
    DataCopyExtParams params{1, count * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
    DataCopyPad(gm[offset], local, params);
    queue.FreeTensor(local);
}

template <typename T>
__aicore__ inline void DeepNormGrad<T>::CopyOutTensorBatch(TQue<QuePosition::VECOUT, 1>& queue, GlobalTensor<T>& gm,
                                                           uint64_t offset, uint32_t rows)
{
    LocalTensor<T> local = queue.template DeQue<T>();
    DataCopyExtParams params{static_cast<uint16_t>(rows),
                             static_cast<uint32_t>(tiling_->numCols * static_cast<uint64_t>(sizeof(T))), 0, 0, 0};
    DataCopyPad(gm[offset], local, params);
    queue.FreeTensor(local);
}

template <typename T>
__aicore__ inline void DeepNormGrad<T>::CopyOutFloat(TQue<QuePosition::VECOUT, 1>& queue, GlobalTensor<float>& gm,
                                                     uint64_t offset, uint32_t count)
{
    LocalTensor<float> local = queue.template DeQue<float>();
    DataCopyExtParams params{1, count * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
    DataCopyPad(gm[offset], local, params);
    queue.FreeTensor(local);
}

template <typename T>
__aicore__ inline void DeepNormGrad<T>::StoreOutputForDtype(__local_mem__ T* dst, RegTensor<float>& src, MaskReg& mask,
                                                            uint32_t offset)
{
    if constexpr (IsSameType<T, float>::value) {
        DataCopy<T, StoreDist::DIST_NORM>(dst + offset, src, mask);
    } else if constexpr (IsSameType<T, half>::value) {
        RegTensor<half> dstReg;
        Cast<half, float, CAST_TRAIT_B32_TO_F16>(dstReg, src, mask);
        DataCopy<half, StoreDist::DIST_PACK_B32>(dst + offset, dstReg, mask);
    } else {
        RegTensor<bfloat16_t> dstReg;
        Cast<bfloat16_t, float, CAST_TRAIT_B32_TO_BF16>(dstReg, src, mask);
        DataCopy<bfloat16_t, StoreDist::DIST_PACK_B32>(dst + offset, dstReg, mask);
    }
}

template <typename T>
__aicore__ inline void DeepNormGrad<T>::InitScalar(LocalTensor<float>& scalar)
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
__aicore__ inline void DeepNormGrad<T>::AccumulateScalar(LocalTensor<float>& dst, LocalTensor<float>& src)
{
    __local_mem__ float* dstPtr = (__local_mem__ float*)dst.GetPhyAddr();
    __local_mem__ float* srcPtr = (__local_mem__ float*)src.GetPhyAddr();
    __VEC_SCOPE__
    {
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        RegTensor<float> dstReg;
        RegTensor<float> srcReg;
        RegTensor<float> outReg;
        MaskReg mask = CreateMask<float, MaskPattern::VL1>();
        DataCopy<float, LoadDist::DIST_BRC_B32>(dstReg, dstPtr);
        DataCopy<float, LoadDist::DIST_BRC_B32>(srcReg, srcPtr);
        Add(outReg, dstReg, srcReg, mask);
        DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dstPtr, outReg, mask);
    }
}

template <typename T>
__aicore__ inline void DeepNormGrad<T>::ScaleScalar(LocalTensor<float>& scalar, float scale)
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
__aicore__ inline uint32_t DeepNormGrad<T>::GetPowerSplit(uint32_t count) const
{
    uint32_t power = VL_FP32;
    while (power <= count / 2) {
        power *= 2;
    }
    return power;
}

template <typename T>
__aicore__ inline void DeepNormGrad<T>::ComputeFirstPass(LocalTensor<T>& dy, LocalTensor<T>& x, LocalTensor<T>& gx,
                                                         LocalTensor<T>& gamma, LocalTensor<float>& mean,
                                                         LocalTensor<float>& rstd, uint32_t count)
{
    __local_mem__ T* dyPtr = (__local_mem__ T*)dy.GetPhyAddr();
    __local_mem__ T* xPtr = (__local_mem__ T*)x.GetPhyAddr();
    __local_mem__ T* gxPtr = (__local_mem__ T*)gx.GetPhyAddr();
    __local_mem__ T* gammaPtr = (__local_mem__ T*)gamma.GetPhyAddr();
    __local_mem__ float* meanPtr = (__local_mem__ float*)mean.GetPhyAddr();
    __local_mem__ float* rstdPtr = (__local_mem__ float*)rstd.GetPhyAddr();
    __local_mem__ float* tmpPtr = (__local_mem__ float*)calc0Buf_.Get<float>().GetPhyAddr();
    __local_mem__ float* tmpNormPtr = (__local_mem__ float*)calc1Buf_.Get<float>().GetPhyAddr();
    uint16_t loops = static_cast<uint16_t>((count + VL_FP32 - 1) / VL_FP32);
    uint32_t remaining = count;
    float alpha = alpha_;
    __VEC_SCOPE__
    {
        RegTensor<float> dyReg;
        RegTensor<float> xReg;
        RegTensor<float> gxReg;
        RegTensor<float> gammaReg;
        RegTensor<float> meanReg;
        RegTensor<float> rstdReg;
        RegTensor<float> tmpReg;
        RegTensor<float> normReg;
        RegTensor<float> productReg;
        MaskReg mask;
        DataCopy<float, LoadDist::DIST_BRC_B32>(meanReg, meanPtr);
        DataCopy<float, LoadDist::DIST_BRC_B32>(rstdReg, rstdPtr);
        for (uint16_t i = 0; i < loops; ++i) {
            uint32_t valid = Min(remaining, VL_FP32);
            remaining -= valid;
            mask = UpdateMask<float>(valid);
            uint32_t offset = static_cast<uint32_t>(i) * VL_FP32;
            NormCommon::NormCommonRegbase::LoadRegForDtype<T>(dyPtr, dyReg, mask, offset);
            NormCommon::NormCommonRegbase::LoadRegForDtype<T>(gammaPtr, gammaReg, mask, offset);
            NormCommon::NormCommonRegbase::LoadRegForDtype<T>(xPtr, xReg, mask, offset);
            NormCommon::NormCommonRegbase::LoadRegForDtype<T>(gxPtr, gxReg, mask, offset);
            Mul(tmpReg, dyReg, gammaReg, mask);
            Muls(xReg, xReg, alpha, mask);
            Add(normReg, xReg, gxReg, mask);
            Sub(normReg, normReg, meanReg, mask);
            Mul(productReg, rstdReg, rstdReg, mask);
            Mul(productReg, productReg, rstdReg, mask);
            Mul(productReg, productReg, normReg, mask);
            Mul(productReg, productReg, tmpReg, mask);
            Mul(normReg, tmpReg, rstdReg, mask);
            DataCopy<float, StoreDist::DIST_NORM_B32>(tmpPtr + offset, productReg, mask);
            DataCopy<float, StoreDist::DIST_NORM_B32>(tmpNormPtr + offset, normReg, mask);
        }
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    }
}

template <typename T>
__aicore__ inline void DeepNormGrad<T>::ComputeSecondPass(LocalTensor<T>& dy, LocalTensor<T>& x, LocalTensor<T>& gx,
                                                          LocalTensor<T>& gamma, LocalTensor<float>& mean,
                                                          LocalTensor<float>& rstd, LocalTensor<float>& avgTmp,
                                                          LocalTensor<float>& avgTmpNorm, uint32_t count)
{
    LocalTensor<T> dx = dxQueue_.template AllocTensor<T>();
    LocalTensor<T> dgx = dgxQueue_.template AllocTensor<T>();
    __local_mem__ T* dyPtr = (__local_mem__ T*)dy.GetPhyAddr();
    __local_mem__ T* xPtr = (__local_mem__ T*)x.GetPhyAddr();
    __local_mem__ T* gxPtr = (__local_mem__ T*)gx.GetPhyAddr();
    __local_mem__ T* gammaPtr = (__local_mem__ T*)gamma.GetPhyAddr();
    __local_mem__ T* dxPtr = (__local_mem__ T*)dx.GetPhyAddr();
    __local_mem__ T* dgxPtr = (__local_mem__ T*)dgx.GetPhyAddr();
    __local_mem__ float* meanPtr = (__local_mem__ float*)mean.GetPhyAddr();
    __local_mem__ float* rstdPtr = (__local_mem__ float*)rstd.GetPhyAddr();
    __local_mem__ float* avgTmpPtr = (__local_mem__ float*)avgTmp.GetPhyAddr();
    __local_mem__ float* avgTmpNormPtr = (__local_mem__ float*)avgTmpNorm.GetPhyAddr();
    uint16_t loops = static_cast<uint16_t>((count + VL_FP32 - 1) / VL_FP32);
    uint32_t remaining = count;
    float alpha = alpha_;
    __VEC_SCOPE__
    {
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        RegTensor<float> dyReg;
        RegTensor<float> xReg;
        RegTensor<float> gxReg;
        RegTensor<float> gammaReg;
        RegTensor<float> meanReg;
        RegTensor<float> rstdReg;
        RegTensor<float> avgTmpReg;
        RegTensor<float> avgTmpNormReg;
        RegTensor<float> tmpReg;
        RegTensor<float> normReg;
        RegTensor<float> dgxReg;
        RegTensor<float> dxReg;
        MaskReg mask;
        DataCopy<float, LoadDist::DIST_BRC_B32>(meanReg, meanPtr);
        DataCopy<float, LoadDist::DIST_BRC_B32>(rstdReg, rstdPtr);
        DataCopy<float, LoadDist::DIST_BRC_B32>(avgTmpReg, avgTmpPtr);
        DataCopy<float, LoadDist::DIST_BRC_B32>(avgTmpNormReg, avgTmpNormPtr);
        for (uint16_t i = 0; i < loops; ++i) {
            uint32_t valid = Min(remaining, VL_FP32);
            remaining -= valid;
            mask = UpdateMask<float>(valid);
            uint32_t offset = static_cast<uint32_t>(i) * VL_FP32;
            NormCommon::NormCommonRegbase::LoadRegForDtype<T>(dyPtr, dyReg, mask, offset);
            NormCommon::NormCommonRegbase::LoadRegForDtype<T>(gammaPtr, gammaReg, mask, offset);
            NormCommon::NormCommonRegbase::LoadRegForDtype<T>(xPtr, xReg, mask, offset);
            NormCommon::NormCommonRegbase::LoadRegForDtype<T>(gxPtr, gxReg, mask, offset);
            Mul(tmpReg, dyReg, gammaReg, mask);
            Muls(xReg, xReg, alpha, mask);
            Add(normReg, xReg, gxReg, mask);
            Sub(normReg, normReg, meanReg, mask);
            Mul(dgxReg, tmpReg, rstdReg, mask);
            Mul(normReg, normReg, avgTmpReg, mask);
            Add(dgxReg, dgxReg, normReg, mask);
            Add(dgxReg, dgxReg, avgTmpNormReg, mask);
            Muls(dxReg, dgxReg, alpha, mask);
            StoreOutputForDtype(dgxPtr, dgxReg, mask, offset);
            StoreOutputForDtype(dxPtr, dxReg, mask, offset);
        }
    }
    dxQueue_.EnQue(dx);
    dgxQueue_.EnQue(dgx);
}

template <typename T>
__aicore__ inline void DeepNormGrad<T>::ComputeGammaBeta(LocalTensor<T>& dy, LocalTensor<T>& x, LocalTensor<T>& gx,
                                                         LocalTensor<float>& mean, LocalTensor<float>& rstd,
                                                         LocalTensor<float>& dgamma, LocalTensor<float>& dbeta,
                                                         uint32_t count)
{
    __local_mem__ T* dyPtr = (__local_mem__ T*)dy.GetPhyAddr();
    __local_mem__ T* xPtr = (__local_mem__ T*)x.GetPhyAddr();
    __local_mem__ T* gxPtr = (__local_mem__ T*)gx.GetPhyAddr();
    __local_mem__ float* meanPtr = (__local_mem__ float*)mean.GetPhyAddr();
    __local_mem__ float* rstdPtr = (__local_mem__ float*)rstd.GetPhyAddr();
    __local_mem__ float* dgammaPtr = (__local_mem__ float*)dgamma.GetPhyAddr();
    __local_mem__ float* dbetaPtr = (__local_mem__ float*)dbeta.GetPhyAddr();
    uint16_t loops = static_cast<uint16_t>((count + VL_FP32 - 1) / VL_FP32);
    uint32_t remaining = count;
    float alpha = alpha_;
    __VEC_SCOPE__
    {
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        RegTensor<float> dyReg;
        RegTensor<float> xReg;
        RegTensor<float> gxReg;
        RegTensor<float> meanReg;
        RegTensor<float> rstdReg;
        RegTensor<float> normReg;
        RegTensor<float> dgammaReg;
        RegTensor<float> dbetaReg;
        RegTensor<float> productReg;
        MaskReg mask;
        DataCopy<float, LoadDist::DIST_BRC_B32>(meanReg, meanPtr);
        DataCopy<float, LoadDist::DIST_BRC_B32>(rstdReg, rstdPtr);
        for (uint16_t i = 0; i < loops; ++i) {
            uint32_t valid = Min(remaining, VL_FP32);
            remaining -= valid;
            mask = UpdateMask<float>(valid);
            uint32_t offset = static_cast<uint32_t>(i) * VL_FP32;
            NormCommon::NormCommonRegbase::LoadRegForDtype<T>(dyPtr, dyReg, mask, offset);
            NormCommon::NormCommonRegbase::LoadRegForDtype<T>(xPtr, xReg, mask, offset);
            NormCommon::NormCommonRegbase::LoadRegForDtype<T>(gxPtr, gxReg, mask, offset);
            DataCopy(dgammaReg, dgammaPtr + offset);
            DataCopy(dbetaReg, dbetaPtr + offset);
            Muls(xReg, xReg, alpha, mask);
            Add(normReg, xReg, gxReg, mask);
            Sub(normReg, normReg, meanReg, mask);
            Mul(normReg, normReg, rstdReg, mask);
            Mul(productReg, dyReg, normReg, mask);
            Add(dgammaReg, dgammaReg, productReg, mask);
            Add(dbetaReg, dbetaReg, dyReg, mask);
            DataCopy<float, StoreDist::DIST_NORM_B32>(dgammaPtr + offset, dgammaReg, mask);
            DataCopy<float, StoreDist::DIST_NORM_B32>(dbetaPtr + offset, dbetaReg, mask);
        }
    }
}

template <typename T>
__aicore__ inline void DeepNormGrad<T>::ComputeSmallDSecondPass(
    LocalTensor<T>& dy, LocalTensor<T>& x, LocalTensor<T>& gx, LocalTensor<T>& gamma, LocalTensor<float>& mean,
    LocalTensor<float>& rstd, LocalTensor<float>& avgTmp, LocalTensor<float>& avgTmpNorm, LocalTensor<T>& dx,
    LocalTensor<T>& dgx, LocalTensor<float>& dgamma, LocalTensor<float>& dbeta, LocalTensor<float>& dgammaComp,
    LocalTensor<float>& dbetaComp, uint32_t count)
{
    __local_mem__ T* dyPtr = (__local_mem__ T*)dy.GetPhyAddr();
    __local_mem__ T* xPtr = (__local_mem__ T*)x.GetPhyAddr();
    __local_mem__ T* gxPtr = (__local_mem__ T*)gx.GetPhyAddr();
    __local_mem__ T* gammaPtr = (__local_mem__ T*)gamma.GetPhyAddr();
    __local_mem__ T* dxPtr = (__local_mem__ T*)dx.GetPhyAddr();
    __local_mem__ T* dgxPtr = (__local_mem__ T*)dgx.GetPhyAddr();
    __local_mem__ float* meanPtr = (__local_mem__ float*)mean.GetPhyAddr();
    __local_mem__ float* rstdPtr = (__local_mem__ float*)rstd.GetPhyAddr();
    __local_mem__ float* avgTmpPtr = (__local_mem__ float*)avgTmp.GetPhyAddr();
    __local_mem__ float* avgTmpNormPtr = (__local_mem__ float*)avgTmpNorm.GetPhyAddr();
    __local_mem__ float* dgammaPtr = (__local_mem__ float*)dgamma.GetPhyAddr();
    __local_mem__ float* dbetaPtr = (__local_mem__ float*)dbeta.GetPhyAddr();
    __local_mem__ float* dgammaCompPtr = (__local_mem__ float*)dgammaComp.GetPhyAddr();
    __local_mem__ float* dbetaCompPtr = (__local_mem__ float*)dbetaComp.GetPhyAddr();
    uint16_t loops = static_cast<uint16_t>((count + VL_FP32 - 1) / VL_FP32);
    uint32_t remaining = count;
    float alpha = alpha_;
    __VEC_SCOPE__
    {
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        RegTensor<float> dyReg;
        RegTensor<float> xReg;
        RegTensor<float> gxReg;
        RegTensor<float> gammaReg;
        RegTensor<float> meanReg;
        RegTensor<float> rstdReg;
        RegTensor<float> avgTmpReg;
        RegTensor<float> avgTmpNormReg;
        RegTensor<float> normReg;
        RegTensor<float> dyGammaReg;
        RegTensor<float> dgxReg;
        RegTensor<float> dxReg;
        RegTensor<float> valueReg;
        RegTensor<float> sumReg;
        RegTensor<float> compReg;
        RegTensor<float> adjustedReg;
        RegTensor<float> newSumReg;
        RegTensor<float> deltaReg;
        MaskReg mask;
        DataCopy<float, LoadDist::DIST_BRC_B32>(meanReg, meanPtr);
        DataCopy<float, LoadDist::DIST_BRC_B32>(rstdReg, rstdPtr);
        DataCopy<float, LoadDist::DIST_BRC_B32>(avgTmpReg, avgTmpPtr);
        DataCopy<float, LoadDist::DIST_BRC_B32>(avgTmpNormReg, avgTmpNormPtr);
        for (uint16_t i = 0; i < loops; ++i) {
            uint32_t valid = Min(remaining, VL_FP32);
            remaining -= valid;
            mask = UpdateMask<float>(valid);
            uint32_t offset = static_cast<uint32_t>(i) * VL_FP32;
            NormCommon::NormCommonRegbase::LoadRegForDtype<T>(dyPtr, dyReg, mask, offset);
            NormCommon::NormCommonRegbase::LoadRegForDtype<T>(xPtr, xReg, mask, offset);
            NormCommon::NormCommonRegbase::LoadRegForDtype<T>(gxPtr, gxReg, mask, offset);
            NormCommon::NormCommonRegbase::LoadRegForDtype<T>(gammaPtr, gammaReg, mask, offset);
            DataCopy(sumReg, dgammaPtr + offset);
            DataCopy(compReg, dgammaCompPtr + offset);

            Muls(xReg, xReg, alpha, mask);
            Add(normReg, xReg, gxReg, mask);
            Sub(normReg, normReg, meanReg, mask);
            Mul(valueReg, dyReg, normReg, mask);
            Mul(valueReg, valueReg, rstdReg, mask);
            Sub(adjustedReg, valueReg, compReg, mask);
            Add(newSumReg, sumReg, adjustedReg, mask);
            Sub(deltaReg, newSumReg, sumReg, mask);
            Sub(compReg, deltaReg, adjustedReg, mask);
            DataCopy<float, StoreDist::DIST_NORM_B32>(dgammaPtr + offset, newSumReg, mask);
            DataCopy<float, StoreDist::DIST_NORM_B32>(dgammaCompPtr + offset, compReg, mask);

            DataCopy(sumReg, dbetaPtr + offset);
            DataCopy(compReg, dbetaCompPtr + offset);
            Sub(adjustedReg, dyReg, compReg, mask);
            Add(newSumReg, sumReg, adjustedReg, mask);
            Sub(deltaReg, newSumReg, sumReg, mask);
            Sub(compReg, deltaReg, adjustedReg, mask);
            DataCopy<float, StoreDist::DIST_NORM_B32>(dbetaPtr + offset, newSumReg, mask);
            DataCopy<float, StoreDist::DIST_NORM_B32>(dbetaCompPtr + offset, compReg, mask);

            Mul(dyGammaReg, dyReg, gammaReg, mask);
            Mul(dgxReg, dyGammaReg, rstdReg, mask);
            Mul(normReg, normReg, avgTmpReg, mask);
            Add(dgxReg, dgxReg, normReg, mask);
            Add(dgxReg, dgxReg, avgTmpNormReg, mask);
            Muls(dxReg, dgxReg, alpha, mask);
            StoreOutputForDtype(dgxPtr, dgxReg, mask, offset);
            StoreOutputForDtype(dxPtr, dxReg, mask, offset);
        }
    }
}

template <typename T>
template <uint32_t COLS>
__aicore__ inline void DeepNormGrad<T>::ComputeTinyDBatch(LocalTensor<T>& dy, LocalTensor<T>& x, LocalTensor<T>& gx,
                                                          LocalTensor<T>& gamma, LocalTensor<float>& mean,
                                                          LocalTensor<float>& rstd, LocalTensor<T>& dx,
                                                          LocalTensor<T>& dgx, LocalTensor<float>& dgamma,
                                                          LocalTensor<float>& dbeta, LocalTensor<float>& dgammaComp,
                                                          LocalTensor<float>& dbetaComp, uint32_t rows)
{
    static_assert(COLS == 1 || COLS == 2, "tiny-D batch only supports one or two columns");
    __local_mem__ T* dyPtr = (__local_mem__ T*)dy.GetPhyAddr();
    __local_mem__ T* xPtr = (__local_mem__ T*)x.GetPhyAddr();
    __local_mem__ T* gxPtr = (__local_mem__ T*)gx.GetPhyAddr();
    __local_mem__ T* gammaPtr = (__local_mem__ T*)gamma.GetPhyAddr();
    __local_mem__ T* dxPtr = (__local_mem__ T*)dx.GetPhyAddr();
    __local_mem__ T* dgxPtr = (__local_mem__ T*)dgx.GetPhyAddr();
    __local_mem__ float* meanPtr = (__local_mem__ float*)mean.GetPhyAddr();
    __local_mem__ float* rstdPtr = (__local_mem__ float*)rstd.GetPhyAddr();
    __local_mem__ float* dgammaPtr = (__local_mem__ float*)dgamma.GetPhyAddr();
    __local_mem__ float* dbetaPtr = (__local_mem__ float*)dbeta.GetPhyAddr();
    __local_mem__ float* dgammaCompPtr = (__local_mem__ float*)dgammaComp.GetPhyAddr();
    __local_mem__ float* dbetaCompPtr = (__local_mem__ float*)dbetaComp.GetPhyAddr();
    uint32_t rowStride = tiling_->smallRowStride;
    float alpha = alpha_;
    float negInvCols = -invCols_;
    __VEC_SCOPE__
    {
        RegTensor<float> dyReg;
        RegTensor<float> xReg;
        RegTensor<float> gxReg;
        RegTensor<float> gammaReg;
        RegTensor<float> meanReg;
        RegTensor<float> rstdReg;
        RegTensor<float> normReg;
        RegTensor<float> dyGammaReg;
        RegTensor<float> tmpNormReg;
        RegTensor<float> productReg;
        RegTensor<float> avgTmpReg;
        RegTensor<float> avgTmpDupReg;
        RegTensor<float> avgTmpNormReg;
        RegTensor<float> avgTmpNormDupReg;
        RegTensor<float> dgxReg;
        RegTensor<float> dxReg;
        RegTensor<float> dgammaSumReg;
        RegTensor<float> dbetaSumReg;
        RegTensor<float> dgammaCompReg;
        RegTensor<float> dbetaCompReg;
        RegTensor<float> adjustedReg;
        RegTensor<float> newSumReg;
        RegTensor<float> deltaReg;
        RegTensor<float> valueReg;
        uint32_t validCols = COLS;
        MaskReg mask = UpdateMask<float>(validCols);
        MaskReg scalarMask = CreateMask<float, MaskPattern::VL1>();

        NormCommon::NormCommonRegbase::LoadRegForDtype<T>(gammaPtr, gammaReg, mask, 0);
        DataCopy(dgammaSumReg, dgammaPtr);
        DataCopy(dbetaSumReg, dbetaPtr);
        DataCopy(dgammaCompReg, dgammaCompPtr);
        DataCopy(dbetaCompReg, dbetaCompPtr);
        for (uint16_t localRow = 0; localRow < rows; ++localRow) {
            uint32_t offset = static_cast<uint32_t>(localRow) * rowStride;
            NormCommon::NormCommonRegbase::LoadRegForDtype<T>(dyPtr, dyReg, mask, offset);
            NormCommon::NormCommonRegbase::LoadRegForDtype<T>(xPtr, xReg, mask, offset);
            NormCommon::NormCommonRegbase::LoadRegForDtype<T>(gxPtr, gxReg, mask, offset);
            DataCopy<float, LoadDist::DIST_BRC_B32>(meanReg, meanPtr + localRow);
            DataCopy<float, LoadDist::DIST_BRC_B32>(rstdReg, rstdPtr + localRow);

            Muls(xReg, xReg, alpha, mask);
            Add(normReg, xReg, gxReg, mask);
            Sub(normReg, normReg, meanReg, mask);

            Mul(valueReg, dyReg, normReg, mask);
            Mul(valueReg, valueReg, rstdReg, mask);
            Sub(adjustedReg, valueReg, dgammaCompReg, mask);
            Add(newSumReg, dgammaSumReg, adjustedReg, mask);
            Sub(deltaReg, newSumReg, dgammaSumReg, mask);
            Sub(dgammaCompReg, deltaReg, adjustedReg, mask);
            AscendC::MicroAPI::Copy<float, AscendC::MicroAPI::MaskMergeMode::MERGING>(dgammaSumReg, newSumReg, mask);

            Sub(adjustedReg, dyReg, dbetaCompReg, mask);
            Add(newSumReg, dbetaSumReg, adjustedReg, mask);
            Sub(deltaReg, newSumReg, dbetaSumReg, mask);
            Sub(dbetaCompReg, deltaReg, adjustedReg, mask);
            AscendC::MicroAPI::Copy<float, AscendC::MicroAPI::MaskMergeMode::MERGING>(dbetaSumReg, newSumReg, mask);

            Mul(dyGammaReg, dyReg, gammaReg, mask);
            Mul(tmpNormReg, dyGammaReg, rstdReg, mask);
            Mul(productReg, rstdReg, rstdReg, mask);
            Mul(productReg, productReg, rstdReg, mask);
            Mul(productReg, productReg, normReg, mask);
            Mul(productReg, productReg, dyGammaReg, mask);
            if constexpr (COLS == 1) {
                Muls(avgTmpReg, productReg, negInvCols, mask);
                Muls(avgTmpNormReg, tmpNormReg, negInvCols, mask);
                Mul(dgxReg, normReg, avgTmpReg, mask);
                Add(dgxReg, tmpNormReg, dgxReg, mask);
                Add(dgxReg, dgxReg, avgTmpNormReg, mask);
            } else {
                ReduceSum(avgTmpReg, productReg, mask);
                Muls(avgTmpReg, avgTmpReg, negInvCols, scalarMask);
                Duplicate(avgTmpDupReg, avgTmpReg, mask);
                ReduceSum(avgTmpNormReg, tmpNormReg, mask);
                Muls(avgTmpNormReg, avgTmpNormReg, negInvCols, scalarMask);
                Duplicate(avgTmpNormDupReg, avgTmpNormReg, mask);
                Mul(dgxReg, normReg, avgTmpDupReg, mask);
                Add(dgxReg, tmpNormReg, dgxReg, mask);
                Add(dgxReg, dgxReg, avgTmpNormDupReg, mask);
            }
            Muls(dxReg, dgxReg, alpha, mask);
            StoreOutputForDtype(dgxPtr, dgxReg, mask, offset);
            StoreOutputForDtype(dxPtr, dxReg, mask, offset);
        }
        DataCopy<float, StoreDist::DIST_NORM_B32>(dgammaPtr, dgammaSumReg, mask);
        DataCopy<float, StoreDist::DIST_NORM_B32>(dbetaPtr, dbetaSumReg, mask);
        DataCopy<float, StoreDist::DIST_NORM_B32>(dgammaCompPtr, dgammaCompReg, mask);
        DataCopy<float, StoreDist::DIST_NORM_B32>(dbetaCompPtr, dbetaCompReg, mask);
    }
}

template <typename T>
__aicore__ inline void DeepNormGrad<T>::AccumulateKahan(LocalTensor<float>& sum, LocalTensor<float>& compensation,
                                                        LocalTensor<float>& value, uint32_t count)
{
    __local_mem__ float* sumPtr = (__local_mem__ float*)sum.GetPhyAddr();
    __local_mem__ float* compensationPtr = (__local_mem__ float*)compensation.GetPhyAddr();
    __local_mem__ float* valuePtr = (__local_mem__ float*)value.GetPhyAddr();
    uint16_t loops = static_cast<uint16_t>((count + VL_FP32 - 1) / VL_FP32);
    uint32_t remaining = count;
    __VEC_SCOPE__
    {
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        RegTensor<float> sumReg;
        RegTensor<float> compensationReg;
        RegTensor<float> valueReg;
        RegTensor<float> adjustedReg;
        RegTensor<float> newSumReg;
        RegTensor<float> deltaReg;
        MaskReg mask;
        for (uint16_t i = 0; i < loops; ++i) {
            uint32_t valid = Min(remaining, VL_FP32);
            remaining -= valid;
            mask = UpdateMask<float>(valid);
            uint32_t offset = static_cast<uint32_t>(i) * VL_FP32;
            DataCopy(sumReg, sumPtr + offset);
            DataCopy(compensationReg, compensationPtr + offset);
            DataCopy(valueReg, valuePtr + offset);
            Sub(adjustedReg, valueReg, compensationReg, mask);
            Add(newSumReg, sumReg, adjustedReg, mask);
            Sub(deltaReg, newSumReg, sumReg, mask);
            Sub(compensationReg, deltaReg, adjustedReg, mask);
            DataCopy<float, StoreDist::DIST_NORM_B32>(sumPtr + offset, newSumReg, mask);
            DataCopy<float, StoreDist::DIST_NORM_B32>(compensationPtr + offset, compensationReg, mask);
        }
    }
}

template <typename T>
__aicore__ inline void DeepNormGrad<T>::ProcessBackwardRow(uint64_t row)
{
    LocalTensor<float> mean = CopyInScalar(meanQueue_, meanGm_, row);
    LocalTensor<float> rstd = CopyInScalar(rstdQueue_, rstdGm_, row);
    LocalTensor<float> sumTmp = sumTmpBuf_.Get<float>();
    LocalTensor<float> sumTmpNorm = sumTmpNormBuf_.Get<float>();
    LocalTensor<float> tileSums = tileSumBuf_.Get<float>();
    LocalTensor<float> tileTmp = tileSums;
    LocalTensor<float> tileTmpNorm = tileSums[SCALAR_BLOCK_ELEMS];
    LocalTensor<float> calc0 = calc0Buf_.Get<float>();
    LocalTensor<float> calc1 = calc1Buf_.Get<float>();
    InitScalar(sumTmp);
    InitScalar(sumTmpNorm);

    uint64_t rowOffset = row * tiling_->numCols;
    for (uint64_t col = 0; col < tiling_->numCols; col += tileLength_) {
        uint32_t count = static_cast<uint32_t>(Min(static_cast<uint64_t>(tileLength_), tiling_->numCols - col));
        LocalTensor<T> dy = CopyInTensor(dyQueue_, dyGm_, rowOffset + col, count);
        LocalTensor<T> x = CopyInTensor(xQueue_, xGm_, rowOffset + col, count);
        LocalTensor<T> gx = CopyInTensor(gxQueue_, gxGm_, rowOffset + col, count);
        LocalTensor<T> gamma = CopyInTensor(gammaQueue_, gammaGm_, col, count);
        ComputeFirstPass(dy, x, gx, gamma, mean, rstd, count);
        NormCommon::NormCommonRegbase::CalculateReduceSum(calc0, tileTmp, reduceTmpBuf_, count, GetPowerSplit(count));
        NormCommon::NormCommonRegbase::CalculateReduceSum(calc1, tileTmpNorm, reduceTmpBuf_, count,
                                                          GetPowerSplit(count));
        AccumulateScalar(sumTmp, tileTmp);
        AccumulateScalar(sumTmpNorm, tileTmpNorm);
        dyQueue_.FreeTensor(dy);
        xQueue_.FreeTensor(x);
        gxQueue_.FreeTensor(gx);
        gammaQueue_.FreeTensor(gamma);
    }
    ScaleScalar(sumTmp, -invCols_);
    ScaleScalar(sumTmpNorm, -invCols_);

    for (uint64_t col = 0; col < tiling_->numCols; col += tileLength_) {
        uint32_t count = static_cast<uint32_t>(Min(static_cast<uint64_t>(tileLength_), tiling_->numCols - col));
        LocalTensor<T> dy = CopyInTensor(dyQueue_, dyGm_, rowOffset + col, count);
        LocalTensor<T> x = CopyInTensor(xQueue_, xGm_, rowOffset + col, count);
        LocalTensor<T> gx = CopyInTensor(gxQueue_, gxGm_, rowOffset + col, count);
        LocalTensor<T> gamma = CopyInTensor(gammaQueue_, gammaGm_, col, count);
        ComputeSecondPass(dy, x, gx, gamma, mean, rstd, sumTmp, sumTmpNorm, count);
        dyQueue_.FreeTensor(dy);
        xQueue_.FreeTensor(x);
        gxQueue_.FreeTensor(gx);
        gammaQueue_.FreeTensor(gamma);
        CopyOutTensor(dxQueue_, dxGm_, rowOffset + col, count);
        CopyOutTensor(dgxQueue_, dgxGm_, rowOffset + col, count);
    }
    meanQueue_.FreeTensor(mean);
    rstdQueue_.FreeTensor(rstd);
}

template <typename T>
__aicore__ inline void DeepNormGrad<T>::ProcessBackward()
{
    uint32_t core = GetBlockIdx();
    if (core >= tiling_->backwardBlockDim) {
        return;
    }
    uint64_t rowBegin = static_cast<uint64_t>(core) * tiling_->rowsPerCore;
    if (rowBegin >= tiling_->numRows) {
        return;
    }
    uint64_t rowEnd = rowBegin + Min(tiling_->rowsPerCore, tiling_->numRows - rowBegin);
    for (uint64_t row = rowBegin; row < rowEnd; ++row) {
        ProcessBackwardRow(row);
    }
}

template <typename T>
__aicore__ inline void DeepNormGrad<T>::ProcessGammaBeta()
{
    uint32_t core = GetBlockIdx();
    if (core >= tiling_->gammaBetaBlockDim) {
        return;
    }
    uint64_t colBegin = static_cast<uint64_t>(core) * tiling_->colsPerCore;
    if (colBegin >= tiling_->numCols) {
        return;
    }
    uint64_t colEnd = colBegin + Min(tiling_->colsPerCore, tiling_->numCols - colBegin);
    for (uint64_t col = colBegin; col < colEnd; col += tileLength_) {
        uint32_t count = static_cast<uint32_t>(Min(static_cast<uint64_t>(tileLength_), colEnd - col));
        LocalTensor<float> dbeta = dbetaQueue_.template AllocTensor<float>();
        LocalTensor<float> dgamma = dgammaQueue_.template AllocTensor<float>();
        Duplicate(dbeta, 0.0f, tileLengthAlign_);
        Duplicate(dgamma, 0.0f, tileLengthAlign_);
        for (uint64_t row = 0; row < tiling_->numRows; ++row) {
            uint64_t offset = row * tiling_->numCols + col;
            LocalTensor<float> mean = CopyInScalar(meanQueue_, meanGm_, row);
            LocalTensor<float> rstd = CopyInScalar(rstdQueue_, rstdGm_, row);
            LocalTensor<T> dy = CopyInTensor(dyQueue_, dyGm_, offset, count);
            LocalTensor<T> x = CopyInTensor(xQueue_, xGm_, offset, count);
            LocalTensor<T> gx = CopyInTensor(gxQueue_, gxGm_, offset, count);
            ComputeGammaBeta(dy, x, gx, mean, rstd, dgamma, dbeta, count);
            meanQueue_.FreeTensor(mean);
            rstdQueue_.FreeTensor(rstd);
            dyQueue_.FreeTensor(dy);
            xQueue_.FreeTensor(x);
            gxQueue_.FreeTensor(gx);
        }
        dbetaQueue_.EnQue(dbeta);
        dgammaQueue_.EnQue(dgamma);
        CopyOutFloat(dbetaQueue_, dbetaGm_, col, count);
        CopyOutFloat(dgammaQueue_, dgammaGm_, col, count);
    }
}

template <typename T>
__aicore__ inline void DeepNormGrad<T>::ProcessSmallD()
{
    uint32_t core = GetBlockIdx();
    uint64_t rowBegin = static_cast<uint64_t>(core) * tiling_->rowsPerCore;
    uint64_t rowEnd = rowBegin + Min(tiling_->rowsPerCore, tiling_->numRows - rowBegin);
    uint32_t count = static_cast<uint32_t>(tiling_->numCols);

    LocalTensor<T> gamma = CopyInTensor(gammaQueue_, gammaGm_, 0, count);
    LocalTensor<float> dgamma = dgammaQueue_.template AllocTensor<float>();
    LocalTensor<float> dbeta = dbetaQueue_.template AllocTensor<float>();
    LocalTensor<float> dgammaComp = dgammaCompBuf_.Get<float>();
    LocalTensor<float> dbetaComp = dbetaCompBuf_.Get<float>();
    Duplicate(dgamma, 0.0f, tiling_->smallColsAlign);
    Duplicate(dbeta, 0.0f, tiling_->smallColsAlign);
    Duplicate(dgammaComp, 0.0f, tiling_->smallColsAlign);
    Duplicate(dbetaComp, 0.0f, tiling_->smallColsAlign);

    for (uint64_t row = rowBegin; row < rowEnd; row += tiling_->smallRowsPerTile) {
        uint32_t rows = static_cast<uint32_t>(
            Min(static_cast<uint64_t>(tiling_->smallRowsPerTile), static_cast<uint64_t>(rowEnd - row)));
        uint64_t tensorOffset = row * tiling_->numCols;
        LocalTensor<T> dy = CopyInTensorBatch(dyQueue_, dyGm_, tensorOffset, rows);
        LocalTensor<T> x = CopyInTensorBatch(xQueue_, xGm_, tensorOffset, rows);
        LocalTensor<T> gx = CopyInTensorBatch(gxQueue_, gxGm_, tensorOffset, rows);
        LocalTensor<float> mean = CopyInScalars(meanQueue_, meanGm_, row, rows);
        LocalTensor<float> rstd = CopyInScalars(rstdQueue_, rstdGm_, row, rows);
        LocalTensor<T> dx = dxQueue_.template AllocTensor<T>();
        LocalTensor<T> dgx = dgxQueue_.template AllocTensor<T>();

        if (count == 1) {
            ComputeTinyDBatch<1>(dy, x, gx, gamma, mean, rstd, dx, dgx, dgamma, dbeta, dgammaComp, dbetaComp, rows);
        } else if (count == 2) {
            ComputeTinyDBatch<2>(dy, x, gx, gamma, mean, rstd, dx, dgx, dgamma, dbeta, dgammaComp, dbetaComp, rows);
        } else {
            for (uint32_t localRow = 0; localRow < rows; ++localRow) {
                uint32_t localOffset = localRow * tiling_->smallRowStride;
                LocalTensor<T> dyRow = dy[localOffset];
                LocalTensor<T> xRow = x[localOffset];
                LocalTensor<T> gxRow = gx[localOffset];
                LocalTensor<T> dxRow = dx[localOffset];
                LocalTensor<T> dgxRow = dgx[localOffset];
                LocalTensor<float> meanRow = mean[localRow];
                LocalTensor<float> rstdRow = rstd[localRow];
                LocalTensor<float> sumTmp = sumTmpBuf_.Get<float>();
                LocalTensor<float> sumTmpNorm = sumTmpNormBuf_.Get<float>();
                LocalTensor<float> tileSums = tileSumBuf_.Get<float>();
                LocalTensor<float> tileTmp = tileSums;
                LocalTensor<float> tileTmpNorm = tileSums[SCALAR_BLOCK_ELEMS];
                LocalTensor<float> calc0 = calc0Buf_.Get<float>();
                LocalTensor<float> calc1 = calc1Buf_.Get<float>();
                InitScalar(sumTmp);
                InitScalar(sumTmpNorm);
                ComputeFirstPass(dyRow, xRow, gxRow, gamma, meanRow, rstdRow, count);
                NormCommon::NormCommonRegbase::CalculateReduceSum(calc0, tileTmp, reduceTmpBuf_, count,
                                                                  GetPowerSplit(count));
                NormCommon::NormCommonRegbase::CalculateReduceSum(calc1, tileTmpNorm, reduceTmpBuf_, count,
                                                                  GetPowerSplit(count));
                AccumulateScalar(sumTmp, tileTmp);
                AccumulateScalar(sumTmpNorm, tileTmpNorm);
                ScaleScalar(sumTmp, -invCols_);
                ScaleScalar(sumTmpNorm, -invCols_);
                ComputeSmallDSecondPass(dyRow, xRow, gxRow, gamma, meanRow, rstdRow, sumTmp, sumTmpNorm, dxRow, dgxRow,
                                        dgamma, dbeta, dgammaComp, dbetaComp, count);
            }
        }

        dxQueue_.EnQue(dx);
        dgxQueue_.EnQue(dgx);
        dyQueue_.FreeTensor(dy);
        xQueue_.FreeTensor(x);
        gxQueue_.FreeTensor(gx);
        meanQueue_.FreeTensor(mean);
        rstdQueue_.FreeTensor(rstd);
        CopyOutTensorBatch(dxQueue_, dxGm_, tensorOffset, rows);
        CopyOutTensorBatch(dgxQueue_, dgxGm_, tensorOffset, rows);
    }
    gammaQueue_.FreeTensor(gamma);

    uint64_t workspaceBase = static_cast<uint64_t>(core) * 2 * tiling_->smallColsAlign;
    dgammaQueue_.EnQue(dgamma);
    dbetaQueue_.EnQue(dbeta);
    CopyOutFloat(dgammaQueue_, workspaceGm_, workspaceBase, count);
    CopyOutFloat(dbetaQueue_, workspaceGm_, workspaceBase + tiling_->smallColsAlign, count);
    ReduceSmallDPartials();
}

template <typename T>
__aicore__ inline void DeepNormGrad<T>::ReduceSmallDPartials()
{
    SyncAll();
    if (GetBlockIdx() != 0) {
        return;
    }

    uint32_t count = static_cast<uint32_t>(tiling_->numCols);
    LocalTensor<float> dgamma = dgammaQueue_.template AllocTensor<float>();
    LocalTensor<float> dbeta = dbetaQueue_.template AllocTensor<float>();
    LocalTensor<float> dgammaComp = calc0Buf_.Get<float>();
    LocalTensor<float> dbetaComp = calc1Buf_.Get<float>();
    Duplicate(dgamma, 0.0f, tiling_->smallColsAlign);
    Duplicate(dbeta, 0.0f, tiling_->smallColsAlign);
    Duplicate(dgammaComp, 0.0f, tiling_->smallColsAlign);
    Duplicate(dbetaComp, 0.0f, tiling_->smallColsAlign);

    for (uint32_t core = 0; core < tiling_->gammaBetaBlockDim; ++core) {
        uint64_t workspaceBase = static_cast<uint64_t>(core) * 2 * tiling_->smallColsAlign;
        LocalTensor<float> dgammaPartial = CopyInFloatTensor(dyQueue_, workspaceGm_, workspaceBase, count);
        LocalTensor<float> dbetaPartial = CopyInFloatTensor(xQueue_, workspaceGm_,
                                                            workspaceBase + tiling_->smallColsAlign, count);
        AccumulateKahan(dgamma, dgammaComp, dgammaPartial, count);
        AccumulateKahan(dbeta, dbetaComp, dbetaPartial, count);
        dyQueue_.FreeTensor(dgammaPartial);
        xQueue_.FreeTensor(dbetaPartial);
    }

    dgammaQueue_.EnQue(dgamma);
    dbetaQueue_.EnQue(dbeta);
    CopyOutFloat(dgammaQueue_, dgammaGm_, 0, count);
    CopyOutFloat(dbetaQueue_, dbetaGm_, 0, count);
}

template <typename T>
__aicore__ inline void DeepNormGrad<T>::Process()
{
    if (tiling_->gammaBetaRowSplit != 0) {
        ProcessSmallD();
    } else {
        ProcessBackward();
        ProcessGammaBeta();
    }
}

} // namespace DeepNormGradArch35

#endif // DEEP_NORM_GRAD_ARCH35_H
