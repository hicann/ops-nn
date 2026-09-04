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
using AscendC::Reg::CreateMask;
using AscendC::Reg::LoadDist;
using AscendC::Reg::LocalMemBar;
using AscendC::Reg::MaskPattern;
using AscendC::Reg::MaskReg;
using AscendC::Reg::MemType;
using AscendC::Reg::RegTensor;
using AscendC::Reg::StoreDist;
using AscendC::Reg::UpdateMask;

constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t VL_FP32 = 256 / sizeof(float);
constexpr uint32_t SCALAR_BLOCK_ELEMS = BLOCK_SIZE / sizeof(float);
constexpr uint32_t REDUCE_TMP_ELEMS = 2 * VL_FP32;
constexpr AscendC::Reg::CastTrait CAST_TRAIT_B32_TO_F16 = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::NO_SAT,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};
constexpr AscendC::Reg::CastTrait CAST_TRAIT_B32_TO_BF16 = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::NO_SAT,
    AscendC::Reg::MaskMergeMode::ZEROING,
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

#include "deep_norm_grad_part1.h"
#include "deep_norm_grad_part2.h"

} // namespace DeepNormGradArch35

#endif // DEEP_NORM_GRAD_ARCH35_H
