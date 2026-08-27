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
 * \file clipped_swiglu_grad_kernel.h
 * \brief Regbase VF kernel for ClippedSwigluGrad (Ascend 950 / arch35)
 *
 * 基于910B版反向逻辑，减少 UB 间搬运：
 * - 单个 __VEC_SCOPE__：Reg RegTensor 加载 a/b/dy → 寄存器内 Compare/Select 做 clamp mask
 *   → 计算 da/db（寄存器内）→ 写回 vecBuf(交错) / dxFloatLocal(前后切分)
 * - interleaved 散开：scope 外用 LocalTensor 级 Interleave（向量化，无标量大循环；仅 <=7 元素 32B 尾标量补齐）
 * - 16-bit：写回后 Cast float→T（与910B一致）
 * - 分核逻辑：与910B一致（CalTilingParam复用）
 */

#ifndef CLIPPED_SWIGLU_GRAD_KERNEL_H
#define CLIPPED_SWIGLU_GRAD_KERNEL_H

#include "kernel_operator.h"
#include "op_kernel/math_util.h"
#include "op_kernel/platform_util.h"
#include "kernel_tiling/kernel_tiling.h"

namespace ClippedSwigluGradArch35Op {
using namespace AscendC;

constexpr int64_t DB_BUFFER = 2;
constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t SWI_FACTOR = 2;
constexpr int64_t ZERO_CHUNK_BYTES = 65535 / BLOCK_SIZE * BLOCK_SIZE;
constexpr uint32_t VF_LEN_FP32 = Ops::Base::GetVRegSize() / sizeof(float);

static constexpr Reg::CastTrait CAST_BF16_FP16_TO_FP32 = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                          Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

template <typename T, bool isInterleaved, bool isGroup>
class ClippedSwigluGradArch35Kernel {
public:
    __aicore__ inline ClippedSwigluGradArch35Kernel(const ClippedSwigluGradTilingData* tilingData, TPipe* pipe)
        : tiling_(tilingData), pipe_(pipe)
    {}

    __aicore__ inline void Init(GM_ADDR gradY, GM_ADDR x, GM_ADDR groupIndex, GM_ADDR gradXOut);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ComputeRealBatchSize();
    __aicore__ inline void CalTilingParam();
    __aicore__ inline void ProcessMainLoop();
    __aicore__ inline void ProcessSingleLoop(int64_t xOffset, int64_t dyOffset, int64_t dxOffset);
    __aicore__ inline void CopyIn(int64_t xOffset, int64_t dyOffset);
    __aicore__ inline void CopyInHalfShortH(LocalTensor<T>& xDTypeLocal, LocalTensor<T>& dyDTypeLocal, int64_t xOffset,
                                            int64_t dyOffset);
    __aicore__ inline void CopyInHalfLongH(LocalTensor<T>& xDTypeLocal, LocalTensor<T>& dyDTypeLocal, int64_t xOffset,
                                           int64_t dyOffset);
    __aicore__ inline void CopyInInterLeaved(LocalTensor<T>& xDTypeLocal, LocalTensor<T>& dyDTypeLocal, int64_t xOffset,
                                             int64_t dyOffset);
    __aicore__ inline void ComputeVfGrad(LocalTensor<T>& xDTypeLocal, LocalTensor<T>& dyDTypeLocal,
                                         LocalTensor<T>& dxDTypeLocal, int64_t onceNum);
    __aicore__ inline void CopyOut(int64_t dxOffset);
    __aicore__ inline void InitZeroBuffer();
    __aicore__ inline void ZeroInvalidRows();
    __aicore__ inline int64_t AlignBytes(int64_t number) { return (number + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE; }
    __aicore__ inline void LoadOneTensor(__local_mem__ void* input, Reg::RegTensor<float>& dst, Reg::MaskReg& preg,
                                         uint32_t offset);

private:
    GlobalTensor<T> xGm_;
    GlobalTensor<T> gradYGm_;
    GlobalTensor<int64_t> groupIndexGm_;
    GlobalTensor<T> gradXOutGm_;

    TPipe* pipe_ = nullptr;
    const ClippedSwigluGradTilingData* tiling_ = nullptr;

    TQue<QuePosition::VECIN, DB_BUFFER> xQueue_;
    TQue<QuePosition::VECIN, DB_BUFFER> dyQueue_;
    TQue<QuePosition::VECOUT, 1> dxQueue_;
    TBuf<TPosition::VECCALC> vectorBuf_;
    TBuf<TPosition::VECCALC> tmpBuf_;
    TBuf<TPosition::VECCALC> maskBufA_;
    TBuf<TPosition::VECCALC> maskBufB_;
    TBuf<TPosition::VECCALC> groupBuf_;
    TBuf<TPosition::VECCALC> zeroBuf_;

    uint32_t blockIdx_ = 0;
    uint32_t usedCoreNum_ = 0;
    int64_t realBatchSize_ = 0;
    int64_t blockOffset_ = 0;
    int64_t loopOffset_ = 0;
    int64_t loopTime_ = 0;
    int64_t pairFrontLoop_ = 0;
    int64_t pairLastLoop_ = 0;
    int64_t pairNum_ = 0;
    int64_t batchPreBlock_ = 0;
    int64_t dimH_ = 0;
    int64_t ubMaxPair_ = 0;
    int64_t xQueSpace_ = 0;
    int64_t dyQueSpace_ = 0;
    int64_t half_ = 0;
    int64_t calPairFrontLoop_ = 0;
    int64_t calPairLastLoop_ = 0;
    int64_t calPairNum_ = 0;
    int64_t xLocalOffset1_ = 0;
    int64_t xLocalOffset2_ = 0;
    int64_t dyLocalOffset_ = 0;
    int64_t dxDbOffset_ = 0;

    float limit_ = 0.0f;
    float alpha_ = 0.0f;
    float bias_ = 0.0f;
};

template <typename T, bool isInterleaved, bool isGroup>
__aicore__ inline void ClippedSwigluGradArch35Kernel<T, isInterleaved, isGroup>::Init(GM_ADDR gradY, GM_ADDR x,
                                                                                      GM_ADDR groupIndex,
                                                                                      GM_ADDR gradXOut)
{
    blockIdx_ = GetBlockIdx();
    ubMaxPair_ = tiling_->ubMaxPair;
    dimH_ = tiling_->dim2H / SWI_FACTOR;
    xQueSpace_ = SWI_FACTOR * AlignBytes(ubMaxPair_ * static_cast<int64_t>(sizeof(float)));
    dyQueSpace_ = AlignBytes(ubMaxPair_ * static_cast<int64_t>(sizeof(float)));
    half_ = xQueSpace_ / sizeof(float) / SWI_FACTOR;
    limit_ = tiling_->limit;
    alpha_ = tiling_->alpha;
    bias_ = tiling_->bias;

    xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
    gradYGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(gradY));
    gradXOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(gradXOut));
    if constexpr (isGroup) {
        groupIndexGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(groupIndex));
    }

    pipe_->InitBuffer(xQueue_, DB_BUFFER, xQueSpace_);
    pipe_->InitBuffer(dyQueue_, DB_BUFFER, dyQueSpace_);
    pipe_->InitBuffer(dxQueue_, 1, xQueSpace_);
    pipe_->InitBuffer(vectorBuf_, xQueSpace_);
}

template <typename T, bool isInterleaved, bool isGroup>
__aicore__ inline void ClippedSwigluGradArch35Kernel<T, isInterleaved, isGroup>::ComputeRealBatchSize()
{
    if constexpr (!isGroup) {
        realBatchSize_ = tiling_->dimBatchSize;
    } else {
        int64_t groupSum = 0;
        for (int64_t i = 0; i < tiling_->groupNum; ++i) {
            groupSum += groupIndexGm_.GetValue(i);
        }
        realBatchSize_ = groupSum < tiling_->dimBatchSize ? groupSum : tiling_->dimBatchSize;
    }
}

template <typename T, bool isInterleaved, bool isGroup>
__aicore__ inline void ClippedSwigluGradArch35Kernel<T, isInterleaved, isGroup>::CalTilingParam()
{
    int64_t coreNum = static_cast<int64_t>(tiling_->coreNumAll);
    int64_t blockIdx = static_cast<int64_t>(blockIdx_);

    if constexpr (!isInterleaved) {
        int64_t base = realBatchSize_ / coreNum;
        int64_t remainder = realBatchSize_ % coreNum;
        usedCoreNum_ = static_cast<uint32_t>(realBatchSize_ < coreNum ? realBatchSize_ : coreNum);
        batchPreBlock_ = base + (blockIdx < remainder ? 1 : 0);
        int64_t coreStartRow = blockIdx * base + (blockIdx < remainder ? blockIdx : remainder);
        blockOffset_ = coreStartRow * tiling_->dim2H;

        if (tiling_->isLongH == 0) {
            int64_t batchSpace = SWI_FACTOR * AlignBytes(dimH_ * static_cast<int64_t>(sizeof(float)));
            int64_t ubMaxBatch = xQueSpace_ / batchSpace;
            loopTime_ = (batchPreBlock_ + ubMaxBatch - 1) / ubMaxBatch;
            int64_t batchLastLoop = batchPreBlock_ - ubMaxBatch * (loopTime_ - 1);
            pairFrontLoop_ = ubMaxBatch * dimH_;
            pairLastLoop_ = batchLastLoop * dimH_;
            loopOffset_ = ubMaxBatch * tiling_->dim2H;
            calPairFrontLoop_ = ubMaxBatch * batchSpace / SWI_FACTOR / sizeof(float);
            calPairLastLoop_ = batchLastLoop * batchSpace / SWI_FACTOR / sizeof(float);
        } else {
            loopTime_ = (dimH_ + ubMaxPair_ - 1) / ubMaxPair_;
            pairLastLoop_ = dimH_ - ubMaxPair_ * (loopTime_ - 1);
            pairFrontLoop_ = ubMaxPair_;
            loopOffset_ = ubMaxPair_;
            calPairFrontLoop_ = pairFrontLoop_;
            calPairLastLoop_ = pairLastLoop_;
        }
    } else {
        int64_t pairTotal = tiling_->dim2H * realBatchSize_ / SWI_FACTOR;
        int64_t base = pairTotal / coreNum;
        int64_t remainder = pairTotal % coreNum;
        usedCoreNum_ = static_cast<uint32_t>(pairTotal < coreNum ? pairTotal : coreNum);
        int64_t pairPreBlock = base + (blockIdx < remainder ? 1 : 0);
        int64_t coreStartPair = blockIdx * base + (blockIdx < remainder ? blockIdx : remainder);
        blockOffset_ = coreStartPair * SWI_FACTOR;

        loopTime_ = (pairPreBlock + ubMaxPair_ - 1) / ubMaxPair_;
        pairLastLoop_ = pairPreBlock - ubMaxPair_ * (loopTime_ - 1);
        pairFrontLoop_ = ubMaxPair_;
        loopOffset_ = SWI_FACTOR * ubMaxPair_;
        calPairFrontLoop_ = pairFrontLoop_;
        calPairLastLoop_ = pairLastLoop_;
    }
}

template <typename T, bool isInterleaved, bool isGroup>
__aicore__ inline void ClippedSwigluGradArch35Kernel<T, isInterleaved, isGroup>::Process()
{
    ComputeRealBatchSize();
    CalTilingParam();

    if (blockIdx_ < usedCoreNum_) {
        ProcessMainLoop();
    }

    SyncAll();

    if constexpr (isGroup) {
        if (realBatchSize_ < tiling_->dimBatchSize) {
            InitZeroBuffer();
            ZeroInvalidRows();
        }
    }
}

template <typename T, bool isInterleaved, bool isGroup>
__aicore__ inline void ClippedSwigluGradArch35Kernel<T, isInterleaved, isGroup>::ProcessMainLoop()
{
    int64_t xOffset = 0;
    int64_t dyOffset = 0;
    int64_t dxOffset = 0;

    if constexpr (!isInterleaved) {
        if (tiling_->isLongH == 1) {
            for (int64_t batchIdx = 0; batchIdx < batchPreBlock_; ++batchIdx) {
                xOffset = blockOffset_ + batchIdx * tiling_->dim2H;
                dyOffset = blockOffset_ / SWI_FACTOR + batchIdx * dimH_;
                dxOffset = blockOffset_ + batchIdx * tiling_->dim2H;
                for (int64_t loopIdx = 0; loopIdx < loopTime_; ++loopIdx) {
                    pairNum_ = loopIdx == (loopTime_ - 1) ? pairLastLoop_ : pairFrontLoop_;
                    calPairNum_ = loopIdx == (loopTime_ - 1) ? calPairLastLoop_ : calPairFrontLoop_;
                    ProcessSingleLoop(xOffset, dyOffset, dxOffset);
                    xOffset += loopOffset_;
                    dyOffset += loopOffset_;
                    dxOffset += loopOffset_;
                }
            }
            return;
        }
    }

    xOffset = blockOffset_;
    dyOffset = blockOffset_ / SWI_FACTOR;
    dxOffset = blockOffset_;
    for (int64_t loopIdx = 0; loopIdx < loopTime_; ++loopIdx) {
        pairNum_ = loopIdx == (loopTime_ - 1) ? pairLastLoop_ : pairFrontLoop_;
        calPairNum_ = loopIdx == (loopTime_ - 1) ? calPairLastLoop_ : calPairFrontLoop_;
        ProcessSingleLoop(xOffset, dyOffset, dxOffset);
        xOffset += loopOffset_;
        dyOffset += loopOffset_ / SWI_FACTOR;
        dxOffset += loopOffset_;
    }
}

template <typename T, bool isInterleaved, bool isGroup>
__aicore__ inline void ClippedSwigluGradArch35Kernel<T, isInterleaved, isGroup>::ProcessSingleLoop(int64_t xOffset,
                                                                                                   int64_t dyOffset,
                                                                                                   int64_t dxOffset)
{
    CopyIn(xOffset, dyOffset);

    LocalTensor<T> xDTypeLocal = xQueue_.template DeQue<T>();
    LocalTensor<T> dyDTypeLocal = dyQueue_.template DeQue<T>();
    LocalTensor<T> dxDTypeLocal = dxQueue_.template AllocTensor<T>();

    ComputeVfGrad(xDTypeLocal, dyDTypeLocal, dxDTypeLocal, calPairNum_);

    xQueue_.FreeTensor(xDTypeLocal);
    dyQueue_.FreeTensor(dyDTypeLocal);
    dxQueue_.EnQue(dxDTypeLocal);
    CopyOut(dxOffset);
}

template <typename T, bool isInterleaved, bool isGroup>
__aicore__ inline void ClippedSwigluGradArch35Kernel<T, isInterleaved, isGroup>::CopyIn(int64_t xOffset,
                                                                                        int64_t dyOffset)
{
    if constexpr (!std::is_same_v<T, float>) {
        int64_t blockElem = BLOCK_SIZE / sizeof(T);
        xLocalOffset1_ = (xQueSpace_ / SWI_FACTOR / static_cast<int64_t>(sizeof(T)) + blockElem - 1) / blockElem *
                         blockElem;
        xLocalOffset2_ = (xLocalOffset1_ / SWI_FACTOR + blockElem - 1) / blockElem * blockElem;
        dyLocalOffset_ = (dyQueSpace_ / static_cast<int64_t>(sizeof(T)) / SWI_FACTOR + blockElem - 1) / blockElem *
                         blockElem;
        dxDbOffset_ = (calPairNum_ * static_cast<int64_t>(sizeof(T)) + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE /
                      sizeof(T);
    } else {
        xLocalOffset1_ = 0;
        xLocalOffset2_ = xQueSpace_ / static_cast<int64_t>(sizeof(float)) / SWI_FACTOR;
        dyLocalOffset_ = 0;
        dxDbOffset_ = half_;
    }

    LocalTensor<T> xDTypeLocal = xQueue_.template AllocTensor<T>();
    LocalTensor<T> dyDTypeLocal = dyQueue_.template AllocTensor<T>();

    if constexpr (isInterleaved) {
        CopyInInterLeaved(xDTypeLocal, dyDTypeLocal, xOffset, dyOffset);
    } else {
        if (tiling_->isLongH == 0) {
            CopyInHalfShortH(xDTypeLocal, dyDTypeLocal, xOffset, dyOffset);
        } else {
            CopyInHalfLongH(xDTypeLocal, dyDTypeLocal, xOffset, dyOffset);
        }
    }

    xQueue_.EnQue(xDTypeLocal);
    dyQueue_.EnQue(dyDTypeLocal);
}

template <typename T, bool isInterleaved, bool isGroup>
__aicore__ inline void ClippedSwigluGradArch35Kernel<T, isInterleaved, isGroup>::CopyInHalfShortH(
    LocalTensor<T>& xDTypeLocal, LocalTensor<T>& dyDTypeLocal, int64_t xOffset, int64_t dyOffset)
{
    DataCopyPadParams padParams{false, 0, 0, 0};
    DataCopyParams dataCopyXParams;
    dataCopyXParams.blockCount = pairNum_ / dimH_;
    dataCopyXParams.blockLen = dimH_ * sizeof(T);
    dataCopyXParams.srcStride = dimH_ * sizeof(T);
    dataCopyXParams.dstStride = 0;
    DataCopyPad(xDTypeLocal[xLocalOffset1_], xGm_[xOffset], dataCopyXParams, padParams);
    DataCopyPad(xDTypeLocal[xLocalOffset1_ + xLocalOffset2_], xGm_[xOffset + dimH_], dataCopyXParams, padParams);
    DataCopyParams dataCopyDyParams;
    dataCopyDyParams.blockCount = pairNum_ / dimH_;
    dataCopyDyParams.blockLen = dimH_ * sizeof(T);
    dataCopyDyParams.srcStride = 0;
    dataCopyDyParams.dstStride = 0;
    DataCopyPad(dyDTypeLocal[dyLocalOffset_], gradYGm_[dyOffset], dataCopyDyParams, padParams);
}

template <typename T, bool isInterleaved, bool isGroup>
__aicore__ inline void ClippedSwigluGradArch35Kernel<T, isInterleaved, isGroup>::CopyInHalfLongH(
    LocalTensor<T>& xDTypeLocal, LocalTensor<T>& dyDTypeLocal, int64_t xOffset, int64_t dyOffset)
{
    DataCopyPadParams padParams{false, 0, 0, 0};
    DataCopyParams dataCopyXParams;
    dataCopyXParams.blockCount = 1;
    dataCopyXParams.blockLen = AlignBytes(pairNum_ * sizeof(T));
    dataCopyXParams.srcStride = 0;
    dataCopyXParams.dstStride = 0;
    DataCopyPad(xDTypeLocal[xLocalOffset1_], xGm_[xOffset], dataCopyXParams, padParams);
    DataCopyPad(xDTypeLocal[xLocalOffset1_ + xLocalOffset2_], xGm_[xOffset + dimH_], dataCopyXParams, padParams);
    DataCopyParams dataCopyDyParams;
    dataCopyDyParams.blockCount = 1;
    dataCopyDyParams.blockLen = AlignBytes(pairNum_ * sizeof(T));
    dataCopyDyParams.srcStride = 0;
    dataCopyDyParams.dstStride = 0;
    DataCopyPad(dyDTypeLocal[dyLocalOffset_], gradYGm_[dyOffset], dataCopyDyParams, padParams);
}

template <typename T, bool isInterleaved, bool isGroup>
__aicore__ inline void ClippedSwigluGradArch35Kernel<T, isInterleaved, isGroup>::CopyInInterLeaved(
    LocalTensor<T>& xDTypeLocal, LocalTensor<T>& dyDTypeLocal, int64_t xOffset, int64_t dyOffset)
{
    DataCopyPadParams padParams{false, 0, 0, 0};
    DataCopyParams dataCopyXParams;
    dataCopyXParams.blockCount = 1;
    dataCopyXParams.blockLen = SWI_FACTOR * pairNum_ * sizeof(T);
    dataCopyXParams.srcStride = 0;
    dataCopyXParams.dstStride = 0;
    DataCopyPad(xDTypeLocal[xLocalOffset1_], xGm_[xOffset], dataCopyXParams, padParams);
    DataCopyParams dataCopyDyParams;
    dataCopyDyParams.blockCount = 1;
    dataCopyDyParams.blockLen = pairNum_ * sizeof(T);
    dataCopyDyParams.srcStride = 0;
    dataCopyDyParams.dstStride = 0;
    DataCopyPad(dyDTypeLocal[dyLocalOffset_], gradYGm_[dyOffset], dataCopyDyParams, padParams);
}

template <typename T, bool isInterleaved, bool isGroup>
__aicore__ inline void ClippedSwigluGradArch35Kernel<T, isInterleaved, isGroup>::LoadOneTensor(
    __local_mem__ void* input, Reg::RegTensor<float>& dst, Reg::MaskReg& preg, uint32_t offset)
{
    if constexpr (std::is_same_v<T, half>) {
        Reg::RegTensor<half> xFp16;
        Reg::LoadAlign<half, Reg::LoadDist::DIST_UNPACK_B16>(xFp16, (__local_mem__ half*)input + offset);
        Cast<float, half, CAST_BF16_FP16_TO_FP32>(dst, xFp16, preg);
    } else if constexpr (std::is_same_v<T, bfloat16_t>) {
        Reg::RegTensor<bfloat16_t> xBf16;
        Reg::LoadAlign<bfloat16_t, Reg::LoadDist::DIST_UNPACK_B16>(xBf16, (__local_mem__ bfloat16_t*)input + offset);
        Cast<float, bfloat16_t, CAST_BF16_FP16_TO_FP32>(dst, xBf16, preg);
    } else {
        Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(dst, (__local_mem__ float*)input + offset);
    }
}

template <typename T, bool isInterleaved, bool isGroup>
__aicore__ inline void ClippedSwigluGradArch35Kernel<T, isInterleaved, isGroup>::ComputeVfGrad(
    LocalTensor<T>& xDTypeLocal, LocalTensor<T>& dyDTypeLocal, LocalTensor<T>& dxDTypeLocal, int64_t onceNum)
{
    float clampLimit = limit_;
    float negClampLimit = -limit_;
    float scalarOne = 1.0f;
    float scalarZero = 0.0f;
    float gluBias = bias_;
    float gluAlpha = alpha_;
    float negAlpha = -alpha_;

    uint32_t halfU32 = static_cast<uint32_t>(half_);

    uint16_t dim1VfTimes = onceNum / VF_LEN_FP32;
    uint32_t tail = onceNum % VF_LEN_FP32;
    uint16_t tailTimes = (tail > 0) ? 1 : 0;

    LocalTensor<float> dxFloatLocal = dxDTypeLocal.template ReinterpretCast<float>();
    LocalTensor<float> vecBufF = vectorBuf_.Get<float>();

    __local_mem__ float* dxFAddr = reinterpret_cast<__local_mem__ float*>(dxDTypeLocal.GetPhyAddr());
    __local_mem__ float* vecAddr = reinterpret_cast<__local_mem__ float*>(vecBufF.GetPhyAddr());
    __local_mem__ T* xAddr = reinterpret_cast<__local_mem__ T*>(xDTypeLocal.GetPhyAddr()) +
                             static_cast<uint32_t>(xLocalOffset1_);
    __local_mem__ T* dyAddr = reinterpret_cast<__local_mem__ T*>(dyDTypeLocal.GetPhyAddr()) +
                              static_cast<uint32_t>(dyLocalOffset_);
    __local_mem__ T* x0Addr = xAddr;
    __local_mem__ T* x1Addr = xAddr + static_cast<uint32_t>(xLocalOffset2_);

    // ---- VF: load a/b/dy, clamp-mask in reg, compute da/db ----
    __VEC_SCOPE__
    {
        Reg::RegTensor<float> vregX0;
        Reg::RegTensor<float> vregX1;
        Reg::RegTensor<float> vregX0DeF;
        Reg::RegTensor<float> vregX1DeF;
        Reg::RegTensor<float> vregDY;
        Reg::RegTensor<float> minsReg;
        Reg::RegTensor<float> mulsReg;
        Reg::RegTensor<float> expReg;
        Reg::RegTensor<float> addsReg;
        Reg::RegTensor<float> sigReg;
        Reg::RegTensor<float> tmpReg;
        Reg::RegTensor<float> oneReg;
        Reg::RegTensor<float> daReg;
        Reg::RegTensor<float> dbReg;
        Reg::RegTensor<float> limitReg;
        Reg::RegTensor<float> negLimitReg;
        Reg::RegTensor<float> zeroReg;

        Reg::MaskReg maskAll = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
        Reg::MaskReg maskT = Reg::UpdateMask<float>(tail);
        Reg::MaskReg maskA;
        Reg::MaskReg maskB;
        Reg::MaskReg maskBn;

        Reg::Duplicate(limitReg, clampLimit);
        Reg::Duplicate(negLimitReg, negClampLimit);
        Reg::Duplicate(zeroReg, scalarZero);

        for (uint16_t vfIdx = 0; vfIdx < dim1VfTimes + tailTimes; vfIdx++) {
            uint32_t offset = vfIdx * static_cast<uint32_t>(VF_LEN_FP32);
            Reg::MaskReg preg = (vfIdx < dim1VfTimes) ? maskAll : maskT;

            if constexpr (isInterleaved) {
                uint32_t vfLenT = VF_LEN_FP32 * SWI_FACTOR;
                Reg::AddrReg srcIdxOffset = Reg::CreateAddrReg<T>(vfIdx, vfLenT);
                if constexpr (std::is_same_v<T, half>) {
                    Reg::RegTensor<half> vregX0Raw;
                    Reg::RegTensor<half> vregX1Raw;
                    Reg::LoadAlign<half, Reg::LoadDist::DIST_UNPACK_B16>(vregX0Raw, xAddr, srcIdxOffset);
                    Reg::LoadAlign<half, Reg::LoadDist::DIST_UNPACK_B16>(
                        vregX1Raw, xAddr + static_cast<uint32_t>(VF_LEN_FP32), srcIdxOffset);
                    Reg::Cast<float, half, CAST_BF16_FP16_TO_FP32>(vregX0, vregX0Raw, maskAll);
                    Reg::Cast<float, half, CAST_BF16_FP16_TO_FP32>(vregX1, vregX1Raw, maskAll);
                } else if constexpr (std::is_same_v<T, bfloat16_t>) {
                    Reg::RegTensor<bfloat16_t> vregX0Raw;
                    Reg::RegTensor<bfloat16_t> vregX1Raw;
                    Reg::LoadAlign<bfloat16_t, Reg::LoadDist::DIST_UNPACK_B16>(vregX0Raw, xAddr, srcIdxOffset);
                    Reg::LoadAlign<bfloat16_t, Reg::LoadDist::DIST_UNPACK_B16>(
                        vregX1Raw, xAddr + static_cast<uint32_t>(VF_LEN_FP32), srcIdxOffset);
                    Reg::Cast<float, bfloat16_t, CAST_BF16_FP16_TO_FP32>(vregX0, vregX0Raw, maskAll);
                    Reg::Cast<float, bfloat16_t, CAST_BF16_FP16_TO_FP32>(vregX1, vregX1Raw, maskAll);
                } else {
                    Reg::LoadAlign<T, Reg::LoadDist::DIST_NORM>((Reg::RegTensor<T>&)vregX0, xAddr, srcIdxOffset);
                    Reg::LoadAlign<T, Reg::LoadDist::DIST_NORM>(
                        (Reg::RegTensor<T>&)vregX1, xAddr + static_cast<uint32_t>(VF_LEN_FP32), srcIdxOffset);
                }
                Reg::DeInterleave(vregX0DeF, vregX1DeF, vregX0, vregX1);
            } else {
                LoadOneTensor(x0Addr, vregX0DeF, preg, offset);
                LoadOneTensor(x1Addr, vregX1DeF, preg, offset);
            }
            LoadOneTensor(dyAddr, vregDY, preg, offset);

            Reg::Compare<float, CMPMODE::LE>(maskA, vregX0DeF, limitReg, preg);
            Reg::Compare<float, CMPMODE::LE>(maskB, vregX1DeF, limitReg, preg);
            Reg::Compare<float, CMPMODE::GE>(maskBn, vregX1DeF, negLimitReg, preg);

            Mins(minsReg, vregX0DeF, clampLimit, preg);

            Muls(mulsReg, minsReg, negAlpha, preg);
            Exp(expReg, mulsReg, preg);
            Adds(addsReg, expReg, scalarOne, preg);
            Muls(oneReg, minsReg, 0.0f, preg);
            Adds(oneReg, oneReg, scalarOne, preg);
            Div(sigReg, oneReg, addsReg, preg);

            Mins(vregX1DeF, vregX1DeF, clampLimit, preg);
            Maxs(vregX1DeF, vregX1DeF, negClampLimit, preg);
            Adds(vregX1DeF, vregX1DeF, gluBias, preg);

            Mul(dbReg, vregDY, minsReg, preg);
            Mul(dbReg, dbReg, sigReg, preg);

            Muls(tmpReg, sigReg, -1.0f, preg);
            Adds(tmpReg, tmpReg, scalarOne, preg);
            Mul(tmpReg, tmpReg, minsReg, preg);
            Muls(tmpReg, tmpReg, gluAlpha, preg);
            Adds(tmpReg, tmpReg, scalarOne, preg);
            Mul(tmpReg, tmpReg, sigReg, preg);
            Mul(tmpReg, tmpReg, vregX1DeF, preg);
            Mul(daReg, tmpReg, vregDY, preg);

            Reg::Select<float>(daReg, daReg, zeroReg, maskA);
            Reg::Select<float>(dbReg, dbReg, zeroReg, maskB);
            Reg::Select<float>(dbReg, dbReg, zeroReg, maskBn);

            if constexpr (isInterleaved) {
                Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM>(vecAddr + offset, daReg, preg);
                Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM>(vecAddr + halfU32 + offset, dbReg, preg);
            } else {
                Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM>(dxFAddr + offset, daReg, preg);
                Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM>(dxFAddr + halfU32 + offset, dbReg, preg);
            }
        }
    }

    PipeBarrier<PIPE_V>();

    if constexpr (isInterleaved) {
        LocalTensor<float> daBuf = vecBufF;
        LocalTensor<float> dbBuf = vecBufF[half_];
        constexpr int64_t ALIGN_ELEMS = 32 / sizeof(float);
        int64_t alignedCount = (calPairNum_ / ALIGN_ELEMS) * ALIGN_ELEMS;
        if (alignedCount > 0) {
            Interleave(dxFloatLocal, dxFloatLocal[alignedCount], daBuf, dbBuf, alignedCount);
            PipeBarrier<PIPE_V>();
        }
        for (int64_t i = alignedCount; i < calPairNum_; ++i) {
            dxFloatLocal.SetValue(2 * i, daBuf.GetValue(i));
            dxFloatLocal.SetValue(2 * i + 1, dbBuf.GetValue(i));
        }
        event_t vToMte3 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>());
        SetFlag<HardEvent::V_MTE3>(vToMte3);
        WaitFlag<HardEvent::V_MTE3>(vToMte3);
        GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::V_MTE3>(vToMte3);
    }

    if constexpr (std::is_same_v<T, bfloat16_t>) {
        if constexpr (isInterleaved) {
            Cast(dxDTypeLocal, dxFloatLocal, RoundMode::CAST_RINT, calPairNum_ * SWI_FACTOR);
        } else {
            Cast(dxDTypeLocal, dxFloatLocal, RoundMode::CAST_RINT, calPairNum_);
            PipeBarrier<PIPE_V>();
            Cast(dxDTypeLocal[dxDbOffset_], dxFloatLocal[half_], RoundMode::CAST_RINT, calPairNum_);
        }
        PipeBarrier<PIPE_V>();
    } else if constexpr (std::is_same_v<T, half>) {
        if constexpr (isInterleaved) {
            Cast(dxDTypeLocal, dxFloatLocal, RoundMode::CAST_NONE, calPairNum_ * SWI_FACTOR);
        } else {
            Cast(dxDTypeLocal, dxFloatLocal, RoundMode::CAST_NONE, calPairNum_);
            PipeBarrier<PIPE_V>();
            Cast(dxDTypeLocal[dxDbOffset_], dxFloatLocal[half_], RoundMode::CAST_NONE, calPairNum_);
        }
        PipeBarrier<PIPE_V>();
    }
}

template <typename T, bool isInterleaved, bool isGroup>
__aicore__ inline void ClippedSwigluGradArch35Kernel<T, isInterleaved, isGroup>::CopyOut(int64_t dxOffset)
{
    LocalTensor<T> dxDTypeLocal = dxQueue_.template DeQue<T>();

    DataCopyParams params;
    if constexpr (!isInterleaved) {
        if (tiling_->isLongH == 0) {
            params.blockCount = pairNum_ / dimH_;
            params.blockLen = dimH_ * sizeof(T);
            params.srcStride = 0;
            params.dstStride = dimH_ * sizeof(T);
        } else {
            params.blockCount = 1;
            params.blockLen = pairNum_ * sizeof(T);
            params.srcStride = 0;
            params.dstStride = 0;
        }
        DataCopyPad(gradXOutGm_[dxOffset], dxDTypeLocal, params);
        DataCopyPad(gradXOutGm_[dxOffset + dimH_], dxDTypeLocal[dxDbOffset_], params);
    } else {
        DataCopyParams dataCopyParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = pairNum_ * SWI_FACTOR * sizeof(T);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;
        DataCopyPad(gradXOutGm_[dxOffset], dxDTypeLocal, dataCopyParams);
    }
    dxQueue_.FreeTensor(dxDTypeLocal);
}

template <typename T, bool isInterleaved, bool isGroup>
__aicore__ inline void ClippedSwigluGradArch35Kernel<T, isInterleaved, isGroup>::InitZeroBuffer()
{
    pipe_->Reset();
    int64_t elemBytes = static_cast<int64_t>(sizeof(T));
    int64_t chunkElems = ZERO_CHUNK_BYTES / elemBytes;
    int64_t zeroBufSize = AlignBytes(ZERO_CHUNK_BYTES);
    pipe_->InitBuffer(zeroBuf_, zeroBufSize);

    LocalTensor<T> zeroLocal = zeroBuf_.Get<T>();
    Duplicate(zeroLocal, static_cast<T>(0), chunkElems);
    event_t vToMte3 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>());
    SetFlag<HardEvent::V_MTE3>(vToMte3);
    WaitFlag<HardEvent::V_MTE3>(vToMte3);
    GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::V_MTE3>(vToMte3);
}

template <typename T, bool isInterleaved, bool isGroup>
__aicore__ inline void ClippedSwigluGradArch35Kernel<T, isInterleaved, isGroup>::ZeroInvalidRows()
{
    int64_t invalidRows = tiling_->dimBatchSize - realBatchSize_;
    if (invalidRows <= 0) {
        return;
    }

    int64_t coreNum = static_cast<int64_t>(tiling_->coreNumAll);
    int64_t blockIdx = static_cast<int64_t>(blockIdx_);
    int64_t base = invalidRows / coreNum;
    int64_t remainder = invalidRows % coreNum;
    int64_t rowsToZero = base + (blockIdx < remainder ? 1 : 0);
    if (rowsToZero <= 0) {
        return;
    }
    int64_t zeroStartRow = realBatchSize_ + blockIdx * base + (blockIdx < remainder ? blockIdx : remainder);

    LocalTensor<T> zeroLocal = zeroBuf_.Get<T>();
    DataCopyPadParams padParams{false, 0, 0, 0};
    DataCopyParams params;
    params.blockCount = 1;
    params.srcStride = 0;
    params.dstStride = 0;
    int64_t elemBytes = static_cast<int64_t>(sizeof(T));
    int64_t chunkElems = ZERO_CHUNK_BYTES / elemBytes;
    int64_t dim2H = tiling_->dim2H;
    int64_t fullChunks = dim2H / chunkElems;
    int64_t tailElems = dim2H % chunkElems;

    for (int64_t row = 0; row < rowsToZero; ++row) {
        int64_t rowBase = (zeroStartRow + row) * dim2H;
        int64_t off = 0;
        for (int64_t c = 0; c < fullChunks; ++c) {
            params.blockLen = ZERO_CHUNK_BYTES;
            DataCopyPad(gradXOutGm_[rowBase + off], zeroLocal, params);
            off += chunkElems;
        }
        if (tailElems > 0) {
            params.blockLen = tailElems * elemBytes;
            DataCopyPad(gradXOutGm_[rowBase + off], zeroLocal, params);
        }
    }
}

} // namespace ClippedSwigluGradArch35Op
#endif // CLIPPED_SWIGLU_GRAD_KERNEL_H
