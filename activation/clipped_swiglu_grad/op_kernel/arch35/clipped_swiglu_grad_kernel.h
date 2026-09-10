/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
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
 * - 分核逻辑：与910B一致（CalTilingParam 复用自公共基类）
 */

#ifndef CLIPPED_SWIGLU_GRAD_KERNEL_H
#define CLIPPED_SWIGLU_GRAD_KERNEL_H

#include "kernel_operator.h"
#include "op_kernel/math_util.h"
#include "op_kernel/platform_util.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../clipped_swiglu_grad_base.h"

namespace ClippedSwigluGradArch35Op {
using namespace AscendC;
using ClippedSwigluGradOps::ClippedSwigluGradSchedBase;
using ClippedSwigluGradOps::SWI_FACTOR;

constexpr uint32_t VF_LEN_FP32 = Ops::Base::GetVRegSize() / sizeof(float);

static constexpr Reg::CastTrait CAST_BF16_FP16_TO_FP32 = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                          Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

template <typename T, bool isInterleaved, bool isGroup>
class ClippedSwigluGradArch35Kernel
    : public ClippedSwigluGradSchedBase<T, isInterleaved, isGroup,
                                        ClippedSwigluGradArch35Kernel<T, isInterleaved, isGroup>> {
public:
    using SchedBase = ClippedSwigluGradSchedBase<T, isInterleaved, isGroup,
                                                 ClippedSwigluGradArch35Kernel<T, isInterleaved, isGroup>>;
    __aicore__ inline ClippedSwigluGradArch35Kernel(const ClippedSwigluGradTilingData* tilingData, TPipe* pipe)
        : SchedBase(tilingData, pipe)
    {}

    __aicore__ inline void Init(GM_ADDR gradY, GM_ADDR x, GM_ADDR groupIndex, GM_ADDR gradXOut);

private:
    friend class ClippedSwigluGradSchedBase<T, isInterleaved, isGroup, ClippedSwigluGradArch35Kernel>;

    __aicore__ inline void ProcessSingleLoop(int64_t xOffset, int64_t dyOffset, int64_t dxOffset);
    __aicore__ inline void ComputeVfGrad(LocalTensor<T>& xDTypeLocal, LocalTensor<T>& dyDTypeLocal,
                                         LocalTensor<T>& dxDTypeLocal, int64_t onceNum);
    __aicore__ inline void LoadOneTensor(__local_mem__ void* input, Reg::RegTensor<float>& dst, Reg::MaskReg& preg,
                                         uint32_t offset);
    __aicore__ inline void LoadInterleavedPair(__local_mem__ T* xAddr, uint32_t vfIdx, Reg::MaskReg& maskAll,
                                               Reg::RegTensor<float>& vregX0, Reg::RegTensor<float>& vregX1,
                                               Reg::RegTensor<float>& vregX0DeF, Reg::RegTensor<float>& vregX1DeF);
    __aicore__ inline void VfGradMath(Reg::RegTensor<float>& vregX0DeF, Reg::RegTensor<float>& vregX1DeF,
                                      Reg::RegTensor<float>& vregDY, Reg::RegTensor<float>& daReg,
                                      Reg::RegTensor<float>& dbReg, Reg::MaskReg& preg);
    __aicore__ inline void VfGradComputeCore(__local_mem__ T* xAddr, __local_mem__ T* x0Addr, __local_mem__ T* x1Addr,
                                             __local_mem__ T* dyAddr, __local_mem__ float* dxFAddr,
                                             __local_mem__ float* vecAddr, int64_t onceNum);
    __aicore__ inline void ScatterInterleaved(LocalTensor<float>& dxFloatLocal, LocalTensor<float>& vecBufF);
    __aicore__ inline void CastDxOut(LocalTensor<T>& dxDTypeLocal, LocalTensor<float>& dxFloatLocal);

    using SchedBase::calPairNum_;
    using SchedBase::CopyIn;
    using SchedBase::CopyOut;
    using SchedBase::dxDbOffset_;
    using SchedBase::dxQueue_;
    using SchedBase::dyLocalOffset_;
    using SchedBase::dyQueue_;
    using SchedBase::half_;
    using SchedBase::InitCommon;
    using SchedBase::pipe_;
    using SchedBase::tiling_;
    using SchedBase::xLocalOffset1_;
    using SchedBase::xLocalOffset2_;
    using SchedBase::xQueSpace_;
    using SchedBase::xQueue_;

    TBuf<TPosition::VECCALC> vectorBuf_;

    float limit_ = 0.0f;
    float alpha_ = 0.0f;
    float bias_ = 0.0f;
};

template <typename T, bool isInterleaved, bool isGroup>
__aicore__ inline void ClippedSwigluGradArch35Kernel<T, isInterleaved, isGroup>::Init(GM_ADDR gradY, GM_ADDR x,
                                                                                      GM_ADDR groupIndex,
                                                                                      GM_ADDR gradXOut)
{
    InitCommon(gradY, x, groupIndex, gradXOut);
    limit_ = tiling_->limit;
    alpha_ = tiling_->alpha;
    bias_ = tiling_->bias;
    pipe_->InitBuffer(vectorBuf_, xQueSpace_);
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
__aicore__ inline void ClippedSwigluGradArch35Kernel<T, isInterleaved, isGroup>::LoadInterleavedPair(
    __local_mem__ T* xAddr, uint32_t vfIdx, Reg::MaskReg& maskAll, Reg::RegTensor<float>& vregX0,
    Reg::RegTensor<float>& vregX1, Reg::RegTensor<float>& vregX0DeF, Reg::RegTensor<float>& vregX1DeF)
{
    uint32_t vfLenT = VF_LEN_FP32 * SWI_FACTOR;
    Reg::AddrReg srcIdxOffset = Reg::CreateAddrReg<T>(vfIdx, vfLenT);
    if constexpr (std::is_same_v<T, half>) {
        Reg::RegTensor<half> vregX0Raw;
        Reg::RegTensor<half> vregX1Raw;
        Reg::LoadAlign<half, Reg::LoadDist::DIST_UNPACK_B16>(vregX0Raw, xAddr, srcIdxOffset);
        Reg::LoadAlign<half, Reg::LoadDist::DIST_UNPACK_B16>(vregX1Raw, xAddr + static_cast<uint32_t>(VF_LEN_FP32),
                                                             srcIdxOffset);
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
        Reg::LoadAlign<T, Reg::LoadDist::DIST_NORM>((Reg::RegTensor<T>&)vregX1,
                                                    xAddr + static_cast<uint32_t>(VF_LEN_FP32), srcIdxOffset);
    }
    Reg::DeInterleave(vregX0DeF, vregX1DeF, vregX0, vregX1);
}

template <typename T, bool isInterleaved, bool isGroup>
__aicore__ inline void ClippedSwigluGradArch35Kernel<T, isInterleaved, isGroup>::VfGradMath(
    Reg::RegTensor<float>& vregX0DeF, Reg::RegTensor<float>& vregX1DeF, Reg::RegTensor<float>& vregDY,
    Reg::RegTensor<float>& daReg, Reg::RegTensor<float>& dbReg, Reg::MaskReg& preg)
{
    Reg::RegTensor<float> minsReg;
    Reg::RegTensor<float> mulsReg;
    Reg::RegTensor<float> expReg;
    Reg::RegTensor<float> addsReg;
    Reg::RegTensor<float> sigReg;
    Reg::RegTensor<float> tmpReg;
    Reg::RegTensor<float> oneReg;
    Reg::RegTensor<float> limitReg;
    Reg::RegTensor<float> negLimitReg;
    Reg::RegTensor<float> zeroReg;
    Reg::MaskReg maskA;
    Reg::MaskReg maskB;
    Reg::MaskReg maskBn;

    Reg::Duplicate(limitReg, limit_);
    Reg::Duplicate(negLimitReg, -limit_);
    Reg::Duplicate(zeroReg, 0.0f);

    Reg::Compare<float, CMPMODE::LE>(maskA, vregX0DeF, limitReg, preg);
    Reg::Compare<float, CMPMODE::LE>(maskB, vregX1DeF, limitReg, preg);
    Reg::Compare<float, CMPMODE::GE>(maskBn, vregX1DeF, negLimitReg, preg);

    Mins(minsReg, vregX0DeF, limit_, preg);
    Muls(mulsReg, minsReg, -alpha_, preg);
    Exp(expReg, mulsReg, preg);
    Adds(addsReg, expReg, 1.0f, preg);
    Muls(oneReg, minsReg, 0.0f, preg);
    Adds(oneReg, oneReg, 1.0f, preg);
    Div(sigReg, oneReg, addsReg, preg);

    Mins(vregX1DeF, vregX1DeF, limit_, preg);
    Maxs(vregX1DeF, vregX1DeF, -limit_, preg);
    Adds(vregX1DeF, vregX1DeF, bias_, preg);

    Mul(dbReg, vregDY, minsReg, preg);
    Mul(dbReg, dbReg, sigReg, preg);

    Muls(tmpReg, sigReg, -1.0f, preg);
    Adds(tmpReg, tmpReg, 1.0f, preg);
    Mul(tmpReg, tmpReg, minsReg, preg);
    Muls(tmpReg, tmpReg, alpha_, preg);
    Adds(tmpReg, tmpReg, 1.0f, preg);
    Mul(tmpReg, tmpReg, sigReg, preg);
    Mul(tmpReg, tmpReg, vregX1DeF, preg);
    Mul(daReg, tmpReg, vregDY, preg);

    Reg::Select<float>(daReg, daReg, zeroReg, maskA);
    Reg::Select<float>(dbReg, dbReg, zeroReg, maskB);
    Reg::Select<float>(dbReg, dbReg, zeroReg, maskBn);
}

template <typename T, bool isInterleaved, bool isGroup>
__aicore__ inline void ClippedSwigluGradArch35Kernel<T, isInterleaved, isGroup>::VfGradComputeCore(
    __local_mem__ T* xAddr, __local_mem__ T* x0Addr, __local_mem__ T* x1Addr, __local_mem__ T* dyAddr,
    __local_mem__ float* dxFAddr, __local_mem__ float* vecAddr, int64_t onceNum)
{
    uint16_t dim1VfTimes = onceNum / VF_LEN_FP32;
    uint32_t tail = onceNum % VF_LEN_FP32;
    uint16_t tailTimes = (tail > 0) ? 1 : 0;
    uint32_t halfU32 = static_cast<uint32_t>(half_);

    // ---- VF: load a/b/dy, clamp-mask in reg, compute da/db ----
    __VEC_SCOPE__
    {
        Reg::RegTensor<float> vregX0;
        Reg::RegTensor<float> vregX1;
        Reg::RegTensor<float> vregX0DeF;
        Reg::RegTensor<float> vregX1DeF;
        Reg::RegTensor<float> vregDY;
        Reg::RegTensor<float> daReg;
        Reg::RegTensor<float> dbReg;

        Reg::MaskReg maskAll = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
        Reg::MaskReg maskT = Reg::UpdateMask<float>(tail);

        for (uint16_t vfIdx = 0; vfIdx < dim1VfTimes + tailTimes; vfIdx++) {
            uint32_t offset = vfIdx * static_cast<uint32_t>(VF_LEN_FP32);
            Reg::MaskReg preg = (vfIdx < dim1VfTimes) ? maskAll : maskT;

            if constexpr (isInterleaved) {
                LoadInterleavedPair(xAddr, vfIdx, maskAll, vregX0, vregX1, vregX0DeF, vregX1DeF);
            } else {
                LoadOneTensor(x0Addr, vregX0DeF, preg, offset);
                LoadOneTensor(x1Addr, vregX1DeF, preg, offset);
            }
            LoadOneTensor(dyAddr, vregDY, preg, offset);

            VfGradMath(vregX0DeF, vregX1DeF, vregDY, daReg, dbReg, preg);

            if constexpr (isInterleaved) {
                Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM>(vecAddr + offset, daReg, preg);
                Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM>(vecAddr + halfU32 + offset, dbReg, preg);
            } else {
                Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM>(dxFAddr + offset, daReg, preg);
                Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM>(dxFAddr + halfU32 + offset, dbReg, preg);
            }
        }
    }
}

template <typename T, bool isInterleaved, bool isGroup>
__aicore__ inline void ClippedSwigluGradArch35Kernel<T, isInterleaved, isGroup>::ScatterInterleaved(
    LocalTensor<float>& dxFloatLocal, LocalTensor<float>& vecBufF)
{
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

template <typename T, bool isInterleaved, bool isGroup>
__aicore__ inline void ClippedSwigluGradArch35Kernel<T, isInterleaved, isGroup>::CastDxOut(
    LocalTensor<T>& dxDTypeLocal, LocalTensor<float>& dxFloatLocal)
{
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
__aicore__ inline void ClippedSwigluGradArch35Kernel<T, isInterleaved, isGroup>::ComputeVfGrad(
    LocalTensor<T>& xDTypeLocal, LocalTensor<T>& dyDTypeLocal, LocalTensor<T>& dxDTypeLocal, int64_t onceNum)
{
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

    VfGradComputeCore(xAddr, x0Addr, x1Addr, dyAddr, dxFAddr, vecAddr, onceNum);

    PipeBarrier<PIPE_V>();

    if constexpr (isInterleaved) {
        ScatterInterleaved(dxFloatLocal, vecBufF);
    }

    CastDxOut(dxDTypeLocal, dxFloatLocal);
}

} // namespace ClippedSwigluGradArch35Op
#endif // CLIPPED_SWIGLU_GRAD_KERNEL_H
