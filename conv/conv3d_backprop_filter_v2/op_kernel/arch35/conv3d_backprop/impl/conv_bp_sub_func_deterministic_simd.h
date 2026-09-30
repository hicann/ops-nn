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
 * \file conv_bp_sub_func_deterministic_simd.h
 * \brief 确定性计算的VB微码原语层: 索引表驱动的 Gather/Scatter 指令序列, 以及索引表构造与搬运上下文准备
 */

#ifndef CONV3D_BP_FILTER_SUB_FUNC_DETERMINISTIC_SIMD_H
#define CONV3D_BP_FILTER_SUB_FUNC_DETERMINISTIC_SIMD_H
#include "conv_bp_sub_func_deterministic_def.h"

namespace ConvolutionBackpropFunc {
template <class Intf>
static __aicore__ inline void CreateIndexBuf4BaseNUndivided(Intf* self, const uint32_t srcStride,
                                                            const uint64_t baseCin)
{
    auto indexBuf = self->ctx.vecBuf_.template GetWithOffset<int32_t>(
        AscendC::VECTOR_REG_WIDTH >> FLOAT_SHIFT_SIZE, AscendC::TOTAL_UB_SIZE - AscendC::VECTOR_REG_WIDTH);
    uint64_t C0_PER_REG = AscendC::VECTOR_REG_WIDTH / (sizeof(typename Intf::DstT) * self->ctx.tiling_->n0);
    uint64_t dstAddr = 0;
    for (uint64_t i = 0; i < C0_PER_REG; i++) {
        CreateVecIndex(indexBuf[dstAddr], (int32_t)0, BLOCK_CUBE); // 从0依次递增
        PipeBarrier<PIPE_V>();
        Adds(indexBuf[dstAddr], indexBuf[dstAddr], (int32_t)(srcStride * baseCin * i), BLOCK_CUBE); // srcStride * 16
        PipeBarrier<PIPE_V>();
        dstAddr += BLOCK_CUBE;
    }
}

template <class Intf>
static __aicore__ inline void CreateIndexBuf4BaseNDivided(Intf* self, const uint32_t mulStride,
                                                          const uint64_t addStride)
{
    auto indexBuf = self->ctx.vecBuf_.template GetWithOffset<int32_t>(
        AscendC::VECTOR_REG_WIDTH >> FLOAT_SHIFT_SIZE, AscendC::TOTAL_UB_SIZE - AscendC::VECTOR_REG_WIDTH);
    uint64_t C0_PER_REG = AscendC::VECTOR_REG_WIDTH / (sizeof(typename Intf::DstT) * self->ctx.tiling_->n0);
    uint64_t dstAddr = 0;
    for (uint64_t i = 0; i < C0_PER_REG; i++) {
        CreateVecIndex(indexBuf[dstAddr], (int32_t)0, BLOCK_CUBE);
        PipeBarrier<PIPE_V>();
        Muls(indexBuf[dstAddr], indexBuf[dstAddr], (int32_t)(mulStride), BLOCK_CUBE);
        PipeBarrier<PIPE_V>();
        Adds(indexBuf[dstAddr], indexBuf[dstAddr], (int32_t)(addStride * i), BLOCK_CUBE);
        PipeBarrier<PIPE_V>();
        dstAddr += BLOCK_CUBE;
    }
}

template <class Intf>
static __aicore__ inline ScatterDeterCtx<typename Intf::DstT> InitScatterDeterCtx(
    Intf* self, const CutDeterMinisticMNSize& cutMNSize, uint32_t hkDivisor)
{
    ScatterDeterCtx<typename Intf::DstT> ctx;
    ctx.baseCin = CeilHkWk(cutMNSize.curNSize, hkDivisor); // Cin1*Cin0
    ctx.coutNum = cutMNSize.curMSize;
    auto indexBuf = self->ctx.vecBuf_.template GetWithOffset<int32_t>(
        AscendC::VECTOR_REG_WIDTH >> FLOAT_SHIFT_SIZE, AscendC::TOTAL_UB_SIZE - AscendC::VECTOR_REG_WIDTH);
    ctx.indexPtr = (__ubuf__ uint32_t*)indexBuf[0].GetPhyAddr();
    ctx.srcPtr = (__ubuf__ typename Intf::DstT*)self->ctx.vecOutBuf_[0].GetPhyAddr();
    uint64_t srcSize = cutMNSize.curNSize * ctx.coutNum;
    ctx.dstPtr = (__ubuf__ typename Intf::DstT*)self->ctx.vecOutBuf_[srcSize].GetPhyAddr();
    ctx.c0PerReg = AscendC::VECTOR_REG_WIDTH / (sizeof(typename Intf::DstT) * self->ctx.tiling_->n0);
    ctx.sreg = 64;
    uint32_t tailNum = ctx.coutNum % ctx.c0PerReg;
    ctx.tailSreg = (tailNum == 0) ? SREG_PROC_NUM : tailNum * BLOCK_CUBE;
    return ctx;
}

template <typename DstT>
__simd_vf__ inline void GatherDepthwiseSimdVf(__ubuf__ DstT* srcPtr, __ubuf__ DstT* dstPtr, __ubuf__ uint32_t* indexPtr,
                                              uint32_t sreg, uint32_t sregTail, uint16_t indexLength, uint16_t loopSize,
                                              uint32_t hwkSrcStride, uint32_t strideHwk, uint32_t baseUseM)
{
    Reg::RegTensor<DstT> srcReg;
    Reg::RegTensor<uint32_t> vIndexReg;
    Reg::MaskReg preg = Reg::UpdateMask<DstT>(sreg);
    Reg::MaskReg pregTail = Reg::UpdateMask<DstT>(sregTail);
    Reg::LoadAlign(vIndexReg, indexPtr);

    Reg::UnalignRegForStore u0;
    for (uint16_t i = 0; i < static_cast<uint16_t>(baseUseM); i++) {
        uint32_t iBlockOffset = i * BLOCK_CUBE + i;
        uint32_t iDstOffset = i * strideHwk;
        uint16_t j = 0;
        for (j = 0; j < (uint16_t)(loopSize - 1); j++) {
            uint32_t srcoffset = iBlockOffset + j * hwkSrcStride;
            uint32_t dstoffset = iDstOffset + j * indexLength;
            Reg::Gather(srcReg, srcPtr + srcoffset, vIndexReg, preg);
            auto tmp = dstPtr + dstoffset;
            Reg::StoreUnAlign(tmp, srcReg, u0, indexLength);
            Reg::StoreUnAlignPost(tmp, u0, 0);
        }
        uint32_t srcoffset = iBlockOffset + j * hwkSrcStride;
        uint32_t dstoffset = iDstOffset + j * indexLength;
        Reg::Gather(srcReg, srcPtr + srcoffset, vIndexReg, pregTail);
        auto tmp = dstPtr + dstoffset;
        Reg::StoreUnAlign(tmp, srcReg, u0, indexLength);
        Reg::StoreUnAlignPost(tmp, u0, 0);
    }
}

template <typename DstT>
__simd_vf__ inline void Rearrange2GmScatterSimdVf(__ubuf__ DstT* srcPtr, __ubuf__ DstT* dstPtr,
                                                  __ubuf__ uint32_t* indexPtr, uint32_t sreg, uint32_t tailSreg,
                                                  uint16_t iterCin, uint16_t iterCout, uint8_t sregU8,
                                                  uint32_t coutDstStride, uint32_t hwkSrcStride, uint32_t cinSrcStride,
                                                  uint32_t cinDstStride, uint16_t indexLength,
                                                  bool enableVecRegWidthMin, uint32_t hwK)
{
    Reg::RegTensor<DstT> srcReg;
    Reg::RegTensor<uint32_t> vIndexReg;
    Reg::MaskReg preg = Reg::UpdateMask<DstT>(sreg);
    Reg::MaskReg pregTail = Reg::UpdateMask<DstT>(tailSreg);
    Reg::LoadAlign(vIndexReg, indexPtr);

    for (uint16_t cinIndex = 0; cinIndex < iterCin; cinIndex++) {
        uint32_t srcOffsetCin = enableVecRegWidthMin ?
                                    ((cinIndex / DOUBLE) * cinSrcStride + (cinIndex % DOUBLE) * indexLength) :
                                    (cinIndex * cinSrcStride);
        uint32_t dstOffsetCin = cinIndex * cinDstStride;
        for (uint16_t hwkIndex = 0; hwkIndex < static_cast<uint16_t>(hwK); hwkIndex++) {
            uint32_t hwkSrcOffset = hwkIndex * hwkSrcStride;
            uint32_t hwkDstOffset = hwkIndex;
            uint16_t coutIndex = 0;
            for (coutIndex = 0; coutIndex < static_cast<uint16_t>(iterCout - 1); coutIndex++) {
                uint32_t srcOffset = srcOffsetCin + hwkSrcOffset + coutIndex * sregU8;
                uint32_t dstOffset = dstOffsetCin + hwkDstOffset + coutIndex * coutDstStride;
                Reg::LoadAlign(srcReg, srcPtr + srcOffset);
                Reg::Scatter(dstPtr + dstOffset, srcReg, vIndexReg, preg);
            }
            uint32_t srcOffset = srcOffsetCin + hwkSrcOffset + coutIndex * sregU8;
            uint32_t dstOffset = dstOffsetCin + hwkDstOffset + coutIndex * coutDstStride;

            Reg::LoadAlign(srcReg, srcPtr + srcOffset);
            Reg::Scatter(dstPtr + dstOffset, srcReg, vIndexReg, pregTail);
        }
    }
}

template <typename DstT>
__simd_vf__ inline void Rearrange2GmScatterBaseNUndividedSimdVf(__ubuf__ DstT* srcPtr, __ubuf__ DstT* dstPtr,
                                                                __ubuf__ uint32_t* indexPtr, uint32_t sreg,
                                                                uint32_t tailSreg, uint16_t iterWk, uint16_t iterCout,
                                                                uint16_t wkSrcStride, uint16_t coutDstStride)
{
    Reg::RegTensor<DstT> srcReg;
    Reg::RegTensor<uint32_t> vIndexReg;
    Reg::MaskReg preg = Reg::UpdateMask<DstT>(sreg);
    Reg::MaskReg pregTail = Reg::UpdateMask<DstT>(tailSreg);
    Reg::LoadAlign(vIndexReg, indexPtr);

    for (uint16_t wkIndex = 0; wkIndex < iterWk; wkIndex++) {
        uint32_t wkSrcOffset = wkIndex * wkSrcStride;
        uint32_t wkDstOffset = wkIndex * BLOCK_CUBE;
        uint16_t coutIndex = 0;
        for (coutIndex = 0; coutIndex < static_cast<uint16_t>(iterCout - 1); coutIndex++) {
            uint32_t srcOffset = coutIndex * SREG_PROC_NUM + wkSrcOffset;
            uint32_t dstOffset = coutIndex * coutDstStride + wkDstOffset;
            Reg::LoadAlign(srcReg, srcPtr + srcOffset);
            Reg::Scatter(dstPtr + dstOffset, srcReg, vIndexReg, preg);
        }
        uint32_t srcOffsetTail = coutIndex * SREG_PROC_NUM + wkSrcOffset;
        uint32_t dstOffsetTail = coutIndex * coutDstStride + wkDstOffset;
        Reg::LoadAlign(srcReg, srcPtr + srcOffsetTail);
        Reg::Scatter(dstPtr + dstOffsetTail, srcReg, vIndexReg, pregTail);
    }
}

template <typename DstT>
__simd_vf__ inline void Rearrange2GmScatterDeterForNDHWCSimdVf(__ubuf__ DstT* srcPtr, __ubuf__ DstT* dstPtr,
                                                               __ubuf__ uint32_t* indexPtr, uint32_t sreg,
                                                               uint32_t tailSreg, uint16_t iterCin, uint16_t iterCout,
                                                               uint16_t iterHkWk, uint16_t hkWkSrcStride,
                                                               uint16_t cinSrcStride, uint16_t coutDstStride,
                                                               uint16_t hkWkDstStride)
{
    Reg::RegTensor<DstT> srcReg;
    Reg::RegTensor<uint32_t> vIndexReg;
    Reg::MaskReg preg = Reg::UpdateMask<DstT>(sreg);
    Reg::MaskReg pregTail = Reg::UpdateMask<DstT>(tailSreg);
    Reg::LoadAlign(vIndexReg, indexPtr);

    for (uint16_t cinIndex = 0; cinIndex < iterCin; cinIndex++) {
        uint32_t cinSrcOffset = cinIndex * cinSrcStride;
        uint32_t cinDstOffset = cinIndex * BLOCK_CUBE;
        for (uint16_t hkWkIndex = 0; hkWkIndex < iterHkWk; hkWkIndex++) {
            uint32_t hkWkSrcOffset = hkWkIndex * hkWkSrcStride;
            uint32_t hkWkDstOffset = hkWkIndex * hkWkDstStride;
            uint16_t coutIndex = 0;
            for (coutIndex = 0; coutIndex < static_cast<uint16_t>(iterCout - 1); coutIndex++) {
                uint32_t srcOffset = coutIndex * SREG_PROC_NUM + hkWkSrcOffset + cinSrcOffset;
                uint32_t dstOffset = coutIndex * coutDstStride + cinDstOffset + hkWkDstOffset;
                Reg::LoadAlign(srcReg, srcPtr + srcOffset);
                Reg::Scatter(dstPtr + dstOffset, srcReg, vIndexReg, preg);
            }
            uint32_t srcOffsetTail = coutIndex * SREG_PROC_NUM + hkWkSrcOffset + cinSrcOffset;
            uint32_t dstOffsetTail = coutIndex * coutDstStride + cinDstOffset + hkWkDstOffset;
            Reg::LoadAlign(srcReg, srcPtr + srcOffsetTail);
            Reg::Scatter(dstPtr + dstOffsetTail, srcReg, vIndexReg, pregTail);
        }
    }
}

template <typename DstT>
__simd_vf__ inline void Rearrange2GmScatterDeterForDHWCNSimdVf(__ubuf__ DstT* srcPtr, __ubuf__ DstT* dstPtr,
                                                               __ubuf__ uint32_t* indexPtr, uint32_t sreg,
                                                               uint32_t tailSreg, uint16_t iterCin, uint16_t iterCout,
                                                               uint16_t hkWkSrcStride, uint16_t cinSrcStride,
                                                               uint32_t srcStride, uint64_t baseCin, uint64_t coutNum,
                                                               uint32_t c0PerReg)
{
    Reg::RegTensor<DstT> srcReg;
    Reg::RegTensor<uint32_t> vIndexReg;
    Reg::MaskReg preg = Reg::UpdateMask<DstT>(sreg);
    Reg::MaskReg pregTail = Reg::UpdateMask<DstT>(tailSreg);
    Reg::LoadAlign(vIndexReg, indexPtr);

    for (uint32_t cinIndex = 0; cinIndex < iterCin; cinIndex++) {
        uint32_t cinSrcOffset = cinIndex * cinSrcStride;
        uint64_t cinDstOffset = cinIndex * BLOCK_CUBE * coutNum;
        for (uint32_t hkWkIndex = 0; hkWkIndex < srcStride; hkWkIndex++) {
            uint32_t hkWkSrcOffset = hkWkIndex * hkWkSrcStride;
            uint64_t hkWkDstOffset = hkWkIndex * baseCin * coutNum;
            uint16_t coutIndex = 0;
            for (coutIndex = 0; coutIndex < static_cast<uint16_t>(iterCout - 1); coutIndex++) {
                uint32_t srcOffset = coutIndex * SREG_PROC_NUM + hkWkSrcOffset + cinSrcOffset;
                uint32_t dstOffset = hkWkDstOffset + cinDstOffset + coutIndex * c0PerReg;
                Reg::LoadAlign(srcReg, srcPtr + srcOffset);
                Reg::Scatter(dstPtr + dstOffset, srcReg, vIndexReg, preg);
            }
            uint32_t srcOffsetTail = coutIndex * SREG_PROC_NUM + hkWkSrcOffset + cinSrcOffset;
            uint32_t dstOffsetTail = hkWkDstOffset + cinDstOffset + coutIndex * c0PerReg;
            Reg::LoadAlign(srcReg, srcPtr + srcOffsetTail);
            Reg::Scatter(dstPtr + dstOffsetTail, srcReg, vIndexReg, pregTail);
        }
    }
}

template <typename DstT>
__simd_vf__ inline void Rearrange2GmScatterDeterSimdVf(__ubuf__ DstT* srcPtr, __ubuf__ DstT* dstPtr,
                                                       __ubuf__ uint32_t* indexPtr, uint32_t sreg, uint32_t tailSreg,
                                                       uint16_t iterCin, uint16_t iterCout, uint16_t hkWkSrcStride,
                                                       uint16_t cinSrcStride, uint16_t cinDstStride,
                                                       uint16_t coutDstStride, uint32_t hwK)
{
    Reg::RegTensor<DstT> srcReg;
    Reg::RegTensor<uint32_t> vIndexReg;
    Reg::MaskReg preg = Reg::UpdateMask<DstT>(sreg);
    Reg::MaskReg pregTail = Reg::UpdateMask<DstT>(tailSreg);
    Reg::LoadAlign(vIndexReg, indexPtr);

    for (uint16_t cinIndex = 0; cinIndex < iterCin; cinIndex++) {
        uint32_t cinSrcOffset = cinIndex * cinSrcStride;
        uint32_t cinDstOffset = cinIndex * cinDstStride;
        for (uint16_t hkWkIndex = 0; hkWkIndex < static_cast<uint16_t>(hwK); hkWkIndex++) {
            uint32_t hkWkSrcOffset = hkWkIndex * hkWkSrcStride;
            uint32_t hkWkDstOffset = hkWkIndex;
            uint16_t coutIndex = 0;
            for (coutIndex = 0; coutIndex < static_cast<uint16_t>(iterCout - 1); coutIndex++) {
                uint32_t srcOffset = coutIndex * SREG_PROC_NUM + hkWkSrcOffset + cinSrcOffset;
                uint32_t dstOffset = coutIndex * coutDstStride + hkWkDstOffset + cinDstOffset;
                Reg::LoadAlign(srcReg, srcPtr + srcOffset);
                Reg::Scatter(dstPtr + dstOffset, srcReg, vIndexReg, preg);
            }
            uint32_t srcOffsetTail = coutIndex * SREG_PROC_NUM + hkWkSrcOffset + cinSrcOffset;
            uint32_t dstOffsetTail = coutIndex * coutDstStride + hkWkDstOffset + cinDstOffset;
            Reg::LoadAlign(srcReg, srcPtr + srcOffsetTail);
            Reg::Scatter(dstPtr + dstOffsetTail, srcReg, vIndexReg, pregTail);
        }
    }
}
} // namespace ConvolutionBackpropFunc

#endif
