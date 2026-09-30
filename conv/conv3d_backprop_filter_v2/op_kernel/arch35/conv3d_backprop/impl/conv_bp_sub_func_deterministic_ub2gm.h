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
 * \file conv_bp_sub_func_deterministic_ub2gm.h
 * \brief 确定性计算的UB->GM搬运编排层: 驱动微码、计算UB偏移并落盘
 */

#ifndef CONV3D_BP_FILTER_SUB_FUNC_DETERMINISTIC_UB2GM_H
#define CONV3D_BP_FILTER_SUB_FUNC_DETERMINISTIC_UB2GM_H
#include "conv_bp_sub_func_deterministic_def.h"
#include "conv_bp_sub_func_deterministic_simd.h"

namespace ConvolutionBackpropFunc {

template <class Intf>
static __aicore__ inline void CalcL0cParams(Intf* self, bool tailCinExist)
{
    bool isNLoop = (self->ctx.lastNIdx_ != self->ctx.curNIdx_);
    // 若cin有尾块，在cin循环开始时对其进行赋值初始化
    if (tailCinExist && isNLoop && (self->ctx.curNIdx_ % self->ctx.cinHkWkLoop_ == 0)) {
        self->ctx.cinRemainLen_ = self->ctx.singleShapeCin_;
    }

    // LOC的搬运次数是nIter*mIter，当mIter循环时，head不应该发生变化
    if (isNLoop) {
        // 当N循环时，更新head和cinRemainLen的值，否则还按照上次循环的值作为head和cinRemainLen的值
        self->ctx.nLoopHead_ = self->ctx.head_;
        self->ctx.nLoopCinRemainLen_ = self->ctx.cinRemainLen_;
    }
    self->ctx.head_ = self->ctx.nLoopHead_;
    self->ctx.cinRemainLen_ = self->ctx.nLoopCinRemainLen_;
    return;
}

template <class Intf>
static __aicore__ inline void GatherDepthwise(Intf* self, uint32_t hwkLength, uint32_t srcSize, uint32_t strideHwk)
{
    uint16_t vLoop = AscendC::VECTOR_REG_WIDTH / sizeof(typename Intf::DstT);
    uint16_t loopSize = CeilDivision(hwkLength, vLoop);
    uint16_t indexLength = vLoop > hwkLength ? hwkLength : vLoop;

    uint32_t coutCin0 = self->ctx.baseUseM_ * BLOCK_CUBE;
    auto indexBuf = self->ctx.vecBuf_.template GetWithOffset<int32_t>(
        AscendC::VECTOR_REG_WIDTH >> FLOAT_SHIFT_SIZE, AscendC::TOTAL_UB_SIZE - AscendC::VECTOR_REG_WIDTH);
    auto indexPtr = (__ubuf__ uint32_t*)indexBuf[0].GetPhyAddr();
    auto srcPtr = (__ubuf__ typename Intf::DstT*)self->ctx.vecOutBuf_[0].GetPhyAddr();
    auto dstPtr = (__ubuf__ typename Intf::DstT*)self->ctx.vecOutBuf_[srcSize].GetPhyAddr();

    CreateVecIndex(indexBuf[0], (int32_t)0, indexLength); // 从0依次递增到indexLength
    PipeBarrier<PIPE_V>();
    Muls(indexBuf[0], indexBuf[0], (int32_t)coutCin0, indexLength); // 每个元素相乘coutCin0
    PipeBarrier<PIPE_V>();

    uint32_t sreg = indexLength;
    uint32_t sregTail = (hwkLength % vLoop == 0) ? indexLength : (hwkLength % vLoop);
    uint32_t hwkSrcStride = indexLength * coutCin0;
    GatherDepthwiseSimdVf<typename Intf::DstT>(srcPtr, dstPtr, indexPtr, sreg, sregTail, indexLength, loopSize,
                                               hwkSrcStride, strideHwk, self->ctx.baseUseM_);
    event_t eventIdVecToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIdVecToMte3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVecToMte3);
}

template <class Intf>
static __aicore__ inline void Rearrange2GmDepthwise(Intf* self, const GlobalTensor<typename Intf::DstT>& output)
{
    LoopModeParams loopParams;
    DataCopyExtParams ub2GmParams;
    bool baseNNoAllHkWk = ((self->ctx.tiling_->baseN >> 4) % self->ctx.hwK_ != 0);

    constexpr uint8_t ALIGN_BYTE = 8; // UB 32Byte对齐
    uint32_t srcSize = self->ctx.baseUseN_ * self->ctx.baseUseM_;

    int64_t dstGmOffset = 0;
    uint32_t nValue = self->ctx.hwK_;

    if (baseNNoAllHkWk) {
        constexpr uint64_t baseCin = 16;
        bool tailCinExist = (self->ctx.singleShapeCin_ % baseCin != 0);
        CalcL0cParams(self, tailCinExist);
        uint32_t c1hkwk = ShiftDivM0(self->ctx.baseUseN_, baseCin);
        self->ctx.tail_ = (c1hkwk + self->ctx.head_) > self->ctx.hwK_ ? self->ctx.hwK_ : (c1hkwk + self->ctx.head_);

        nValue = self->ctx.tail_ - self->ctx.head_; // 表示每次转多少hkwk
        dstGmOffset = self->ctx.head_;

        self->ctx.head_ = self->ctx.tail_ == self->ctx.hwK_ ? 0 : self->ctx.tail_;
        self->ctx.lastNIdx_ = self->ctx.curNIdx_;
    }

    uint32_t strideHwk = Ceil((self->ctx.curSingleCoreDk_ * nValue), ALIGN_BYTE) * ALIGN_BYTE;
    GatherDepthwise(self, nValue, srcSize, strideHwk);

    loopParams.loop2Size = 1;
    loopParams.loop1Size = self->ctx.baseUseM_;
    loopParams.loop2SrcStride = loopParams.loop1SrcStride;
    loopParams.loop2DstStride = loopParams.loop1DstStride;
    loopParams.loop1SrcStride = strideHwk * DST_DTYPE_BYTES;
    loopParams.loop1DstStride = self->ctx.dhwK_ * DST_DTYPE_BYTES;

    SetLoopModePara(loopParams, DataCopyMVType::UB_TO_OUT);

    ub2GmParams.blockLen = nValue * DST_DTYPE_BYTES;     // len
    ub2GmParams.blockCount = self->ctx.curSingleCoreDk_; // num
    ub2GmParams.srcStride = 0;                           // compact mode ignore srcstride
    ub2GmParams.dstStride = 0;

    DataCopyPad<typename Intf::DstT, PaddingMode::Compact>(output[dstGmOffset], self->ctx.vecOutBuf_[srcSize],
                                                           ub2GmParams);
    ResetLoopModePara(DataCopyMVType::UB_TO_OUT);
}

template <class Intf>
static __aicore__ inline void NormalGroupDataCopyPadDkEqOne(Intf* self, const GlobalTensor<typename Intf::DstT>& output,
                                                            uint32_t srcStride, uint32_t coutNum)
{
    uint32_t srcSize = self->ctx.baseUseN_ * coutNum;
    uint32_t coutPerGroup = self->ctx.tiling_->cout / self->ctx.tiling_->group;
    uint32_t cinPerGroup = self->ctx.tiling_->cin / self->ctx.tiling_->group;
    uint32_t baseCin = Ceil(Ceil(self->ctx.baseUseN_, self->ctx.curSingleCoreDk_), self->ctx.hwK_);
    LoopModeParams loopParams;
    loopParams.loop2Size = 1;
    loopParams.loop1Size = self->ctx.baseUseM_ / coutPerGroup;
    loopParams.loop1SrcStride = (baseCin * self->ctx.curSingleCoreDk_ * self->ctx.hwK_ * coutPerGroup +
                                 cinPerGroup * self->ctx.hwK_) *
                                DST_DTYPE_BYTES;
    loopParams.loop1DstStride = self->ctx.hwK_ * DST_DTYPE_BYTES * cinPerGroup * coutPerGroup;

    loopParams.loop2SrcStride = loopParams.loop1SrcStride;
    loopParams.loop2DstStride = loopParams.loop1DstStride;

    SetLoopModePara(loopParams, DataCopyMVType::UB_TO_OUT);

    DataCopyExtParams ub2GmParams;
    constexpr uint8_t ALIGN_BYTE = 32;                                     // UB 32Byte对齐
    ub2GmParams.blockLen = self->ctx.hwK_ * DST_DTYPE_BYTES * cinPerGroup; // len:byte
    ub2GmParams.blockCount = coutPerGroup;
    ub2GmParams.srcStride = (baseCin - cinPerGroup) * self->ctx.hwK_ * DST_DTYPE_BYTES / ALIGN_BYTE;
    ub2GmParams.dstStride = 0;
    DataCopyPad<typename Intf::DstT>(output[0], self->ctx.vecOutBuf_[srcSize], ub2GmParams);
    ResetLoopModePara(DataCopyMVType::UB_TO_OUT);
}

template <class Intf>
static __aicore__ inline void NormalGroupDataCopyPad(Intf* self, const GlobalTensor<typename Intf::DstT>& output,
                                                     uint32_t srcStride, uint32_t coutNum)
{
    uint32_t coutPerGroup = self->ctx.tiling_->cout / self->ctx.tiling_->group;
    uint32_t cinPerGroup = self->ctx.tiling_->cin / self->ctx.tiling_->group;
    uint32_t srcSize = self->ctx.baseUseN_ * coutNum;
    uint32_t baseCin = Ceil(Ceil(self->ctx.baseUseN_, self->ctx.curSingleCoreDk_), self->ctx.hwK_);

    LoopModeParams loopParams;
    loopParams.loop2Size = self->ctx.baseUseM_ / coutPerGroup;
    loopParams.loop1Size = coutPerGroup;
    loopParams.loop2SrcStride = (baseCin * self->ctx.curSingleCoreDk_ * srcStride * coutPerGroup +
                                 cinPerGroup * srcStride) *
                                DST_DTYPE_BYTES;
    loopParams.loop2DstStride = self->ctx.dhwK_ * DST_DTYPE_BYTES * cinPerGroup * coutPerGroup;

    loopParams.loop1SrcStride = baseCin * self->ctx.curSingleCoreDk_ * srcStride * DST_DTYPE_BYTES;
    loopParams.loop1DstStride = self->ctx.dhwK_ * DST_DTYPE_BYTES * cinPerGroup;

    SetLoopModePara(loopParams, DataCopyMVType::UB_TO_OUT);

    DataCopyExtParams ub2GmParams;
    constexpr uint8_t ALIGN_BYTE = 32;                                                    // UB 32Byte对齐
    ub2GmParams.blockLen = self->ctx.hwK_ * DST_DTYPE_BYTES * self->ctx.curSingleCoreDk_; // len:byte
    ub2GmParams.blockCount = cinPerGroup;
    ub2GmParams.srcStride = (srcStride - self->ctx.hwK_) * DST_DTYPE_BYTES * self->ctx.curSingleCoreDk_ / ALIGN_BYTE;
    // GM单位：byte
    ub2GmParams.dstStride = (self->ctx.tiling_->dk - self->ctx.curSingleCoreDk_) * self->ctx.hwK_ * DST_DTYPE_BYTES;
    DataCopyPad<typename Intf::DstT>(output[0], self->ctx.vecOutBuf_[srcSize], ub2GmParams);
    ResetLoopModePara(DataCopyMVType::UB_TO_OUT);
}

template <class Intf>
static __aicore__ inline void Rearrange2GmScatter(Intf* self, uint32_t srcStride, uint16_t loopSize,
                                                  uint16_t indexLength, uint32_t coutNum)
{
    uint32_t baseCin = Ceil(Ceil(self->ctx.baseUseN_, self->ctx.curSingleCoreDk_), self->ctx.hwK_);
    auto indexBuf = self->ctx.vecBuf_.template GetWithOffset<int32_t>(
        AscendC::VECTOR_REG_WIDTH >> FLOAT_SHIFT_SIZE, AscendC::TOTAL_UB_SIZE - AscendC::VECTOR_REG_WIDTH);
    auto indexPtr = (__ubuf__ uint32_t*)indexBuf[0].GetPhyAddr();
    auto srcPtr = (__ubuf__ typename Intf::DstT*)self->ctx.vecOutBuf_[0].GetPhyAddr();
    uint32_t srcSize = self->ctx.baseUseN_ * coutNum;
    auto dstPtr = (__ubuf__ typename Intf::DstT*)self->ctx.vecOutBuf_[srcSize].GetPhyAddr();

    bool enableVecRegWidthMin = (indexLength < BLOCK_CUBE);
    uint32_t sreg = AscendC::VECTOR_REG_WIDTH / sizeof(typename Intf::DstT);
    uint8_t sregU8 = sreg > BLOCK_CUBE ? sreg : BLOCK_CUBE;
    uint16_t iterCin = baseCin / indexLength;
    uint16_t iterCout = Ceil(coutNum, loopSize);
    uint32_t coutDstStride = baseCin * loopSize * srcStride;
    uint32_t hwkSrcStride = coutNum * BLOCK_CUBE;
    uint32_t cinSrcStride = hwkSrcStride * self->ctx.hwK_;
    uint32_t cinDstStride = indexLength * srcStride;

    uint32_t tailNum = coutNum % loopSize;
    uint32_t tailSreg = (tailNum == 0) ? sreg : tailNum * BLOCK_CUBE;

    Rearrange2GmScatterSimdVf<typename Intf::DstT>(srcPtr, dstPtr, indexPtr, sreg, tailSreg, iterCin, iterCout, sregU8,
                                                   coutDstStride, hwkSrcStride, cinSrcStride, cinDstStride, indexLength,
                                                   enableVecRegWidthMin, self->ctx.hwK_);
}

template <class Intf>
static __aicore__ inline void Rearrange2GmNormalGroup(Intf* self, const GlobalTensor<typename Intf::DstT>& output,
                                                      bool isCoutNumAligned)
{
    uint16_t vLoop = AscendC::VECTOR_REG_WIDTH / sizeof(typename Intf::DstT);
    uint16_t loopSize = CeilDivision(vLoop, BLOCK_CUBE);
    uint16_t indexLength = vLoop > BLOCK_CUBE ? BLOCK_CUBE : vLoop;

    uint32_t cinPerGroup = self->ctx.tiling_->cin / self->ctx.tiling_->group;
    uint32_t baseCin = Ceil(Ceil(self->ctx.baseUseN_, self->ctx.curSingleCoreDk_), self->ctx.hwK_);
    auto indexBuf = self->ctx.vecBuf_.template GetWithOffset<int32_t>(
        AscendC::VECTOR_REG_WIDTH >> FLOAT_SHIFT_SIZE, AscendC::TOTAL_UB_SIZE - AscendC::VECTOR_REG_WIDTH);

    uint32_t srcStride = self->ctx.hwK_;
    constexpr uint32_t ALIGN_BYTE = 8; // loop_src_stride 32 byte align

    bool flag = (self->ctx.tiling_->dk != 1 || (cinPerGroup * self->ctx.hwK_) % ALIGN_BYTE != 0);
    if (flag) {
        srcStride = Ceil(self->ctx.hwK_, ALIGN_BYTE) * ALIGN_BYTE;
    }

    uint16_t dstOffset = 0;
    for (uint16_t i = 0; i < loopSize; i++) {
        CreateVecIndex(indexBuf[dstOffset], (int32_t)0, indexLength); // 从0依次递增
        PipeBarrier<PIPE_V>();
        Muls(indexBuf[dstOffset], indexBuf[dstOffset], (int32_t)(srcStride), indexLength);
        PipeBarrier<PIPE_V>();
        Adds(indexBuf[dstOffset], indexBuf[dstOffset], (int32_t)(baseCin * i * srcStride), indexLength);
        PipeBarrier<PIPE_V>();
        dstOffset += indexLength;
    }
    // group重排有两个通路，NO_STREAMK: l0c2ub, STREAMK: gm2ub
    // l0c2ub时候，coutNum是16对齐的，而gm2ub时候coutNum和baseUseM一致
    uint32_t coutNum = isCoutNumAligned ? ShiftCeilM0(self->ctx.baseUseM_, BLOCK_CUBE) * BLOCK_CUBE :
                                          self->ctx.baseUseM_;
    Rearrange2GmScatter(self, srcStride, loopSize, indexLength, coutNum);

    event_t eventIdVecToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIdVecToMte3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVecToMte3);

    if (!flag) {
        NormalGroupDataCopyPadDkEqOne(self, output, srcStride, coutNum);
    } else {
        NormalGroupDataCopyPad(self, output, srcStride, coutNum);
    }
}

template <class Intf>
static __aicore__ inline void Rearrange2Gm(Intf* self, const GlobalTensor<typename Intf::DstT>& output,
                                           uint8_t enAtomic = 0, bool isCoutNumAligned = 1)
{
    if ASCEND_IS_AIC {
        return;
    }
    if (GetSubBlockIdx() > 0) {
        return;
    }

    if (enAtomic == 1) {
        SetAtomicAdd<typename Intf::DstT>();
    }

    self->ctx.vecOutBuf_ = self->ctx.vecBuf_.template Get<typename Intf::DstT>();
    if (self->ctx.tiling_->group != self->ctx.tiling_->cin || self->ctx.tiling_->group != self->ctx.tiling_->cout) {
        Rearrange2GmNormalGroup(self, output, isCoutNumAligned);
    } else if (self->ctx.tiling_->group == self->ctx.tiling_->cin &&
               self->ctx.tiling_->group == self->ctx.tiling_->cout) {
        Rearrange2GmDepthwise(self, output);
    }

    if (enAtomic == 1) {
        SetAtomicNone();
    }
}

template <class Intf>
static __aicore__ inline void Rearrange2GmScatterBaseNUndivided(Intf* self, const uint32_t srcStride,
                                                                const GlobalTensor<typename Intf::DstT>& output,
                                                                const CutDeterMinisticMNSize& cutMNSize)
{
    uint64_t coutNum = cutMNSize.curMSize;
    auto indexBuf = self->ctx.vecBuf_.template GetWithOffset<int32_t>(
        AscendC::VECTOR_REG_WIDTH >> FLOAT_SHIFT_SIZE, AscendC::TOTAL_UB_SIZE - AscendC::VECTOR_REG_WIDTH);
    auto srcPtr = (__ubuf__ typename Intf::DstT*)self->ctx.vecOutBuf_[0].GetPhyAddr();
    auto indexPtr = (__ubuf__ uint32_t*)indexBuf[0].GetPhyAddr();
    uint64_t srcSize = cutMNSize.curNSize * coutNum;
    auto dstPtr = (__ubuf__ typename Intf::DstT*)self->ctx.vecOutBuf_[srcSize].GetPhyAddr();
    uint32_t C0_PER_REG = AscendC::VECTOR_REG_WIDTH / (sizeof(typename Intf::DstT) * self->ctx.tiling_->n0);
    uint32_t sreg = 64; // 64: reg处理的数目
    uint32_t tailNum = coutNum % C0_PER_REG;
    uint32_t tailSreg = (tailNum == 0) ? SREG_PROC_NUM : tailNum * BLOCK_CUBE;

    CreateIndexBuf4BaseNUndivided(self, srcStride, BLOCK_CUBE);
    uint16_t iterWk = ShiftCeilM0(cutMNSize.curNSize, BLOCK_CUBE);
    uint16_t iterCout = Ceil(coutNum, C0_PER_REG);
    uint16_t wkSrcStride = coutNum * BLOCK_CUBE;
    uint16_t coutDstStride = C0_PER_REG * cutMNSize.curNSize;

    Rearrange2GmScatterBaseNUndividedSimdVf<typename Intf::DstT>(srcPtr, dstPtr, indexPtr, sreg, tailSreg, iterWk,
                                                                 iterCout, wkSrcStride, coutDstStride);

    event_t eventIdVecToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIdVecToMte3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVecToMte3);
    if constexpr (Intf::Config::dType::format == ConvolutionBackprop::CubeFormat::NCDHW) {
        DataCopyPadBaseNUndivided(self, output, srcStride, cutMNSize);
    } else {
        DataCopyPadBaseNUndividedForNDHWC(self, output, srcStride, cutMNSize);
    }
}

template <class Intf>
static __aicore__ inline void Rearrange2GmScatterDeterForNDHWC(Intf* self, const uint32_t srcStride,
                                                               const GlobalTensor<typename Intf::DstT>& output,
                                                               const CutDeterMinisticMNSize& cutMNSize)
{
    auto ctx = InitScatterDeterCtx<Intf>(self, cutMNSize, self->ctx.hwK_);

    CreateIndexBuf4BaseNUndivided(self, self->ctx.hwK_, ctx.baseCin);
    uint16_t iterCin = ShiftDivM0(ctx.baseCin, BLOCK_CUBE);
    uint16_t iterCout = Ceil(ctx.coutNum, ctx.c0PerReg);
    uint16_t iterHkWk = static_cast<uint16_t>(self->ctx.hwK_);
    uint16_t hkWkSrcStride = ctx.coutNum * BLOCK_CUBE;
    uint16_t cinSrcStride = hkWkSrcStride * self->ctx.hwK_;
    uint16_t coutDstStride = ctx.c0PerReg * cutMNSize.curNSize;
    uint16_t hkWkDstStride = ctx.baseCin;

    Rearrange2GmScatterDeterForNDHWCSimdVf<typename Intf::DstT>(
        ctx.srcPtr, ctx.dstPtr, ctx.indexPtr, ctx.sreg, ctx.tailSreg, iterCin, iterCout, iterHkWk, hkWkSrcStride,
        cinSrcStride, coutDstStride, hkWkDstStride);

    event_t eventIdVecToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIdVecToMte3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVecToMte3);
    DataCopyPadDkEqOneForNDHWC(self, output, cutMNSize, ctx.baseCin);
}

template <class Intf>
static __aicore__ inline void Rearrange2GmScatterDeterForDHWCN(Intf* self, const uint32_t srcStride,
                                                               const GlobalTensor<typename Intf::DstT>& output,
                                                               const CutDeterMinisticMNSize& cutMNSize)
{
    auto ctx = InitScatterDeterCtx<Intf>(self, cutMNSize, srcStride);

    CreateIndexBuf4BaseNDivided(self, ctx.coutNum, 1);
    uint16_t iterCin = ShiftDivM0(ctx.baseCin, BLOCK_CUBE);
    uint16_t iterCout = Ceil(ctx.coutNum, ctx.c0PerReg);
    uint16_t hkWkSrcStride = ctx.coutNum * BLOCK_CUBE;
    uint16_t cinSrcStride = hkWkSrcStride * srcStride;

    Rearrange2GmScatterDeterForDHWCNSimdVf<typename Intf::DstT>(
        ctx.srcPtr, ctx.dstPtr, ctx.indexPtr, ctx.sreg, ctx.tailSreg, iterCin, iterCout, hkWkSrcStride, cinSrcStride,
        srcStride, ctx.baseCin, ctx.coutNum, ctx.c0PerReg);

    event_t eventIdVecToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIdVecToMte3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVecToMte3);
    DataCopyPadDkEqOneForDHWCN(self, output, cutMNSize, srcStride);
}

template <class Intf>
static __aicore__ inline void Rearrange2GmScatterDeter(Intf* self, const uint32_t srcStride,
                                                       const GlobalTensor<typename Intf::DstT>& output,
                                                       const CutDeterMinisticMNSize& cutMNSize)
{
    auto ctx = InitScatterDeterCtx<Intf>(self, cutMNSize, self->ctx.hwK_);

    CreateIndexBuf4BaseNDivided(self, srcStride, ctx.baseCin * srcStride);
    // BaseN整除hwk场景，其值较小，均在uint16_t范围内
    uint16_t iterCin = ShiftDivM0(ctx.baseCin, BLOCK_CUBE);
    uint16_t iterCout = Ceil(ctx.coutNum, ctx.c0PerReg);
    uint16_t hkWkSrcStride = ctx.coutNum * BLOCK_CUBE;
    uint16_t cinSrcStride = hkWkSrcStride * self->ctx.hwK_;
    uint16_t cinDstStride = BLOCK_CUBE * srcStride;
    uint16_t coutDstStride = ctx.c0PerReg * cutMNSize.curNSize;

    Rearrange2GmScatterDeterSimdVf<typename Intf::DstT>(ctx.srcPtr, ctx.dstPtr, ctx.indexPtr, ctx.sreg, ctx.tailSreg,
                                                        iterCin, iterCout, hkWkSrcStride, cinSrcStride, cinDstStride,
                                                        coutDstStride, self->ctx.hwK_);

    event_t eventIdVecToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIdVecToMte3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVecToMte3);
    DataCopyPadDkEqOne(self, output, cutMNSize, ctx.baseCin);
}

template <class Intf>
static __aicore__ inline void DataCopyPadBaseNUndivided(Intf* self, const GlobalTensor<typename Intf::DstT>& output,
                                                        const uint32_t srcStride,
                                                        const CutDeterMinisticMNSize& cutMNSize)
{
    uint64_t srcSize = cutMNSize.curNSize * cutMNSize.curMSize;
    uint64_t wkCnt = ShiftCeilM0(self->ctx.tiling_->baseN, BLOCK_CUBE) < self->ctx.hwK_ ?
                         ShiftCeilM0(self->ctx.tiling_->baseN, BLOCK_CUBE) :
                         self->ctx.hwK_;
    LoopModeParams loopParams;
    DataCopyExtParams ub2GmParams;
    loopParams.loop2Size = 1;
    loopParams.loop1Size = cutMNSize.curMSize;
    loopParams.loop1SrcStride = BLOCK_CUBE * srcStride * DST_DTYPE_BYTES;
    loopParams.loop1DstStride = self->ctx.tiling_->cin1G * self->ctx.dhwK_ * DST_DTYPE_BYTES;
    ub2GmParams.blockLen = DST_DTYPE_BYTES; // len:byte
    ub2GmParams.dstStride = (self->ctx.dhwK_ - 1) * DST_DTYPE_BYTES;
    for (uint64_t j = 0; j < srcStride; j++) {
        uint64_t totalHwk = wkCnt * self->ctx.curNIdx_ + j;
        uint64_t tailNSize = self->ctx.singleShapeCin_ - cutMNSize.usedNSize / self->ctx.hwK_ -
                             j / self->ctx.hwK_ * BLOCK_CUBE;
        ub2GmParams.blockCount = (tailNSize > BLOCK_CUBE) ? BLOCK_CUBE : tailNSize;
        uint64_t srcOffset = j * BLOCK_CUBE;
        uint64_t dstOffset = (static_cast<uint64_t>(self->ctx.curMIdx_) * self->ctx.tiling_->baseM +
                              static_cast<uint64_t>(cutMNSize.usedMSize)) *
                                 self->ctx.dhwK_ * self->ctx.tiling_->cin1G +
                             (totalHwk / self->ctx.hwK_) * (self->ctx.dhwK_ * BLOCK_CUBE) + totalHwk % self->ctx.hwK_ +
                             static_cast<uint64_t>(cutMNSize.usedNSize) * self->ctx.tiling_->dk;
        SetLoopModePara(loopParams, DataCopyMVType::UB_TO_OUT);
        DataCopyPad<typename Intf::DstT, PaddingMode::Compact>(output[dstOffset],
                                                               self->ctx.vecOutBuf_[srcSize + srcOffset], ub2GmParams);
        ResetLoopModePara(DataCopyMVType::UB_TO_OUT);
    }
}

template <class Intf>
static __aicore__ inline void DataCopyPadBaseNUndividedForNDHWC(Intf* self,
                                                                const GlobalTensor<typename Intf::DstT>& output,
                                                                const uint32_t srcStride,
                                                                const CutDeterMinisticMNSize& cutMNSize)
{
    uint64_t srcSize = cutMNSize.curNSize * cutMNSize.curMSize;
    uint64_t wkCnt = ShiftCeilM0(self->ctx.tiling_->baseN, BLOCK_CUBE) < self->ctx.hwK_ ?
                         ShiftCeilM0(self->ctx.tiling_->baseN, BLOCK_CUBE) :
                         self->ctx.hwK_;
    LoopModeParams loopParams;
    loopParams.loop2Size = 1;
    loopParams.loop1Size = cutMNSize.curMSize;
    loopParams.loop1SrcStride = BLOCK_CUBE * srcStride * DST_DTYPE_BYTES;
    loopParams.loop1DstStride = self->ctx.tiling_->cin1G * self->ctx.dhwK_ * DST_DTYPE_BYTES;
    DataCopyExtParams ub2GmParams;
    ub2GmParams.blockLen = self->ctx.singleShapeCin_ * DST_DTYPE_BYTES;
    ub2GmParams.dstStride = (self->ctx.tiling_->cin1G - self->ctx.singleShapeCin_) * DST_DTYPE_BYTES;
    ub2GmParams.srcStride = (BLOCK_CUBE - self->ctx.singleShapeCin_) * DST_DTYPE_BYTES >> 5;
    ub2GmParams.blockCount = srcStride;
    uint64_t coutIdx = static_cast<uint64_t>(self->ctx.curMIdx_) * self->ctx.tiling_->baseM +
                       static_cast<uint64_t>(cutMNSize.usedMSize);
    uint64_t dstOffset = coutIdx * self->ctx.hwK_ * self->ctx.tiling_->cin1G +
                         wkCnt * self->ctx.tiling_->cin1G * self->ctx.curNIdx_ +
                         static_cast<uint64_t>(cutMNSize.usedNSize) * self->ctx.tiling_->dk;
    SetLoopModePara(loopParams, DataCopyMVType::UB_TO_OUT);
    DataCopyPad(output[dstOffset], self->ctx.vecOutBuf_[srcSize], ub2GmParams);
    ResetLoopModePara(DataCopyMVType::UB_TO_OUT);
}

template <class Intf>
static __aicore__ inline void DataCopyPadDkEqOne(Intf* self, const GlobalTensor<typename Intf::DstT>& output,
                                                 const CutDeterMinisticMNSize& cutMNSize, const uint64_t baseCin)
{
    uint64_t srcSize = cutMNSize.curNSize * cutMNSize.curMSize;
    DataCopyExtParams ub2GmParams;
    uint64_t dstOffset = (static_cast<uint64_t>(self->ctx.curMIdx_) * self->ctx.tiling_->baseM +
                          static_cast<uint64_t>(cutMNSize.usedMSize)) *
                             self->ctx.dhwK_ * self->ctx.tiling_->cin1G +
                         static_cast<uint64_t>(self->ctx.curNIdx_) * self->ctx.tiling_->baseN +
                         static_cast<uint64_t>(cutMNSize.usedNSize);
    if (cutMNSize.isNTail) {
        uint64_t tailNSize = (self->ctx.singleShapeCin_ * self->ctx.hwK_ - cutMNSize.usedNSize);
        ub2GmParams.srcStride = ((cutMNSize.curNSize - tailNSize) * DST_DTYPE_BYTES) >> ONE_BLK_SHIFT_SIZE;
        ub2GmParams.blockLen = tailNSize * DST_DTYPE_BYTES;
        ub2GmParams.dstStride = (self->ctx.tiling_->cin1G * self->ctx.hwK_ - tailNSize) * DST_DTYPE_BYTES;
    } else {
        ub2GmParams.srcStride = 0;
        ub2GmParams.blockLen = cutMNSize.curNSize * DST_DTYPE_BYTES;
        uint64_t shapeCin = self->ctx.singleShapeCin_ < baseCin ? self->ctx.singleShapeCin_ : baseCin;
        ub2GmParams.dstStride = (self->ctx.tiling_->cin1G - shapeCin) * self->ctx.hwK_ * DST_DTYPE_BYTES;
    }
    ub2GmParams.blockCount = cutMNSize.curMSize;
    DataCopyPad(output[dstOffset], self->ctx.vecOutBuf_[srcSize], ub2GmParams);
}

template <class Intf>
static __aicore__ inline void DataCopyPadDkEqOneForDHWCN(Intf* self, const GlobalTensor<typename Intf::DstT>& output,
                                                         const CutDeterMinisticMNSize& cutMNSize,
                                                         const uint64_t srcStride)
{
    uint64_t srcSize = cutMNSize.curNSize * cutMNSize.curMSize;
    uint64_t wkCnt = ShiftCeilM0(self->ctx.tiling_->baseN, BLOCK_CUBE) < self->ctx.hwK_ ?
                         ShiftCeilM0(self->ctx.tiling_->baseN, BLOCK_CUBE) :
                         self->ctx.hwK_;
    uint64_t dstOffset = (static_cast<uint64_t>(self->ctx.curMIdx_) * self->ctx.tiling_->baseM +
                          static_cast<uint64_t>(cutMNSize.usedMSize)) +
                         static_cast<uint64_t>(self->ctx.curNIdx_) * wkCnt * self->ctx.tiling_->cin1G *
                             self->ctx.tiling_->cout +
                         static_cast<uint64_t>(cutMNSize.usedNSize) / srcStride * self->ctx.tiling_->cout;
    LoopModeParams loopParams;
    loopParams.loop2Size = 1;
    loopParams.loop1Size = srcStride;
    uint64_t baseCin = CeilHkWk(cutMNSize.curNSize, srcStride); // Cin1*Cin0
    loopParams.loop1SrcStride = baseCin * cutMNSize.curMSize * DST_DTYPE_BYTES;
    loopParams.loop1DstStride = self->ctx.tiling_->cin1G * self->ctx.tiling_->cout * DST_DTYPE_BYTES;

    uint64_t shapeCin = self->ctx.singleShapeCin_ < baseCin ? self->ctx.singleShapeCin_ : baseCin;
    uint64_t tailNSize = cutMNSize.curMSize;
    DataCopyExtParams ub2GmParams;
    ub2GmParams.srcStride = 0;
    ub2GmParams.blockLen = tailNSize * DST_DTYPE_BYTES;
    ub2GmParams.dstStride = (self->ctx.tiling_->cout - tailNSize) * DST_DTYPE_BYTES;
    uint64_t tailN = shapeCin;
    if (cutMNSize.isNTail) {
        tailN = (self->ctx.singleShapeCin_ * srcStride - cutMNSize.usedNSize) / srcStride;
    }
    ub2GmParams.blockCount = tailN;
    SetLoopModePara(loopParams, DataCopyMVType::UB_TO_OUT);
    DataCopyPad<typename Intf::DstT, PaddingMode::Compact>(output[dstOffset], self->ctx.vecOutBuf_[srcSize],
                                                           ub2GmParams);
    ResetLoopModePara(DataCopyMVType::UB_TO_OUT);
}

template <class Intf>
static __aicore__ inline void DataCopyPadDkEqOneForNDHWC(Intf* self, const GlobalTensor<typename Intf::DstT>& output,
                                                         const CutDeterMinisticMNSize& cutMNSize,
                                                         const uint64_t baseCin)
{
    uint64_t srcSize = cutMNSize.curNSize * cutMNSize.curMSize;
    DataCopyExtParams ub2GmParams;
    uint64_t dstOffset = (static_cast<uint64_t>(self->ctx.curMIdx_) * self->ctx.tiling_->baseM +
                          static_cast<uint64_t>(cutMNSize.usedMSize)) *
                             self->ctx.hwK_ * self->ctx.tiling_->cin1G +
                         static_cast<uint64_t>(self->ctx.curNIdx_) * self->ctx.tiling_->baseN +
                         static_cast<uint64_t>(cutMNSize.usedNSize) / self->ctx.hwK_;
    uint64_t shapeCin = self->ctx.singleShapeCin_ < baseCin ? self->ctx.singleShapeCin_ : baseCin;
    uint64_t tailN = shapeCin;
    if (cutMNSize.isNTail) {
        tailN = (self->ctx.singleShapeCin_ * self->ctx.hwK_ - cutMNSize.usedNSize) / self->ctx.hwK_;
    }

    uint64_t tailNSize = tailN;
    ub2GmParams.srcStride = ((baseCin - tailNSize) * DST_DTYPE_BYTES) >> ONE_BLK_SHIFT_SIZE;
    ub2GmParams.blockLen = tailNSize * DST_DTYPE_BYTES;
    ub2GmParams.dstStride = (self->ctx.tiling_->cin1G - tailNSize) * DST_DTYPE_BYTES;
    ub2GmParams.blockCount = cutMNSize.curMSize * self->ctx.hwK_;
    DataCopyPad(output[dstOffset], self->ctx.vecOutBuf_[srcSize], ub2GmParams);
}

template <class Intf>
static __aicore__ inline void UBRearrange2Gm(Intf* self, const GlobalTensor<typename Intf::DstT>& output,
                                             const DeterMinisticShape& deterShape)
{
    SetAtomicNone();
    CutDeterMinisticMNSize cutMNSize;
    cutMNSize.curMSize = deterShape.mSize[self->ctx.subCoreInx_];
    cutMNSize.curNSize = deterShape.nSize[self->ctx.subCoreInx_];
    if (cutMNSize.curMSize == 0 || cutMNSize.curNSize == 0) {
        return;
    }
    cutMNSize.usedMSize = deterShape.usedMSize[self->ctx.subCoreInx_];
    cutMNSize.usedNSize = deterShape.usedNSize[self->ctx.subCoreInx_];
    cutMNSize.isNTail = deterShape.isNTail[self->ctx.subCoreInx_];

    uint32_t hwNum = ShiftCeilM0(cutMNSize.curNSize, BLOCK_CUBE);
    if constexpr (Intf::conv3ddwConfig.groupEnlarge) {
        Rearrange2Gm(self, output, 1, 0);
    } else if ((self->ctx.tiling_->dk == 1 && hwNum < self->ctx.hwK_) || (self->ctx.tiling_->dk != 1)) {
        uint32_t srcStride = hwNum;
        if constexpr (Intf::Config::dType::format == ConvolutionBackprop::CubeFormat::DHWCN) {
            Rearrange2GmScatterDeterForDHWCN(self, srcStride, output, cutMNSize);
        } else { // NCDHW or NDHWC
            Rearrange2GmScatterBaseNUndivided(self, srcStride, output, cutMNSize);
        }
    } else {
        uint32_t srcStride = self->ctx.hwK_;
        if constexpr (Intf::Config::dType::format == ConvolutionBackprop::CubeFormat::NCDHW) {
            Rearrange2GmScatterDeter(self, srcStride, output, cutMNSize);
        } else if constexpr (Intf::Config::dType::format == ConvolutionBackprop::CubeFormat::NDHWC) {
            Rearrange2GmScatterDeterForNDHWC(self, srcStride, output, cutMNSize);
        } else { // DHWCN
            Rearrange2GmScatterDeterForDHWCN(self, srcStride, output, cutMNSize);
        }
    }
    event_t eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
    WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
}
} // namespace ConvolutionBackpropFunc

#endif
