/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file conv_bp_sub_func_deterministic_common.h
 * \brief
 */

#ifndef CONV3D_BP_FILTER_SUB_FUNC_DETERMINISTIC_COMMON_H
#define CONV3D_BP_FILTER_SUB_FUNC_DETERMINISTIC_COMMON_H

namespace ConvolutionBackpropFunc {
constexpr uint8_t SYNC_MODE0 = 0;
constexpr uint8_t SYNC_MODE2 = 2;
constexpr uint8_t SYNC_MODE4 = 4;
constexpr uint8_t DST_DTYPE_BYTES = 4;
constexpr uint8_t ONE_BLK_SHIFT_SIZE = 5;
constexpr uint32_t VECTOR_UB_SIZE_HALF = AscendC::TOTAL_UB_SIZE >> 1;   // UB大小的一半
constexpr uint32_t FLOAT_SHIFT_SIZE = 2;
constexpr uint32_t CUT_FOUR = 4;
constexpr uint32_t RELATED_CORE_NUM = 2;
constexpr uint32_t DOUBLE = 2;
constexpr uint32_t PINGPONG_NUM = 2;
constexpr uint64_t CUBE_WORKSPACE = AscendC::TOTAL_L0C_SIZE >> 2; // 2: sizeof(float)
constexpr uint64_t HALF_CUBE_WORKSPACE = CUBE_WORKSPACE >> 1;
constexpr uint64_t QUARTER_CUBE_WORKSPACE = CUBE_WORKSPACE >> 2;
constexpr FixpipeConfig CFG_NZ = {CO2Layout::NZ, true};
static constexpr uint16_t SYNC_AIV_AIC_DET_FLAG = 6;

struct DeterMinisticShape {
    uint32_t mSize[CUT_FOUR] = {0, 0, 0, 0};
    uint32_t nSize[CUT_FOUR] = {0, 0, 0, 0};
    uint64_t mnSize[CUT_FOUR] = {0, 0, 0, 0};
    uint64_t addrOffset[CUT_FOUR] = {0, 0, 0, 0};
    uint32_t usedMSize[CUT_FOUR] = {0, 0, 0, 0};
    uint32_t usedNSize[CUT_FOUR] = {0, 0, 0, 0};
    bool isNTail[CUT_FOUR] = {false, false, false, false};
};

struct CutDeterMinisticMNSize {
    uint32_t curMSize = 0;
    uint32_t curNSize = 0;
    uint32_t usedMSize = 0;
    uint32_t usedNSize = 0;
    bool isNTail = false;
};

static __aicore__ inline bool IsDivisible2(const uint64_t a)
{
    return a == ((a >> 1) << 1);
}

static __aicore__ inline void DivTwoNumersInHalf(uint64_t &a, uint64_t &b)
{
    // 能对半切就切，不能就切大的数
    if (IsDivisible2(b)) {
        b = b >> 1;
    } else if (IsDivisible2(a)) {
        a = a >> 1;
    } else if (b > a) {
        b = (b + 1) >> 1;
    } else {
        a = (a + 1) >> 1;
    }
    // 最小要为1， 不能是0
    a = a < 1 ? 1 : a;
    b = b < 1 ? 1 : b;
}

template <class Intf>
static __aicore__ inline void CalcL0cParams(Intf *self, bool tailCinExist)
{
    bool isNLoop = (self->ctx.lastNIdx_ != self->ctx.curNL0Idx_);
    // 若cin有尾块，在cin循环开始时对其进行赋值初始化
    if (tailCinExist && isNLoop && (self->ctx.curNL0Idx_ % self->ctx.cinHkWkLoop_ == 0)) {
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
static __aicore__ inline void GatherDepthwise(Intf *self, uint32_t hwkLength, uint32_t srcSize, uint32_t strideHwk)
{
    uint16_t vLoop = AscendC::VECTOR_REG_WIDTH / sizeof(typename Intf::DstT);
    uint16_t loopSize = DivCeil(hwkLength, vLoop);
    uint16_t indexLength = vLoop > hwkLength ? hwkLength : vLoop;

    uint32_t coutCin0 = self->ctx.baseUseM_ * BLOCK_CUBE;
    auto indexBuf = self->ctx.vecBuf_.template GetWithOffset<int32_t>(
        AscendC::VECTOR_REG_WIDTH >> FLOAT_SHIFT_SIZE, AscendC::TOTAL_UB_SIZE - AscendC::VECTOR_REG_WIDTH);
    auto indexPtr = (__ubuf__ uint32_t *)indexBuf[0].GetPhyAddr();
    auto srcPtr = (__ubuf__ typename Intf::DstT *)self->ctx.vecOutBuf_[0].GetPhyAddr();
    auto dstPtr = (__ubuf__ typename Intf::DstT *)self->ctx.vecOutBuf_[srcSize].GetPhyAddr();

    CreateVecIndex(indexBuf[0], (int32_t)0, indexLength); // 从0依次递增到indexLength
    PipeBarrier<PIPE_V>();
    Muls(indexBuf[0], indexBuf[0], (int32_t)coutCin0, indexLength); // 每个元素相乘coutCin0
    PipeBarrier<PIPE_V>();

    uint32_t sreg = indexLength;
    uint32_t sregTail = (hwkLength % vLoop == 0) ? indexLength : (hwkLength % vLoop);
    uint32_t hwkSrcStride = indexLength * coutCin0;
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<typename Intf::DstT> srcReg;
        MicroAPI::RegTensor<uint32_t> vIndexReg;
        MicroAPI::MaskReg preg = MicroAPI::UpdateMask<typename Intf::DstT>(sreg);
        MicroAPI::MaskReg pregTail = MicroAPI::UpdateMask<typename Intf::DstT>(sregTail);
        MicroAPI::DataCopy(vIndexReg, indexPtr);

        MicroAPI::UnalignReg u0;
        for (uint16_t i = 0; i < static_cast<uint16_t>(self->ctx.baseUseM_); i++) {
            uint16_t j = 0;
            for (j = 0; j < (uint16_t)(loopSize - 1); j++) {
                uint32_t srcoffset = i * BLOCK_CUBE + i + j * hwkSrcStride;
                uint32_t dstoffset = i * strideHwk + j * indexLength;
                MicroAPI::DataCopyGather(srcReg, srcPtr + srcoffset, vIndexReg, preg);
                auto tmp = dstPtr + dstoffset;
                MicroAPI::DataCopyUnAlign(tmp, srcReg, u0, indexLength);
                MicroAPI::DataCopyUnAlignPost(tmp, u0, 0);
            }
            uint32_t srcoffset = i * BLOCK_CUBE + i + j * hwkSrcStride;
            uint32_t dstoffset = i * strideHwk + j * indexLength;
            MicroAPI::DataCopyGather(srcReg, srcPtr + srcoffset, vIndexReg, pregTail);
            auto tmp = dstPtr + dstoffset;
            MicroAPI::DataCopyUnAlign(tmp, srcReg, u0, indexLength);
            MicroAPI::DataCopyUnAlignPost(tmp, u0, 0);
        }
    }
    event_t eventIdVecToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIdVecToMte3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVecToMte3);
}

template <class Intf>
static __aicore__ inline void Rearrange2GmDepthwise(Intf *self, const GlobalTensor<typename Intf::DstT> &output)
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
        self->ctx.lastNIdx_ = self->ctx.curNL0Idx_;
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

    ub2GmParams.blockLen = nValue * DST_DTYPE_BYTES; // len
    ub2GmParams.blockCount = self->ctx.curSingleCoreDk_; // num
    ub2GmParams.srcStride = 0; // compact mode ignore srcstride
    ub2GmParams.dstStride = 0;

    DataCopyPad<typename Intf::DstT, PaddingMode::Compact>(output[dstGmOffset],
        self->ctx.vecOutBuf_[srcSize], ub2GmParams);
    ResetLoopModePara(DataCopyMVType::UB_TO_OUT);
}

template <class Intf>
static __aicore__ inline void NormalGroupDataCopyPadDkEqOne(Intf *self,
    const GlobalTensor<typename Intf::DstT> &output, uint32_t srcStride, uint32_t coutNum)
{
    uint32_t srcSize = self->ctx.baseUseN_ * coutNum;
    uint32_t coutPerGroup = self->ctx.tiling_->cout / self->ctx.tiling_->group;
    uint32_t cinPerGroup = self->ctx.tiling_->cin / self->ctx.tiling_->group;
    uint32_t baseCin = Ceil(Ceil(self->ctx.baseUseN_, self->ctx.curSingleCoreDk_), self->ctx.hwK_);
    LoopModeParams loopParams;
    loopParams.loop2Size = 1;
    loopParams.loop1Size = self->ctx.baseUseM_ / coutPerGroup;
    loopParams.loop1SrcStride =
        (baseCin * self->ctx.curSingleCoreDk_ * self->ctx.hwK_ * coutPerGroup +
        cinPerGroup * self->ctx.hwK_) * DST_DTYPE_BYTES;
    loopParams.loop1DstStride = self->ctx.hwK_ * DST_DTYPE_BYTES * cinPerGroup * coutPerGroup;

    loopParams.loop2SrcStride = loopParams.loop1SrcStride;
    loopParams.loop2DstStride = loopParams.loop1DstStride;

    SetLoopModePara(loopParams, DataCopyMVType::UB_TO_OUT);

    DataCopyExtParams ub2GmParams;
    constexpr uint8_t ALIGN_BYTE = 32; // UB 32Byte对齐
    ub2GmParams.blockLen = self->ctx.hwK_ * DST_DTYPE_BYTES * cinPerGroup; // len:byte
    ub2GmParams.blockCount = coutPerGroup;
    ub2GmParams.srcStride = (baseCin - cinPerGroup) * self->ctx.hwK_ * DST_DTYPE_BYTES / ALIGN_BYTE;
    ub2GmParams.dstStride = 0;
    DataCopyPad<typename Intf::DstT>(output[0], self->ctx.vecOutBuf_[srcSize], ub2GmParams);
    ResetLoopModePara(DataCopyMVType::UB_TO_OUT);
}

template <class Intf>
static __aicore__ inline void NormalGroupDataCopyPad(Intf *self,
    const GlobalTensor<typename Intf::DstT> &output, uint32_t srcStride, uint32_t coutNum)
{
    uint32_t coutPerGroup = self->ctx.tiling_->cout / self->ctx.tiling_->group;
    uint32_t cinPerGroup = self->ctx.tiling_->cin / self->ctx.tiling_->group;
    uint32_t srcSize = self->ctx.baseUseN_ * coutNum;
    uint32_t baseCin = Ceil(Ceil(self->ctx.baseUseN_, self->ctx.curSingleCoreDk_), self->ctx.hwK_);

    LoopModeParams loopParams;
    loopParams.loop2Size = self->ctx.baseUseM_ / coutPerGroup;
    loopParams.loop1Size = coutPerGroup;
    loopParams.loop2SrcStride =
        (baseCin * self->ctx.curSingleCoreDk_ * srcStride * coutPerGroup + cinPerGroup * srcStride) * DST_DTYPE_BYTES;
    loopParams.loop2DstStride = self->ctx.dhwK_ * DST_DTYPE_BYTES * cinPerGroup * coutPerGroup;

    loopParams.loop1SrcStride = baseCin * self->ctx.curSingleCoreDk_ * srcStride * DST_DTYPE_BYTES;
    loopParams.loop1DstStride = self->ctx.dhwK_ * DST_DTYPE_BYTES * cinPerGroup;

    SetLoopModePara(loopParams, DataCopyMVType::UB_TO_OUT);

    DataCopyExtParams ub2GmParams;
    constexpr uint8_t ALIGN_BYTE = 32; // UB 32Byte对齐
    ub2GmParams.blockLen = self->ctx.hwK_ * DST_DTYPE_BYTES * self->ctx.curSingleCoreDk_; // len:byte
    ub2GmParams.blockCount = cinPerGroup;
    ub2GmParams.srcStride = (srcStride - self->ctx.hwK_) * DST_DTYPE_BYTES * self->ctx.curSingleCoreDk_/ ALIGN_BYTE;
    // GM单位：byte
    ub2GmParams.dstStride = (self->ctx.tiling_->dk - self->ctx.curSingleCoreDk_) * self->ctx.hwK_ * DST_DTYPE_BYTES;
    DataCopyPad<typename Intf::DstT>(output[0], self->ctx.vecOutBuf_[srcSize], ub2GmParams);
    ResetLoopModePara(DataCopyMVType::UB_TO_OUT);
}

template <class Intf>
static __aicore__ inline void Rearrange2GmScatter(Intf *self, uint32_t srcStride,
    uint16_t loopSize, uint16_t indexLength, uint32_t coutNum)
{
    uint32_t baseCin = Ceil(Ceil(self->ctx.baseUseN_, self->ctx.curSingleCoreDk_), self->ctx.hwK_);
    auto indexBuf = self->ctx.vecBuf_.template GetWithOffset<int32_t>(
        AscendC::VECTOR_REG_WIDTH >> FLOAT_SHIFT_SIZE, AscendC::TOTAL_UB_SIZE - AscendC::VECTOR_REG_WIDTH);
    auto indexPtr = (__ubuf__ uint32_t *)indexBuf[0].GetPhyAddr();
    auto srcPtr = (__ubuf__ typename Intf::DstT *)self->ctx.vecOutBuf_[0].GetPhyAddr();
    uint32_t srcSize = self->ctx.baseUseN_ * coutNum;
    auto dstPtr = (__ubuf__ typename Intf::DstT *)self->ctx.vecOutBuf_[srcSize].GetPhyAddr();

    bool enableVecRegWidthMin = (indexLength < BLOCK_CUBE);
    uint32_t sreg = AscendC::VECTOR_REG_WIDTH / sizeof(typename Intf::DstT);
    uint8_t sregU8 = sreg > BLOCK_CUBE ? sreg : BLOCK_CUBE;
    uint16_t iterCin = baseCin / indexLength;
    uint16_t iterCout = Ceil(coutNum , loopSize);
    uint32_t coutDstStride = baseCin * loopSize * srcStride;
    uint32_t hwkSrcStride = coutNum * BLOCK_CUBE;
    uint32_t cinSrcStride = hwkSrcStride * self->ctx.hwK_;
    uint32_t cinDstStride = indexLength * srcStride;

    uint32_t tailNum = coutNum % loopSize;
    uint32_t tailSreg = (tailNum == 0) ? sreg : tailNum * BLOCK_CUBE;

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<typename Intf::DstT> srcReg;
        MicroAPI::RegTensor<uint32_t> vIndexReg;
        MicroAPI::MaskReg preg = MicroAPI::UpdateMask<typename Intf::DstT>(sreg);
        MicroAPI::MaskReg pregTail = MicroAPI::UpdateMask<typename Intf::DstT>(tailSreg);
        MicroAPI::DataCopy(vIndexReg, indexPtr);

        for (uint16_t cinIndex = 0; cinIndex < iterCin; cinIndex++) {
            uint32_t srcOffsetCin = enableVecRegWidthMin ?
                                        ((cinIndex / DOUBLE) * cinSrcStride + (cinIndex % DOUBLE) * indexLength) :
                                        (cinIndex * cinSrcStride);
            uint32_t dstOffsetCin = cinIndex * cinDstStride;
            for (uint16_t hwkIndex = 0; hwkIndex < static_cast<uint16_t>(self->ctx.hwK_); hwkIndex++) {
                uint16_t coutIndex = 0;
                for (coutIndex = 0; coutIndex < static_cast<uint16_t>(iterCout - 1); coutIndex++) {
                    uint32_t srcOffset = srcOffsetCin + hwkIndex * hwkSrcStride + coutIndex * sregU8;
                    uint32_t dstOffset = dstOffsetCin + hwkIndex + coutIndex * coutDstStride;
                    MicroAPI::DataCopy(srcReg, srcPtr + srcOffset);
                    MicroAPI::DataCopyScatter(dstPtr + dstOffset, srcReg, vIndexReg, preg);
                }
                uint32_t srcOffset = srcOffsetCin + hwkIndex * hwkSrcStride + coutIndex * sregU8;
                uint32_t dstOffset = dstOffsetCin + hwkIndex + coutIndex * coutDstStride;

                MicroAPI::DataCopy(srcReg, srcPtr + srcOffset);
                MicroAPI::DataCopyScatter(dstPtr + dstOffset, srcReg, vIndexReg, pregTail);
            }
        }
    }
}

template <class Intf>
static __aicore__ inline void Rearrange2GmNormalGroup(
    Intf* self, const GlobalTensor<typename Intf::DstT>& output, bool isCoutNumAligned)
{
    uint16_t vLoop = AscendC::VECTOR_REG_WIDTH / sizeof(typename Intf::DstT);
    uint16_t loopSize = DivCeil(vLoop, BLOCK_CUBE);
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
    uint32_t coutNum = isCoutNumAligned ? ShiftCeilM0(self->ctx.baseUseM_, BLOCK_CUBE) * BLOCK_CUBE : self->ctx.baseUseM_;
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
static __aicore__ inline void Rearrange2Gm(Intf *self, const GlobalTensor<typename Intf::DstT> &output,
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
static __aicore__ inline void DataCopyPadBaseNUndivided(Intf *self, const GlobalTensor<typename Intf::DstT> &output,
    const uint32_t srcStride, const CutDeterMinisticMNSize &cutMNSize)
{
    uint64_t srcSize = cutMNSize.curNSize * cutMNSize.curMSize;
    uint64_t wkCnt = ShiftCeilM0(self->ctx.tiling_->baseN, BLOCK_CUBE) < self->ctx.hwK_ ?
        ShiftCeilM0(self->ctx.tiling_->baseN, BLOCK_CUBE) : self->ctx.hwK_;
    LoopModeParams loopParams;
    DataCopyExtParams ub2GmParams;
    loopParams.loop2Size = 1;
    loopParams.loop1Size = cutMNSize.curMSize;
    loopParams.loop1SrcStride = BLOCK_CUBE * srcStride * DST_DTYPE_BYTES;
    loopParams.loop1DstStride = self->ctx.tiling_->cin1G * self->ctx.dhwK_ * DST_DTYPE_BYTES;
    ub2GmParams.blockLen = DST_DTYPE_BYTES; // len:byte
    ub2GmParams.dstStride = (self->ctx.dhwK_ - 1) * DST_DTYPE_BYTES;
    for (uint64_t j = 0; j < srcStride; j++) {
        uint64_t totalHwk = wkCnt * self->ctx.curNL0Idx_ + j;
        uint64_t tailNSize = self->ctx.singleShapeCin_ - cutMNSize.usedNSize / self->ctx.hwK_ - j / self->ctx.hwK_ * BLOCK_CUBE;
        ub2GmParams.blockCount = (tailNSize > BLOCK_CUBE) ? BLOCK_CUBE : tailNSize;
        uint64_t srcOffset = j * BLOCK_CUBE;
        uint64_t dstOffset = (static_cast<uint64_t>(self->ctx.curML0Idx_) * self->ctx.tiling_->baseM + static_cast<uint64_t>(cutMNSize.usedMSize)) *
                        self->ctx.dhwK_ * self->ctx.tiling_->cin1G +
                        (totalHwk / self->ctx.hwK_) * (self->ctx.dhwK_ * BLOCK_CUBE) +
                        totalHwk % self->ctx.hwK_ + static_cast<uint64_t>(cutMNSize.usedNSize) * self->ctx.tiling_->dk;
        SetLoopModePara(loopParams, DataCopyMVType::UB_TO_OUT);
        DataCopyPad<typename Intf::DstT, PaddingMode::Compact>(output[dstOffset], self->ctx.vecOutBuf_[srcSize + srcOffset],
            ub2GmParams);
        ResetLoopModePara(DataCopyMVType::UB_TO_OUT);
    }
}

template <class Intf>
static __aicore__ inline void DataCopyPadDkEqOne(Intf *self, const GlobalTensor<typename Intf::DstT> &output,
    const CutDeterMinisticMNSize &cutMNSize, const uint64_t baseCin)
{
    // 在ub做完重排后，需将数据搬到output中
    // 对于主块和尾块，数据在UB中按32Bytes对齐，根据数据排布特点给blockLen和dstStride即可，按Normal模式搬出
    // 对于Cin<16的场景，数据在UB中紧凑存没有气泡，按Compact模式搬出
    uint64_t srcSize = cutMNSize.curNSize * cutMNSize.curMSize;
    DataCopyExtParams ub2GmParams;
    uint64_t dstOffset = (static_cast<uint64_t>(self->ctx.curML0Idx_) * self->ctx.tiling_->baseM + static_cast<uint64_t>(cutMNSize.usedMSize)) *
                        self->ctx.dhwK_ * self->ctx.tiling_->cin1G +
                        static_cast<uint64_t>(self->ctx.curNL0Idx_) * self->ctx.tiling_->baseN +
                        static_cast<uint64_t>(cutMNSize.usedNSize);
    if (cutMNSize.isNTail) {
        uint64_t tailNSize = (self->ctx.singleShapeCin_ * self->ctx.hwK_ - cutMNSize.usedNSize);
        ub2GmParams.srcStride = ((cutMNSize.curNSize - tailNSize) * DST_DTYPE_BYTES) >> ONE_BLK_SHIFT_SIZE;
        ub2GmParams.blockLen = tailNSize * DST_DTYPE_BYTES;
        ub2GmParams.dstStride = (self->ctx.tiling_->cin1G * self->ctx.hwK_ - tailNSize) * DST_DTYPE_BYTES;
    } else {
        ub2GmParams.srcStride = 0;
        ub2GmParams.blockLen = cutMNSize.curNSize * DST_DTYPE_BYTES;
        uint64_t shapeCin = self->ctx.singleShapeCin_ < baseCin ?  self->ctx.singleShapeCin_ : baseCin;
        ub2GmParams.dstStride = (self->ctx.tiling_->cin1G - shapeCin) * self->ctx.hwK_ * DST_DTYPE_BYTES;
    }
    ub2GmParams.blockCount = cutMNSize.curMSize;
    DataCopyPad(output[dstOffset], self->ctx.vecOutBuf_[srcSize], ub2GmParams);
}

template <class Intf>
static __aicore__ inline void CreateIndexBuf4BaseNUndivided(Intf *self, const uint32_t srcStride)
{
    auto indexBuf = self->ctx.vecBuf_.template GetWithOffset<int32_t>(
        AscendC::VECTOR_REG_WIDTH >> FLOAT_SHIFT_SIZE, AscendC::TOTAL_UB_SIZE - AscendC::VECTOR_REG_WIDTH);
    uint64_t C0_PER_REG = AscendC::VECTOR_REG_WIDTH / (sizeof(typename Intf::DstT) * self->ctx.tiling_->n0);
    uint64_t dstAddr = 0;
    for (uint64_t i = 0; i < C0_PER_REG; i++) {
        CreateVecIndex(indexBuf[dstAddr], (int32_t)0, BLOCK_CUBE); // 从0依次递增
        PipeBarrier<PIPE_V>();
        Adds(indexBuf[dstAddr], indexBuf[dstAddr], (int32_t)(srcStride * BLOCK_CUBE * i), BLOCK_CUBE); // srcStride * 16
        PipeBarrier<PIPE_V>();
        dstAddr += BLOCK_CUBE;
    }
}

template <class Intf>
static __aicore__ inline void Rearrange2GmScatterBaseNUndivided(Intf *self, const uint32_t srcStride,
    const GlobalTensor<typename Intf::DstT> &output, const CutDeterMinisticMNSize &cutMNSize)
{
    uint64_t coutNum = cutMNSize.curMSize;
    auto indexBuf = self->ctx.vecBuf_.template GetWithOffset<int32_t>(
        AscendC::VECTOR_REG_WIDTH >> FLOAT_SHIFT_SIZE, AscendC::TOTAL_UB_SIZE - AscendC::VECTOR_REG_WIDTH);
    auto srcPtr = (__ubuf__ typename Intf::DstT *)self->ctx.vecOutBuf_[0].GetPhyAddr();
    auto indexPtr = (__ubuf__ uint32_t *)indexBuf[0].GetPhyAddr();
    uint64_t srcSize = cutMNSize.curNSize * coutNum;
    auto dstPtr = (__ubuf__ typename Intf::DstT *)self->ctx.vecOutBuf_[srcSize].GetPhyAddr();
    uint32_t C0_PER_REG = AscendC::VECTOR_REG_WIDTH / (sizeof(typename Intf::DstT) * self->ctx.tiling_->n0);
    constexpr uint8_t SREG_PROC_NUM = 64;
    uint32_t sreg = 64; // 64: reg处理的数目
    uint32_t tailNum = coutNum % C0_PER_REG;
    uint32_t tailSreg = (tailNum == 0) ? SREG_PROC_NUM : tailNum * BLOCK_CUBE;

    CreateIndexBuf4BaseNUndivided(self, srcStride);
    uint16_t iterWk = ShiftCeilM0(cutMNSize.curNSize, BLOCK_CUBE);
    uint16_t iterCout = Ceil(coutNum, C0_PER_REG);
    uint16_t wkSrcStride = coutNum * BLOCK_CUBE;
    uint16_t coutDstStride = C0_PER_REG * cutMNSize.curNSize;

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<typename Intf::DstT> srcReg;
        MicroAPI::RegTensor<uint32_t> vIndexReg;
        MicroAPI::MaskReg preg = MicroAPI::UpdateMask<typename Intf::DstT>(sreg);
        MicroAPI::MaskReg pregTail = MicroAPI::UpdateMask<typename Intf::DstT>(tailSreg);
        MicroAPI::DataCopy(vIndexReg, indexPtr);

        for (uint16_t wkIndex = 0; wkIndex < iterWk; wkIndex++) {
            uint16_t coutIndex = 0;
            for (coutIndex = 0; coutIndex < static_cast<uint16_t>(iterCout - 1); coutIndex++) {
                uint32_t srcOffset = coutIndex * SREG_PROC_NUM + wkIndex * wkSrcStride;
                uint32_t dstOffset = coutIndex * coutDstStride + wkIndex * BLOCK_CUBE;
                MicroAPI::DataCopy(srcReg, srcPtr + srcOffset);
                MicroAPI::DataCopyScatter(dstPtr + dstOffset, srcReg, vIndexReg, preg);
            }
            uint32_t srcOffsetTail = coutIndex * SREG_PROC_NUM + wkIndex * wkSrcStride;
            uint32_t dstOffsetTail = coutIndex * coutDstStride + wkIndex * BLOCK_CUBE;
            MicroAPI::DataCopy(srcReg, srcPtr + srcOffsetTail);
            MicroAPI::DataCopyScatter(dstPtr + dstOffsetTail, srcReg, vIndexReg, pregTail);
        }
    }

    event_t eventIdVecToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIdVecToMte3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVecToMte3);
    DataCopyPadBaseNUndivided(self, output, srcStride, cutMNSize);
}

template <class Intf>
static __aicore__ inline void CreateIndexBuf4BaseNDivided(Intf *self, const uint32_t srcStride,
    const uint64_t baseCin)
{
    auto indexBuf = self->ctx.vecBuf_.template GetWithOffset<int32_t>(
        AscendC::VECTOR_REG_WIDTH >> FLOAT_SHIFT_SIZE, AscendC::TOTAL_UB_SIZE - AscendC::VECTOR_REG_WIDTH);
    uint64_t C0_PER_REG = AscendC::VECTOR_REG_WIDTH / (sizeof(typename Intf::DstT) * self->ctx.tiling_->n0);
    uint64_t dstAddr = 0;
    for (uint64_t i = 0; i < C0_PER_REG; i++) {
        CreateVecIndex(indexBuf[dstAddr], (int32_t)0, BLOCK_CUBE); // 从0依次递增
        PipeBarrier<PIPE_V>();
        Muls(indexBuf[dstAddr], indexBuf[dstAddr], (int32_t)(srcStride), BLOCK_CUBE);
        PipeBarrier<PIPE_V>();
        Adds(indexBuf[dstAddr], indexBuf[dstAddr], (int32_t)(srcStride * baseCin * i), BLOCK_CUBE); // srcStride * 16
        PipeBarrier<PIPE_V>();
        dstAddr += BLOCK_CUBE;
    }
}

template <class Intf>
static __aicore__ inline void Rearrange2GmScatterDeter(Intf *self, const uint32_t srcStride,
    const GlobalTensor<typename Intf::DstT> &output, const CutDeterMinisticMNSize &cutMNSize)
{
    uint64_t baseCin = CeilHkWk(cutMNSize.curNSize, self->ctx.hwK_);
    uint64_t coutNum = cutMNSize.curMSize;
    auto indexBuf = self->ctx.vecBuf_.template GetWithOffset<int32_t>(
        AscendC::VECTOR_REG_WIDTH >> FLOAT_SHIFT_SIZE, AscendC::TOTAL_UB_SIZE - AscendC::VECTOR_REG_WIDTH);
    auto indexPtr = (__ubuf__ uint32_t *)indexBuf[0].GetPhyAddr();
    auto srcPtr = (__ubuf__ typename Intf::DstT *)self->ctx.vecOutBuf_[0].GetPhyAddr();
    uint64_t srcSize = cutMNSize.curNSize * coutNum;
    auto dstPtr = (__ubuf__ typename Intf::DstT *)self->ctx.vecOutBuf_[srcSize].GetPhyAddr();
    uint32_t C0_PER_REG = AscendC::VECTOR_REG_WIDTH / (sizeof(typename Intf::DstT) * self->ctx.tiling_->n0);
    uint32_t sreg = 64; // 64: reg处理的数目
    constexpr uint8_t SREG_PROC_NUM = 64;
    uint32_t tailNum = coutNum % C0_PER_REG;
    uint32_t tailSreg = (tailNum == 0) ? SREG_PROC_NUM : tailNum * BLOCK_CUBE;

    CreateIndexBuf4BaseNDivided(self, srcStride, baseCin);
    // BaseN整除hwk场景，其值较小，均在uint16_t范围内
    uint16_t iterCin = ShiftDivM0(baseCin, BLOCK_CUBE);
    uint16_t iterCout = Ceil(coutNum, C0_PER_REG);
    uint16_t hkWkSrcStride = coutNum * BLOCK_CUBE;
    uint16_t cinSrcStride = hkWkSrcStride * self->ctx.hwK_;
    uint16_t cinDstStride = BLOCK_CUBE * srcStride;
    uint16_t coutDstStride = C0_PER_REG * cutMNSize.curNSize;

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<typename Intf::DstT> srcReg;
        MicroAPI::RegTensor<uint32_t> vIndexReg;
        MicroAPI::MaskReg preg = MicroAPI::UpdateMask<typename Intf::DstT>(sreg);
        MicroAPI::MaskReg pregTail = MicroAPI::UpdateMask<typename Intf::DstT>(tailSreg);
        MicroAPI::DataCopy(vIndexReg, indexPtr);

        for (uint16_t cinIndex = 0; cinIndex < iterCin; cinIndex++) {
            for (uint16_t hkWkIndex = 0; hkWkIndex < static_cast<uint16_t>(self->ctx.hwK_); hkWkIndex++) {
                uint16_t coutIndex = 0;
                for (coutIndex = 0; coutIndex < static_cast<uint16_t>(iterCout - 1); coutIndex++) {
                    uint32_t srcOffset = coutIndex * SREG_PROC_NUM + hkWkIndex * hkWkSrcStride +
                        cinIndex * cinSrcStride;
                    uint32_t dstOffset = coutIndex * coutDstStride + hkWkIndex + cinIndex * cinDstStride;
                    MicroAPI::DataCopy(srcReg, srcPtr + srcOffset);
                    MicroAPI::DataCopyScatter(dstPtr + dstOffset, srcReg, vIndexReg, preg);
                }
                uint32_t srcOffsetTail = coutIndex * SREG_PROC_NUM + hkWkIndex * hkWkSrcStride +
                    cinIndex * cinSrcStride;
                uint32_t dstOffsetTail = coutIndex * coutDstStride + hkWkIndex + cinIndex * cinDstStride;
                MicroAPI::DataCopy(srcReg, srcPtr + srcOffsetTail);
                MicroAPI::DataCopyScatter(dstPtr + dstOffsetTail, srcReg, vIndexReg, pregTail);
            }
        }
    }

    event_t eventIdVecToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIdVecToMte3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVecToMte3);
    DataCopyPadDkEqOne(self, output, cutMNSize, baseCin);
}

template <class Intf>
static __aicore__ inline void UBRearrange2Gm(Intf *self, const GlobalTensor<typename Intf::DstT> &output,
                                        const DeterMinisticShape &deterShape)
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
    if (self->ctx.tiling_->group != self->ctx.tiling_->realGroup) {
        Rearrange2Gm(self, output, 1, 0);
    } else if ((self->ctx.tiling_->dk == 1 && hwNum < self->ctx.hwK_) || (self->ctx.tiling_->dk != 1)) {
        uint32_t srcStride = hwNum;
        Rearrange2GmScatterBaseNUndivided(self, srcStride, output, cutMNSize);
    } else {
        uint32_t srcStride = self->ctx.hwK_;
        Rearrange2GmScatterDeter(self, srcStride, output, cutMNSize);
    }
    event_t eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
    WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
}

static __aicore__ inline void BarrierVector()
{
#ifndef __CCE_KT_TEST__
    static constexpr uint16_t SYNC_AIV_BAR_FLAG = 0;
    CrossCoreSetFlag<SYNC_MODE0, PIPE_MTE3>(SYNC_AIV_BAR_FLAG);
    CrossCoreWaitFlag(SYNC_AIV_BAR_FLAG);
#endif
}

template <class Intf>
static __aicore__ inline void GetCutShape(Intf *self, DeterMinisticShape &deterShape)
{
    uint64_t splitedCin1 = ShiftCeilM0(Ceil(
        Ceil(self->ctx.baseUseN_, self->ctx.curSingleCoreDk_), self->ctx.hwK_), BLOCK_CUBE);
    uint64_t splitedCout1 = ShiftCeilM0(self->ctx.baseUseM_, BLOCK_CUBE);
    // group != realgroup是扩维场景，扩维场景在host侧已经保证的数据不超UBsize，因此不需要对数据做切分
    if(self->ctx.tiling_->group == self->ctx.tiling_->realGroup){
        // 默认切2份，L0C不开double buffer时切4份
        DivTwoNumersInHalf(splitedCin1, splitedCout1);
#if defined(__DAV_C310__)
        if (self->ctx.tiling_->cl0Pbuffer <= 1) {
            DivTwoNumersInHalf(splitedCin1, splitedCout1);
        }
#endif
    }

    uint64_t cutNSize = ShiftMulM0(splitedCin1 * self->ctx.curSingleCoreDk_ * self->ctx.hwK_, BLOCK_CUBE);
    cutNSize = self->ctx.baseUseN_ < cutNSize ? self->ctx.baseUseN_ : cutNSize;
    uint64_t cutMSize = ShiftMulM0(splitedCout1, BLOCK_CUBE);
    uint64_t splitedMIter = Ceil(self->ctx.baseUseM_, cutMSize);
    uint64_t splitedNIter = Ceil(self->ctx.baseUseN_, cutNSize);
    uint64_t workspaceOffset = 0;
    uint64_t pingpongIdx = 0;

    for (uint64_t i = 0; i < splitedNIter; i++) {
        for (uint64_t j = 0; j < splitedMIter; j++) {
            uint64_t inx = pingpongIdx + i * splitedMIter + j;
            deterShape.mSize[inx] = (j == (splitedMIter - 1)) ? (self->ctx.baseUseM_ - cutMSize * j) : cutMSize;
            deterShape.nSize[inx] = (i == (splitedNIter - 1)) ? (self->ctx.baseUseN_ - cutNSize * i) : cutNSize;
            deterShape.mnSize[inx] = deterShape.mSize[inx] * deterShape.nSize[inx];
            deterShape.usedMSize[inx] = j * cutMSize;
            deterShape.usedNSize[inx] = i * cutNSize;
            deterShape.addrOffset[inx] = workspaceOffset;
            deterShape.isNTail[inx] = (self->ctx.curNL0Idx_ == self->ctx.nIter_ - 1) && (i == splitedNIter - 1);
            workspaceOffset += deterShape.mnSize[inx];
        }
    }
}

template <class Intf>
static __aicore__ inline void MovOutL0cForDeterministicRefactor(Intf *self, LocalTensor<typename Intf::L0cT> &l0c,
    const GlobalTensor<typename Intf::DstT> &output)
{
    uint64_t splitedCin1 =
        ShiftCeilM0(Ceil(Ceil(self->ctx.baseUseN_, self->ctx.curSingleCoreDk_), self->ctx.hwK_), BLOCK_CUBE);
    uint64_t splitedCout1 = ShiftCeilM0(self->ctx.baseUseM_, BLOCK_CUBE);
    // group != realgroup是扩维场景，扩维场景在host侧已经保证的数据不超UBsize，因此不需要对数据做切分
    if(self->ctx.tiling_->group == self->ctx.tiling_->realGroup){
        // 默认切2份，L0C不开double buffer时切4份
        DivTwoNumersInHalf(splitedCin1, splitedCout1);
#if defined(__DAV_C310__)
        if (self->ctx.tiling_->cl0Pbuffer <= 1) {
            DivTwoNumersInHalf(splitedCin1, splitedCout1);
        }
#endif
    }

    uint64_t cutNSize =
        ShiftMulM0(static_cast<uint64_t>(splitedCin1) * self->ctx.curSingleCoreDk_ * self->ctx.hwK_, BLOCK_CUBE);

    cutNSize = self->ctx.baseUseN_ < cutNSize ? self->ctx.baseUseN_ : cutNSize;
    uint64_t cutMSize = ShiftMulM0(splitedCout1, BLOCK_CUBE);

    FixpipeParamsC310<CO2Layout::NZ> fixPipeParams;
    fixPipeParams.quantPre = QuantMode_t::NoQuant;
    fixPipeParams.unitFlag = 0;

    uint64_t splitedMIter = Ceil(self->ctx.baseUseM_, cutMSize);
    uint64_t splitedNIter = Ceil(self->ctx.baseUseN_, cutNSize);
    uint64_t workspaceOffset = 0;

    uint64_t alignedUseM = ShiftMulM0(ShiftCeilM0(self->ctx.baseUseM_, BLOCK_CUBE), BLOCK_CUBE);
    for (uint64_t i = 0; i < splitedNIter; i++) {
        fixPipeParams.nSize = (i == (splitedNIter - 1)) ? (self->ctx.baseUseN_ - cutNSize * i) : cutNSize;
        for (uint64_t j = 0; j < splitedMIter; j++) {
            fixPipeParams.mSize = (j == (splitedMIter - 1)) ? (self->ctx.baseUseM_ - cutMSize * j) : cutMSize;
            fixPipeParams.srcStride = alignedUseM;
            fixPipeParams.dstStride = ShiftMulM0(fixPipeParams.mSize, BLOCK_CUBE);
            // l0c: cin1 hkwk cout1 cout0 cin0, gm: cin1 hkwk cout cin0, cin0=16
            uint64_t l0cAddress = i * cutNSize * alignedUseM + ShiftMulM0(j * cutMSize, BLOCK_CUBE);
            Fixpipe<typename Intf::DstT, float, CFG_NZ>(output[workspaceOffset], l0c[l0cAddress], fixPipeParams);
            workspaceOffset += fixPipeParams.mSize * fixPipeParams.nSize;
        }
    }
}

template <class Intf>
static __aicore__ inline void DeterministicUb2Gm(Intf *self, const GlobalTensor<typename Intf::DstT> &userGm,
    DeterMinisticShape &deterShape)
{
    DataCopyExtParams ub2GmParams;
    ub2GmParams.srcStride = 0;
    ub2GmParams.dstStride = 0;
    ub2GmParams.blockCount = 1;
    ub2GmParams.blockLen = deterShape.mnSize[self->ctx.subCoreInx_] * sizeof(typename Intf::DstT);

    event_t eventIdVecToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIdVecToMte3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVecToMte3);
    DataCopyPad<typename Intf::DstT, PaddingMode::Compact>(userGm, self->ctx.vecOutBuf_, ub2GmParams);
}

template <class Intf>
static __aicore__ inline void DeterministicUb2GmNoPingPong(Intf *self, const GlobalTensor<typename Intf::DstT> &userGm,
    DeterMinisticShape &deterShape)
{
    DataCopyExtParams ub2GmParams;
    ub2GmParams.srcStride = 0;
    ub2GmParams.dstStride = 0;
    ub2GmParams.blockCount = 1;
    ub2GmParams.blockLen =  deterShape.mnSize[self->ctx.subCoreInx_ + RELATED_CORE_NUM] * sizeof(typename Intf::DstT);

    auto vecMiddleBuf = self->ctx.vecBuf_.template GetWithOffset<float>(VECTOR_UB_SIZE_HALF >> FLOAT_SHIFT_SIZE,
        VECTOR_UB_SIZE_HALF);
    DataCopyPad<typename Intf::DstT, PaddingMode::Compact>(userGm, vecMiddleBuf, ub2GmParams);
}

static __aicore__ inline bool IsAddTreeLoopEnd(const uint64_t coreAddCnt, const uint64_t deterAddCoreNum)
{
    return (coreAddCnt << 1) >= deterAddCoreNum;
}

template <class Intf>
static __aicore__ inline bool GetIsLastPieceOut(Intf *self, const uint64_t coreAddCnt, const uint64_t coreRelatedIndexTotal)
{
    return coreAddCnt == 1 && self->ctx.tiling_->cl0Pbuffer == 1 && (self->ctx.deterAddCoreNum_ & 1) != 0 &&
        self->ctx.deterAddCoreNum_ == coreRelatedIndexTotal - self->ctx.coreStartIndexTotal_;
}

template <class Intf>
static __aicore__ inline bool GetLoopEndFlag(Intf *self, const uint64_t coreAddCnt)
{
    // 1、二叉树计算中是输出核，需要提前退出累加过程
    // 2、累加核数量为奇数时，最后一个核只需要提前输出一次，提前退出累加过程
    uint64_t doublecoreAddCnt = (coreAddCnt << 1);
    bool isLastCoreOdd = (self->ctx.deterAddCoreIndex_ == self->ctx.deterAddCoreNum_ - 1) &&
        (self->ctx.deterAddCoreNum_ & 1) != 0;
    if (((self->ctx.deterAddCoreIndex_ >> 1) % doublecoreAddCnt > 0) || isLastCoreOdd) {
        return true;
    }
    // 3、累加核数量超过累加次数，完成计算退出
    if (IsAddTreeLoopEnd(coreAddCnt, self->ctx.deterAddCoreNum_)) {
        return true;
    }

    return false;
}

template <class Intf>
static __aicore__ inline uint64_t GetRdGmAddr(Intf *self, const uint64_t coreRelatedIndexTotal,
    DeterMinisticShape &deterShape)
{
    uint64_t cubeUserGmSize = GetBlockNum() * CUBE_WORKSPACE; // cube输出gm总大小
#if defined(__DAV_C310__)
    return cubeUserGmSize + ((coreRelatedIndexTotal << 1) + GetSubBlockIdx()) * QUARTER_CUBE_WORKSPACE;
#elif defined(__NPU_ARCH__) && (__NPU_ARCH__ == 9201)
    uint64_t addCoreRdGmAddr = 0;
    if ((coreRelatedIndexTotal - self->ctx.coreStartIndexTotal_ == self->ctx.deterAddCoreNum_ - 1) &&
        (self->ctx.deterAddCoreNum_ & 1) != 0) { // 最后一个奇数核需要从CUBE_WORKSPACE搬入
        addCoreRdGmAddr = coreRelatedIndexTotal * CUBE_WORKSPACE +
            deterShape.addrOffset[self->ctx.subCoreInx_];
    } else {
        addCoreRdGmAddr =
            cubeUserGmSize + ((coreRelatedIndexTotal << 1) + self->ctx.subCoreInx_) * HALF_CUBE_WORKSPACE;
    }
    return addCoreRdGmAddr;
#endif
}

template <class Intf>
static __aicore__ inline uint64_t GetStGmAddr(Intf *self, const uint64_t cubeUserGmSize)
{
    uint64_t coreIndexTotal = self->ctx.coreStartIndexTotal_ + self->ctx.deterAddCoreIndex_; // 当前核的索引
#if defined(__DAV_C310__)
    return cubeUserGmSize + ((coreIndexTotal << 1) + GetSubBlockIdx()) * QUARTER_CUBE_WORKSPACE;
#elif defined(__NPU_ARCH__) && (__NPU_ARCH__ == 9201)
    return cubeUserGmSize + ((coreIndexTotal << 1) + self->ctx.subCoreInx_) * HALF_CUBE_WORKSPACE;
#endif
}
}  // namespace ConvolutionBackpropFunc

#endif