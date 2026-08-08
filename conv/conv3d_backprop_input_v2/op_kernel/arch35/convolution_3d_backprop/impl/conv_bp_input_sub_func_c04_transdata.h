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
 * \file conv_bp_input_sub_func_c04_transdata.h
 * \brief C04 mode weight transdata for conv3d backprop input
 */

#ifndef CONV3D_BP_INPUT_SUB_FUNC_C04_TRANSDATA_H
#define CONV3D_BP_INPUT_SUB_FUNC_C04_TRANSDATA_H

#include "conv_bp_input_sub_func_utils.h"
#include "conv_bp_input_sub_func_sync.h"
#include "conv_bp_input_sub_func_vector_intrinsics.h"

using AscendC::DivCeil;
using AscendC::LocalTensor;

namespace Convolution3DBackpropFunc {

template <class Intf>
static __aicore__ inline void InitUbZero4C04(Intf* self, uint32_t b1CinSize)
{
    self->ctx.ndVecTensor_ = self->ctx.ndVecBuf_.template Get<typename Intf::SrcBT>();
    uint32_t ubCinSize = (b1CinSize < self->ctx.vecBlockN_) ? b1CinSize : self->ctx.vecBlockN_;
    uint64_t ubPixCount = static_cast<uint64_t>(ubCinSize) * self->ctx.tiling_->dkHkWk;
    uint64_t ubOffset = DivDtypeByte<typename Intf::SrcBT>(AscendC::ONE_BLOCK_SIZE);
    ubOffset += (self->ctx.tiling_->cout * self->ctx.vecBlockN_ * self->ctx.tiling_->dkHkWk);
    for (uint8_t i = self->ctx.tiling_->cout; i < C04_COUT_SIZE; i++) {
        Duplicate<typename Intf::SrcBT>(self->ctx.ndVecTensor_[ubOffset], 0, ubPixCount);
        ubOffset += (self->ctx.vecBlockN_ * self->ctx.tiling_->dkHkWk);
    }
}

template <class Intf>
static __aicore__ inline void LoadUb4C04(Intf* self, uint32_t cinBlockSize, uint64_t srcGmOffset)
{
    LoopModeParams loopParams;
    loopParams.loop2Size = 1;
    loopParams.loop2SrcStride = 0;
    loopParams.loop2DstStride = 0;
    loopParams.loop1Size = 1;
    loopParams.loop1SrcStride = 0;
    loopParams.loop1DstStride = 0;
    SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);

    // The size of 'self->ctx.tiling_->dkHkWk * sizeof(typename Intf::SrcBT)' is not larger than the size of UB.
    // It will be checked in CheckC04Enable(), in range of uint32_t.
    uint32_t cinDataLen = static_cast<uint32_t>(self->ctx.tiling_->dkHkWk) * sizeof(typename Intf::SrcBT);
    uint32_t oriBlockLen = cinBlockSize * cinDataLen;
    uint32_t alignedBlockLen = AlignUp(oriBlockLen, AscendC::ONE_BLOCK_SIZE);
    uint8_t rightPadding = DivDtypeByte<typename Intf::SrcBT>(alignedBlockLen - oriBlockLen);
    DataCopyPadExtParams<typename Intf::SrcBT> padParams{true, 0, rightPadding, 0};

    DataCopyExtParams gm2UbParams;
    gm2UbParams.blockLen = oriBlockLen;
    gm2UbParams.blockCount = self->ctx.tiling_->cout;
    gm2UbParams.srcStride = (self->ctx.tiling_->cin - cinBlockSize) * cinDataLen;
    gm2UbParams.dstStride = (self->ctx.vecBlockN_ * cinDataLen - alignedBlockLen) >> ONE_BLK_SHIFT_SIZE;

    uint32_t ubOffset = DivDtypeByte<typename Intf::SrcBT>(AscendC::ONE_BLOCK_SIZE);
    DataCopyPad<typename Intf::SrcBT, PaddingMode::Normal>(
        self->ctx.ndVecTensor_[ubOffset], self->ctx.weightGlobal_[srcGmOffset], gm2UbParams, padParams);
}

template <class Intf>
static __aicore__ inline void SetGatherIdx4C04(Intf* self)
{
    self->ctx.idxVecTensor_ = self->ctx.idxVecBuf_.template Get<typename Intf::IndexT>();
    typename Intf::IndexT idxVal = 0;
    uint8_t idx = 0;
    for (int8_t kernelIdx = self->ctx.tiling_->c0 / C04_COUT_SIZE - 1; kernelIdx >= 0; --kernelIdx) {
        idxVal = kernelIdx;
        for (uint8_t coutIdx = 0; coutIdx < C04_COUT_SIZE; ++coutIdx) {
            self->ctx.idxVecTensor_.SetValue(idx, idxVal);
            ++idx;
            idxVal += (self->ctx.vecBlockN_ * self->ctx.tiling_->dkHkWk);
        }
    }

    event_t eventId = static_cast<event_t>(self->ctx.pipe_.FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventId);
    WaitFlag<HardEvent::S_V>(eventId);

    auto idxAddr = (__ubuf__ typename Intf::IndexT*)self->ctx.idxVecTensor_.GetPhyAddr();
    uint16_t repeatTimes = static_cast<uint16_t>(DivDtypeByte<typename Intf::IndexT>(AscendC::VECTOR_REG_WIDTH) >>
                                                 self->ctx.tiling_->c0BitsB) -
                           1;
    uint32_t mask = self->ctx.tiling_->c0;
    uint16_t numPerRepeat = self->ctx.tiling_->c0;
    uint16_t dstOffset = self->ctx.tiling_->c0;
    auto cinStride = static_cast<typename Intf::IndexT>(self->ctx.tiling_->dkHkWk);

    ExpandGatherIdxByStride<typename Intf::IndexT>(idxAddr, repeatTimes, numPerRepeat, dstOffset, mask, cinStride);
}

template <class Intf>
static __aicore__ inline void SetGatherTailMask4C04(Intf* self)
{
    self->ctx.maskVecTensor_ = self->ctx.maskVecBuf_.template Get<uint32_t>();
    uint32_t maskVal = 0xffffffff;
    uint32_t kernelNumInC0 = self->ctx.tiling_->c0 >> C04_SHIFT_SIZE;
    uint32_t tmpVal = self->ctx.tiling_->hkWk % kernelNumInC0;
    if constexpr (std::is_same<typename Intf::SrcBT, float>::value) {
        if (tmpVal == 1) { // 最后一个分形，VGather只需要从UB中取hkwk_rev中的最后1个点
            maskVal = 0xffff;
        }
    } else if constexpr (std::is_same<typename Intf::SrcBT, bfloat16_t>::value ||
                         std::is_same<typename Intf::SrcBT, half>::value) {
        if (tmpVal == 3) { // 最后一个分形，VGather只需要从UB中取hkwk_rev中的最后3个点
            maskVal = 0xffffff;
        } else if (tmpVal == 2) { // 最后一个分形，VGather只需要从UB中取hkwk_rev中的最后2个点
            maskVal = 0xffff;
        } else if (tmpVal == 1) {
            maskVal = 0xff;
        }
    }
    uint8_t repeatTimes = static_cast<uint8_t>(DivDtypeByte<typename Intf::IndexT>(AscendC::VECTOR_REG_WIDTH) >>
                                               self->ctx.tiling_->c0BitsB);
    for (uint8_t idx = 0; idx < repeatTimes; ++idx) {
        self->ctx.maskVecTensor_.SetValue(idx, maskVal);
    }
}

template <class Intf>
static __aicore__ inline void SetIdxAndMask4C04(Intf* self, uint32_t loopIdx)
{
    // VGather的index只需要生成一次
    if (likely(self->ctx.c04LoadToB1IterIdx_ != GetSubBlockIdx() || loopIdx > 0)) {
        return;
    }
    SetGatherIdx4C04<Intf>(self);
    PipeBarrier<PIPE_V>();

    SetGatherTailMask4C04<Intf>(self);
    event_t eventId = static_cast<event_t>(self->ctx.pipe_.FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventId);
    WaitFlag<HardEvent::S_V>(eventId);
}

template <class Intf>
static __aicore__ inline void Dn2Nz4C04(Intf* self, uint32_t cinBlockSize, uint32_t curDkIdx, uint32_t loopIdx)
{
    SetIdxAndMask4C04<Intf>(self, loopIdx);

    uint32_t C0PerReg = DivDtypeByte<typename Intf::IndexT>(AscendC::VECTOR_REG_WIDTH) >> self->ctx.tiling_->c0BitsB;
    uint16_t C0LoopTimes = BLOCK_CUBE / C0PerReg;
    uint32_t kernelNumInC0 = self->ctx.tiling_->c0 >> C04_SHIFT_SIZE;
    uint16_t k1 = static_cast<uint16_t>(DivCeil(self->ctx.tiling_->hkWk, kernelNumInC0));
    uint16_t n1 = static_cast<uint16_t>(DivCeil16(cinBlockSize));
    uint16_t n1IterMax = n1 * C0LoopTimes;
    uint32_t srcN1Stride = C0PerReg * self->ctx.tiling_->dkHkWk;
    uint32_t srcK1Stride = kernelNumInC0;
    uint32_t dstN1Stride = C0PerReg << self->ctx.tiling_->c0BitsB;
    uint32_t dstK1Stride = AlignUp16(cinBlockSize) << self->ctx.tiling_->c0BitsB;

    self->ctx.nzVecTensor_ = self->ctx.nzVecBuf_.template Get<typename Intf::SrcBT>();

    auto idxAddr = (__ubuf__ typename Intf::IndexT*)self->ctx.idxVecTensor_.GetPhyAddr();
    auto srcAddr = (__ubuf__ typename Intf::SrcBT*)self->ctx.ndVecTensor_.GetPhyAddr();
    srcAddr += (DivDtypeByte<typename Intf::SrcBT>(AscendC::ONE_BLOCK_SIZE) + curDkIdx * self->ctx.tiling_->hkWk);
    srcAddr -= ((k1 * kernelNumInC0) - self->ctx.tiling_->hkWk);
    auto dstAddr = (__ubuf__ typename Intf::SrcBT*)self->ctx.nzVecTensor_.GetPhyAddr();
    auto maskAddr = (__ubuf__ uint32_t*)self->ctx.maskVecTensor_.GetPhyAddr();

    GatherDn2Nz4C04<typename Intf::SrcBT, typename Intf::IndexT>(idxAddr, srcAddr, dstAddr, maskAddr, k1, n1IterMax,
                                                                 srcN1Stride, srcK1Stride, dstN1Stride, dstK1Stride);
}

template <class Intf>
static __aicore__ inline void CopyUb2L14C04(Intf* self, uint32_t cinBlockSize, uint32_t b1CinSize,
                                            LocalTensor<typename Intf::SrcBT>& useB1Buf, uint32_t dstB1Offset)
{
    if (cinBlockSize == b1CinSize) { // UB上的格式转换不需要切块
        DataCopyParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = (AlignUp16(cinBlockSize) *
                               AlignUp(self->ctx.tiling_->hkWk * C04_COUT_SIZE, self->ctx.tiling_->c0) *
                               sizeof(typename Intf::SrcBT)) >>
                              ONE_BLK_SHIFT_SIZE;
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        DataCopy(useB1Buf, self->ctx.nzVecTensor_, copyParams);
    } else {
        DataCopyParams copyParams;
        uint32_t kernelNumInC0 = self->ctx.tiling_->c0 >> C04_SHIFT_SIZE;
        copyParams.blockCount = DivCeil(self->ctx.tiling_->hkWk, kernelNumInC0);
        copyParams.blockLen = AlignUp16(cinBlockSize);
        copyParams.srcStride = 0;
        copyParams.dstStride = AlignUp16(b1CinSize) - AlignUp16(cinBlockSize);
        DataCopy(useB1Buf[dstB1Offset], self->ctx.nzVecTensor_, copyParams);
    }
}

template <class Intf>
static __aicore__ inline void C04TransdataWeightCore(Intf* self, uint32_t b1CinSize, uint64_t srcGmOffset,
                                                     LocalTensor<typename Intf::SrcBT>& useB1Buf, uint32_t curDkIdx)
{
    uint32_t loopCnt = DivCeil(b1CinSize, self->ctx.vecBlockN_);
    uint32_t cinRemain = b1CinSize;
    uint32_t cinBlockSize = (cinRemain < self->ctx.vecBlockN_) ? cinRemain : self->ctx.vecBlockN_;
    uint32_t dstB1Offset = 0;
    for (uint32_t i = 0; i < loopCnt; i++) {
        // DataCopy wait VDup or prev loop VGather
        event_t eventId = static_cast<event_t>(self->ctx.pipe_.FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(eventId);
        WaitFlag<HardEvent::V_MTE2>(eventId);
        // DataCopy (MTE2)
        LoadUb4C04<Intf>(self, cinBlockSize, srcGmOffset);
        srcGmOffset += (static_cast<uint64_t>(cinBlockSize) * self->ctx.tiling_->dkHkWk);

        // VGather wait DataCopy
        eventId = static_cast<event_t>(self->ctx.pipe_.FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventId);
        WaitFlag<HardEvent::MTE2_V>(eventId);
        // VGather wait prev loop MOV_UB_TO_L1
        eventId = static_cast<event_t>(self->ctx.pipe_.FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(eventId);
        WaitFlag<HardEvent::MTE3_V>(eventId);
        // VGather
        Dn2Nz4C04<Intf>(self, cinBlockSize, curDkIdx, i);

        // MOV_UB_TO_L1 wait prev VGather
        eventId = static_cast<event_t>(self->ctx.pipe_.FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventId);
        WaitFlag<HardEvent::V_MTE3>(eventId);
        // MOV_UB_TO_L1 (MTE3)
        CopyUb2L14C04<Intf>(self, cinBlockSize, b1CinSize, useB1Buf, dstB1Offset);
        dstB1Offset += (AlignUp16(cinBlockSize) << self->ctx.tiling_->c0BitsB);

        cinRemain -= cinBlockSize;
        cinBlockSize = (cinRemain < self->ctx.vecBlockN_) ? cinRemain : self->ctx.vecBlockN_;
    }
}

template <class Intf>
__aicore__ inline void C04TransdataWeight(Intf* self, const uint32_t kIdx, uint32_t curDkIdx)
{
    WaitForCubeBeforeLoadToB1<Intf>(self);
    LocalTensor<typename Intf::SrcBT> useB1Tbuf = GetB1TbufByFlag<Intf>(self, self->ctx.b1PingPongFlag_);

    if (GetSubBlockIdx() == (self->ctx.c04LoadToB1IterIdx_ & 1)) {
        uint32_t curCinIdx = self->ctx.curNIdx_ * self->ctx.tiling_->baseN;
        uint32_t b1CinSize = CalcCurCinSizeB1(self, curCinIdx);
        uint64_t srcGmOffset = static_cast<uint64_t>(curCinIdx) * self->ctx.tiling_->dkHkWk;

        // 调用指令不支持hif8，暂时隔离开，否则编译不通过
        if constexpr (!std::is_same<typename Intf::SrcBT, hifloat8_t>::value &&
                      !std::is_same<typename Intf::SrcBT, fp8_e4m3fn_t>::value &&
                      !std::is_same<typename Intf::SrcBT, int8_t>::value) {
            // 每个AIV只在第一次计算时需要清零
            if (GetSubBlockIdx() == self->ctx.c04LoadToB1IterIdx_) {
                InitUbZero4C04<Intf>(self, b1CinSize); // VDup
            }
            C04TransdataWeightCore<Intf>(self, b1CinSize, srcGmOffset, useB1Tbuf, curDkIdx);
        }
    }

    NotifyCubeAfterLoadToB1<Intf>(self);
}

} // namespace Convolution3DBackpropFunc

#endif
