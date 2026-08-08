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
 * \file conv_bp_input_sub_func_group_transdata.h
 * \brief Group mode weight transdata for conv3d backprop input
 */

#ifndef CONV3D_BP_INPUT_SUB_FUNC_GROUP_TRANSDATA_H
#define CONV3D_BP_INPUT_SUB_FUNC_GROUP_TRANSDATA_H

#include "conv_bp_input_sub_func_utils.h"
#include "conv_bp_input_sub_func_sync.h"
#include "conv_bp_input_sub_func_index_calc.h"
#include "conv_bp_input_sub_func_vector_intrinsics.h"

using AscendC::LocalTensor;
using AscendC::NdDmaConfig;

namespace Convolution3DBackpropFunc {

template <class Intf>
static __aicore__ inline void InitUbZero4Group(Intf* self)
{
    // Set ndVecBuf to zero.
    // size is cout1G * hk * wk * cinG * BLOCK_CUBE * c0
    self->ctx.ndVecTensor_ = self->ctx.ndVecBuf_.template Get<typename Intf::SrcBT>();
    uint32_t groupHalfUbSize = self->ctx.tiling_->cout1G * self->ctx.tiling_->hkWk * self->ctx.tiling_->cin1G *
                                   BLOCK_CUBE * sizeof(typename Intf::SrcBT)
                               << self->ctx.tiling_->c0BitsB;
    Duplicate<typename Intf::SrcBT>(self->ctx.ndVecTensor_, 0, groupHalfUbSize / sizeof(typename Intf::SrcBT));
}

/*
 * B matrix format: [enlarge*coutPerGroup, cinPerGroup, 1, hk, wk] -> [enlarge*coutPerGroup, enlarge*cinPerGroup, hk,
 * wk]
 */
template <class Intf>
static __aicore__ inline void LoadUbDiag4GroupNCDHW(Intf* self, uint64_t srcGmOffset)
{
    if (unlikely(self->ctx.groupIterIdx_ == GetSubBlockIdx())) {
        uint32_t cinPerGroup = self->ctx.tiling_->cinG / self->ctx.tiling_->enlarge;
        uint32_t coutPerGroup = self->ctx.tiling_->coutG / self->ctx.tiling_->enlarge;
        // 因为扩维，所以是cinG
        uint64_t dstCoutGStride = (self->ctx.curEnlargeCin1_ * self->ctx.tiling_->hkWk) << SHIFT_BIT_4;
        // 因为对角化，所以dstEnlargeStride加上了cinPerGroup * hk * wk
        uint64_t dstEnlargeStride = coutPerGroup * dstCoutGStride + cinPerGroup * self->ctx.tiling_->hkWk;
        // NDDMA Loop0 params
        self->ctx.groupCopyParams_.loopInfo.loopSize[INDEX_0] = self->ctx.tiling_->hkWk;
        self->ctx.groupCopyParams_.loopInfo.loopSrcStride[INDEX_0] = 1;
        self->ctx.groupCopyParams_.loopInfo.loopDstStride[INDEX_0] = 1;
        // NDDMA Loop1 params
        self->ctx.groupCopyParams_.loopInfo.loopSize[INDEX_1] = cinPerGroup;
        self->ctx.groupCopyParams_.loopInfo.loopSrcStride[INDEX_1] = self->ctx.tiling_->dkHkWk;
        self->ctx.groupCopyParams_.loopInfo.loopDstStride[INDEX_1] = self->ctx.tiling_->hkWk;
        // NDDMA Loop2 params
        self->ctx.groupCopyParams_.loopInfo.loopSize[INDEX_2] = coutPerGroup;
        self->ctx.groupCopyParams_.loopInfo.loopSrcStride[INDEX_2] = cinPerGroup * self->ctx.tiling_->dkHkWk;
        self->ctx.groupCopyParams_.loopInfo.loopDstStride[INDEX_2] = dstCoutGStride;
        // NDDMA Loop3 params
        self->ctx.groupCopyParams_.loopInfo.loopSrcStride[INDEX_3] = coutPerGroup * cinPerGroup *
                                                                     self->ctx.tiling_->dkHkWk;
        self->ctx.groupCopyParams_.loopInfo.loopDstStride[INDEX_3] = dstEnlargeStride;
    }
    self->ctx.groupCopyParams_.loopInfo.loopSize[INDEX_3] = self->ctx.curEnlarge;
    DataCopy<typename Intf::SrcBT, GROUP_NDDMA_DIM_NUM, nddmaConfig>(
        self->ctx.ndVecTensor_, self->ctx.weightGlobal_[srcGmOffset], self->ctx.groupCopyParams_);
}

template <class Intf>
static __aicore__ inline void LoadUbDiag4GroupNDHWC(Intf* self, uint64_t srcGmOffset)
{
    if (unlikely(self->ctx.groupIterIdx_ == GetSubBlockIdx())) {
        uint32_t cinPerGroup = self->ctx.tiling_->cinG / self->ctx.tiling_->enlarge;
        uint32_t coutPerGroup = self->ctx.tiling_->coutG / self->ctx.tiling_->enlarge;
        uint64_t dstCoutGStride = (self->ctx.curEnlargeCin1_ * self->ctx.tiling_->hkWk) << SHIFT_BIT_4;
        // NDDMA Loop0 params
        self->ctx.groupCopyParams_.loopInfo.loopSize[INDEX_0] = cinPerGroup;
        self->ctx.groupCopyParams_.loopInfo.loopSrcStride[INDEX_0] = 1;
        self->ctx.groupCopyParams_.loopInfo.loopDstStride[INDEX_0] = self->ctx.tiling_->hkWk;
        // NDDMA Loop1 params
        self->ctx.groupCopyParams_.loopInfo.loopSize[INDEX_1] = self->ctx.tiling_->hkWk;
        self->ctx.groupCopyParams_.loopInfo.loopSrcStride[INDEX_1] = cinPerGroup;
        self->ctx.groupCopyParams_.loopInfo.loopDstStride[INDEX_1] = 1;
        // NDDMA Loop2 params
        self->ctx.groupCopyParams_.loopInfo.loopSize[INDEX_2] = coutPerGroup;
        self->ctx.groupCopyParams_.loopInfo.loopSrcStride[INDEX_2] = cinPerGroup * self->ctx.tiling_->dkHkWk;
        self->ctx.groupCopyParams_.loopInfo.loopDstStride[INDEX_2] = dstCoutGStride;
        // NDDMA Loop3 params
        self->ctx.groupCopyParams_.loopInfo.loopSrcStride[INDEX_3] = coutPerGroup * cinPerGroup *
                                                                     self->ctx.tiling_->dkHkWk;
        self->ctx.groupCopyParams_.loopInfo.loopDstStride[INDEX_3] = coutPerGroup * dstCoutGStride +
                                                                     cinPerGroup * self->ctx.tiling_->hkWk;
    }
    self->ctx.groupCopyParams_.loopInfo.loopSize[INDEX_3] = self->ctx.curEnlarge;
    DataCopy<typename Intf::SrcBT, GROUP_NDDMA_DIM_NUM, nddmaConfig>(
        self->ctx.ndVecTensor_, self->ctx.weightGlobal_[srcGmOffset], self->ctx.groupCopyParams_);
}

template <class Intf>
static __aicore__ inline void LoadUbDiag4GroupDHWCN(Intf* self, uint64_t srcGmOffset)
{
    if (unlikely(self->ctx.groupIterIdx_ == GetSubBlockIdx())) {
        uint32_t cinPerGroup = self->ctx.tiling_->cinG / self->ctx.tiling_->enlarge;
        uint32_t coutPerGroup = self->ctx.tiling_->coutG / self->ctx.tiling_->enlarge;
        uint64_t dstCoutGStride = (self->ctx.curEnlargeCin1_ * self->ctx.tiling_->hkWk) << SHIFT_BIT_4;
        // NDDMA Loop0 params
        self->ctx.groupCopyParams_.loopInfo.loopSize[INDEX_0] = coutPerGroup;
        self->ctx.groupCopyParams_.loopInfo.loopSrcStride[INDEX_0] = 1;
        self->ctx.groupCopyParams_.loopInfo.loopDstStride[INDEX_0] = dstCoutGStride;
        // NDDMA Loop1 params
        self->ctx.groupCopyParams_.loopInfo.loopSize[INDEX_1] = cinPerGroup;
        self->ctx.groupCopyParams_.loopInfo.loopSrcStride[INDEX_1] = self->ctx.tiling_->cout;
        self->ctx.groupCopyParams_.loopInfo.loopDstStride[INDEX_1] = self->ctx.tiling_->hkWk;
        // NDDMA Loop2 params
        self->ctx.groupCopyParams_.loopInfo.loopSize[INDEX_2] = self->ctx.tiling_->hkWk;
        self->ctx.groupCopyParams_.loopInfo.loopSrcStride[INDEX_2] = cinPerGroup * self->ctx.tiling_->cout;
        self->ctx.groupCopyParams_.loopInfo.loopDstStride[INDEX_2] = 1;
        // NDDMA Loop3 params
        self->ctx.groupCopyParams_.loopInfo.loopSrcStride[INDEX_3] = coutPerGroup;
        self->ctx.groupCopyParams_.loopInfo.loopDstStride[INDEX_3] = coutPerGroup * dstCoutGStride +
                                                                     cinPerGroup * self->ctx.tiling_->hkWk;
    }
    self->ctx.groupCopyParams_.loopInfo.loopSize[INDEX_3] = self->ctx.curEnlarge;
    DataCopy<typename Intf::SrcBT, GROUP_NDDMA_DIM_NUM, nddmaConfig>(
        self->ctx.ndVecTensor_, self->ctx.weightGlobal_[srcGmOffset], self->ctx.groupCopyParams_);
}

template <class Intf>
static __aicore__ inline void SetGatherIdxDn2Nz(Intf* self)
{
    // cinG * hk * wk * 16 should be in uint16 range(ubSize constrain satisfy this)
    /* gen gather index for: [coutG, cinG, hk, wk] -> [coutG1, hk, wk, cinG1, cin0, cout0]
       [cinG * hk * wk * [0, 1, ..., c0 - 1] + 0 * hk * wk,
        cinG * hk * wk * [0, 1, ..., c0 - 1] + 1 * hk * wk,
        ...
        cinG * hk * wk * [0, 1, ..., c0 - 1] + (repeatTimes - 1) * hk * wk]
     */
    self->ctx.idxVecTensor_ = self->ctx.idxVecBuf_.template Get<typename Intf::IndexT>();
    typename Intf::IndexT idxVal = 0;
    // cinG * hk * wk * [0, 1, ..., c0 - 1]
    for (uint8_t idx = 0; idx < self->ctx.tiling_->c0; ++idx) {
        self->ctx.idxVecTensor_.SetValue(idx, idxVal);
        idxVal += static_cast<typename Intf::IndexT>((self->ctx.curEnlargeCin1_ * self->ctx.tiling_->hkWk)
                                                     << SHIFT_BIT_4);
    }

    event_t eventId = static_cast<event_t>(self->ctx.pipe_.FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventId);
    WaitFlag<HardEvent::S_V>(eventId);

    auto idxAddr = (__ubuf__ typename Intf::IndexT*)self->ctx.idxVecTensor_.GetPhyAddr();
    uint16_t repeatTimes = static_cast<uint16_t>(
        AscendC::VECTOR_REG_WIDTH / (sizeof(typename Intf::IndexT) << self->ctx.tiling_->c0BitsB) - 1);
    uint16_t numPerRepeat = self->ctx.tiling_->c0;
    uint16_t dstOffset = self->ctx.tiling_->c0;
    uint32_t mask = self->ctx.tiling_->c0;
    auto cinStride = static_cast<typename Intf::IndexT>(self->ctx.tiling_->hkWk);

    ExpandGatherIdxByStride<typename Intf::IndexT>(idxAddr, repeatTimes, numPerRepeat, dstOffset, mask, cinStride);
}

template <class Intf>
static __aicore__ inline void Dn2Nz4Group(Intf* self)
{
    // [coutG, cinG, hk, wk] -> [cinG1, hk, wk, coutG1, cout0, cin0]
    // 其中coutG = enlarge * coutPerGroup, cinG = enlarge * cinPerGroup
    // gather index only need set once
    if (unlikely(self->ctx.groupIterIdx_ == GetSubBlockIdx())) {
        SetGatherIdxDn2Nz<Intf>(self);
        PipeBarrier<PIPE_V>();
    }

    uint32_t C0PerReg = AscendC::VECTOR_REG_WIDTH / (sizeof(typename Intf::IndexT) << self->ctx.tiling_->c0BitsB);
    uint16_t C0LoopTimes = BLOCK_CUBE / C0PerReg;
    uint32_t srcCout1GStride = (BLOCK_CUBE * self->ctx.curEnlargeCin1_ * self->ctx.tiling_->hkWk)
                               << self->ctx.tiling_->c0BitsB;
    uint32_t srcCin1GStride = C0PerReg * self->ctx.tiling_->hkWk;
    uint32_t dstCout1GStride = (self->ctx.tiling_->hkWk * self->ctx.curEnlargeCin1_ << self->ctx.tiling_->c0BitsB)
                               << SHIFT_BIT_4;
    uint32_t dstKStride = (self->ctx.curEnlargeCin1_ << self->ctx.tiling_->c0BitsB) << SHIFT_BIT_4;
    uint32_t dstCin1GStride = C0PerReg << self->ctx.tiling_->c0BitsB;

    self->ctx.nzVecTensor_ = self->ctx.nzVecBuf_.template Get<typename Intf::SrcBT>();

    auto idxAddr = (__ubuf__ typename Intf::IndexT*)self->ctx.idxVecTensor_.GetPhyAddr();
    auto srcAddr = (__ubuf__ typename Intf::SrcBT*)self->ctx.ndVecTensor_.GetPhyAddr();
    auto dstAddr = (__ubuf__ typename Intf::SrcBT*)self->ctx.nzVecTensor_.GetPhyAddr();

    // ub size is 253952, half is 126976, bfloat16 data num max is 63488, which is smaller than uint16_max.
    // that is cout1G * hk * wk * cinG * BLOCK_CUBE * c0 < 63488
    uint16_t cout1G = static_cast<uint16_t>(DivCeilC0<Intf>(self, self->ctx.singleShapeCout_));
    uint16_t hkWk = static_cast<uint16_t>(self->ctx.tiling_->hkWk);
    uint16_t cin1G = static_cast<uint16_t>(self->ctx.curEnlargeCin1_);
    uint16_t cin1GIterMax = cin1G * C0LoopTimes;

    GatherDn2Nz4Group<typename Intf::SrcBT, typename Intf::IndexT>(idxAddr, srcAddr, dstAddr, cout1G, hkWk,
                                                                   cin1GIterMax, srcCout1GStride, srcCin1GStride,
                                                                   dstCout1GStride, dstKStride, dstCin1GStride);
}

template <class Intf>
static __aicore__ inline void CopyUb2L14Group(Intf* self, uint32_t curCout1Size, uint32_t curCin1Cin0Size,
                                              uint32_t srcUbOffset, LocalTensor<typename Intf::SrcBT>& useB1Buf)
{
    // from [coutG1, hk, wk, cinG1, cin0, cout0] extract [curCout1Size, hk, wk, curCin1Cin0Size, cout0],
    // copy data from ub to L1
    DataCopyParams copyParams;
    copyParams.blockCount = curCout1Size * self->ctx.tiling_->hkWk;
    copyParams.blockLen = curCin1Cin0Size;
    copyParams.srcStride = (self->ctx.curEnlargeCin1_ << SHIFT_BIT_4) - curCin1Cin0Size;
    copyParams.dstStride = 0;
    DataCopy(useB1Buf, self->ctx.nzVecTensor_[srcUbOffset], copyParams);
}

template <class Intf>
static __aicore__ inline void GroupTransdataWeightCore(Intf* self, uint32_t curCinSize, uint32_t curCoutSize,
                                                       uint64_t srcGmOffset, uint32_t srcUbOffset,
                                                       LocalTensor<typename Intf::SrcBT>& useB1Buf)
{
    uint32_t curCin1Cin0Size = AlignUp16(curCinSize);
    uint32_t curCout1Size = DivCeilC0<Intf>(self, curCoutSize);
    uint32_t curCin1Size = DivCeil16(curCinSize);
#ifndef __CCE_KT_TEST__
    CvCrossCoreWait<Intf, PIPE_MTE1, PIPE_V>(self, FLAG_MTE1_ID_2);
#endif
    // vdup wait vgather
    if (self->ctx.groupIterIdx_ > GetSubBlockIdx()) {
        PipeBarrier<PIPE_V>();
    }
    InitUbZero4Group<Intf>(self); // vDup (pipe_v)

    // nddma wait vdup
    event_t eventId = static_cast<event_t>(self->ctx.pipe_.FetchEventID(HardEvent::V_MTE2));
    SetFlag<HardEvent::V_MTE2>(eventId);
    WaitFlag<HardEvent::V_MTE2>(eventId);
    if constexpr (Intf::Config::xType::format == Convolution3DBackprop::CubeFormat::NCDHW) {
        LoadUbDiag4GroupNCDHW<Intf>(self, srcGmOffset); // NDDMA (MTE2)
    } else if constexpr (Intf::Config::xType::format == Convolution3DBackprop::CubeFormat::NDHWC) {
        LoadUbDiag4GroupNDHWC<Intf>(self, srcGmOffset);
    } else { // DHWCN
        LoadUbDiag4GroupDHWCN<Intf>(self, srcGmOffset);
    }

    // vgather wait nddma
    eventId = static_cast<event_t>(self->ctx.pipe_.FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventId);
    WaitFlag<HardEvent::MTE2_V>(eventId);
    Dn2Nz4Group<Intf>(self); // vGather (pipe_v)

    // next loop SetValue wait current loop vGather
    eventId = static_cast<event_t>(self->ctx.pipe_.FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventId);
    WaitFlag<HardEvent::V_S>(eventId);

    // ub2l1 wait vgather
    eventId = static_cast<event_t>(self->ctx.pipe_.FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventId);
    WaitFlag<HardEvent::V_MTE3>(eventId);
    CopyUb2L14Group<Intf>(self, curCout1Size, curCin1Cin0Size, srcUbOffset, useB1Buf); // MOV_UB_TO_L1 (MTE3)
#ifndef __CCE_KT_TEST__
    CvCrossCoreSet<Intf, PIPE_MTE3, PIPE_MTE1>(self, FLAG_MTE1_ID_1);
#endif
}

template <class Intf>
__aicore__ inline void GroupTransdataWeight(Intf* self, uint32_t kIdx, uint32_t curDkIdx)
{
    if (GetSubBlockIdx() != 0) {
        return;
    }

    LocalTensor<typename Intf::SrcBT> useB1Tbuf = GetB1TbufByFlag<Intf>(self, self->ctx.b1PingPongFlag_);

    uint32_t curCinIdx = self->ctx.curNIdx_ * self->ctx.tiling_->baseN;
    uint32_t curCinSize = CalcCurCinSizeB1(self, curCinIdx);
    uint32_t curCoutIdx = 0;
    uint32_t curCoutSize = 0;
    CalcCoutIndexAndSizeB1<Intf, false>(self, kIdx, curCoutIdx, curCoutSize);

    uint32_t curCin1Idx = (self->ctx.curCinStartIdx_ + curCinIdx) >> SHIFT_BIT_4;
    uint32_t curCout1Idx = (self->ctx.curCoutStartIdx_ + curCoutIdx) >> self->ctx.tiling_->c0BitsB;

    uint64_t srcGmOffset = 0;
    if constexpr (Intf::Config::xType::format == Convolution3DBackprop::CubeFormat::NCDHW) {
        srcGmOffset = static_cast<uint64_t>(curDkIdx) * self->ctx.tiling_->hkWk;
    } else if (Intf::Config::xType::format == Convolution3DBackprop::CubeFormat::NDHWC) {
        srcGmOffset = static_cast<uint64_t>(curDkIdx) * self->ctx.tiling_->hkWk * self->ctx.tiling_->cinG /
                      self->ctx.tiling_->enlarge;
    } else { // DHWCN
        srcGmOffset = static_cast<uint64_t>(curDkIdx) * self->ctx.tiling_->hkWk * self->ctx.tiling_->cinG /
                      self->ctx.tiling_->enlarge * self->ctx.tiling_->cout;
    }
    uint32_t srcUbOffset = (curCout1Idx * self->ctx.tiling_->hkWk * self->ctx.curEnlargeCin1_ + curCin1Idx)
                           << self->ctx.tiling_->c0BitsB << SHIFT_BIT_4;
    // 调用指令不支持hif8，暂时隔离开，否则编译不通过
    if constexpr (!std::is_same<typename Intf::SrcBT, hifloat8_t>::value &&
                  !std::is_same<typename Intf::SrcBT, fp8_e4m3fn_t>::value &&
                  !std::is_same<typename Intf::SrcBT, int8_t>::value) {
        GroupTransdataWeightCore<Intf>(self, curCinSize, curCoutSize, srcGmOffset, srcUbOffset, useB1Tbuf);
    }
}

} // namespace Convolution3DBackpropFunc

#endif
