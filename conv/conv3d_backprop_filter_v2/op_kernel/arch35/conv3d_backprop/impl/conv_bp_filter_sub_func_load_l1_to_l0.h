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
 * \file conv_bp_filter_sub_func_load_l1_to_l0.h
 * \brief L1 to L0 load functions for conv3d backprop filter
 */

#ifndef CONV3D_BP_FILTER_SUB_FUNC_LOAD_L1_TO_L0_H
#define CONV3D_BP_FILTER_SUB_FUNC_LOAD_L1_TO_L0_H

namespace ConvolutionBackpropFunc {

template <class Intf>
static __aicore__ inline void CalcParamsL12L0a(Intf* self)
{
    // (N, Co1, (Ho*Wo)/C16, C16, Co0) -> (N, Co1, (Ho*Wo)/C16, Co0, C16)
    // (m1, k1, k0, m0) Zn -> (k1, m1, m0, k0) Nz
    // mStartPosition/mStep(continuous axis): k1
    // kStartPosition/kStep(non-continuous axis): m1
    if constexpr (IsSameType<typename Intf::SrcT, float>::value) {
        // (N, (Ho*Wo)/C8, Co1, Co0, C8) -> (N, (Ho*Wo)/C8, Co1, Co0, C8)
        if (likely(!self->ctx.isSplitWo_)) {
            self->ctx.load2dv2_.ifTranspose = 0;
            self->ctx.load2dv2_.mStep = ShiftCeilM0(self->ctx.baseUseM_, self->ctx.tiling_->m0);
            self->ctx.load2dv2_.srcStride = self->ctx.load2dv2_.mStep;
            self->ctx.load2dv2_.dstStride = self->ctx.load2dv2_.mStep;
        } else {
            // baseM需要保持m0对齐，fp32场景下，kstep是按照8为单位的，因此kstep一定要是2的倍数
            self->ctx.load2dv2_.kStep = ShiftCeilM0(self->ctx.baseUseM_, self->ctx.tiling_->m0) *
                                        self->ctx.tiling_->m0 / self->ctx.tiling_->channelSize;
            // kstep由于按照8为单位，转置后放到目的地址，分形单位为m0=16
            self->ctx.load2dv2_.dstStride = self->ctx.load2dv2_.kStep >> 1;
            self->ctx.load2dv2_.ifTranspose = 1;
        }
    } else if (self->ctx.baseUseM_ == 1) { // fp16 且self->ctx.baseUseM_ == 1
        self->ctx.load2dv2_.ifTranspose = 0;
        self->ctx.load2dv2_.mStep = ShiftCeilM0(self->ctx.baseUseM_, self->ctx.tiling_->m0);
        self->ctx.load2dv2_.srcStride = self->ctx.load2dv2_.mStep;
        self->ctx.load2dv2_.dstStride = self->ctx.load2dv2_.mStep;
    } else {
        self->ctx.load2dv2_.kStep = ShiftCeilM0(self->ctx.baseUseM_, self->ctx.tiling_->m0);
        self->ctx.load2dv2_.dstStride = self->ctx.load2dv2_.kStep;
        self->ctx.load2dv2_.ifTranspose = 1;
    }
}

template <class Intf>
__aicore__ inline void CalcParamsL12L0b(Intf* self)
{
    // load3dStepK
    if ((self->ctx.dhwK_ != 1) && (IsSameType<typename Intf::SrcT, float>::value)) {
        self->ctx.load3d_.kExtension = self->ctx.tiling_->channelSize;
    } else {
        self->ctx.load3d_.kExtension = self->ctx.baseUseN_;
    }

    // posK
    // 当存在dk循环，且tailN不等于baseN时，如果curNL1Idx_不对self->ctx.cinHkWkLoop_取余，则会精度不对
    // 当dk循环第二次，cinhkwk第一次循环，kStartPt应为0
    uint32_t hkwk16 = self->ctx.hwK_ * self->ctx.tiling_->n0;
    // cinHkWkLoop_不等于1时，stepN偏移需考虑dk
    uint32_t curNL0Idx =
        (self->ctx.enableStepNTail_) ? (self->ctx.curNL0Idx_ % self->ctx.cinHkWkLoop_) : self->ctx.curNL0Idx_;
    uint32_t curStepN = (self->ctx.enableStepNIncludeDkNocinhwk_) ?
                            (self->ctx.tiling_->stepN / self->ctx.cinHkWkLoop_ * self->ctx.cinHkWkLoop_) :
                            self->ctx.tiling_->stepN;
    self->ctx.load3d_.kStartPt = ((self->ctx.curNL1Idx_ % self->ctx.cinHkWkLoop_) * self->ctx.tiling_->baseN) % hkwk16 +
                                 (curNL0Idx % curStepN) * self->ctx.tiling_->baseN;
    if constexpr (IsSameType<typename Intf::SrcT, float>::value) {
        self->ctx.load3d_.channelSize =
            Ceil(self->ctx.load3d_.kStartPt + self->ctx.baseUseN_, hkwk16) * self->ctx.tiling_->n0;
    } else {
        self->ctx.load3d_.channelSize =
            Ceil(self->ctx.load3d_.kStartPt + self->ctx.baseUseN_, hkwk16) * self->ctx.tiling_->channelSize;
    }
}

template <class Intf>
static __aicore__ inline void LoadL12L0a(
    Intf* self, const LocalTensor<typename Intf::SrcT>& l1AMatrix, uint32_t kPos, LocalTensor<typename Intf::SrcT>& l0a)
{
    uint32_t kOffset =
        ShiftDivChannelSize<Intf>((kPos % self->ctx.tiling_->stepKa) * self->ctx.tiling_->baseK, self->ctx.tiling_->k0);
    uint32_t mOffset = ShiftCeilM0(
        (self->ctx.curML0Idx_ % self->ctx.tiling_->stepM) * self->ctx.tiling_->baseM, self->ctx.tiling_->m0);
    if (self->ctx.isSplitWo_ && IsSameType<typename Intf::SrcT, float>::value) {
        mOffset = ShiftDivM0(
            (self->ctx.curML0Idx_ % self->ctx.tiling_->stepM) * self->ctx.tiling_->baseM, self->ctx.tiling_->m0);
        kOffset = ShiftDivM0((kPos % self->ctx.tiling_->stepKa) * self->ctx.tiling_->baseK, self->ctx.tiling_->m0);
    }

    if constexpr (IsSameType<typename Intf::SrcT, float>::value) {
        if (likely(!self->ctx.isSplitWo_)) {
            self->ctx.load2dv2_.kStep = ShiftCeilChannelSize<Intf>(self->ctx.baseUseK_, self->ctx.tiling_->k0);
            self->ctx.srcL12L0aOffset_ =
                (mOffset * self->ctx.tiling_->m0 + kOffset * self->ctx.alignedL1UseM_) * self->ctx.tiling_->k0;
        } else {
            self->ctx.load2dv2_.srcStride = ShiftDivM0(self->ctx.alignedL1UseKa_, self->ctx.tiling_->m0);
            self->ctx.load2dv2_.mStep = ShiftCeilM0(self->ctx.baseUseK_, self->ctx.tiling_->m0);
            self->ctx.srcL12L0aOffset_ =
                (mOffset * self->ctx.alignedL1UseKa_ + kOffset * self->ctx.tiling_->m0) * self->ctx.tiling_->k0 +
                (kPos / self->ctx.tiling_->stepKa) * self->ctx.kal1_ % self->ctx.singleShapeWo_ * self->ctx.tiling_->k0;
        }
    } else if (self->ctx.baseUseM_ == 1) { // fp16 且 baseUseM_ == 1
        self->ctx.load2dv2_.kStep = ShiftCeilChannelSize<Intf>(self->ctx.baseUseK_, self->ctx.tiling_->k0);
        self->ctx.srcL12L0aOffset_ =
            (mOffset * self->ctx.tiling_->m0 + kOffset * self->ctx.alignedL1UseM_) * self->ctx.tiling_->k0;
    } else {
        self->ctx.load2dv2_.srcStride = ShiftDivM0(self->ctx.alignedL1UseKa_, self->ctx.tiling_->k0);
        self->ctx.load2dv2_.mStep = ShiftCeilM0(self->ctx.baseUseK_, self->ctx.tiling_->k0);
        if (likely(!self->ctx.isSplitWo_)) {
            self->ctx.srcL12L0aOffset_ =
                (mOffset * self->ctx.alignedL1UseKa_ + kOffset * self->ctx.tiling_->k0) * self->ctx.tiling_->m0;
        } else {
            self->ctx.srcL12L0aOffset_ =
                (mOffset * self->ctx.alignedL1UseKa_ + kOffset * self->ctx.tiling_->k0) * self->ctx.tiling_->m0 +
                (kPos / self->ctx.tiling_->stepKa) * self->ctx.kal1_ % self->ctx.singleShapeWo_ * self->ctx.tiling_->m0;
        }
    }

    LoadData(l0a[self->ctx.dstL12L0aOffset_], l1AMatrix[self->ctx.srcL12L0aOffset_], self->ctx.load2dv2_);
}

template <class Intf>
__aicore__ inline void LoadL12L0bFp32(
    Intf* self, const LocalTensor<typename Intf::SrcT>& l1BMatrix, LocalTensor<typename Intf::SrcT>& l0b)
{
    // 由于fp32是通过k_repeat重复载入，先加载c0上数据，再加载c1数据（两者并不连续，中间隔着hkwkc0），两者合并为整体数据
    // 当前计算的kStart是c0连续的情况，c0不连续时，跳过已计算的数据为kStart需除以k_repeat，kStart为src跳过的地址
    uint32_t curNL0Idx =
        (self->ctx.enableStepNTail_) ? (self->ctx.curNL0Idx_ % self->ctx.cinHkWkLoop_) : self->ctx.curNL0Idx_;
    uint32_t curStepN = (self->ctx.enableStepNIncludeDkNocinhwk_) ?
                            (self->ctx.tiling_->stepN / self->ctx.cinHkWkLoop_ * self->ctx.cinHkWkLoop_) :
                            self->ctx.tiling_->stepN;
    uint32_t calculatedL1N = (curNL0Idx % curStepN) * self->ctx.tiling_->baseN;
    uint32_t hkwk16 = self->ctx.hwK_ * self->ctx.tiling_->n0;
    self->ctx.load3d_.kStartPt =
        (((self->ctx.curNL1Idx_ % self->ctx.cinHkWkLoop_) * self->ctx.tiling_->baseN) % hkwk16) / DOUBLE +
        calculatedL1N / hkwk16 * hkwk16 + (calculatedL1N % hkwk16) / DOUBLE;

    // 先加载c0上的数据， 使用k_repeat方式来加载c1上的8个数， 从而凑齐16个数
    uint16_t repeatStride = (self->ctx.tiling_->singleCoreCin <= 8) ?
                                self->ctx.hwK_ * self->ctx.curSingleCoreDk_ :
                                self->ctx.hwK_; // 跳hkwk个c0就能读取c1上的数据，src跳过的地址
    uint8_t repeatTime = DOUBLE;                // repeat 2次
    uint8_t repeatMode = 1;                     // K方向repeat
    // 跳c1hkwk16放howo方向的数据，即m_dst_stride
    uint16_t dstStride = static_cast<uint16_t>(ShiftCeilM0(self->ctx.baseUseN_, self->ctx.tiling_->n0));
#if defined(ASC_DEVKIT_VERSION_NUM) && (ASC_DEVKIT_VERSION_NUM >= 90000000)
    LoadDataRepeatParamWithStride repeatParam = {repeatStride, repeatTime, repeatMode, dstStride};
    SetLoadDataRepeatWithStride(repeatParam);
#else
    LoadDataRepeatParam repeatParam = {repeatStride, repeatTime, repeatMode, dstStride};
    SetLoadDataRepeat(repeatParam);
#endif

    // 由于load3d不支持配置k_dst_stride, 因此循环下hkwk条load3d命令去实现k_dst_stride
    for (uint16_t i = 0; i < dstStride; i++) {
        // 每次k方向的dst地址跳512B
        constexpr uint32_t baseBlock = 128; // self->ctx.tiling_->n0 * self->ctx.tiling_->k0
#if defined(ASC_DEVKIT_VERSION_NUM) && (ASC_DEVKIT_VERSION_NUM >= 90000000)
        LoadDataWithStride(l0b[i * baseBlock], l1BMatrix, self->ctx.load3d_);
#else
        LoadData(l0b[i * baseBlock], l1BMatrix, self->ctx.load3d_);
#endif
        // 每次k方向上src跳一个cin0,取的是连续cin0， 即hkwk上的cin0
        self->ctx.load3d_.kStartPt += self->ctx.tiling_->channelSize;
        if (self->ctx.tiling_->singleCoreCin > 8 &&
            // 判断是否读完一轮的hkwk的条件为kStart是否对hkwkc0整除
            (self->ctx.load3d_.kStartPt % (self->ctx.hwK_ * self->ctx.tiling_->channelSize) == 0)) {
            // 跳过已经repeat的数据
            self->ctx.load3d_.kStartPt += (self->ctx.hwK_ * self->ctx.tiling_->channelSize);
        }
    }
}

template <class Intf>
__aicore__ inline void LoadL12L0b(
    Intf* self, const LocalTensor<typename Intf::SrcT>& l1BMatrix, LocalTensor<typename Intf::SrcT>& l0b)
{
    // load3dStepM
    // fp32时，K轴为8，N轴为16，M轴为16，因此kstep为baseUseK_/8
    self->ctx.load3d_.mExtension = self->ctx.baseUseK_; // 尾块时，可以不用对齐，而非尾块时，刚好对齐的
    if ((self->ctx.dhwK_ != 1) && (IsSameType<typename Intf::SrcT, float>::value)) {
        if constexpr (!Intf::conv3ddwConfig.isSplitKernelHW) {
            LoadL12L0bFp32(self, l1BMatrix, l0b);
        } else {
#if defined(ASC_DEVKIT_VERSION_NUM) && (ASC_DEVKIT_VERSION_NUM >= 90000000)
            LoadDataRepeatParamWithStride repeatParam = {0, 1, 0, 1};
            SetLoadDataRepeatWithStride(repeatParam);
            LoadDataWithStride(l0b, l1BMatrix, self->ctx.load3d_);
#else
            LoadDataRepeatParam repeatParam = {0, 1, 0, 1};
            SetLoadDataRepeat(repeatParam);
            LoadData(l0b, l1BMatrix, self->ctx.load3d_);
#endif
        }
    } else {
#if defined(ASC_DEVKIT_VERSION_NUM) && (ASC_DEVKIT_VERSION_NUM >= 90000000)
        LoadDataRepeatParamWithStride repeatParam = {
            0, 1, 0, static_cast<uint16_t>(ShiftCeilM0(self->ctx.baseUseN_, self->ctx.tiling_->n0))};
        if constexpr (Intf::conv3ddwConfig.isSplitKernelHW) {
            repeatParam = {0, 1, 0, 1}; // 切hk/wk后，每次循环一个hkwk，因此修改dstStride为1个单元
        }
        SetLoadDataRepeatWithStride(repeatParam);
        LoadDataWithStride(l0b, l1BMatrix, self->ctx.load3d_);
#else
        LoadDataRepeatParam repeatParam = {
            0, 1, 0, static_cast<uint16_t>(ShiftCeilM0(self->ctx.baseUseN_, self->ctx.tiling_->n0))};
        if constexpr (Intf::conv3ddwConfig.isSplitKernelHW) {
            repeatParam = {0, 1, 0, 1}; // 切hk/wk后，每次循环一个hkwk，因此修改dstStride为1个单元
        }
        SetLoadDataRepeat(repeatParam);
        LoadData(l0b, l1BMatrix, self->ctx.load3d_);
#endif
    }
}

} // namespace ConvolutionBackpropFunc

#endif
