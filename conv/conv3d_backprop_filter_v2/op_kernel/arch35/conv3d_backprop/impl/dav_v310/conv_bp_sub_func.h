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
 * \file conv_bp_sub_func.h
 * \brief
 */

#ifndef CONV3D_BP_FILTER_SUB_FUNC_H
#define CONV3D_BP_FILTER_SUB_FUNC_H
#include "conv_bp_sub_func_load_gm_to_l1.h"
#include "conv_bp_sub_func_deterministic_common.h"

namespace ConvolutionBackpropFunc {
constexpr int32_t MAX_BLOCK_COUNT = 4095;
constexpr int32_t MIN_BLOCK_COUNT = 1;
constexpr int32_t MAX_BLOCK_LEN = 65535;
constexpr int32_t MAX_16BITS_STRIDE = 65535;
constexpr uint8_t FLAG_FIXP_ID = 5;
constexpr uint8_t FLAG_MTE3_ID = 6;
constexpr uint8_t FLAG_ID_MAX = 16;
constexpr FixpipeConfig CFG_NZ_UB = {CO2Layout::NZ, true};

template <class Intf>
static __aicore__ inline void InitLoadToA2Params(Intf *self)
{
    self->ctx.dstL12L0aOffset_ = 0;
    self->ctx.srcL12L0aOffset_ = 0;
    self->ctx.alignedL1UseKa_ = 0;
    self->ctx.load2dv2_.mStartPosition = 0;
    self->ctx.load2dv2_.kStartPosition = 0;
    self->ctx.load2dv2_.mStep = 0;
    self->ctx.load2dv2_.kStep = 0;
    self->ctx.load2dv2_.srcStride = 0;
    self->ctx.load2dv2_.dstStride = 0;
    self->ctx.load2dv2_.ifTranspose = 1;
    if constexpr (IsSameType<typename Intf::SrcT, float>::value || IsSameType<typename Intf::SrcT, hifloat8_t>::value) {
        self->ctx.alignedL1UseM_ = 0;
    }
}

// 计算Load2B2的指令参数
template <class Intf>
static __aicore__ inline void InitLoadToB2Params(Intf *self)
{
    // load3dStepK
    self->ctx.load3d_.kExtension = self->ctx.tiling_->baseN;
    // load3dStepM
    self->ctx.load3d_.mExtension = self->ctx.tiling_->baseM;
    // posK
    self->ctx.load3d_.kStartPt = 0;
     // posM
    self->ctx.load3d_.mStartPt = 0;
    self->ctx.load3d_.strideW = self->ctx.tiling_->strideW;
    self->ctx.load3d_.strideH = self->ctx.tiling_->strideH;
    self->ctx.load3d_.filterW = self->ctx.tiling_->wk;
    self->ctx.load3d_.filterH = self->ctx.tiling_->hk;
    self->ctx.load3d_.dilationFilterW = self->ctx.tiling_->dilationW;
    self->ctx.load3d_.dilationFilterH = self->ctx.tiling_->dilationH;
    self->ctx.load3d_.filterSizeW = (self->ctx.tiling_->wk >> 8) & 255;
    self->ctx.load3d_.filterSizeH = (self->ctx.tiling_->hk >> 8) & 255;
    self->ctx.load3d_.enTranspose = 0;
    self->ctx.load3d_.fMatrixCtrl = 0;
    self->ctx.load3d_.channelSize = 16;
}

template <class Intf>
static __aicore__ inline void InitSetFmatrixParams(Intf *self)
{
    // W
    self->ctx.load3d_.l1W = self->ctx.tiling_->wi;
    // H
    self->ctx.load3d_.l1H = 1;
    self->ctx.load3d_.padList[0] = self->ctx.tiling_->padLeft;
    self->ctx.load3d_.padList[1] = self->ctx.tiling_->padRight;
    // padUp now is set in Load BL1
    self->ctx.load3d_.padList[3] = 255;  // 255: set max pad_bottom for compatibility
}

template <class Intf>
static __aicore__ inline void CalcParamsMmad(Intf *self)
{
    self->ctx.mmad_.m = self->ctx.baseUseM_;
    self->ctx.mmad_.n = self->ctx.baseUseN_;
}

template <class Intf>
static __aicore__ inline void InitMmadParams(Intf *self)
{
    self->ctx.dstL0cOffset_ = 0;
    self->ctx.srcL0aOffset_ = 0;
    self->ctx.srcL0bOffset_ = 0;
    self->ctx.mmad_.m = self->ctx.tiling_->baseM;
    self->ctx.mmad_.k = self->ctx.tiling_->baseK;
    self->ctx.mmad_.n = self->ctx.tiling_->baseN;
    self->ctx.mmad_.unitFlag = 0;
    self->ctx.mmad_.kDirectionAlign = 0;
    self->ctx.mmad_.cmatrixSource = 0;
    self->ctx.mmad_.cmatrixInitVal = 0;

    // cinG表示每组cin的个数
    self->ctx.cinG_ = self->ctx.tiling_->cin;
    if (self->ctx.tiling_->group != 1) {
        // 扩维情况和depthwise情况均不会使用此参数
        self->ctx.cinG_ = self->ctx.tiling_->cin1G;
    }
}

template <class Intf>
static __aicore__ inline void CalcParamsL12L0a(Intf *self)
{
    // (N, Co1, (Ho*Wo)/C16, C16, Co0) -> (N, Co1, (Ho*Wo)/C16, Co0, C16)
    // (m1, k1, k0, m0) Zn -> (k1, m1, m0, k0) Nz
    // mStartPosition/mStep(continuous axis): k1
    // kStartPosition/kStep(non-continuous axis): m1
    if constexpr (IsSameType<typename Intf::SrcT, float>::value || IsSameType<typename Intf::SrcT, hifloat8_t>::value) {
    // (N, (Ho*Wo)/C8, Co1, Co0, C8) -> (N, (Ho*Wo)/C8, Co1, Co0, C8)
        if (!self->ctx.isSplitWo_) {
            self->ctx.load2dv2_.ifTranspose = 0;
            self->ctx.load2dv2_.mStep = ShiftCeilM0(self->ctx.baseUseM_, self->ctx.tiling_->m0);
            self->ctx.load2dv2_.srcStride = self->ctx.load2dv2_.mStep;
            self->ctx.load2dv2_.dstStride = self->ctx.load2dv2_.mStep;
        } else {
            //baseM需要保持m0对齐，fp32场景下，kstep是按照8为单位的，因此kstep一定要是2的倍数
            self->ctx.load2dv2_.kStep = ShiftCeilM0(self->ctx.baseUseM_, self->ctx.tiling_->m0) * self->ctx.tiling_->m0
             / self->ctx.tiling_->channelSize;
            //kstep由于按照8为单位，转置后放到目的地址，分形单位为m0=16
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
__aicore__ inline void CalcParamsL12L0b(Intf *self)
{
    // load3dStepK
    if ((self->ctx.dhwK_ != 1) &&
        (IsSameType<typename Intf::SrcT, float>::value || IsSameType<typename Intf::SrcT, hifloat8_t>::value)) {
        self->ctx.load3d_.kExtension = self->ctx.tiling_->channelSize;
    } else {
        self->ctx.load3d_.kExtension = self->ctx.baseUseN_;
    }

    // posK
    // 当存在dk循环，且tailN不等于baseN时，如果curNL1Idx_不对self->ctx.cinHkWkLoop_取余，则会精度不对
    // 当dk循环第二次，cinhkwk第一次循环，kStartPt应为0
    uint32_t hkwk16 = self->ctx.hwK_ * self->ctx.tiling_->n0;
    // cinHkWkLoop_不等于1时，stepN偏移需考虑dk
    uint32_t curNL0Idx = (self->ctx.enableStepNTail_) ? (self->ctx.curNL0Idx_ % self->ctx.cinHkWkLoop_) :
            self->ctx.curNL0Idx_;
    uint32_t curStepN = (self->ctx.enableStepNIncludeDkNocinhwk_) ? (self->ctx.tiling_->stepN /
        self->ctx.cinHkWkLoop_ * self->ctx.cinHkWkLoop_) : self->ctx.tiling_->stepN;
    self->ctx.load3d_.kStartPt = ((self->ctx.curNL1Idx_ % self->ctx.cinHkWkLoop_) * self->ctx.tiling_->baseN) % hkwk16 +
        (curNL0Idx % curStepN) * self->ctx.tiling_->baseN;
    if constexpr (IsSameType<typename Intf::SrcT, float>::value) {
        self->ctx.load3d_.channelSize = Ceil(self->ctx.load3d_.kStartPt + self->ctx.baseUseN_, hkwk16) *
            self->ctx.tiling_->n0;
    } else {
        self->ctx.load3d_.channelSize = Ceil(self->ctx.load3d_.kStartPt + self->ctx.baseUseN_, hkwk16) *
            self->ctx.tiling_->channelSize;
    }
}

template <class Intf>
static __aicore__ inline void LoadL12L0a(Intf *self, const LocalTensor<typename Intf::SrcT> &l1AMatrix,
                                         uint32_t kPos, LocalTensor<typename Intf::SrcT> &l0a)
{
    uint32_t kOffset = ShiftDivChannelSize<Intf>((kPos % self->ctx.tiling_->stepKa) * self->ctx.tiling_->baseK,
        self->ctx.tiling_->k0);
    uint32_t mOffset = ShiftCeilM0((self->ctx.curML0Idx_ % self->ctx.tiling_->stepM) * self->ctx.tiling_->baseM,
        self->ctx.tiling_->m0);
    if (self->ctx.isSplitWo_ && IsSameType<typename Intf::SrcT, float>::value) {
        mOffset = ShiftDivM0((self->ctx.curML0Idx_ % self->ctx.tiling_->stepM) * self->ctx.tiling_->baseM, self->ctx.tiling_->m0);
        kOffset = ShiftDivM0((kPos % self->ctx.tiling_->stepKa) * self->ctx.tiling_->baseK, self->ctx.tiling_->m0);
    }

    if constexpr (IsSameType<typename Intf::SrcT, float>::value || IsSameType<typename Intf::SrcT, hifloat8_t>::value) {
        if (!self->ctx.isSplitWo_) {
            self->ctx.load2dv2_.kStep = ShiftCeilChannelSize<Intf>(self->ctx.baseUseK_, self->ctx.tiling_->k0);
            self->ctx.srcL12L0aOffset_ =
                (mOffset * self->ctx.tiling_->m0 + kOffset * self->ctx.alignedL1UseM_) * self->ctx.tiling_->k0;
        } else {
        self->ctx.load2dv2_.srcStride = ShiftDivM0(self->ctx.alignedL1UseKa_, self->ctx.tiling_->m0);
        self->ctx.load2dv2_.mStep = ShiftCeilM0(self->ctx.baseUseK_, self->ctx.tiling_->m0);
        self->ctx.srcL12L0aOffset_ = (mOffset * self->ctx.alignedL1UseKa_ + kOffset * self->ctx.tiling_->m0) * self->ctx.tiling_->k0 +
            (kPos / self->ctx.tiling_->stepKa) * self->ctx.kal1_ % self->ctx.singleShapeWo_ * self->ctx.tiling_->k0;
        }
    } else if (self->ctx.baseUseM_ == 1) { // fp16 且 baseUseM_ == 1
        self->ctx.load2dv2_.kStep = ShiftCeilChannelSize<Intf>(self->ctx.baseUseK_, self->ctx.tiling_->k0);
        self->ctx.srcL12L0aOffset_ =
            (mOffset * self->ctx.tiling_->m0 + kOffset * self->ctx.alignedL1UseM_) * self->ctx.tiling_->k0;
    } else {
        self->ctx.load2dv2_.srcStride = ShiftDivM0(self->ctx.alignedL1UseKa_, self->ctx.tiling_->k0);
        self->ctx.load2dv2_.mStep = ShiftCeilM0(self->ctx.baseUseK_, self->ctx.tiling_->k0);
        if (!self->ctx.isSplitWo_) {
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
__aicore__ inline void LoadL12L0bFp32(Intf *self, const LocalTensor<typename Intf::SrcT> &l1BMatrix,
                                  LocalTensor<typename Intf::SrcT> &l0b)
{
    // 由于fp32是通过k_repeat重复载入，先加载c0上数据，再加载c1数据（两者并不连续，中间隔着hkwkc0），两者合并为整体数据
    // 当前计算的kStart是c0连续的情况，c0不连续时，跳过已计算的数据为kStart需除以k_repeat，kStart为src跳过的地址
    uint32_t curNL0Idx = (self->ctx.enableStepNTail_) ? (self->ctx.curNL0Idx_ % self->ctx.cinHkWkLoop_) :
        self->ctx.curNL0Idx_;
    uint32_t curStepN = (self->ctx.enableStepNIncludeDkNocinhwk_) ? (self->ctx.tiling_->stepN /
        self->ctx.cinHkWkLoop_ * self->ctx.cinHkWkLoop_) : self->ctx.tiling_->stepN;
    uint32_t calculatedL1N = (curNL0Idx % curStepN) * self->ctx.tiling_->baseN;
    uint32_t hkwk16 = self->ctx.hwK_ * self->ctx.tiling_->n0;
    self->ctx.load3d_.kStartPt = (((self->ctx.curNL1Idx_ % self->ctx.cinHkWkLoop_) * self->ctx.tiling_->baseN) %
        hkwk16) / DOUBLE + calculatedL1N / hkwk16 * hkwk16 + (calculatedL1N % hkwk16) / DOUBLE;

    // 先加载c0上的数据， 使用k_repeat方式来加载c1上的8个数， 从而凑齐16个数
    uint16_t repeatStride =(self->ctx.tiling_->singleCoreCin <= 8) ?
        self->ctx.hwK_ * self->ctx.curSingleCoreDk_ : self->ctx.hwK_; // 跳hkwk个c0就能读取c1上的数据，src跳过的地址
    uint8_t repeatTime = DOUBLE; // repeat 2次
    uint8_t repeatMode = 1; // K方向repeat
    // 跳c1hkwk16放howo方向的数据，即m_dst_stride
    uint16_t dstStride = static_cast<uint16_t>(ShiftCeilM0(self->ctx.baseUseN_, self->ctx.tiling_->n0));
    LoadDataRepeatParam repeatParam = {repeatStride, repeatTime, repeatMode, dstStride};
    SetLoadDataRepeat(repeatParam);

    // 由于load3d不支持配置k_dst_stride, 因此循环下hkwk条load3d命令去实现k_dst_stride
    for (uint16_t i = 0; i < dstStride; i++) {
        // 每次k方向的dst地址跳512B
        constexpr uint32_t baseBlock = 128; //self->ctx.tiling_->n0 * self->ctx.tiling_->k0
        LoadData(l0b[i * baseBlock], l1BMatrix, self->ctx.load3d_);
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
__aicore__ inline void LoadL12L0bFp8(Intf *self, const LocalTensor<typename Intf::SrcT> &l1BMatrix,
                                  LocalTensor<typename Intf::SrcT> &l0b)
{
    self->ctx.load3d_.mExtension = self->ctx.baseUseK_ > self->ctx.tiling_->k0 ? self->ctx.tiling_->k0 :
        self->ctx.baseUseK_;
    // 由于fp8的c0为32，即dkc12hkwkc11c0，先加载c0上的数据， 使用k_repeat方式来加载hkwk个数，循环dkc12c11次
    // 每个c0之间间隔c1数据，kStartPt需考虑下一次加载的是c0数据还是c1数据
    bool baseNNoAllHkWk = ((self->ctx.tiling_->baseN >> 4) % self->ctx.hwK_ != 0);
    uint32_t curNL0Idx = (self->ctx.enableStepNTail_) ? (self->ctx.curNL0Idx_ % self->ctx.cinHkWkLoop_) :
        self->ctx.curNL0Idx_;
    uint32_t curStepN = (self->ctx.enableStepNIncludeDkNocinhwk_) ? (self->ctx.tiling_->stepN /
        self->ctx.cinHkWkLoop_ * self->ctx.cinHkWkLoop_) : self->ctx.tiling_->stepN;
    uint32_t calculatedL1N = (curNL0Idx % curStepN) * self->ctx.tiling_->baseN;
    uint32_t hkwk16 = self->ctx.hwK_ * self->ctx.tiling_->n0;
    // kStartPt为BL1到BL0的起始跳过的地址，它由两部分组成，一部分是baseN，一部分是StepN
    // kStartPt的位置可能是wk、hk、等多种情况，以下计算确定hkwk的位置
    self->ctx.load3d_.kStartPt = ((self->ctx.curNL1Idx_ * self->ctx.tiling_->baseN) %
        hkwk16) * DOUBLE + (calculatedL1N % hkwk16) * DOUBLE;

    LoadDataRepeatParam repeatParam = {0, 1, 0, 0};
    SetLoadDataRepeat(repeatParam);
    uint32_t mLoopTime = ShiftCeilChannelSize<Intf>(self->ctx.baseUseK_, self->ctx.tiling_->k0);
    uint32_t hwkLoopTime = self->ctx.hwK_;
    if (baseNNoAllHkWk) {
        hwkLoopTime = ShiftCeilM0(self->ctx.baseUseN_, self->ctx.tiling_->n0) % self->ctx.hwK_;
    }
    constexpr uint32_t baseBlock = 512;
    uint32_t l0bDstStride = 0;
    auto kStartPt = self->ctx.load3d_.kStartPt;

    for (uint16_t i = 0; i < mLoopTime; i++) {
        self->ctx.load3d_.kStartPt = kStartPt;
        for (uint16_t j = 0; j < hwkLoopTime; j++) {
            LoadData(l0b[l0bDstStride], l1BMatrix, self->ctx.load3d_);
            AscendC::PipeBarrier<PIPE_MTE1>();
            l0bDstStride += baseBlock;
            self->ctx.load3d_.kStartPt += self->ctx.tiling_->k0;
        }
        self->ctx.load3d_.mStartPt += self->ctx.tiling_->k0;
    }
}

template <class Intf>
__aicore__ inline void LoadL12L0b(Intf *self, const LocalTensor<typename Intf::SrcT> &l1BMatrix,
                                  LocalTensor<typename Intf::SrcT> &l0b)
{
    // load3dStepM
    // fp32时，K轴为8，N轴为16，M轴为16，因此kstep为baseUseK_/8
    self->ctx.load3d_.mExtension = self->ctx.baseUseK_; // 尾块时，可以不用对齐，而非尾块时，刚好对齐的
    if ((self->ctx.dhwK_ != 1) && (IsSameType<typename Intf::SrcT, float>::value)) {
        if (!self->ctx.tiling_->isSplitKernelHW) {
            LoadL12L0bFp32(self, l1BMatrix, l0b);
        } else {
            LoadDataRepeatParam repeatParam = {0, 1, 0, 1};
            SetLoadDataRepeat(repeatParam);
            LoadData(l0b, l1BMatrix, self->ctx.load3d_);
        }
    } else if ((self->ctx.dhwK_ != 1) && (IsSameType<typename Intf::SrcT, hifloat8_t>::value)) {
        LoadL12L0bFp8(self, l1BMatrix, l0b);
    } else {
        LoadDataRepeatParam repeatParam = {0, 1, 0,
            static_cast<uint16_t>(ShiftCeilM0(self->ctx.baseUseN_, self->ctx.tiling_->n0))};
        if (self->ctx.tiling_->isSplitKernelHW) {
            repeatParam = {0, 1, 0, 1}; //切hk/wk后，每次循环一个hkwk，因此修改dstStride为1个单元
        }
        SetLoadDataRepeat(repeatParam);
        LoadData(l0b, l1BMatrix, self->ctx.load3d_);
    }
}


template <class Intf>
static __aicore__ inline void MmadLocal(Intf *self, const LocalTensor<typename Intf::SrcT> &l0a,
    const LocalTensor<typename Intf::SrcT> &l0b, LocalTensor<typename Intf::L0cT> &l0c)
{
    Mmad(l0c[self->ctx.dstL0cOffset_],
        l0a[self->ctx.srcL0aOffset_],
        l0b[self->ctx.srcL0bOffset_],
        self->ctx.mmad_);

    // MMAD计算量baseM*baseN小于一定阈值时需要添加PIPE_M同步,当前平台阈值为10*256
    if (self->ctx.mmad_.m * self->ctx.mmad_.n < 2560) {
        AscendC::PipeBarrier<PIPE_M>();
    }
}

template <class Intf>
static __aicore__ inline void LoadL0c2GmDkhkwkEqOne(Intf *self, const GlobalTensor<typename Intf::DstT> &output,
    LocalTensor<typename Intf::L0cT> &l0c)
{
    bool tailCinExist = (self->ctx.singleShapeCin_ & 0xF) != 0;
    CalcL0cParams(self, tailCinExist);

    int64_t dstOffset = self->ctx.curML0Idx_ * self->ctx.tiling_->baseM * self->ctx.dhwK_ *
        self->ctx.cinG_ + self->ctx.curNL0Idx_ * self->ctx.tiling_->baseN * self->ctx.tiling_->dk;

    // the problem is simplified to (ci1, co1, co0, ci0) -> (co, ci) NZ2ND
    AscendC::FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixPipeParams;
    fixPipeParams.params.ndNum = 1;
    fixPipeParams.mSize = self->ctx.baseUseM_;

    if (tailCinExist && (self->ctx.cinRemainLen_ < self->ctx.baseUseN_)) {
        fixPipeParams.nSize = self->ctx.cinRemainLen_;
    } else {
        fixPipeParams.nSize = self->ctx.baseUseN_;
    }

    // N
    fixPipeParams.srcStride = ShiftCeilM0(self->ctx.baseUseM_, BLOCK_CUBE) * BLOCK_CUBE;
    // D
    fixPipeParams.dstStride = self->ctx.cinG_;

    fixPipeParams.quantPre = QuantMode_t::NoQuant;
    fixPipeParams.unitFlag = 0;

    AscendC::Fixpipe<typename Intf::DstT, float, CFG_ROW_MAJOR>(output[dstOffset],
        l0c, fixPipeParams);
    if (tailCinExist) {
        self->ctx.cinRemainLen_ -= self->ctx.baseUseN_;
    }
    // 更新上一次N的Idx
    self->ctx.lastNIdx_ = self->ctx.curNL0Idx_;
    return;
}

template <class Intf>
static __aicore__ inline void LoadL0c2GmDkhkwkEqOneNz2DHWCN(Intf *self, const GlobalTensor<typename Intf::DstT> &output,
    LocalTensor<typename Intf::L0cT> &l0c)
{
    int64_t dstOffset = self->ctx.curML0Idx_ * self->ctx.tiling_->baseM +
            self->ctx.curNL0Idx_ * self->ctx.tiling_->baseN * self->ctx.tiling_->cout;

    // the problem is simplified to (ci1, co1, co0, ci0) -> (ci, co) NZ2DN
    AscendC::FixpipeParamsC310<CO2Layout::COLUMN_MAJOR> fixPipeParams;
    fixPipeParams.params.dnNum = 1;
    fixPipeParams.mSize = self->ctx.baseUseM_;
    if (self->ctx.bL1cin1CopyLen >= self->ctx.baseUseN_) {
        fixPipeParams.nSize = self->ctx.baseUseN_;
    } else {
        fixPipeParams.nSize = self->ctx.bL1cin1CopyLen % self->ctx.baseUseN_;
    }

    // N
    fixPipeParams.srcStride = ShiftCeilM0(self->ctx.baseUseM_, BLOCK_CUBE) * BLOCK_CUBE;
    // D
    fixPipeParams.dstStride = self->ctx.tiling_->cout;
    // loop0_src_stride, n_src_stride
    fixPipeParams.params.srcNzC0Stride = 1;
    // loop3_src_stride, nd_src_stride
    fixPipeParams.params.srcNzMatrixStride = 1;
    // loop3_dst_stride, dn_dst_stride
    fixPipeParams.params.dstDnMatrixStride = 1;

    fixPipeParams.quantPre = QuantMode_t::NoQuant;
    fixPipeParams.unitFlag = 0;

    AscendC::Fixpipe<typename Intf::DstT, float, CFG_COLUMN_MAJOR>(output[dstOffset], l0c, fixPipeParams);

    return;
}

template <class Intf>
static __aicore__ inline void LoadL0c2GmBaseNUndivided(Intf *self, const GlobalTensor<typename Intf::DstT> &output,
    LocalTensor<typename Intf::L0cT> &l0c)
{
    constexpr uint64_t baseCin = BLOCK_CUBE;
    bool tailCinExist = (self->ctx.singleShapeCin_ % baseCin != 0);
    CalcL0cParams(self, tailCinExist);

    AscendC::FixpipeParamsC310<CO2Layout::COLUMN_MAJOR> fixPipeParams;
    uint64_t nValueSum = 0;
    int64_t dstOffset = self->ctx.curML0Idx_ * self->ctx.tiling_->baseM * self->ctx.cinG_ * self->ctx.dhwK_;
    uint32_t c1hkwk = ShiftDivM0(self->ctx.baseUseN_, baseCin);
    uint32_t numBaseNIncludeHwk = (c1hkwk + self->ctx.head_ >= self->ctx.hwK_) ?
        (c1hkwk - (self->ctx.hwK_ - self->ctx.head_)) : 0;
    uint64_t baseNIter = Ceil(numBaseNIncludeHwk, self->ctx.hwK_) + 1;

    for (uint64_t j = 0; j < baseNIter; j++) {
        self->ctx.tail_ = (c1hkwk - nValueSum + self->ctx.head_) > self->ctx.hwK_ ? self->ctx.hwK_ :
            (c1hkwk - nValueSum + self->ctx.head_);

        fixPipeParams.params.dnNum = self->ctx.baseUseM_;
        // loop3_src_stride, dn_src_stride
        fixPipeParams.params.srcNzMatrixStride = 1;
        // loop3_dst_stride, dn_dst_stride
        fixPipeParams.params.dstDnMatrixStride = self->ctx.cinG_ * self->ctx.dhwK_;
        // 当dk>1且分核时，nSize必须为真实值，原因是多写的部分会跟下一个核出现同地址，不同核atomicAdd，导致精度错误
        if (tailCinExist && (self->ctx.cinRemainLen_ < baseCin)) {
            fixPipeParams.nSize = self->ctx.cinRemainLen_;
        } else {
            fixPipeParams.nSize = baseCin;
        }
        // loop1_src_stride, d_src_stride
        fixPipeParams.srcStride = self->ctx.hwK_ * ShiftCeilM0(self->ctx.baseUseM_, BLOCK_CUBE) * BLOCK_CUBE;
        // loop2_dst_stride, d_dst_stride
        fixPipeParams.dstStride = self->ctx.dhwK_;

        uint32_t nValue = self->ctx.tail_ - self->ctx.head_;
        fixPipeParams.mSize = nValue;

        // loop0_src_stride, n_src_stride
        fixPipeParams.params.srcNzC0Stride = ShiftCeilM0(self->ctx.baseUseM_, BLOCK_CUBE) * BLOCK_CUBE;

        fixPipeParams.quantPre = QuantMode_t::NoQuant;
        fixPipeParams.unitFlag = 0;

        int64_t srcOffset = nValueSum * ShiftCeilM0(self->ctx.baseUseM_, BLOCK_CUBE) * BLOCK_CUBE * BLOCK_CUBE;
        nValueSum += nValue;
        int64_t baseNAlignLength = baseCin * self->ctx.hwK_;
        int64_t dstDkOffset = (self->ctx.curNL0Idx_ / self->ctx.cinHkWkLoop_) * self->ctx.hwK_;
        int64_t dstCinOffset = (self->ctx.curNL0Idx_ % self->ctx.cinHkWkLoop_) *
            self->ctx.tiling_->baseN / baseNAlignLength * baseNAlignLength * self->ctx.tiling_->dk;

        int64_t dstBaseNOffset = dstOffset + dstDkOffset + dstCinOffset + self->ctx.head_ +
            j * baseNAlignLength * self->ctx.tiling_->dk;
        self->ctx.head_ = self->ctx.tail_ == self->ctx.hwK_ ? 0 : self->ctx.tail_;

        AscendC::Fixpipe<typename Intf::DstT, float, CFG_COLUMN_MAJOR>(output[dstBaseNOffset],
            l0c[srcOffset], fixPipeParams);
        // 当cin有尾块时，在cin跨行时更新cinRemainLen_的值
        if (tailCinExist && self->ctx.head_ == 0) {
            self->ctx.cinRemainLen_ -= baseCin;
        }
    }
    // 更新上一次N的Idx
    self->ctx.lastNIdx_ = self->ctx.curNL0Idx_;
    return;
}

template <class Intf>
static __aicore__ inline void LoadL0c2GmBaseNUndividedNz2Nd(Intf *self, const GlobalTensor<typename Intf::DstT> &output,
    LocalTensor<typename Intf::L0cT> &l0c)
{
    constexpr uint64_t baseCin = BLOCK_CUBE;
    bool tailCinExist = (self->ctx.singleShapeCin_ % baseCin != 0);
    CalcL0cParams(self, tailCinExist);

    AscendC::FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixPipeParams;
    uint64_t nValueSum = 0;
    int64_t dstOffset = self->ctx.curML0Idx_ * self->ctx.tiling_->baseM * self->ctx.cinG_ * self->ctx.dhwK_;
    uint32_t c1hkwk = ShiftDivM0(self->ctx.baseUseN_, baseCin);
    uint64_t baseNIter = Ceil(c1hkwk - (self->ctx.hwK_ - self->ctx.head_), self->ctx.hwK_) + 1;

    for (uint64_t j = 0; j < baseNIter; j++) {
        self->ctx.tail_ = (c1hkwk - nValueSum + self->ctx.head_) > self->ctx.hwK_ ? self->ctx.hwK_ :
            (c1hkwk - nValueSum + self->ctx.head_);
        uint32_t nValue = self->ctx.tail_ - self->ctx.head_;

        fixPipeParams.params.ndNum = nValue;
        // loop3_src_stride, nd_src_stride
        fixPipeParams.params.srcNdStride = ShiftCeilM0(self->ctx.baseUseM_, BLOCK_CUBE) * BLOCK_CUBE;
        // loop3_dst_stride, nd_dst_stride
        fixPipeParams.params.dstNdStride = self->ctx.cinG_;
        // 当dk>1且分核时，nSize必须为真实值，原因是多写的部分会跟下一个核出现同地址，不同核atomicAdd，导致精度错误
        if (tailCinExist && (self->ctx.cinRemainLen_ < baseCin)) {
            fixPipeParams.nSize = self->ctx.cinRemainLen_;
        } else {
            fixPipeParams.nSize = baseCin;
        }
        // loop1_src_stride, d_src_stride
        fixPipeParams.srcStride = self->ctx.hwK_ * ShiftCeilM0(self->ctx.baseUseM_, BLOCK_CUBE) * BLOCK_CUBE;
        // loop2_dst_stride, d_dst_stride
        fixPipeParams.dstStride = self->ctx.cinG_ * self->ctx.dhwK_;
        fixPipeParams.mSize = self->ctx.baseUseM_;

        fixPipeParams.quantPre = QuantMode_t::NoQuant;
        fixPipeParams.unitFlag = 0;

        int64_t srcOffset = nValueSum * ShiftCeilM0(self->ctx.baseUseM_, BLOCK_CUBE) * BLOCK_CUBE * BLOCK_CUBE;
        nValueSum += nValue;
        int64_t baseNAlignLength = baseCin * self->ctx.hwK_;
        int64_t dstDkOffset = (self->ctx.curNL0Idx_ / self->ctx.cinHkWkLoop_) * self->ctx.hwK_ * self->ctx.cinG_;
        int64_t dstCinOffset = (self->ctx.curNL0Idx_ % self->ctx.cinHkWkLoop_) *
            self->ctx.tiling_->baseN / baseNAlignLength * baseCin;

        int64_t dstBaseNOffset = dstOffset + dstDkOffset + dstCinOffset + self->ctx.head_ * self->ctx.cinG_ +
            j * baseCin;
        self->ctx.head_ = self->ctx.tail_ == self->ctx.hwK_ ? 0 : self->ctx.tail_;

        AscendC::Fixpipe<typename Intf::DstT, float, CFG_ROW_MAJOR>(output[dstBaseNOffset],
            l0c[srcOffset], fixPipeParams);
        // 当cin有尾块时，在cin跨行时更新cinRemainLen_的值
        if (tailCinExist && self->ctx.head_ == 0) {
            self->ctx.cinRemainLen_ -= baseCin;
        }
    }
    // 更新上一次N的Idx
    self->ctx.lastNIdx_ = self->ctx.curNL0Idx_;
    return;
}

template <class Intf>
static __aicore__ inline void LoadL0c2GmBaseNUndividedNz2DHWCN(Intf *self, const GlobalTensor<typename Intf::DstT> &output,
    LocalTensor<typename Intf::L0cT> &l0c)
{
    // (dk, ci1, hkwk, co1, co0, ci0) -> (dk, hk, wk, ci, co) NZ2DN
    constexpr uint64_t baseCin = BLOCK_CUBE;
    bool tailCinExist = (self->ctx.singleShapeCin_ % baseCin != 0);
    CalcL0cParams(self, tailCinExist);

    AscendC::FixpipeParamsC310<CO2Layout::COLUMN_MAJOR> fixPipeParams;
    uint64_t nValueSum = 0;
    int64_t dstOffset = self->ctx.curML0Idx_ * self->ctx.tiling_->baseM;
    uint32_t c1hkwk = ShiftDivM0(self->ctx.baseUseN_, baseCin);
    uint64_t baseNIter = Ceil(c1hkwk - (self->ctx.hwK_ - self->ctx.head_), self->ctx.hwK_) + 1;

    for (uint64_t j = 0; j < baseNIter; j++) {
        self->ctx.tail_ = (c1hkwk - nValueSum + self->ctx.head_) > self->ctx.hwK_ ? self->ctx.hwK_ :
            (c1hkwk - nValueSum + self->ctx.head_);
        uint32_t nValue = self->ctx.tail_ - self->ctx.head_;

        fixPipeParams.params.dnNum = nValue;
        // loop3_src_stride, dn_src_stride
        fixPipeParams.params.srcNzMatrixStride = ShiftCeilM0(self->ctx.baseUseM_, BLOCK_CUBE) * BLOCK_CUBE;
        // loop3_dst_stride, dn_dst_stride
        fixPipeParams.params.dstDnMatrixStride = self->ctx.cinG_ * self->ctx.tiling_->cout;
        // 当dk>1且分核时，nSize必须为真实值，原因是多写的部分会跟下一个核出现同地址，不同核atomicAdd，导致精度错误
        if (tailCinExist && (self->ctx.cinRemainLen_ < baseCin)) {
            fixPipeParams.nSize = self->ctx.cinRemainLen_;
        } else {
            fixPipeParams.nSize = baseCin;
        }
        // loop1_src_stride, d_src_stride
        fixPipeParams.srcStride = self->ctx.hwK_ * ShiftCeilM0(self->ctx.baseUseM_, BLOCK_CUBE) * BLOCK_CUBE;
        // loop2_dst_stride, d_dst_stride
        fixPipeParams.dstStride = self->ctx.tiling_->cout;

        fixPipeParams.mSize = self->ctx.baseUseM_;

        // loop0_src_stride, n_src_stride
        fixPipeParams.params.srcNzC0Stride = 1;

        fixPipeParams.quantPre = QuantMode_t::NoQuant;
        fixPipeParams.unitFlag = 0;

        int64_t srcOffset = nValueSum * ShiftCeilM0(self->ctx.baseUseM_, BLOCK_CUBE) * BLOCK_CUBE * BLOCK_CUBE;
        nValueSum += nValue;
        int64_t baseNAlignLength = baseCin * self->ctx.hwK_;
        int64_t dstDkOffset = (self->ctx.curNL0Idx_ / self->ctx.cinHkWkLoop_) *
            self->ctx.hwK_ *  self->ctx.cinG_ * self->ctx.tiling_->cout;
        int64_t dstCinOffset = (self->ctx.curNL0Idx_ % self->ctx.cinHkWkLoop_) *
            self->ctx.tiling_->baseN / baseNAlignLength * baseCin * self->ctx.tiling_->cout;

        int64_t dstBaseNOffset = dstOffset + dstDkOffset + dstCinOffset +
            self->ctx.head_ * self->ctx.cinG_ * self->ctx.tiling_->cout +
            j * baseCin * self->ctx.tiling_->cout;
        self->ctx.head_ = self->ctx.tail_ == self->ctx.hwK_ ? 0 : self->ctx.tail_;

        AscendC::Fixpipe<typename Intf::DstT, float, CFG_COLUMN_MAJOR>(output[dstBaseNOffset], l0c[srcOffset],
            fixPipeParams);
        // 当cin有尾块时，在cin跨行时更新cinRemainLen_
        if (tailCinExist && self->ctx.head_ == 0) {
            self->ctx.cinRemainLen_ -= baseCin;
        }
    }
    // 更新last N的Idx
    self->ctx.lastNIdx_ = self->ctx.curNL0Idx_;
    return;
}

template <class Intf>
static __aicore__ inline void LoadL0c2GmNormal(Intf *self, const GlobalTensor<typename Intf::DstT> &output,
    LocalTensor<typename Intf::L0cT> &l0c)
{
    bool tailCinExist = (self->ctx.singleShapeCin_ & 0xF) != 0;
    CalcL0cParams(self, tailCinExist);

    int64_t dstOffset = self->ctx.curML0Idx_ * self->ctx.tiling_->baseM * self->ctx.dhwK_ * self->ctx.cinG_ +
        (self->ctx.curNL0Idx_ % self->ctx.cinHkWkLoop_) * Ceil(self->ctx.tiling_->baseN, self->ctx.curSingleCoreDk_) *
        self->ctx.tiling_->dk + (self->ctx.curNL0Idx_ / self->ctx.cinHkWkLoop_) * self->ctx.curSingleCoreDk_ *
        self->ctx.hwK_;

    AscendC::FixpipeParamsC310<CO2Layout::COLUMN_MAJOR> fixPipeParams;
    fixPipeParams.params.dnNum = self->ctx.baseUseM_;
    // 当dk>1且分核时，nSize必须为真实值，原因是多写的部分会跟下一个核出现同地址，不同核atomicAdd，导致精度错误
    uint32_t baseCin = Ceil(Ceil(self->ctx.baseUseN_, self->ctx.curSingleCoreDk_), self->ctx.hwK_);

    if (tailCinExist && (self->ctx.cinRemainLen_ < baseCin)) {
        fixPipeParams.nSize = self->ctx.cinRemainLen_;
    } else {
        fixPipeParams.nSize = baseCin;
    }
    fixPipeParams.mSize = self->ctx.hwK_;

    // loop3_src_stride, dn_src_stride
    fixPipeParams.params.srcNzMatrixStride = 1;
    // loop1_src_stride, d_src_stride
    fixPipeParams.srcStride = self->ctx.hwK_ * ShiftCeilM0(self->ctx.baseUseM_, BLOCK_CUBE) * BLOCK_CUBE;
    // loop0_src_stride, n_src_stride
    fixPipeParams.params.srcNzC0Stride = ShiftCeilM0(self->ctx.baseUseM_, BLOCK_CUBE) * BLOCK_CUBE;

    // loop3_dst_stride, dn_dst_stride
    fixPipeParams.params.dstDnMatrixStride = self->ctx.cinG_ * self->ctx.dhwK_;
    // loop2_dst_stride, d_dst_stride
    fixPipeParams.dstStride = self->ctx.dhwK_;

    fixPipeParams.quantPre = QuantMode_t::NoQuant;
    fixPipeParams.unitFlag = 0;
    for (uint32_t i = 0; i < self->ctx.curSingleCoreDk_; i++) {
        int64_t srcDkStride = i * ShiftCeilM0(fixPipeParams.nSize, BLOCK_CUBE) * BLOCK_CUBE * self->ctx.hwK_
                                * ShiftCeilM0(self->ctx.baseUseM_, BLOCK_CUBE) * BLOCK_CUBE;
        int64_t dstDkStride = i * self->ctx.hwK_;
        AscendC::Fixpipe<typename Intf::DstT, float, CFG_COLUMN_MAJOR>(output[dstOffset + dstDkStride],
            l0c[srcDkStride], fixPipeParams);
    }
    if (tailCinExist) {
        self->ctx.cinRemainLen_ -= baseCin;
    }
    // 更新上一次N的Idx
    self->ctx.lastNIdx_ = self->ctx.curNL0Idx_;
    return;
}

template <class Intf>
static __aicore__ inline void LoadL0c2GmNormalNz2Nd(Intf *self, const GlobalTensor<typename Intf::DstT> &output,
    LocalTensor<typename Intf::L0cT> &l0c)
{
    bool tailCinExist = (self->ctx.singleShapeCin_ & 0xF) != 0;
    CalcL0cParams(self, tailCinExist);

    uint32_t curCin = Ceil(self->ctx.tiling_->baseN, self->ctx.curSingleCoreDk_ * self->ctx.hwK_);
    int64_t dstOffset = self->ctx.curML0Idx_ * self->ctx.tiling_->baseM * self->ctx.dhwK_ * self->ctx.cinG_ +
        (self->ctx.curNL0Idx_ % self->ctx.cinHkWkLoop_) * curCin + (self->ctx.curNL0Idx_ / self->ctx.cinHkWkLoop_) *
        self->ctx.curSingleCoreDk_ * self->ctx.hwK_ * self->ctx.cinG_;

    AscendC::FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixPipeParams;
    fixPipeParams.params.ndNum = self->ctx.hwK_;
    // 当dk>1且分核时，nSize必须为真实值，原因是多写的部分会跟下一个核出现同地址，不同核atomicAdd，导致精度错误
    uint32_t baseCin = Ceil(Ceil(self->ctx.baseUseN_, self->ctx.curSingleCoreDk_), self->ctx.hwK_);
    if (tailCinExist && (self->ctx.cinRemainLen_ < baseCin)) {
        fixPipeParams.nSize = self->ctx.cinRemainLen_;
    } else {
        fixPipeParams.nSize = baseCin;
    }
    fixPipeParams.mSize = self->ctx.baseUseM_;

    // loop3_src_stride, nd_src_stride
    fixPipeParams.params.srcNdStride = ShiftCeilM0(self->ctx.baseUseM_, BLOCK_CUBE) * BLOCK_CUBE;
    // loop1_src_stride, d_src_stride
    fixPipeParams.srcStride = self->ctx.hwK_ * ShiftCeilM0(self->ctx.baseUseM_, BLOCK_CUBE) * BLOCK_CUBE;

    // loop3_dst_stride, dn_dst_stride
    fixPipeParams.params.dstNdStride = self->ctx.cinG_;
    // loop2_dst_stride, d_dst_stride
    fixPipeParams.dstStride = self->ctx.cinG_ * self->ctx.dhwK_;

    fixPipeParams.quantPre = QuantMode_t::NoQuant;
    fixPipeParams.unitFlag = 0;
    for (uint32_t i = 0; i < self->ctx.curSingleCoreDk_; i++) {
        int64_t srcDkStride = i * ShiftCeilM0(fixPipeParams.nSize, BLOCK_CUBE) * BLOCK_CUBE * self->ctx.hwK_
                                * ShiftCeilM0(self->ctx.baseUseM_, BLOCK_CUBE) * BLOCK_CUBE;
        int64_t dstDkStride = i * self->ctx.hwK_ * self->ctx.cinG_;
        AscendC::Fixpipe<typename Intf::DstT, float, CFG_ROW_MAJOR>(output[dstOffset + dstDkStride],
            l0c[srcDkStride], fixPipeParams);
    }
    if (tailCinExist) {
        self->ctx.cinRemainLen_ -= baseCin;
    }
    // 更新上一次N的Idx
    self->ctx.lastNIdx_ = self->ctx.curNL0Idx_;
    return;
}

template <class Intf>
static __aicore__ inline void LoadL0c2GmNormalNz2DHWCN(Intf *self, const GlobalTensor<typename Intf::DstT> &output,
    LocalTensor<typename Intf::L0cT> &l0c)
{
    uint32_t curCin = Ceil(self->ctx.tiling_->baseN, self->ctx.curSingleCoreDk_ * self->ctx.hwK_);
    int64_t dstOffset = self->ctx.curML0Idx_ * self->ctx.tiling_->baseM +
        (self->ctx.curNL0Idx_ % self->ctx.cinHkWkLoop_) * curCin * self->ctx.tiling_->cout +
        (self->ctx.curNL0Idx_ / self->ctx.cinHkWkLoop_) * self->ctx.curSingleCoreDk_ *
        self->ctx.hwK_ * self->ctx.cinG_ * self->ctx.tiling_->cout;

    AscendC::FixpipeParamsC310<CO2Layout::COLUMN_MAJOR> fixPipeParams;
    fixPipeParams.params.dnNum = self->ctx.hwK_;
    // 当dk>1且分核时，nSize必须为真实值，原因是多写的部分会跟下一个核出现同地址，不同核atomicAdd，导致精度错误
    uint32_t baseCin = Ceil(Ceil(self->ctx.baseUseN_, self->ctx.curSingleCoreDk_), self->ctx.hwK_);
    if (self->ctx.bL1cin1CopyLen >= baseCin) {
        fixPipeParams.nSize = baseCin;
    } else {
        fixPipeParams.nSize = self->ctx.bL1cin1CopyLen;
    }
    fixPipeParams.mSize = self->ctx.baseUseM_;

    // loop3_src_stride, dn_src_stride
    fixPipeParams.params.srcNzMatrixStride = ShiftCeilM0(self->ctx.baseUseM_, BLOCK_CUBE) * BLOCK_CUBE;
    // loop1_src_stride, d_src_stride
    fixPipeParams.srcStride = self->ctx.hwK_ * ShiftCeilM0(self->ctx.baseUseM_, BLOCK_CUBE) * BLOCK_CUBE;
    // loop0_src_stride, n_src_stride
    fixPipeParams.params.srcNzC0Stride = 1;

    // loop3_dst_stride, dn_dst_stride
    fixPipeParams.params.dstDnMatrixStride = self->ctx.cinG_ * self->ctx.tiling_->cout;
    // loop2_dst_stride, d_dst_stride
    fixPipeParams.dstStride = self->ctx.tiling_->cout;

    fixPipeParams.quantPre = QuantMode_t::NoQuant;
    fixPipeParams.unitFlag = 0;
    for (uint32_t i = 0; i < self->ctx.curSingleCoreDk_; i++) {
        int64_t srcDkStride = i * ShiftCeilM0(fixPipeParams.nSize, BLOCK_CUBE) * BLOCK_CUBE * self->ctx.hwK_
                                * ShiftCeilM0(self->ctx.baseUseM_, BLOCK_CUBE) * BLOCK_CUBE;
        int64_t dstDkStride = i * self->ctx.hwK_ * self->ctx.cinG_ * self->ctx.tiling_->cout;
        AscendC::Fixpipe<typename Intf::DstT, float, CFG_COLUMN_MAJOR>(output[dstOffset + dstDkStride],
            l0c[srcDkStride], fixPipeParams);
    }
    return;
}

// group输出到ub上进行重排
template <class Intf>
static __aicore__ inline void LoadL0c2UbForGroup(Intf *self, LocalTensor<typename Intf::L0cT> &l0c)
{
    self->ctx.vecOutBuf_ = self->ctx.vecBuf_.template Get<typename Intf::DstT>();
    FixpipeParamsC310<CO2Layout::NZ> fixPipeParams;

    fixPipeParams.mSize = self->ctx.baseUseM_; // M: cout
    fixPipeParams.nSize = self->ctx.baseUseN_;
    fixPipeParams.srcStride = ShiftCeilM0(self->ctx.baseUseM_, BLOCK_CUBE) * BLOCK_CUBE;
    if (self->ctx.tiling_->group != self->ctx.tiling_->cin || self->ctx.tiling_->group != self->ctx.tiling_->cout) {
        fixPipeParams.dstStride = fixPipeParams.srcStride * BLOCK_CUBE;
    } else {
        fixPipeParams.dstStride = self->ctx.baseUseM_ << 4;
    }
    fixPipeParams.quantPre = QuantMode_t::NoQuant;
    fixPipeParams.unitFlag = 0;

    Fixpipe<typename Intf::DstT, float, CFG_NZ_UB>(self->ctx.vecOutBuf_, l0c, fixPipeParams);
}

template <class Intf>
static __aicore__ inline void LoadL0c2GmForTransFormatNCDHW(Intf *self, const GlobalTensor<typename Intf::DstT> &output,
    LocalTensor<typename Intf::L0cT> &l0c, bool enSequentialWrite = false)
{
    if (!enSequentialWrite) {
        if (self->ctx.tiling_->group != self->ctx.tiling_->realGroup) {
            LoadL0c2UbForGroup(self, l0c);
        } else if (self->ctx.dhwK_ == 1) {
            LoadL0c2GmDkhkwkEqOne(self, output, l0c);
        } else if ((self->ctx.tiling_->baseN >> 4) % self->ctx.hwK_ != 0) { // 4:用来替换除法运算
            LoadL0c2GmBaseNUndivided(self, output, l0c);
        } else {
            LoadL0c2GmNormal(self, output, l0c);
        }
    } else {
        MovOutL0cForDeterministicRefactor(self, l0c, output);
    }
}

template <class Intf>
static __aicore__ inline void LoadL0c2GmForTransFormatNDHWC(Intf *self, const GlobalTensor<typename Intf::DstT> &output,
    LocalTensor<typename Intf::L0cT> &l0c, bool enSequentialWrite = false)
{
    if (!enSequentialWrite) {
        if (self->ctx.dhwK_ == 1) {
            LoadL0c2GmDkhkwkEqOne(self, output, l0c);
        } else if ((self->ctx.tiling_->baseN >> 4) % self->ctx.hwK_ != 0) { // 4:用来替换除法运算
            LoadL0c2GmBaseNUndividedNz2Nd(self, output, l0c);
        } else {
            LoadL0c2GmNormalNz2Nd(self, output, l0c);
        }
    }
}

template <class Intf>
static __aicore__ inline void LoadL0c2GmForTransFormatDHWCN(Intf *self, const GlobalTensor<typename Intf::DstT> &output,
    LocalTensor<typename Intf::L0cT> &l0c, bool enSequentialWrite = false)
{
    if (!enSequentialWrite) {
        if (self->ctx.dhwK_ == 1) {
            LoadL0c2GmDkhkwkEqOneNz2DHWCN(self, output, l0c);
        } else if ((self->ctx.tiling_->baseN >> 4) % self->ctx.hwK_ != 0) { // 4:用来替换除法运算
            LoadL0c2GmBaseNUndividedNz2DHWCN(self, output, l0c);
        } else {
            LoadL0c2GmNormalNz2DHWCN(self, output, l0c);
        }
    }
}

template <class Intf>
static __aicore__ inline void LoadL0c2GmForTransFormat(Intf *self, const GlobalTensor<typename Intf::DstT> &output,
    LocalTensor<typename Intf::L0cT> &l0c, bool enSequentialWrite = false)
{
    if constexpr (Intf::Config::dType::format == ConvolutionBackprop::CubeFormat::NCDHW) {
        LoadL0c2GmForTransFormatNCDHW(self, output, l0c, enSequentialWrite);
    } else if constexpr (Intf::Config::dType::format == ConvolutionBackprop::CubeFormat::NDHWC) {
        LoadL0c2GmForTransFormatNDHWC(self, output, l0c, enSequentialWrite);
    } else { // DHWCN
        LoadL0c2GmForTransFormatDHWCN(self, output, l0c, enSequentialWrite);
    }
}

template <class Intf>
static __aicore__ inline void LoadL0c2Gm(
    Intf *self, const GlobalTensor<typename Intf::DstT> &output, uint8_t enAtomic = 0, bool enSequentialWrite = false)
{
    LocalTensor<typename Intf::L0cT> l0c;
    if (self->ctx.l0cPingPongFlag_) {
        l0c = self->ctx.l0cPing_.template DeQue<typename Intf::L0cT>();
    } else {
        l0c = self->ctx.l0cPong_.template DeQue<typename Intf::L0cT>();
    }

    if (enAtomic == 1) {
        SetAtomicAdd<typename Intf::DstT>();
    }

    LoadL0c2GmForTransFormat(self, output, l0c, enSequentialWrite);

    if (enAtomic == 1) {
        SetAtomicNone();
    }
    
    if (self->ctx.l0cPingPongFlag_) {
        self->ctx.l0cPing_.FreeTensor(l0c);
    } else {
        self->ctx.l0cPong_.FreeTensor(l0c);
    }
    if (self->ctx.tiling_->cl0Pbuffer > 1) {
        self->ctx.l0cPingPongFlag_ = !self->ctx.l0cPingPongFlag_;
    }
}
}  // namespace ConvolutionBackpropFunc

#endif
