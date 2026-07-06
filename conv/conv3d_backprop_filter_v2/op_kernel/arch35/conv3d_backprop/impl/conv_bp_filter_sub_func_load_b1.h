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
 * \file conv_bp_filter_sub_func_load_b1.h
 * \brief B1 matrix load functions for conv3d backprop filter
 */

#ifndef CONV3D_BP_FILTER_SUB_FUNC_LOAD_B1_H
#define CONV3D_BP_FILTER_SUB_FUNC_LOAD_B1_H
#include "conv_bp_filter_sub_func_load_common.h"

namespace ConvolutionBackpropFunc {
constexpr uint64_t SINGLE_SHAPE_CIN0 = 16;

template <class Intf>
static __aicore__ inline void CalOut2B1ParamsBaseNUndivided(Intf* self, Out2L1ScalarParams& params)
{
    // 计算L1上cin起始idx, 去掉cin1HkWkCin里的HkWk
    uint64_t b1SrCin = (self->ctx.curNIdx_ % self->ctx.cinHkWkLoop_) * self->ctx.tiling_->baseN /
                       (self->ctx.hwK_ * self->ctx.curSingleCoreDk_ * self->ctx.tiling_->n0);
    if constexpr (Intf::Config::xType::format == ConvolutionBackprop::CubeFormat::NCDHW) {
        params.out2B1SrcAddr = static_cast<uint64_t>(b1SrCin) * self->ctx.tiling_->n0 * self->ctx.dhwI_ +
                               (self->ctx.curNIdx_ / self->ctx.cinHkWkLoop_) * self->ctx.hwI_;
    } else {
        params.out2B1SrcAddr = static_cast<uint64_t>(b1SrCin) * self->ctx.tiling_->n0 +
                               (self->ctx.curNIdx_ / self->ctx.cinHkWkLoop_) * self->ctx.hwI_ * self->ctx.tiling_->cin;
    }

    uint64_t bL1N = ShiftCeilM0(self->ctx.baseUseN_ / self->ctx.curSingleCoreDk_, self->ctx.tiling_->n0);
    uint32_t bL1cin1CopyLen = Ceil(bL1N, self->ctx.hwK_);
    if (self->ctx.singleShapeCin_ > SINGLE_SHAPE_CIN0) {
        // 当singleShapeCin_小于16时，搬运的bL1cin1CopyLen最大为1，无需多搬
        if (self->ctx.hwK_ > bL1N) {
            // 需判断过hwK_对bL1N取余不等于0了
            if (self->ctx.hwK_ % bL1N != 0) {
                ++bL1cin1CopyLen; // bL1cin1CopyLen本身就向上取整，也就是说会多搬一行，再多搬一行也就是说覆盖了跨三行的情况
            }
        } else if (2 * bL1N % self->ctx.hwK_ != 0) {
            // 如果2 * bL1N % self->ctx.hwK_ == 0，仅需搬两行，不会出现跨三行的情况
            ++bL1cin1CopyLen;
        }
    }
    uint32_t cin1RemainLen = ShiftCeilM0(self->ctx.singleShapeCin_, self->ctx.tiling_->n0) - b1SrCin;
    // L1上不能多载入数据，否则会导致underflow或者overflow问题
    self->ctx.bL1cin1CopyLen = cin1RemainLen > bL1cin1CopyLen ?
                                   (bL1cin1CopyLen * self->ctx.tiling_->n0) :
                                   (self->ctx.singleShapeCin_ - b1SrCin * self->ctx.tiling_->n0);
}

template <class Intf>
static __aicore__ inline void CalOut2B1Params(Intf* self, Out2L1ScalarParams& params)
{
    // 计算L1上cin起始idx, 去掉cin1HkWkCin里的HkWk
    uint64_t b1SrCin = (self->ctx.curNIdx_ % self->ctx.cinHkWkLoop_) * self->ctx.tiling_->baseN /
                       (self->ctx.hwK_ * self->ctx.curSingleCoreDk_);
    if constexpr (Intf::Config::xType::format == ConvolutionBackprop::CubeFormat::NCDHW) {
        params.out2B1SrcAddr = static_cast<uint64_t>(b1SrCin) * self->ctx.dhwI_ +
                               (self->ctx.curNIdx_ / self->ctx.cinHkWkLoop_) * self->ctx.curSingleCoreDk_ *
                                   self->ctx.hwI_;
    } else {
        params.out2B1SrcAddr = static_cast<uint64_t>(b1SrCin) + (self->ctx.curNIdx_ / self->ctx.cinHkWkLoop_) *
                                                                    self->ctx.curSingleCoreDk_ * self->ctx.hwI_ *
                                                                    self->ctx.tiling_->cin;
    }

    uint64_t bL1N = self->ctx.baseUseN_ / self->ctx.curSingleCoreDk_;
    uint32_t bL1cin1CopyLen = Ceil(bL1N, self->ctx.hwK_);
    uint32_t cin1RemainLen = self->ctx.singleShapeCin_ - b1SrCin;
    self->ctx.bL1cin1CopyLen = cin1RemainLen > bL1cin1CopyLen ? bL1cin1CopyLen : cin1RemainLen;
}

template <class Intf>
static __aicore__ inline void LoadToB1Dn2Nz(Intf* self, const uint32_t hiCopyLen, const uint64_t out2B1SrcAddrOffset,
                                            const Out2L1ScalarParams& params,
                                            LocalTensor<typename Intf::SrcT>& useB1Buf)
{
    if (likely(!self->ctx.isSplitWo_)) {
        Dn2NzParams dn2NzParams;
        dn2NzParams.dnNum = self->ctx.curSingleCoreDk_;

        dn2NzParams.nValue = static_cast<uint64_t>(hiCopyLen) * self->ctx.tiling_->wi;
        dn2NzParams.dValue = self->ctx.bL1cin1CopyLen;
        dn2NzParams.srcDValue = self->ctx.dhwI_;
        dn2NzParams.srcDnMatrixStride = self->ctx.hwI_;

        dn2NzParams.dstNzC0Stride = dn2NzParams.nValue;
        dn2NzParams.dstNzNStride = 1;
        dn2NzParams.dstNzMatrixStride = ShiftCeilChannelSize<Intf>(dn2NzParams.dValue, self->ctx.tiling_->k0) *
                                        dn2NzParams.nValue * self->ctx.tiling_->k0;

        DataCopy(useB1Buf, self->ctx.fmapGlobal_[out2B1SrcAddrOffset], dn2NzParams);
    } else { // 切分出来的singleshapedi=1才可以，否则需要通过多条ND2NZ指令进行拼接
        Dn2NzParams dn2NzParams;
        dn2NzParams.dnNum = hiCopyLen;
        // 需要考虑左右边pad的情况，不能直接等于singleshapewi
        dn2NzParams.nValue = params.singleShapeWi - self->ctx.load3d_.padList[0] - self->ctx.load3d_.padList[1];
        dn2NzParams.dValue = self->ctx.bL1cin1CopyLen;
        dn2NzParams.srcDValue = self->ctx.dhwI_;
        dn2NzParams.srcDnMatrixStride = self->ctx.tiling_->wi;

        dn2NzParams.dstNzC0Stride = dn2NzParams.nValue * dn2NzParams.dnNum;
        dn2NzParams.dstNzNStride = 1;
        dn2NzParams.dstNzMatrixStride = dn2NzParams.nValue * self->ctx.tiling_->k0;

        DataCopy(useB1Buf, self->ctx.fmapGlobal_[out2B1SrcAddrOffset], dn2NzParams);
    }
}

template <class Intf>
static __aicore__ inline void LoadToB1Nd2Nz(Intf* self, const uint32_t hiCopyLen, const uint64_t out2B1SrcAddrOffset,
                                            const Out2L1ScalarParams& params,
                                            LocalTensor<typename Intf::SrcT>& useB1Buf)
{
    if (likely(!self->ctx.isSplitWo_)) {
        Nd2NzParams nd2NzParams;
        nd2NzParams.ndNum = self->ctx.curSingleCoreDk_;
        nd2NzParams.srcNdMatrixStride = self->ctx.hwI_ * self->ctx.tiling_->cin;

        nd2NzParams.nValue = static_cast<uint64_t>(hiCopyLen) * self->ctx.tiling_->wi;
        nd2NzParams.dValue = self->ctx.bL1cin1CopyLen;

        nd2NzParams.srcDValue = self->ctx.tiling_->cin;
        nd2NzParams.dstNzC0Stride = nd2NzParams.nValue;
        nd2NzParams.dstNzNStride = 1;
        nd2NzParams.dstNzMatrixStride = ShiftCeilChannelSize<Intf>(nd2NzParams.dValue, self->ctx.tiling_->k0) *
                                        nd2NzParams.nValue * self->ctx.tiling_->k0;

        DataCopy(useB1Buf, self->ctx.fmapGlobal_[out2B1SrcAddrOffset], nd2NzParams);
    } else {
        Nd2NzParams nd2NzParams;
        nd2NzParams.ndNum = hiCopyLen;
        // 需要考虑左右边pad的情况，不能直接等于singleshapewi
        nd2NzParams.nValue = params.singleShapeWi - self->ctx.load3d_.padList[0] - self->ctx.load3d_.padList[1];
        nd2NzParams.dValue = self->ctx.bL1cin1CopyLen;
        nd2NzParams.srcDValue = self->ctx.tiling_->cin;
        nd2NzParams.srcNdMatrixStride = self->ctx.tiling_->wi * self->ctx.tiling_->cin;

        nd2NzParams.dstNzC0Stride = nd2NzParams.nValue * nd2NzParams.ndNum;
        nd2NzParams.dstNzNStride = 1;
        nd2NzParams.dstNzMatrixStride = nd2NzParams.nValue * self->ctx.tiling_->k0;

        DataCopy(useB1Buf, self->ctx.fmapGlobal_[out2B1SrcAddrOffset], nd2NzParams);
    }
}

template <class Intf>
static __aicore__ inline bool CalB1HiCopyParams(Intf* self, uint64_t kbStepIdx, const Out2L1ScalarParams& params,
                                                B1HiCopyParams& hiParams)
{
    uint64_t b1SrcKAlign = kbStepIdx * self->ctx.kbl1_;
    uint32_t b1SrcHo = b1SrcKAlign / self->ctx.singleShapeWo_;
    uint32_t b1SrcHoGm = b1SrcHo + self->ctx.hoStartIdx_;
    int64_t b1SrcHiGm = static_cast<uint64_t>(b1SrcHoGm) * self->ctx.tiling_->strideH - self->ctx.tiling_->padUp;
    if (b1SrcHiGm > 0 && self->ctx.hiStartIdx_ > 0) {
        hiParams.b1SrcHi = b1SrcHiGm - self->ctx.hiStartIdx_;
    } else if (b1SrcHiGm > 0) {
        hiParams.b1SrcHi = b1SrcHiGm;
    }

    uint32_t kbl1 = self->ctx.kbl1_;
    if (self->ctx.stepKbRound == (kbStepIdx + 1)) {
        kbl1 = self->ctx.singleShapeHo_ * self->ctx.singleShapeWo_ - b1SrcKAlign;
    }
    uint32_t ho = CalRows2Copy(kbl1, self->ctx.singleShapeWo_);
    uint32_t hiCopyLen = ho * self->ctx.tiling_->strideH + self->ctx.strideKernelDilationH;
    if ((b1SrcHiGm + hiCopyLen <= 0) || (b1SrcHiGm >= self->ctx.tiling_->hi)) {
        return true;
    }

    if (b1SrcHiGm < 0) {
        hiCopyLen = hiCopyLen + b1SrcHiGm;
        hiParams.padUp = -b1SrcHiGm;
    }
    if (b1SrcHiGm + hiCopyLen > self->ctx.tiling_->hi) {
        hiCopyLen = self->ctx.tiling_->hi - b1SrcHiGm;
    }
    uint32_t hiRemainLen = params.singleShapeHi - hiParams.b1SrcHi;
    hiParams.hiCopyLen = hiRemainLen > hiCopyLen ? hiCopyLen : hiRemainLen;
    if (hiParams.hiCopyLen == 0) {
        return true;
    }
    return false;
}

template <class Intf>
static __aicore__ inline uint64_t CalB1GmOffset(Intf* self, uint32_t b1SrcHi, const Out2L1ScalarParams& params)
{
    if constexpr (Intf::Config::xType::format == ConvolutionBackprop::CubeFormat::NCDHW) {
        return params.out2B1SrcAddr + static_cast<uint64_t>(b1SrcHi) * self->ctx.tiling_->wi;
    } else {
        return params.out2B1SrcAddr + static_cast<uint64_t>(b1SrcHi) * self->ctx.tiling_->wi * self->ctx.tiling_->cin;
    }
}

/*
 * 为规避load3d限制stride<=63的限制，H方向和W方向均跳着搬运有效数据
 * 因此循环搬运hiCopyLen次，每次搬运dnNum数量为W方向有效数据个数
 */
template <class Intf>
static __aicore__ inline void LoadToB1Dn2NzSplitKernelHW(Intf* self, const uint32_t hiCopyLen, const uint32_t wiCopyLen,
                                                         uint64_t out2B1SrcAddrOffset, const Out2L1ScalarParams& params,
                                                         LocalTensor<typename Intf::SrcT>& useB1Buf)
{
    Dn2NzParams dn2NzParams;
    dn2NzParams.dnNum = self->ctx.load3d_.l1W;
    dn2NzParams.nValue = 1; // hiwi每次搬运大小为1
    dn2NzParams.dValue = self->ctx.bL1cin1CopyLen;
    dn2NzParams.srcDValue = self->ctx.dhwI_;
    dn2NzParams.srcDnMatrixStride = self->ctx.tiling_->strideW; // dn中每次跳过的长度

    dn2NzParams.dstNzC0Stride = hiCopyLen * self->ctx.load3d_.l1W;
    dn2NzParams.dstNzNStride = 1;
    dn2NzParams.dstNzMatrixStride = dn2NzParams.nValue * self->ctx.tiling_->k0;
    InitZeroValue(self, useB1Buf);

    uint32_t dstAddrOffset = 0;
    for (uint32_t i = 0; i < hiCopyLen; ++i) {
        DataCopy(useB1Buf[dstAddrOffset], self->ctx.fmapGlobal_[out2B1SrcAddrOffset], dn2NzParams);

        out2B1SrcAddrOffset += static_cast<uint64_t>(self->ctx.tiling_->wi) * self->ctx.tiling_->strideH;
        dstAddrOffset += self->ctx.load3d_.l1W * self->ctx.tiling_->k0;
    }
}

template <class Intf>
static __aicore__ inline void LoadToB1Nd2NzSplitKernelHW(Intf* self, const uint32_t hiCopyLen, const uint32_t wiCopyLen,
                                                         uint64_t out2B1SrcAddrOffset, const Out2L1ScalarParams& params,
                                                         LocalTensor<typename Intf::SrcT>& useB1Buf)
{
    Nd2NzParams nd2NzParams;
    nd2NzParams.ndNum = self->ctx.load3d_.l1W;
    nd2NzParams.nValue = 1;
    nd2NzParams.dValue = self->ctx.bL1cin1CopyLen;
    nd2NzParams.srcDValue = self->ctx.tiling_->cin;
    nd2NzParams.srcNdMatrixStride = static_cast<uint64_t>(self->ctx.tiling_->strideW) * self->ctx.tiling_->cin;

    nd2NzParams.dstNzC0Stride = hiCopyLen * self->ctx.load3d_.l1W;
    nd2NzParams.dstNzNStride = 1;
    nd2NzParams.dstNzMatrixStride = static_cast<uint64_t>(nd2NzParams.nValue) * self->ctx.tiling_->k0;

    InitZeroValue(self, useB1Buf);
    uint32_t dstAddrOffset = 0;
    for (uint32_t i = 0; i < hiCopyLen; ++i) {
        DataCopy(useB1Buf[dstAddrOffset], self->ctx.fmapGlobal_[out2B1SrcAddrOffset], nd2NzParams);
        dstAddrOffset += self->ctx.load3d_.l1W * self->ctx.tiling_->k0;
        out2B1SrcAddrOffset += static_cast<uint64_t>(self->ctx.tiling_->wi) * self->ctx.tiling_->strideH *
                               self->ctx.tiling_->cin;
    }
}

template <class Intf>
static __aicore__ inline bool CalB1HiCopyParamsSplitKernelHW(Intf* self, uint64_t kbStepIdx, uint64_t hkIdx,
                                                             const Out2L1ScalarParams& params, B1HiCopyParams& hiParams)
{
    // L0shape到orgShape的对应关系，L0和L1是16对齐的，orgShape是Wi对齐的,先算Wo对齐再算Wi对齐
    // 先算L0B所在BL1块的起始地址，16对齐的
    uint64_t b1SrcKAlign = kbStepIdx * self->ctx.kbl1_;
    // load3d必须有完整Wo，做Wo对齐，计算起始地址所在的Ho
    uint32_t b1SrcHo = b1SrcKAlign / self->ctx.singleShapeWo_;
    uint32_t b1SrcHoGm = b1SrcHo + self->ctx.hoStartIdx_;
    // 计算Ho对应的Hi，根据卷积原理
    int64_t b1SrcHiGm = static_cast<uint64_t>(b1SrcHoGm) * self->ctx.tiling_->strideH +
                        hkIdx * self->ctx.tiling_->dilationH - self->ctx.tiling_->padUp;
    if (b1SrcHiGm > 0 && self->ctx.hiStartIdx_ > 0) {
        hiParams.b1SrcHi = b1SrcHiGm - self->ctx.hiStartIdx_;
    } else if (b1SrcHiGm > 0) {
        hiParams.b1SrcHi = b1SrcHiGm;
    }

    uint32_t kbl1Tmp = self->ctx.kbl1_;
    if (self->ctx.stepKbRound == (kbStepIdx + 1)) {
        kbl1Tmp = self->ctx.singleShapeHo_ * self->ctx.singleShapeWo_ - b1SrcKAlign;
    }
    uint32_t ho = CalRows2Copy(kbl1Tmp, self->ctx.singleShapeWo_);
    // hk采用循环，每次只循环一个hk,filterH=1
    uint32_t hiCopyLen = (ho - 1) * self->ctx.tiling_->strideH + 1;
    int64_t b1SrcHiGmDown = b1SrcHiGm + hiCopyLen;
    // 当拷贝的行完全处于padUp部分或是padDown部分时跳出搬运, 注意此处的hicopyLen包含pad行
    if ((b1SrcHiGmDown < 0) || (b1SrcHiGm >= self->ctx.tiling_->hi)) {
        return true;
    }

    if (b1SrcHiGm < 0) {
        //起始地址计算
        hiParams.padUp = -b1SrcHiGm;
        hiParams.hiUpValidOffset = Ceil(hiParams.padUp, self->ctx.tiling_->strideH) * self->ctx.tiling_->strideH -
                                   hiParams.padUp;
    }

    uint32_t padDown = 0;
    if (b1SrcHiGmDown >= self->ctx.tiling_->hi) {
        padDown = b1SrcHiGmDown - self->ctx.tiling_->hi;
    }

    hiCopyLen = ho - Ceil(hiParams.padUp, self->ctx.tiling_->strideH) - Ceil(padDown, self->ctx.tiling_->strideH);
    hiParams.hiCopyLen = hiCopyLen < 0 ? 0 : hiCopyLen;
    if (hiParams.hiCopyLen == 0) {
        return true;
    }
    return false;
}

} // namespace ConvolutionBackpropFunc

#endif
