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
 * \file conv_bp_filter_sub_func_store_l0c.h
 * \brief L0c to Gm store implementation functions for conv3d backprop filter
 */

#ifndef CONV3D_BP_FILTER_SUB_FUNC_STORE_L0C_H
#define CONV3D_BP_FILTER_SUB_FUNC_STORE_L0C_H

namespace ConvolutionBackpropFunc {
constexpr FixpipeConfig CFG_NZ_UB = {CO2Layout::NZ, true};

template <class Intf>
static __aicore__ inline void UpdateHwKIterState(Intf* self, bool tailCinExist, uint64_t baseCin)
{
    self->ctx.head_ = self->ctx.tail_ == self->ctx.hwK_ ? 0 : self->ctx.tail_;
    if (tailCinExist && self->ctx.head_ == 0) {
        self->ctx.cinRemainLen_ -= baseCin;
    }
}

template <class Intf>
static __aicore__ inline void LoadL0c2GmDkhkwkEqOne(
    Intf* self, const GlobalTensor<typename Intf::DstT>& output, LocalTensor<typename Intf::L0cT>& l0c)
{
    bool tailCinExist = (self->ctx.singleShapeCin_ & 0xF) != 0;
    CalcL0cParams(self, tailCinExist);

    int64_t dstOffset = self->ctx.curML0Idx_ * self->ctx.tiling_->baseM * self->ctx.dhwK_ * self->ctx.cinG_ +
                        self->ctx.curNL0Idx_ * self->ctx.tiling_->baseN * self->ctx.tiling_->dk;

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

    AscendC::Fixpipe<typename Intf::DstT, float, CFG_ROW_MAJOR>(output[dstOffset], l0c, fixPipeParams);
    if (tailCinExist) {
        self->ctx.cinRemainLen_ -= self->ctx.baseUseN_;
    }
    // 更新上一次N的Idx
    self->ctx.lastNIdx_ = self->ctx.curNL0Idx_;
    return;
}

template <class Intf>
static __aicore__ inline void LoadL0c2GmDkhkwkEqOneNz2DHWCN(
    Intf* self, const GlobalTensor<typename Intf::DstT>& output, LocalTensor<typename Intf::L0cT>& l0c)
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
static __aicore__ inline void LoadL0c2GmBaseNUndivided(
    Intf* self, const GlobalTensor<typename Intf::DstT>& output, LocalTensor<typename Intf::L0cT>& l0c)
{
    constexpr uint64_t baseCin = BLOCK_CUBE;
    bool tailCinExist = (self->ctx.singleShapeCin_ % baseCin != 0);
    CalcL0cParams(self, tailCinExist);

    AscendC::FixpipeParamsC310<CO2Layout::COLUMN_MAJOR> fixPipeParams;
    uint64_t nValueSum = 0;
    int64_t dstOffset = self->ctx.curML0Idx_ * self->ctx.tiling_->baseM * self->ctx.cinG_ * self->ctx.dhwK_;
    uint32_t c1hkwk = ShiftDivM0(self->ctx.baseUseN_, baseCin);
    uint32_t numBaseNIncludeHwk =
        (c1hkwk + self->ctx.head_ >= self->ctx.hwK_) ? (c1hkwk - (self->ctx.hwK_ - self->ctx.head_)) : 0;
    uint64_t baseNIter = Ceil(numBaseNIncludeHwk, self->ctx.hwK_) + 1;

    for (uint64_t j = 0; j < baseNIter; j++) {
        self->ctx.tail_ = (c1hkwk - nValueSum + self->ctx.head_) > self->ctx.hwK_ ?
                              self->ctx.hwK_ :
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
        int64_t dstCinOffset = (self->ctx.curNL0Idx_ % self->ctx.cinHkWkLoop_) * self->ctx.tiling_->baseN /
                               baseNAlignLength * baseNAlignLength * self->ctx.tiling_->dk;

        int64_t dstBaseNOffset =
            dstOffset + dstDkOffset + dstCinOffset + self->ctx.head_ + j * baseNAlignLength * self->ctx.tiling_->dk;

        AscendC::Fixpipe<typename Intf::DstT, float, CFG_COLUMN_MAJOR>(
            output[dstBaseNOffset], l0c[srcOffset], fixPipeParams);
        UpdateHwKIterState<Intf>(self, tailCinExist, baseCin);
    }
    self->ctx.lastNIdx_ = self->ctx.curNL0Idx_;
    return;
}

template <class Intf>
static __aicore__ inline void LoadL0c2GmBaseNUndividedNz2Nd(
    Intf* self, const GlobalTensor<typename Intf::DstT>& output, LocalTensor<typename Intf::L0cT>& l0c)
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
        self->ctx.tail_ = (c1hkwk - nValueSum + self->ctx.head_) > self->ctx.hwK_ ?
                              self->ctx.hwK_ :
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
        int64_t dstCinOffset =
            (self->ctx.curNL0Idx_ % self->ctx.cinHkWkLoop_) * self->ctx.tiling_->baseN / baseNAlignLength * baseCin;

        int64_t dstBaseNOffset =
            dstOffset + dstDkOffset + dstCinOffset + self->ctx.head_ * self->ctx.cinG_ + j * baseCin;
        AscendC::Fixpipe<typename Intf::DstT, float, CFG_ROW_MAJOR>(
            output[dstBaseNOffset], l0c[srcOffset], fixPipeParams);
        UpdateHwKIterState<Intf>(self, tailCinExist, baseCin);
    }
    self->ctx.lastNIdx_ = self->ctx.curNL0Idx_;
    return;
}

template <class Intf>
static __aicore__ inline void LoadL0c2GmBaseNUndividedNz2DHWCN(
    Intf* self, const GlobalTensor<typename Intf::DstT>& output, LocalTensor<typename Intf::L0cT>& l0c)
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
        self->ctx.tail_ = (c1hkwk - nValueSum + self->ctx.head_) > self->ctx.hwK_ ?
                              self->ctx.hwK_ :
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
        int64_t dstDkOffset = (self->ctx.curNL0Idx_ / self->ctx.cinHkWkLoop_) * self->ctx.hwK_ * self->ctx.cinG_ *
                              self->ctx.tiling_->cout;
        int64_t dstCinOffset = (self->ctx.curNL0Idx_ % self->ctx.cinHkWkLoop_) * self->ctx.tiling_->baseN /
                               baseNAlignLength * baseCin * self->ctx.tiling_->cout;

        int64_t dstBaseNOffset = dstOffset + dstDkOffset + dstCinOffset +
                                 self->ctx.head_ * self->ctx.cinG_ * self->ctx.tiling_->cout +
                                 j * baseCin * self->ctx.tiling_->cout;
        AscendC::Fixpipe<typename Intf::DstT, float, CFG_COLUMN_MAJOR>(
            output[dstBaseNOffset], l0c[srcOffset], fixPipeParams);
        UpdateHwKIterState<Intf>(self, tailCinExist, baseCin);
    }
    self->ctx.lastNIdx_ = self->ctx.curNL0Idx_;
    return;
}

template <class Intf>
static __aicore__ inline void LoadL0c2GmNormal(
    Intf* self, const GlobalTensor<typename Intf::DstT>& output, LocalTensor<typename Intf::L0cT>& l0c)
{
    bool tailCinExist = (self->ctx.singleShapeCin_ & 0xF) != 0;
    CalcL0cParams(self, tailCinExist);

    int64_t dstOffset = self->ctx.curML0Idx_ * self->ctx.tiling_->baseM * self->ctx.dhwK_ * self->ctx.cinG_ +
                        (self->ctx.curNL0Idx_ % self->ctx.cinHkWkLoop_) *
                            Ceil(self->ctx.tiling_->baseN, self->ctx.curSingleCoreDk_) * self->ctx.tiling_->dk +
                        (self->ctx.curNL0Idx_ / self->ctx.cinHkWkLoop_) * self->ctx.curSingleCoreDk_ * self->ctx.hwK_;

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
        int64_t srcDkStride = i * ShiftCeilM0(fixPipeParams.nSize, BLOCK_CUBE) * BLOCK_CUBE * self->ctx.hwK_ *
                              ShiftCeilM0(self->ctx.baseUseM_, BLOCK_CUBE) * BLOCK_CUBE;
        int64_t dstDkStride = i * self->ctx.hwK_;
        AscendC::Fixpipe<typename Intf::DstT, float, CFG_COLUMN_MAJOR>(
            output[dstOffset + dstDkStride], l0c[srcDkStride], fixPipeParams);
    }
    if (tailCinExist) {
        self->ctx.cinRemainLen_ -= baseCin;
    }
    // 更新上一次N的Idx
    self->ctx.lastNIdx_ = self->ctx.curNL0Idx_;
    return;
}

template <class Intf>
static __aicore__ inline void LoadL0c2GmNormalNz2Nd(
    Intf* self, const GlobalTensor<typename Intf::DstT>& output, LocalTensor<typename Intf::L0cT>& l0c)
{
    bool tailCinExist = (self->ctx.singleShapeCin_ & 0xF) != 0;
    CalcL0cParams(self, tailCinExist);

    uint32_t curCin = Ceil(self->ctx.tiling_->baseN, self->ctx.curSingleCoreDk_ * self->ctx.hwK_);
    int64_t dstOffset =
        self->ctx.curML0Idx_ * self->ctx.tiling_->baseM * self->ctx.dhwK_ * self->ctx.cinG_ +
        (self->ctx.curNL0Idx_ % self->ctx.cinHkWkLoop_) * curCin +
        (self->ctx.curNL0Idx_ / self->ctx.cinHkWkLoop_) * self->ctx.curSingleCoreDk_ * self->ctx.hwK_ * self->ctx.cinG_;

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
        int64_t srcDkStride = i * ShiftCeilM0(fixPipeParams.nSize, BLOCK_CUBE) * BLOCK_CUBE * self->ctx.hwK_ *
                              ShiftCeilM0(self->ctx.baseUseM_, BLOCK_CUBE) * BLOCK_CUBE;
        int64_t dstDkStride = i * self->ctx.hwK_ * self->ctx.cinG_;
        AscendC::Fixpipe<typename Intf::DstT, float, CFG_ROW_MAJOR>(
            output[dstOffset + dstDkStride], l0c[srcDkStride], fixPipeParams);
    }
    if (tailCinExist) {
        self->ctx.cinRemainLen_ -= baseCin;
    }
    // 更新上一次N的Idx
    self->ctx.lastNIdx_ = self->ctx.curNL0Idx_;
    return;
}

template <class Intf>
static __aicore__ inline void LoadL0c2GmNormalNz2DHWCN(
    Intf* self, const GlobalTensor<typename Intf::DstT>& output, LocalTensor<typename Intf::L0cT>& l0c)
{
    uint32_t curCin = Ceil(self->ctx.tiling_->baseN, self->ctx.curSingleCoreDk_ * self->ctx.hwK_);
    int64_t dstOffset = self->ctx.curML0Idx_ * self->ctx.tiling_->baseM +
                        (self->ctx.curNL0Idx_ % self->ctx.cinHkWkLoop_) * curCin * self->ctx.tiling_->cout +
                        (self->ctx.curNL0Idx_ / self->ctx.cinHkWkLoop_) * self->ctx.curSingleCoreDk_ * self->ctx.hwK_ *
                            self->ctx.cinG_ * self->ctx.tiling_->cout;

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
        int64_t srcDkStride = i * ShiftCeilM0(fixPipeParams.nSize, BLOCK_CUBE) * BLOCK_CUBE * self->ctx.hwK_ *
                              ShiftCeilM0(self->ctx.baseUseM_, BLOCK_CUBE) * BLOCK_CUBE;
        int64_t dstDkStride = i * self->ctx.hwK_ * self->ctx.cinG_ * self->ctx.tiling_->cout;
        AscendC::Fixpipe<typename Intf::DstT, float, CFG_COLUMN_MAJOR>(
            output[dstOffset + dstDkStride], l0c[srcDkStride], fixPipeParams);
    }
    return;
}

// group输出到ub上进行重排
template <class Intf>
static __aicore__ inline void LoadL0c2UbForGroup(Intf* self, LocalTensor<typename Intf::L0cT>& l0c)
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

} // namespace ConvolutionBackpropFunc

#endif
