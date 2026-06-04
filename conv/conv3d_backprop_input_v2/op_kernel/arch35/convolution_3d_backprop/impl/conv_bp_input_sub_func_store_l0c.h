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
 * \file conv_bp_input_sub_func_store_l0c.h
 * \brief L0c to Gm store implementation functions for conv3d backprop input
 */

#ifndef CONV3D_BP_INPUT_SUB_FUNC_STORE_L0C_H
#define CONV3D_BP_INPUT_SUB_FUNC_STORE_L0C_H

namespace Convolution3DBackpropFunc {

template <class Intf>
static __aicore__ inline uint64_t CalSplitKWorkspaceOffset(Intf* self, uint64_t useN, uint64_t blockIdx)
{
    uint64_t singleCoreCinAlignBaseN = AlignUp(self->ctx.tiling_->singleCoreCin, self->ctx.tiling_->baseN);
    uint64_t singleCoreMAlignBaseM = AlignUp(self->ctx.tiling_->singleCoreM, self->ctx.tiling_->baseM);
    uint64_t singleCoreWorkspaceSize =
        singleCoreCinAlignBaseN * singleCoreMAlignBaseM * self->ctx.tiling_->singleCoreDin;
    uint64_t dinOffset =
        (self->ctx.curDinIdx_ - self->ctx.curDinStartIdx_) * singleCoreCinAlignBaseN * singleCoreMAlignBaseM;
    uint64_t nOffset = self->ctx.curNIdx_ * useN * singleCoreMAlignBaseM;
    uint64_t mOffset = self->ctx.curMIdx_ * useN * self->ctx.tiling_->baseM;
    return blockIdx * singleCoreWorkspaceSize + dinOffset + nOffset + mOffset;
}

template <class Intf>
static __aicore__ inline uint64_t ComputeDstOffset(
    Intf* self, FixpipeParamsC310<CO2Layout::COLUMN_MAJOR>& fixPipeParams)
{
    uint64_t dstOffset = 0;
    if (self->ctx.enableSplitDk_) {
        // workspace格式: D singleCoreCin/baseN singleCoreM/baseM baseN baseM
        fixPipeParams.dstStride = self->ctx.baseUseM_;
        uint64_t singleCoreCinAlignBaseN = AlignUp(self->ctx.tiling_->singleCoreCin, self->ctx.tiling_->baseN);
        uint64_t singleCoreMAlignBaseM = AlignUp(self->ctx.tiling_->singleCoreM, self->ctx.tiling_->baseM);
        uint64_t singleCoreWorkspaceSize =
            singleCoreCinAlignBaseN * singleCoreMAlignBaseM * self->ctx.tiling_->singleCoreDin;
        dstOffset =
            (self->ctx.curDinIdx_ - self->ctx.curDinStartIdx_) * singleCoreCinAlignBaseN * singleCoreMAlignBaseM +
            self->ctx.curNIdx_ * self->ctx.tiling_->baseN * singleCoreMAlignBaseM +
            self->ctx.curMIdx_ * self->ctx.tiling_->baseN * self->ctx.tiling_->baseM +
            GetBlockIdx() * singleCoreWorkspaceSize;
    } else if (self->ctx.useUbAccumForSplitK_) {
        fixPipeParams.dstStride = self->ctx.baseUseM_;
        dstOffset = CalSplitKWorkspaceOffset<Intf>(self, self->ctx.baseUseN_, GetBlockIdx());
    } else {
        // loop2_dst_stride, element, c
        fixPipeParams.dstStride = self->ctx.diHiWi_; // dst N stride, loop2_dst_stride (unit: element)
        dstOffset =
            static_cast<uint64_t>(self->ctx.curNIdx_) * self->ctx.tiling_->baseN * self->ctx.diHiWi_ + // cin offset
            static_cast<uint64_t>(self->ctx.curDinIdx_) * self->ctx.hiWi_ +       // di offset, remove useless data
            static_cast<uint64_t>(self->ctx.curMIdx_) * self->ctx.tiling_->baseM; // hi&wi offset
    }
    return dstOffset;
}

template <class Intf>
static __aicore__ inline void LoadL0c2GmForNz2Dn(
    Intf* self, const GlobalTensor<typename Intf::DstT>& output, const LocalTensor<typename Intf::L0cT>& useC1Buf)
{
    // NZ (1, cin1, hi, wi, cin0) -> DN (n, cin, di, hi, wi)
    FixpipeParamsC310<CO2Layout::COLUMN_MAJOR> fixPipeParams;

    fixPipeParams.params.dnNum = 1;             // not use
    fixPipeParams.params.srcNzMatrixStride = 0; // loop3_src_stride
    fixPipeParams.params.dstDnMatrixStride = 0; // loop3_dst_stride

    fixPipeParams.mSize = self->ctx.baseUseM_; // M: hi&wi
    fixPipeParams.params.srcNzC0Stride = 1;    // src M stride, loop0_src_stride (unit: 32B)

    fixPipeParams.nSize = self->ctx.baseUseN_; // N: cin
    // loop1_src_stride, c0_size, cin1
    fixPipeParams.srcStride = AlignUp16(self->ctx.baseUseM_); // src N stride, loop1_src_stride (unit: 32B)
    fixPipeParams.reluEn = self->ctx.tiling_->enRelu;
#if (__NPU_ARCH__ == 5102)
    fixPipeParams.preReluMode = static_cast<ReluMode>(self->ctx.tiling_->enRelu);
#endif
    uint64_t dstOffset = ComputeDstOffset(self, fixPipeParams);
    if constexpr (std::is_same<typename Intf::DstT, bfloat16_t>::value) {
        fixPipeParams.quantPre = (self->ctx.enableSplitDk_ || self->ctx.useUbAccumForSplitK_) ? QuantMode_t::NoQuant : QuantMode_t::F322BF16;
    } else if constexpr (
        (std::is_same<typename Intf::DstT, half>::value) && (std::is_same<typename Intf::L0cT, int32_t>::value)) {
        SetQuantInt32ToHalf(self, fixPipeParams);
    } else if constexpr (std::is_same<typename Intf::DstT, half>::value) {
        fixPipeParams.quantPre = (self->ctx.enableSplitDk_ || self->ctx.useUbAccumForSplitK_) ? QuantMode_t::NoQuant : QuantMode_t::F322F16;
    } else if constexpr (std::is_same<typename Intf::DstT, hifloat8_t>::value) {
        fixPipeParams.quantPre = (self->ctx.useUbAccumForSplitK_) ? QuantMode_t::NoQuant : QuantMode_t::QF322HIF8_PRE; // Half to Away Round
        fixPipeParams.deqScalar = DQ_SCALAR_ONE;
    } else if constexpr (std::is_same<typename Intf::DstT, int8_t>::value) {
        SetQuantInt8(self, fixPipeParams);
    } else if constexpr (std::is_same<typename Intf::DstT, fp8_e4m3fn_t>::value) {
        fixPipeParams.quantPre = (self->ctx.useUbAccumForSplitK_) ? QuantMode_t::NoQuant : QuantMode_t::QF322FP8_PRE;
        fixPipeParams.deqScalar = DQ_SCALAR_ONE;
    }
    if (self->ctx.enableSplitDk_ || self->ctx.useUbAccumForSplitK_) {
#if (__NPU_ARCH__ != 5102)
        if constexpr (std::is_same<typename Intf::L0cT, int32_t>::value) {
            Fixpipe<float, float, CFG_COLUMN_MAJOR_GM>(self->ctx.l0cOutGm_[dstOffset], useC1Buf.template ReinterpretCast<float>(), fixPipeParams);
        } else {
            Fixpipe<float, float, CFG_COLUMN_MAJOR_GM>(self->ctx.l0cOutGm_[dstOffset], useC1Buf, fixPipeParams);
        }
#endif
    } else if (
        Intf::Config::fType::format != Convolution3DBackprop::CubeFormat::UNSUPPORT &&
        self->ctx.tiling_->quantMode == static_cast<uint8_t>(Convolution3DBackprop::QuantMode::VECTOR_QUANT)) {
        uint64_t scaleAddr = self->ctx.curNIdx_ * self->ctx.tiling_->baseN;
        Fixpipe<typename Intf::DstT, typename Intf::L0cT, CFG_COLUMN_MAJOR_GM>(output[dstOffset], useC1Buf, self->ctx.scaleL1Buf_[scaleAddr], fixPipeParams);
    } else {
        Fixpipe<typename Intf::DstT, typename Intf::L0cT, CFG_COLUMN_MAJOR_GM>(output[dstOffset], useC1Buf, fixPipeParams);
    }
}

template <class Intf>
static __aicore__ inline void LoadL0c2GmForNz2Nd(
    Intf* self, const GlobalTensor<typename Intf::DstT>& output, LocalTensor<typename Intf::L0cT>& useC1Buf)
{
    uint64_t dstOffset = 0;
    if (self->ctx.useUbAccumForSplitK_) {
        dstOffset = CalSplitKWorkspaceOffset<Intf>(self, self->ctx.baseUseN_, GetBlockIdx());
    } else {
        dstOffset = static_cast<uint64_t>(self->ctx.curNIdx_) * self->ctx.tiling_->baseN + // cin offset
                    static_cast<uint64_t>(self->ctx.curDinIdx_) * self->ctx.hiWi_ *
                        self->ctx.tiling_->cin + // di offset, remove useless data
                    static_cast<uint64_t>(self->ctx.curMIdx_) * self->ctx.tiling_->baseM *
                        self->ctx.tiling_->cin; // hi&wi offset
    }

    // NZ (1, cin1, hi, wi, cin0) -> ND (n, di, hi, wi, cin)
    FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixPipeParams;
    fixPipeParams.params.ndNum = 1;            // not use
    fixPipeParams.mSize = self->ctx.baseUseM_; // M: hi&wi
    fixPipeParams.nSize = self->ctx.baseUseN_; // N: cin

    // loop1_src_stride, c0_size,cin1
    fixPipeParams.srcStride = AlignUp16(self->ctx.baseUseM_); // src N stride, loop1_src_stride (unit: 32B)
    // loop2_dst_stride, element
    fixPipeParams.dstStride = self->ctx.tiling_->cin; // dst N stride, loop2_dst_stride (unit: element)

    if constexpr (std::is_same<typename Intf::DstT, bfloat16_t>::value) {
        fixPipeParams.quantPre =
            (self->ctx.enableSplitDk_ || self->ctx.useUbAccumForSplitK_) ? QuantMode_t::NoQuant : QuantMode_t::F322BF16;
    } else if constexpr (std::is_same<typename Intf::DstT, half>::value) {
        fixPipeParams.quantPre =
            (self->ctx.enableSplitDk_ || self->ctx.useUbAccumForSplitK_) ? QuantMode_t::NoQuant : QuantMode_t::F322F16;
    } else if constexpr (std::is_same<typename Intf::DstT, hifloat8_t>::value) {
        fixPipeParams.quantPre =
            (self->ctx.useUbAccumForSplitK_) ? QuantMode_t::NoQuant : QuantMode_t::QF322HIF8_PRE; // Half to Away Round
        fixPipeParams.deqScalar = DQ_SCALAR_ONE;
    } else if constexpr (std::is_same<typename Intf::DstT, fp8_e4m3fn_t>::value) {
        fixPipeParams.quantPre = (self->ctx.useUbAccumForSplitK_) ? QuantMode_t::NoQuant : QuantMode_t::QF322FP8_PRE;
        fixPipeParams.deqScalar = DQ_SCALAR_ONE;
    }
    if (self->ctx.useUbAccumForSplitK_) {
#if (__NPU_ARCH__ != 5102)
        // 写入workspace需要保证数据连续便于UB搬运和Cast
        fixPipeParams.dstStride = self->ctx.baseUseN_;
        Fixpipe<float, float, CFG_ROW_MAJOR>(self->ctx.l0cOutGm_[dstOffset], useC1Buf, fixPipeParams);
#endif
    } else {
        Fixpipe<typename Intf::DstT, float, CFG_ROW_MAJOR>(output[dstOffset], useC1Buf, fixPipeParams);
    }
}

template <class Intf>
static __aicore__ inline void CalcCutInWIndexForOnlyH(Intf* self)
{
    // singleCoreM 不对齐wi，M起始位置不一定在每行开头，需要加上curMStartIdx_来计算此时首地址对应位置
    uint32_t mSize = self->ctx.curMIdx_ * self->ctx.tiling_->baseM + self->ctx.curMStartIdx_;
    uint32_t curWiPos = self->ctx.tiling_->wi - (mSize % self->ctx.tiling_->wi);
    self->ctx.headWi_ = (curWiPos == self->ctx.tiling_->wi) ? 0 : curWiPos;
    if (self->ctx.headWi_ > self->ctx.baseUseM_) {
        self->ctx.headWi_ = self->ctx.baseUseM_;
    }
    int32_t leftBaseUseM = self->ctx.baseUseM_ - self->ctx.headWi_;
    if (leftBaseUseM < 0) {
        leftBaseUseM = 0;
    }
    self->ctx.midHi_ = static_cast<uint32_t>(leftBaseUseM) / self->ctx.tiling_->wi;
    self->ctx.tailWi_ = static_cast<uint32_t>(leftBaseUseM) - self->ctx.midHi_ * self->ctx.tiling_->wi;
}

template <class Intf>
static __aicore__ inline void LoadL0c2GmForKernelSplitHFixPipe(
    Intf* self, const int64_t srcOffset, const int64_t dstOffset, const GlobalTensor<typename Intf::DstT>& output,
    LocalTensor<typename Intf::L0cT>& useC1Buf, FixpipeParamsC310<CO2Layout::COLUMN_MAJOR>& fixPipeParams)
{
    if (Intf::Config::fType::format != Convolution3DBackprop::CubeFormat::UNSUPPORT &&
        self->ctx.tiling_->quantMode == static_cast<uint8_t>(Convolution3DBackprop::QuantMode::VECTOR_QUANT)) {
        uint64_t scaleAddr = self->ctx.curNIdx_ * self->ctx.tiling_->baseN;
        Fixpipe<typename Intf::DstT, typename Intf::L0cT, CFG_COLUMN_MAJOR_GM>(
            output[dstOffset], useC1Buf[srcOffset], self->ctx.scaleL1Buf_[scaleAddr], fixPipeParams);
    } else {
        Fixpipe<typename Intf::DstT, typename Intf::L0cT, CFG_COLUMN_MAJOR_GM>(
            output[dstOffset], useC1Buf[srcOffset], fixPipeParams);
    }
}

template <class Intf>
static __aicore__ inline void LoadL0c2GmRowForKernelSplitHFixPipe(
    Intf* self, const int64_t srcOffset, const int64_t dstOffset, const GlobalTensor<typename Intf::DstT>& output,
    LocalTensor<typename Intf::L0cT>& useC1Buf, FixpipeParamsC310<CO2Layout::ROW_MAJOR>& fixPipeParams)
{
    if (Intf::Config::fType::format != Convolution3DBackprop::CubeFormat::UNSUPPORT &&
        self->ctx.tiling_->quantMode == static_cast<uint8_t>(Convolution3DBackprop::QuantMode::VECTOR_QUANT)) {
        uint64_t scaleAddr = self->ctx.curNIdx_ * self->ctx.tiling_->baseN;
        Fixpipe<typename Intf::DstT, typename Intf::L0cT, CFG_ROW_MAJOR>(
            output[dstOffset], useC1Buf[srcOffset], self->ctx.scaleL1Buf_[scaleAddr], fixPipeParams);
    } else {
        Fixpipe<typename Intf::DstT, typename Intf::L0cT, CFG_ROW_MAJOR>(
            output[dstOffset], useC1Buf[srcOffset], fixPipeParams);
    }
}

template <class Intf>
static __aicore__ inline void LoadL0c2GmDnForKernelSplitH(
    Intf* self, const GlobalTensor<typename Intf::DstT>& output, LocalTensor<typename Intf::L0cT>& useC1Buf)
{
    // NZ (1, cin1, splithi, wi, cin0) -> DN (n, cin, di, splithi, wi)
    uint32_t mSize = self->ctx.curMIdx_ * self->ctx.tiling_->baseM;
    int64_t skipDstOffset = self->ctx.tiling_->wi * (self->ctx.tiling_->strideH - 1);
    uint32_t curSkipMSize = (mSize / self->ctx.tiling_->wi) * skipDstOffset;
    int64_t dstOffset = self->ctx.curNIdx_ * self->ctx.tiling_->baseN * self->ctx.diHiWi_ + // cin offset
                        self->ctx.curDinIdx_ * self->ctx.hiWi_ + // di offset, remove useless data
                        mSize + curSkipMSize;                    // hi&wi offset

    uint64_t srcWi = (self->ctx.baseUseM_ < self->ctx.splitWi_) ? self->ctx.baseUseM_ : self->ctx.splitWi_;
    CalcCutInWIndexForOnlyH<Intf>(self);

    FixpipeParamsC310<CO2Layout::COLUMN_MAJOR> fixPipeParams;
    SetFixPipeQuantVal<Intf>(self, fixPipeParams);
    fixPipeParams.nSize = self->ctx.baseUseN_; // N: cin
    fixPipeParams.params.srcNzC0Stride = 1;    // src M stride, loop0_src_stride (unit: 32B)
    // loop1_src_stride, c0_size, cin1
    fixPipeParams.srcStride = AlignUp16(self->ctx.baseUseM_); // src N stride, loop1_src_stride (unit: 32B)
    // loop2_dst_stride, element, c
    fixPipeParams.dstStride = self->ctx.diHiWi_; // dst N stride, loop2_dst_stride (unit: element)
    fixPipeParams.reluEn = self->ctx.tiling_->enRelu;
#if (__NPU_ARCH__ == 5102)
    fixPipeParams.preReluMode = static_cast<ReluMode>(self->ctx.tiling_->enRelu);
#endif
    int64_t srcOffset = 0;
    // fixpipe->gm 需要分首块，中间块，尾块分别对齐到16，然后再搬到gm
    if (self->ctx.headWi_ != 0) {                   // 需要首块
        fixPipeParams.params.dnNum = 1;             // not use
        fixPipeParams.params.srcNzMatrixStride = 0; // loop3_src_stride
        fixPipeParams.params.dstDnMatrixStride = 0; // loop3_dst_stride

        fixPipeParams.mSize = self->ctx.headWi_; // M: 首块的长度
        LoadL0c2GmForKernelSplitHFixPipe(self, srcOffset, dstOffset, output, useC1Buf, fixPipeParams);
        // BLOCK_CUBE: MMAD一次计算为16*16, fixpipe搬到gm的时候取L0c的数据应该固定c0为16，不能随数据类型变化
        srcOffset += self->ctx.headWi_ * BLOCK_CUBE; // headWi_/2是一个子kernel首块w的长度
        dstOffset += self->ctx.headWi_ + skipDstOffset;
    }

    if (self->ctx.midHi_ != 0) {                                                        // 需要中间块
        fixPipeParams.params.dnNum = self->ctx.midHi_;                                  // 循环中间块的行数
        fixPipeParams.params.srcNzMatrixStride = srcWi;                                 // loop3_src_stride
        fixPipeParams.params.dstDnMatrixStride = skipDstOffset + self->ctx.tiling_->wi; // loop3_dst_stride

        fixPipeParams.mSize = srcWi; // M: 中间块一行的长度
        LoadL0c2GmForKernelSplitHFixPipe(self, srcOffset, dstOffset, output, useC1Buf, fixPipeParams);
        // BLOCK_CUBE: MMAD一次计算为16*16, fixpipe搬到gm的时候取L0c的数据应该固定c0为16，不能随数据类型变化
        srcOffset += self->ctx.midHi_ * srcWi * BLOCK_CUBE; // srcWi是一个子kernel中间块w的长度
        dstOffset += self->ctx.midHi_ * (fixPipeParams.params.dstDnMatrixStride);
    }

    if (self->ctx.tailWi_ != 0) {                   // 需要尾块
        fixPipeParams.params.dnNum = 1;             // not use
        fixPipeParams.params.srcNzMatrixStride = 0; // loop3_src_stride
        fixPipeParams.params.dstDnMatrixStride = 0; // loop3_dst_stride

        fixPipeParams.mSize = self->ctx.tailWi_; // M: 尾块的长度
        LoadL0c2GmForKernelSplitHFixPipe(self, srcOffset, dstOffset, output, useC1Buf, fixPipeParams);
    }
}

template <class Intf>
static __aicore__ inline void LoadL0c2GmNdForKernelSplitH(
    Intf* self, const GlobalTensor<typename Intf::DstT>& output, LocalTensor<typename Intf::L0cT>& useC1Buf)
{
    // NZ (1, cin1, splithi, wi, cin0) -> ND (n, di, splithi, wi, cin)
    uint32_t mSize = self->ctx.curMIdx_ * self->ctx.tiling_->baseM;
    int64_t skipDstOffset = self->ctx.tiling_->wi * (self->ctx.tiling_->strideH - 1) * self->ctx.tiling_->cin;
    uint32_t curSkipMSize = (mSize / self->ctx.tiling_->wi) * skipDstOffset;
    int64_t dstOffset = self->ctx.curNIdx_ * self->ctx.tiling_->baseN +                   // cin offset
                        self->ctx.curDinIdx_ * self->ctx.hiWi_ * self->ctx.tiling_->cin + // di offset
                        mSize * self->ctx.tiling_->cin + curSkipMSize;                    // hi&wi offset

    uint64_t srcWi = (self->ctx.baseUseM_ < self->ctx.splitWi_) ? self->ctx.baseUseM_ : self->ctx.splitWi_;
    CalcCutInWIndexForOnlyH<Intf>(self);

    FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixPipeParams;
    SetFixPipeQuantVal<Intf, CO2Layout::ROW_MAJOR>(self, fixPipeParams);
    fixPipeParams.nSize = self->ctx.baseUseN_; // N: cin
    // loop1_src_stride, c0_size, cin1
    fixPipeParams.srcStride = AlignUp16(self->ctx.baseUseM_); // src N stride, loop1_src_stride (unit: 32B)
    // loop2_dst_stride, element, c
    fixPipeParams.dstStride = self->ctx.tiling_->cin; // dst N stride, loop2_dst_stride (unit: element)
#if (__NPU_ARCH__ == 5102)
    fixPipeParams.reluEn = self->ctx.tiling_->enRelu;
    fixPipeParams.preReluMode = static_cast<ReluMode>(self->ctx.tiling_->enRelu);
#endif
    int64_t srcOffset = 0;
    if (self->ctx.headWi_ != 0) {             // 需要首块
        fixPipeParams.params.ndNum = 1;       // not use
        fixPipeParams.params.srcNdStride = 0; // loop3_src_stride
        fixPipeParams.params.dstNdStride = 0; // loop3_dst_stride

        fixPipeParams.mSize = self->ctx.headWi_; // M: 首块的长度
        LoadL0c2GmRowForKernelSplitHFixPipe(self, srcOffset, dstOffset, output, useC1Buf, fixPipeParams);
        // BLOCK_CUBE: MMAD一次计算为16*16, fixpipe搬到gm的时候取L0c的数据应该固定c0为16，不能随数据类型变化
        srcOffset += self->ctx.headWi_ * BLOCK_CUBE; // headWi_/2是一个子kernel首块w的长度
        dstOffset += self->ctx.headWi_ * self->ctx.tiling_->cin + skipDstOffset;
    }

    if (self->ctx.midHi_ != 0) {                       // 需要中间块
        fixPipeParams.params.ndNum = self->ctx.midHi_; // 循环中间块的行数
        fixPipeParams.params.srcNdStride = srcWi;      // loop3_src_stride
        fixPipeParams.params.dstNdStride =
            skipDstOffset + self->ctx.tiling_->wi * self->ctx.tiling_->cin; // loop3_dst_stride

        fixPipeParams.mSize = srcWi; // M: 中间块一行的长度
        LoadL0c2GmRowForKernelSplitHFixPipe(self, srcOffset, dstOffset, output, useC1Buf, fixPipeParams);
        // BLOCK_CUBE: MMAD一次计算为16*16, fixpipe搬到gm的时候取L0c的数据应该固定c0为16，不能随数据类型变化
        srcOffset += self->ctx.midHi_ * srcWi * BLOCK_CUBE; // srcWi是一个子kernel中间块w的长度
        dstOffset += self->ctx.midHi_ * (fixPipeParams.params.dstNdStride);
    }

    if (self->ctx.tailWi_ != 0) {             // 需要尾块
        fixPipeParams.params.ndNum = 1;       // not use
        fixPipeParams.params.srcNdStride = 0; // loop3_src_stride
        fixPipeParams.params.dstNdStride = 0; // loop3_dst_stride

        fixPipeParams.mSize = self->ctx.tailWi_; // M: 尾块的长度
        LoadL0c2GmRowForKernelSplitHFixPipe(self, srcOffset, dstOffset, output, useC1Buf, fixPipeParams);
    }
}

} // namespace Convolution3DBackpropFunc

#endif
