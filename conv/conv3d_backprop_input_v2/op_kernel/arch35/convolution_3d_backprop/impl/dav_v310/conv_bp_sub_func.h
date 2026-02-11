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

#ifndef CONV3D_BP_INPUT_SUB_FUNC_ADVANCE_H
#define CONV3D_BP_INPUT_SUB_FUNC_ADVANCE_H

#include "conv_bp_sub_func_load_gm_to_l1.h"
#include "conv_bp_sub_func_load_gm_to_l1a.h"

using AscendC::DivCeil;
using AscendC::GlobalTensor;
using AscendC::LocalTensor;
using AscendC::SetAtomicAdd;
using AscendC::SetAtomicNone;
using AscendC::FixpipeParamsC310;
using AscendC::Fixpipe;

namespace Convolution3DBackpropFunc {
template <class Intf>
static __aicore__ inline void InitMmadParams(Intf *self)
{
    self->ctx.mmad_.m = self->ctx.baseUseM_;
    if (unlikely(self->ctx.curML0Idx_ == self->ctx.mIter_ - 1)) {
        // 4用来替换除法运算
        self->ctx.mmad_.m = ((self->ctx.baseUseM_ + BLOCK_CUBE - 1) >> 4) << 4;
    }
    self->ctx.mmad_.k = self->ctx.tiling_->baseK;
    self->ctx.mmad_.n = self->ctx.baseUseN_;
    self->ctx.mmad_.unitFlag = 0;
    self->ctx.mmad_.kDirectionAlign = 0;
    self->ctx.mmad_.cmatrixSource = 0;
    self->ctx.mmad_.cmatrixInitVal = 1;
}

template <class Intf>
static __aicore__ inline void CalcParamsMmad(Intf *self, uint32_t kPos, bool isFirstDk)
{
    self->ctx.mmad_.cmatrixInitVal = (kPos == 0 && isFirstDk);
}

template <class Intf>
static __aicore__ inline void MmadLocal(Intf *self, const LocalTensor<typename Intf::SrcT> &l0a,
    const LocalTensor<typename Intf::SrcT> &l0b, LocalTensor<typename Intf::L0cT> &l0c)
{
#if defined(__DAV_310R6__)
    // eType is bias Class
    if constexpr (Intf::Config::eType::format != Convolution3DBackprop::CubeFormat::UNSUPPORT) {
        if (self->ctx.mmad_.cmatrixInitVal) {
            // bias 通路，C矩阵初始值通过BT（C2）进行初始化
            self->ctx.mmad_.cmatrixInitVal = 0; //不初始化，使用bias的值
            self->ctx.mmad_.cmatrixSource = 1; // 第一次mmad，cmatrix值从BT buffer获取
            uint64_t biasOffset = self->ctx.tiling_->isBiasFullLoad ? (self->ctx.curNL0Idx_ * self->ctx.tiling_->baseN) : 0;
            Mmad(l0c, l0a, l0b, self->ctx.biasBTBuf_[biasOffset], self->ctx.mmad_);
            self->ctx.mmad_.cmatrixSource = 0; // 后续值从l0c读取
        } else {
            Mmad(l0c, l0a, l0b, self->ctx.mmad_);
        }
    } else {
        Mmad(l0c, l0a, l0b, self->ctx.mmad_);
    }
#else
    Mmad(l0c, l0a, l0b, self->ctx.mmad_);
#endif
}

// 计算Load2A2的指令参数
template <class Intf>
static __aicore__ inline void InitLoadToA2Params(Intf *self)
{
    // load3dStepM
    self->ctx.load3d_.mExtension = self->ctx.tiling_->baseM;
    // load3dStepK
    self->ctx.load3d_.kExtension = self->ctx.tiling_->baseK;
    // posM
    self->ctx.load3d_.mStartPt = 0;
    // posK
    self->ctx.load3d_.kStartPt = 0;
    // 前放大预处理，统一转换成stride=1的操作
    self->ctx.load3d_.strideW = 1;
    self->ctx.load3d_.strideH = 1;
    if constexpr (Intf::conv3dConfig.kernelSplitMode == TPL_SPLIT_KERNEL_HW) {
        self->ctx.load3d_.filterW = self->ctx.splitWkList_[self->ctx.splitIndex_];
        self->ctx.load3d_.filterH = self->ctx.splitHkList_[self->ctx.splitIndex_];
        self->ctx.load3d_.filterSizeW = (self->ctx.splitWkList_[self->ctx.splitIndex_] >> 8) & 255;
        self->ctx.load3d_.filterSizeH = (self->ctx.splitHkList_[self->ctx.splitIndex_] >> 8) & 255;
    } else if constexpr (Intf::conv3dConfig.kernelSplitMode == TPL_NO_SPLIT_KERNEL) {
        self->ctx.load3d_.filterW = self->ctx.tiling_->wk;
        self->ctx.load3d_.filterH = self->ctx.tiling_->hk;
        self->ctx.load3d_.filterSizeW = (self->ctx.tiling_->wk >> 8) & 255;
        self->ctx.load3d_.filterSizeH = (self->ctx.tiling_->hk >> 8) & 255;
    }

    if constexpr (Intf::conv3dConfig.loadB1Condition == TPL_GM_TO_L1_NO_HK) {
        self->ctx.load3d_.filterH = 1;
        self->ctx.load3d_.filterSizeH = (1 >> 8) & 255;
        self->ctx.load3d_.dilationFilterW = self->ctx.tiling_->dilationW;
        self->ctx.load3d_.dilationFilterH = 1;
    } else if constexpr (Intf::conv3dConfig.loadB1Condition == TPL_GM_TO_L1_NO_HK_WK) {
        self->ctx.load3d_.filterW = 1;
        self->ctx.load3d_.filterH = 1;
        self->ctx.load3d_.filterSizeW = (1 >> 8) & 255;
        self->ctx.load3d_.filterSizeH = (1 >> 8) & 255;
        self->ctx.load3d_.dilationFilterW = 1;
        self->ctx.load3d_.dilationFilterH = 1;
    } else {
        self->ctx.load3d_.dilationFilterW = self->ctx.tiling_->dilationW;
        self->ctx.load3d_.dilationFilterH = self->ctx.tiling_->dilationH;
    }

    self->ctx.load3d_.enTranspose = 0;
    self->ctx.load3d_.fMatrixCtrl = 0;
    // l1 only cut cout in k
    self->ctx.load3d_.channelSize = self->ctx.channelSize_;
}

template <class Intf>
static __aicore__ inline void UpdateL1KParams(Intf *self, const uint64_t kIdx, uint32_t &curStepKa, uint32_t &curStepKb)
{
    if (kIdx == 0) {
        self->ctx.curLoadKbl1_ = self->ctx.curStepKb_ * self->ctx.tiling_->baseK;
        self->ctx.baseUseK_ = self->ctx.tiling_->baseK;
        self->ctx.load3d_.kExtension = self->ctx.tiling_->baseK;
        self->ctx.mmad_.k = self->ctx.tiling_->baseK;
    }

    if (unlikely(kIdx + 1 == self->ctx.kIter_)) {
        self->ctx.baseUseK_ = self->ctx.tailK_;
        self->ctx.load3d_.kExtension = self->ctx.tailK_;
        if constexpr (Intf::conv3dConfig.enableC04Flag) {
            self->ctx.mmad_.k = AlignUp(self->ctx.tailK_, self->ctx.tiling_->c0); // C04场景尾块不一定对齐到c0
        } else {
            self->ctx.mmad_.k = self->ctx.tailK_;
        }
    }

    if (unlikely(kIdx == self->ctx.kIterStepKaTail)) {
        curStepKa = self->ctx.stepKaTail;
    }

    if (unlikely(kIdx == self->ctx.kIterStepKbTail)) {
        curStepKb = self->ctx.stepKbTail;
        self->ctx.curLoadKbl1_ = (curStepKb - 1) * self->ctx.tiling_->baseK + self->ctx.tailK_;
    }
}

template <class Intf>
static __aicore__ inline void UpdateLoadToA2ParamsM(Intf *self)
{
    // load3dStepM
    self->ctx.load3d_.mExtension = self->ctx.baseUseM_;
    // posM: 当前默认stepM = 1
    LoadDataRepeatParam repeatParam = {0, 1, 0, static_cast<uint16_t>(DivCeil16(self->ctx.baseUseM_))};
    SetLoadDataRepeat(repeatParam);
}

template <class Intf>
static __aicore__ inline void UpdateLoadToA2ParamsK(Intf *self, uint32_t kPos)
{
    // posK
    self->ctx.load3d_.kStartPt = kPos * self->ctx.tiling_->baseK;
}

// 计算B2参数
template <class Intf>
static __aicore__ inline void InitLoadToB2Params(Intf *self)
{
    self->ctx.load2dv2_.mStartPosition = 0;
    self->ctx.load2dv2_.kStartPosition = 0;
    self->ctx.load2dv2_.ifTranspose = 0;
    self->ctx.load2dv2_.mStep = 1;
    self->ctx.load2dv2_.dstStride = 1;
    if constexpr (Intf::conv3dConfig.loadB2Condition != TPL_REVERSE_ONLY &&
        Intf::conv3dConfig.loadB2Condition != TPL_NO_TRANSPOSE_NO_REVERSE) {
        self->ctx.load2dv2_.ifTranspose = 1;
    }
}

template <class Intf>
static __aicore__ inline void UpdateLoadToB2ParamsN(Intf *self)
{
    self->ctx.blockBaseN_ = DivCeil16(self->ctx.baseUseN_);
    uint32_t blockSize = self->ctx.tiling_->c0 << 4;
    if constexpr (Intf::conv3dConfig.loadB2Condition != TPL_REVERSE_ONLY) {
        self->ctx.load2dv2_.kStep = self->ctx.blockBaseN_;
        self->ctx.baseB1Offset_ = 0;
        self->ctx.dstB2Stride_ = self->ctx.blockBaseN_ * blockSize;
    } else {
        if constexpr (Intf::conv3dConfig.kernelSplitMode == TPL_SPLIT_KERNEL_HW) {
            if (self->ctx.kSCoutFullLoad_) {
                if (self->ctx.tiling_->strideH == self->ctx.tiling_->hk) {
                    self->ctx.load2dv2_.srcStride = self->ctx.blockBaseN_;
                } else {
                    self->ctx.load2dv2_.srcStride = -self->ctx.blockBaseN_ * self->ctx.tiling_->strideW; // 逆序
                }
                self->ctx.dstB2Stride_ = self->ctx.blockBaseN_ * self->ctx.splitHkWkList_[self->ctx.splitIndex_] * blockSize;
            } else {
                self->ctx.load2dv2_.srcStride = -self->ctx.blockBaseN_; // 逆序
                self->ctx.dstB2Stride_ = 0;
            }
        } else {
            self->ctx.load2dv2_.srcStride = -self->ctx.blockBaseN_; // 逆序
            self->ctx.dstB2Stride_ = 0;
        }
        self->ctx.load2dv2_.mStep = self->ctx.blockBaseN_;
        self->ctx.load2dv2_.dstStride = self->ctx.blockBaseN_;
    }
}

template <class Intf>
static __aicore__ inline void UpdateLoadToB2ParamsK(Intf *self, uint32_t kPos)
{
    if constexpr (Intf::conv3dConfig.loadB2Condition != TPL_REVERSE_ONLY) {
        self->ctx.load2dv2_.srcStride = self->ctx.curLoadKbl1_ >> self->ctx.tiling_->c0BitsB;
    }
}

template <class Intf>
static __aicore__ inline void UpdateLoadToB2ParamsKForKernelSplit(Intf *self, uint32_t kPos)
{
    uint32_t kRepeat = self->ctx.tiling_->baseK / self->ctx.splitHkWkC0List_[self->ctx.splitIndex_]; // cout1
    self->ctx.startAddrOffset_ = self->ctx.hkWk_ * kPos * kRepeat; // 跳过多少个cout
    self->ctx.load2dv2_.kStep = self->ctx.splitWkList_[self->ctx.splitIndex_]; // 当前kernel拆分场景只可以为1或者2
    self->ctx.load2dv2_.mStartPosition = ((self->ctx.splitHkList_[self->ctx.splitIndex_] - 1) *
        self->ctx.tiling_->strideH * self->ctx.tiling_->wk +
        (self->ctx.splitWkList_[self->ctx.splitIndex_] - 1) * self->ctx.tiling_->strideW +
        (self->ctx.splitHIndex_ * self->ctx.tiling_->wk + self->ctx.splitWIndex_)) * self->ctx.blockBaseN_;
}

template <class Intf>
static __aicore__ inline void LoadToB2NoTransposeNoReverse(Intf *self, const LocalTensor<typename Intf::SrcT> &l1B1Matrix,
    uint32_t srcB1Offset, uint32_t kPos, LocalTensor<typename Intf::SrcT> &l0b)
{
    self->ctx.load2dv2_.mStartPosition = 0;
    self->ctx.load2dv2_.kStartPosition = 0;
    self->ctx.load2dv2_.kStep = 1;
    self->ctx.load2dv2_.mStep = self->ctx.blockBaseN_ * DivCeilC0<Intf>(self, self->ctx.baseUseK_);
    srcB1Offset = kPos * AlignUp16(self->ctx.baseUseN_) * self->ctx.tiling_->baseK;
    LoadData(l0b, l1B1Matrix[srcB1Offset], self->ctx.load2dv2_);
}

template <class Intf>
static __aicore__ inline void LoadToB2ReverseOnlyCommon(Intf *self, const LocalTensor<typename Intf::SrcT> &l1B1Matrix,
    uint32_t blockSize, uint32_t srcB1Offset, uint32_t dstB2Offset, const uint32_t kPos, LocalTensor<typename Intf::SrcT> &l0b)
{
    // baseK可能会跨越HkWk,分多组HkWk进行逆序,以baseK=256,HkWk=9,kPos=1为例
    // *******kk  loop1: kStep=2
    // kkkkkkkkk  loop2: kStep=9
    // kkkkk****  loop3: kStep=5
    uint32_t curHkWkSize = self->ctx.singleShapeHWk_;
    if constexpr (Intf::conv3dConfig.kernelSplitMode != TPL_NO_SPLIT_KERNEL) {
        curHkWkSize = self->ctx.splitHkWkList_[self->ctx.splitIndex_];
    }
    uint32_t kStartPos = kPos * (self->ctx.tiling_->baseK >> self->ctx.tiling_->c0BitsB);
    uint32_t kEndPos = kStartPos + ((self->ctx.baseUseK_ + self->ctx.tiling_->c0 -1) >> self->ctx.tiling_->c0BitsB);
    uint32_t hwKStartIdx = DivHkWk<Intf>(self, kStartPos);
    uint32_t hwKEndIdx = DivCeilHkWk<Intf>(self, kEndPos);
    uint32_t curKRepeat = hwKEndIdx - hwKStartIdx;
    for (uint32_t i = 0; i < curKRepeat; i++) {
        uint32_t curHWkStart = (hwKStartIdx + i) * curHkWkSize;
        uint32_t curHWkEnd = curHWkStart + curHkWkSize;
        uint32_t kStepStart = curHWkStart < kStartPos ? kStartPos : curHWkStart;
        uint32_t kStepEnd = curHWkEnd > kEndPos ? kEndPos : curHWkEnd;
        self->ctx.load2dv2_.kStep = kStepEnd - kStepStart;
        self->ctx.load2dv2_.kStartPosition = curHWkStart;
        self->ctx.load2dv2_.mStartPosition = (curHWkEnd - 1 - kStepStart) * self->ctx.blockBaseN_;
        LoadData(l0b[dstB2Offset], l1B1Matrix[srcB1Offset], self->ctx.load2dv2_);
        dstB2Offset += (kStepEnd - kStepStart) * self->ctx.blockBaseN_ * blockSize;
    }
}

template <class Intf>
static __aicore__ inline void LoadToB2ForKernelSplitB1FullLoad(Intf *self, const LocalTensor<typename Intf::SrcT> &l1B1Matrix,
    uint32_t blockSize, uint32_t srcB1Offset, uint32_t dstB2Offset, LocalTensor<typename Intf::SrcT> &l0b)
{
    if (self->ctx.tiling_->strideH == self->ctx.tiling_->hk) {
        uint32_t kRepeat = self->ctx.baseUseK_ >> self->ctx.tiling_->c0BitsB;
        for (uint32_t i = 0; i < kRepeat; i++) {
            self->ctx.load2dv2_.kStartPosition = self->ctx.startAddrOffset_ + i * self->ctx.hkWk_;
            LoadData(l0b[dstB2Offset], l1B1Matrix[srcB1Offset], self->ctx.load2dv2_);
            dstB2Offset += self->ctx.dstB2Stride_;
        }
    } else {
        auto firstMStartPosition = self->ctx.load2dv2_.mStartPosition;
        uint64_t curMPos = static_cast<uint64_t>(self->ctx.tiling_->wk) *
            static_cast<uint64_t>(self->ctx.tiling_->strideH) * static_cast<uint64_t>(self->ctx.blockBaseN_);
        uint32_t dstStride = self->ctx.splitWkList_[self->ctx.splitIndex_] * self->ctx.blockBaseN_ * blockSize;
        uint32_t curKRepeat = self->ctx.baseUseK_ / self->ctx.splitHkWkC0List_[self->ctx.splitIndex_];
        curKRepeat = curKRepeat > 0 ? curKRepeat : 1;
        for (uint32_t i = 0; i < curKRepeat; i++) {
            self->ctx.load2dv2_.mStartPosition = firstMStartPosition;
            self->ctx.load2dv2_.kStartPosition = 0;
            srcB1Offset = (self->ctx.startAddrOffset_ + i * self->ctx.hkWk_) * self->ctx.blockBaseN_ * blockSize;
            for (uint32_t j = 0; j < self->ctx.splitHkList_[self->ctx.splitIndex_]; j++) {
                self->ctx.load2dv2_.mStartPosition -= j * curMPos;
                dstB2Offset = j * dstStride + i * self->ctx.dstB2Stride_;
                LoadData(l0b[dstB2Offset], l1B1Matrix[srcB1Offset], self->ctx.load2dv2_);
            }
        }
    }
}

template <class Intf>
static __aicore__ inline void LoadToB2ReverseOnly(Intf *self, const LocalTensor<typename Intf::SrcT> &l1B1Matrix,
    uint32_t blockSize, uint32_t srcB1Offset, uint32_t dstB2Offset, const uint32_t kPos, LocalTensor<typename Intf::SrcT> &l0b)
{
    if constexpr (Intf::conv3dConfig.kernelSplitMode == TPL_SPLIT_KERNEL_HW) {
        if (self->ctx.kSCoutFullLoad_) {  // B1全载时，在load2b2阶段完成子kernel重排
            LoadToB2ForKernelSplitB1FullLoad(self, l1B1Matrix, blockSize, srcB1Offset, dstB2Offset, l0b);
        } else {
            LoadToB2ReverseOnlyCommon(self, l1B1Matrix, blockSize, srcB1Offset, dstB2Offset, kPos, l0b);
        }
    } else {
        if (self->ctx.dkHkWk_ == 1) {
            // dkhkwk == 1, 此时已提前做好transpose,且不需要逆序，直接读取数据
            LoadToB2NoTransposeNoReverse(self, l1B1Matrix, srcB1Offset, kPos, l0b);
        } else {
            LoadToB2ReverseOnlyCommon(self, l1B1Matrix, blockSize, srcB1Offset, dstB2Offset, kPos, l0b);
        }
    }
}

template <class Intf>
static __aicore__ inline void LoadToB2(Intf *self, const LocalTensor<typename Intf::SrcT> &l1B1Matrix, uint32_t kPos, LocalTensor<typename Intf::SrcT> &l0b)
{
    uint32_t blockSize = self->ctx.tiling_->c0 << 4;
    uint32_t srcB1Offset = 0;
    uint32_t dstB2Offset = 0;
    if constexpr (Intf::conv3dConfig.loadB2Condition == TPL_TRANSPOSE_ONLY) {
        uint32_t kBlockC0Num = kPos * self->ctx.tiling_->baseK >> self->ctx.tiling_->c0BitsB;   // 转置逆序
        uint32_t curKRepeat = self->ctx.baseUseK_ >> self->ctx.tiling_->c0BitsB;
        for (uint32_t i = 0; i < curKRepeat; i++) {
            uint32_t idxC1out = kBlockC0Num + i;
            dstB2Offset = i * self->ctx.blockBaseN_ * blockSize;
            srcB1Offset = self->ctx.baseB1Offset_ + idxC1out * blockSize;
            LoadData(l0b[dstB2Offset], l1B1Matrix[srcB1Offset], self->ctx.load2dv2_);
        }
    } else if constexpr (Intf::conv3dConfig.loadB2Condition == TPL_REVERSE_ONLY) {
        LoadToB2ReverseOnly(self, l1B1Matrix, blockSize, srcB1Offset, dstB2Offset, kPos, l0b);
    } else if constexpr (Intf::conv3dConfig.loadB2Condition == TPL_NO_TRANSPOSE_NO_REVERSE) {
        LoadToB2NoTransposeNoReverse(self, l1B1Matrix, srcB1Offset, kPos, l0b);
    } else {
        uint32_t kBlockC0Num = kPos * self->ctx.tiling_->baseK >> self->ctx.tiling_->c0BitsB;
        uint32_t curL1Cout = DivHkWk<Intf>(self, self->ctx.curLoadKbl1_) << self->ctx.tiling_->c0BitsB;
        uint32_t curKRepeat = self->ctx.baseUseK_ >> self->ctx.tiling_->c0BitsB;
        for (uint32_t i = 0; i < curKRepeat; i++) {
            uint32_t iRev = kBlockC0Num + i;
            uint32_t idxC1out = DivHkWk<Intf>(self, iRev);
            srcB1Offset = self->ctx.baseB1Offset_ + idxC1out * blockSize +
                                   (self->ctx.singleShapeHWk_ - 1 + idxC1out * self->ctx.singleShapeHWk_ - iRev) * curL1Cout;
            LoadData(l0b[dstB2Offset], l1B1Matrix[srcB1Offset], self->ctx.load2dv2_);
            dstB2Offset += self->ctx.dstB2Stride_;
        }
    }
}

// 数据从A1加载到A2
template <class Intf>
static __aicore__ inline void LoadToA2(Intf *self, const LocalTensor<typename Intf::SrcT> &l1A1Matrix, LocalTensor<typename Intf::SrcT> &l0a)
{
    LoadData(l0a, l1A1Matrix, self->ctx.load3d_);
}

template <class Intf>
static __aicore__ inline void LoadL0c2GmForNz2Dn(Intf *self, const GlobalTensor<typename Intf::DstT> &output,
    const LocalTensor<typename Intf::L0cT> &useC1Buf)
{
    // NZ (1, cin1, hi, wi, cin0) -> DN (n, cin, di, hi, wi)
    //
    FixpipeParamsC310<CO2Layout::COLUMN_MAJOR> fixPipeParams;

    fixPipeParams.params.dnNum = 1; // not use
    fixPipeParams.params.srcNzMatrixStride = 0; // loop3_src_stride
    fixPipeParams.params.dstDnMatrixStride = 0; // loop3_dst_stride

    fixPipeParams.mSize = self->ctx.baseUseM_; // M: hi&wi
    fixPipeParams.params.srcNzC0Stride = 1; // src M stride, loop0_src_stride (unit: 32B)

    fixPipeParams.nSize = self->ctx.baseUseN_; // N: cin
    // loop1_src_stride, c0_size, cin1
    fixPipeParams.srcStride = AlignUp16(self->ctx.baseUseM_); // src N stride, loop1_src_stride (unit: 32B)
    uint64_t dstOffset = 0;
    if (self->ctx.enableSplitDk_) {
        // workspace格式: D singleCoreCin/baseN singleCoreM/baseM baseN baseM
        fixPipeParams.dstStride = self->ctx.baseUseM_;
        uint64_t singleCoreCinAlignBaseN = AlignUp(self->ctx.tiling_->singleCoreCin, self->ctx.tiling_->baseN);
        uint64_t singleCoreMAlignBaseM = AlignUp(self->ctx.tiling_->singleCoreM, self->ctx.tiling_->baseM);
        uint64_t singleCoreWorkspaceSize = singleCoreCinAlignBaseN * singleCoreMAlignBaseM *
            self->ctx.tiling_->singleCoreDin;
        dstOffset = (self->ctx.curDinIdx_ - self->ctx.curDinStartIdx_) * singleCoreCinAlignBaseN *
            singleCoreMAlignBaseM + self->ctx.curNL0Idx_ * self->ctx.tiling_->baseN *
            singleCoreMAlignBaseM + self->ctx.curML0Idx_ * self->ctx.tiling_->baseN *
            self->ctx.tiling_->baseM + GetBlockIdx() * singleCoreWorkspaceSize;
    } else {
        // loop2_dst_stride, element, c
        fixPipeParams.dstStride = self->ctx.diHiWi_; // dst N stride, loop2_dst_stride (unit: element)
        dstOffset = static_cast<uint64_t>(self->ctx.curNL0Idx_) * self->ctx.tiling_->baseN * self->ctx.diHiWi_ + // cin offset
            static_cast<uint64_t>(self->ctx.curDinIdx_) * self->ctx.hiWi_ + // di offset, remove useless data
            static_cast<uint64_t>(self->ctx.curML0Idx_) * self->ctx.tiling_->baseM; // hi&wi offset
    }
    if constexpr(std::is_same<typename Intf::DstT, bfloat16_t>::value) {
        fixPipeParams.quantPre = self->ctx.enableSplitDk_ ? QuantMode_t::NoQuant : QuantMode_t::F322BF16;
    } else if constexpr(std::is_same<typename Intf::DstT, half>::value) {
        fixPipeParams.quantPre = self->ctx.enableSplitDk_ ? QuantMode_t::NoQuant : QuantMode_t::F322F16;
    } else if constexpr(std::is_same<typename Intf::DstT, hifloat8_t>::value) {
        fixPipeParams.quantPre = QuantMode_t::QF322HIF8_PRE; // Half to Away Round
        fixPipeParams.deqScalar = DQ_SCALAR_ONE;
    } else if constexpr(std::is_same<typename Intf::DstT, fp8_e4m3fn_t>::value) {
        fixPipeParams.quantPre = QuantMode_t::QF322FP8_PRE;
        fixPipeParams.deqScalar = DQ_SCALAR_ONE;
    }
    if (self->ctx.enableSplitDk_) {
        Fixpipe<float, float, CFG_COLUMN_MAJOR>(self->ctx.l0cOutGm_[dstOffset],
            useC1Buf, fixPipeParams);
    } else {
        Fixpipe<typename Intf::DstT, float, CFG_COLUMN_MAJOR>(output[dstOffset],
            useC1Buf, fixPipeParams);
    }
}

template <class Intf>
static __aicore__ inline void LoadL0c2GmForNz2Nd(Intf *self, const GlobalTensor<typename Intf::DstT> &output,
    LocalTensor<typename Intf::L0cT> &useC1Buf)
{
    uint64_t dstOffset = static_cast<uint64_t>(self->ctx.curNL0Idx_) * self->ctx.tiling_->baseN + // cin offset
                    static_cast<uint64_t>(self->ctx.curDinIdx_) * self->ctx.hiWi_ * self->ctx.tiling_->cin + // di offset, remove useless data
                    static_cast<uint64_t>(self->ctx.curML0Idx_) * self->ctx.tiling_->baseM * self->ctx.tiling_->cin; // hi&wi offset

    // NZ (1, cin1, hi, wi, cin0) -> ND (n, di, hi, wi, cin)
    //
    FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixPipeParams;
    fixPipeParams.params.ndNum = 1; // not use
    fixPipeParams.mSize = self->ctx.baseUseM_; // M: hi&wi
    fixPipeParams.nSize = self->ctx.baseUseN_; // N: cin

    // loop1_src_stride, c0_size,cin1
    fixPipeParams.srcStride = AlignUp16(self->ctx.baseUseM_); // src N stride, loop1_src_stride (unit: 32B)
    // loop2_dst_stride, element, c
    fixPipeParams.dstStride = self->ctx.tiling_->cin; // dst N stride, loop2_dst_stride (unit: element)

    if constexpr(std::is_same<typename Intf::DstT, bfloat16_t>::value) {
        fixPipeParams.quantPre = QuantMode_t::F322BF16;
    } else if constexpr(std::is_same<typename Intf::DstT, half>::value) {
        fixPipeParams.quantPre = QuantMode_t::F322F16;
    } else if constexpr(std::is_same<typename Intf::DstT, hifloat8_t>::value) {
        fixPipeParams.quantPre = QuantMode_t::QF322HIF8_PRE; // Half to Away Round
        fixPipeParams.deqScalar = DQ_SCALAR_ONE;
    }
    Fixpipe<typename Intf::DstT, float, CFG_ROW_MAJOR>(output[dstOffset],
        useC1Buf, fixPipeParams);
}

template <class Intf>
static __aicore__ inline void CalcCutInWIndexForOnlyH(Intf *self)
{
    // singleCoreM 不对齐wi，M起始位置不一定在每行开头，需要加上curMStartIdx_来计算此时首地址对应位置
    uint32_t mSize = self->ctx.curML0Idx_ * self->ctx.tiling_->baseM + self->ctx.curMStartIdx_;
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
static __aicore__ inline void LoadL0c2GmForKernelSplitH(Intf *self, const GlobalTensor<typename Intf::DstT> &output,
    LocalTensor<typename Intf::L0cT> &useC1Buf)
{
    // NZ (1, cin1, splithi, wi, cin0) -> DN (n, cin, di, splithi, wi)
    uint32_t mSize = self->ctx.curML0Idx_ * self->ctx.tiling_->baseM;
    int64_t skipDstOffset = self->ctx.tiling_->wi * (self->ctx.tiling_->strideH - 1);
    uint32_t curSkipMSize = (mSize / self->ctx.tiling_->wi) * skipDstOffset;
    int64_t dstOffset = self->ctx.curNL0Idx_ * self->ctx.tiling_->baseN * self->ctx.diHiWi_ + // cin offset
            self->ctx.curDinIdx_ * self->ctx.hiWi_ + // di offset, remove useless data
            mSize + curSkipMSize; // hi&wi offset

    uint64_t srcWi = (self->ctx.baseUseM_ < self->ctx.splitWi_) ? self->ctx.baseUseM_ : self->ctx.splitWi_;
    CalcCutInWIndexForOnlyH<Intf>(self);

    FixpipeParamsC310<CO2Layout::COLUMN_MAJOR> fixPipeParams;
    SetFixPipeQuantVal<Intf>(fixPipeParams);
    fixPipeParams.nSize = self->ctx.baseUseN_; // N: cin
    fixPipeParams.params.srcNzC0Stride = 1; // src M stride, loop0_src_stride (unit: 32B)
    // loop1_src_stride, c0_size, cin1
    fixPipeParams.srcStride = AlignUp16(self->ctx.baseUseM_); // src N stride, loop1_src_stride (unit: 32B)
    // loop2_dst_stride, element, c
    fixPipeParams.dstStride = self->ctx.diHiWi_; // dst N stride, loop2_dst_stride (unit: element)
    int64_t srcOffset = 0;
    // fixpipe->ub 需要分首块，中间块，尾块分别对齐到16，然后再搬到ub
    if (self->ctx.headWi_ != 0) { // 需要首块
        fixPipeParams.params.dnNum = 1; // not use
        fixPipeParams.params.srcNzMatrixStride = 0; // loop3_src_stride
        fixPipeParams.params.dstDnMatrixStride = 0; // loop3_dst_stride

        fixPipeParams.mSize = self->ctx.headWi_; // M: 首块的长度
        Fixpipe<typename Intf::DstT, float, CFG_COLUMN_MAJOR>(output[dstOffset], useC1Buf[srcOffset], fixPipeParams);
        // BLOCK_CUBE: MMAD一次计算为16*16, fixpipe搬到ub的时候取L0c的数据应该固定c0为16，不能随数据类型变化
        srcOffset += self->ctx.headWi_ * BLOCK_CUBE; // headWi_/2是一个子kernel首块w的长度
        dstOffset += self->ctx.headWi_ + skipDstOffset;
    }

    if (self->ctx.midHi_ != 0) { // 需要中间块
        fixPipeParams.params.dnNum = self->ctx.midHi_; // 循环中间块的行数
        fixPipeParams.params.srcNzMatrixStride = srcWi; // loop3_src_stride
        fixPipeParams.params.dstDnMatrixStride = skipDstOffset + self->ctx.tiling_->wi; // loop3_dst_stride

        fixPipeParams.mSize = srcWi; // M: 中间块一行的长度
        Fixpipe<typename Intf::DstT, float, CFG_COLUMN_MAJOR>(output[dstOffset],
            useC1Buf[srcOffset], fixPipeParams);
        // BLOCK_CUBE: MMAD一次计算为16*16, fixpipe搬到ub的时候取L0c的数据应该固定c0为16，不能随数据类型变化
        srcOffset += self->ctx.midHi_ * srcWi * BLOCK_CUBE; // srcWi是一个子kernel中间块w的长度
        dstOffset += self->ctx.midHi_ * (fixPipeParams.params.dstDnMatrixStride);
    }

    if (self->ctx.tailWi_ != 0) { // 需要尾块
        fixPipeParams.params.dnNum = 1; // not use
        fixPipeParams.params.srcNzMatrixStride = 0; // loop3_src_stride
        fixPipeParams.params.dstDnMatrixStride = 0; // loop3_dst_stride

        fixPipeParams.mSize = self->ctx.tailWi_; // M: 尾块的长度
        Fixpipe<typename Intf::DstT, float, CFG_COLUMN_MAJOR>(output[dstOffset],
            useC1Buf[srcOffset], fixPipeParams);
    }
}

template <class Intf>
static __aicore__ inline void FreeTensorC1Buf(Intf *self, LocalTensor<typename Intf::L0cT> &useC1Buf)
{
    if ASCEND_IS_AIC {
        // l0c不搬运数据，需释放l0c的空间，否则会等待卡死
        if (self->ctx.l0cPingPongFlag_) {
            self->ctx.l0cPing_.FreeTensor(useC1Buf);
        } else {
            self->ctx.l0cPong_.FreeTensor(useC1Buf);
        }
        if (self->ctx.tiling_->cl0Pbuffer > 1) {
            self->ctx.l0cPingPongFlag_ = !self->ctx.l0cPingPongFlag_;
        }
    }
}

template <class Intf>
static __aicore__ inline void L0CDeQue(Intf *self, LocalTensor<typename Intf::L0cT> &useC1Buf)
{
    if ASCEND_IS_AIV {
        return;
    }
    
    if (self->ctx.l0cPingPongFlag_) {
        useC1Buf = self->ctx.l0cPing_.template DeQue<typename Intf::L0cT>();
    } else {
        useC1Buf = self->ctx.l0cPong_.template DeQue<typename Intf::L0cT>();
    }
    
}

template <class Intf>
__aicore__ inline void FullLoadBias(Intf *self)
{
    if ASCEND_IS_AIV {
        return ;
    }
    // GM -> L1
    LocalTensor<typename Intf::BiasT> useBiasL1 = self->ctx.biasL1Que_.template AllocTensor<typename Intf::BiasT>();
    uint16_t blockBytes = self->ctx.singleShapeCin_ * sizeof(typename Intf::BiasT);
    DataCopyParams dataCopyParams(1, blockBytes, 0, 0);
    uint8_t rightPadding = DivCeil(blockBytes, ONE_BLK_SIZE) * ONE_BLK_SIZE / sizeof(typename Intf::BiasT) -
        self->ctx.singleShapeCin_;
    DataCopyPadParams padParams(true, 0, rightPadding, 0);
#ifndef __CCE_KT_TEST__
    DataCopyPad<typename Intf::BiasT>(useBiasL1, self->ctx.biasGlobal_, dataCopyParams, padParams);
#endif
    self->ctx.biasL1Que_.EnQue(useBiasL1);
    self->ctx.biasL1Buf_ = self->ctx.biasL1Que_.template DeQue<typename Intf::BiasT>();

    // L1 -> BT
    LocalTensor<typename Intf::L0cT> useBT = self->ctx.biasBTQue_.template AllocTensor<typename Intf::L0cT>();
    // BT Buffer 需要64字节对齐
    dataCopyParams.blockLen = DivCeil(blockBytes, 64) << 1;
    DataCopy(useBT, self->ctx.biasL1Buf_, dataCopyParams);
    self->ctx.biasBTQue_.EnQue(useBT);
    self->ctx.biasBTBuf_ = self->ctx.biasBTQue_.template DeQue<typename Intf::L0cT>();
}

template <class Intf>
__aicore__ inline void LoadBiasToBT(Intf *self)
{
    if ASCEND_IS_AIV {
        return ;
    }
    uint32_t btCinSize = self->ctx.singleShapeCin_ < self->ctx.baseUseN_ ? self->ctx.singleShapeCin_ : self->ctx.baseUseN_;
    // GM -> L1
    LocalTensor<typename Intf::BiasT> useBiasL1 = self->ctx.biasL1Que_.template AllocTensor<typename Intf::BiasT>();
    uint16_t blockBytes = btCinSize  * sizeof(typename Intf::BiasT);
    DataCopyParams dataCopyParams(1, blockBytes, 0, 0);
    uint8_t rightPadding = DivCeil(blockBytes, ONE_BLK_SIZE) * ONE_BLK_SIZE / sizeof(typename Intf::BiasT) - btCinSize;
    DataCopyPadParams padParams(true, 0, rightPadding, 0);
    uint64_t biasOffset = self->ctx.curCinStartIdx_ + self->ctx.curNL0Idx_ * self->ctx.tiling_->baseN;
#ifndef __CCE_KT_TEST__
    DataCopyPad<typename Intf::BiasT>(useBiasL1, self->ctx.biasGlobal_[biasOffset], dataCopyParams, padParams);
#endif
    self->ctx.biasL1Que_.EnQue(useBiasL1);
    self->ctx.biasL1Buf_ = self->ctx.biasL1Que_.template DeQue<typename Intf::BiasT>();

    // L1 -> BT
    LocalTensor<typename Intf::L0cT> useBT = self->ctx.biasBTQue_.template AllocTensor<typename Intf::L0cT>();
    // BT Buffer 需要64字节对齐
    dataCopyParams.blockLen = DivCeil(blockBytes, 64) << 1;
    DataCopy(useBT, self->ctx.biasL1Buf_, dataCopyParams);
    self->ctx.biasL1Que_.FreeTensor(self->ctx.biasL1Buf_);
    self->ctx.biasBTQue_.EnQue(useBT);
    self->ctx.biasBTBuf_ = self->ctx.biasBTQue_.template DeQue<typename Intf::L0cT>();
}

template <class Intf>
static __aicore__ inline void LoadL0c2OutForNz2Dn(Intf *self, const GlobalTensor<typename Intf::DstT> &output,
    LocalTensor<typename Intf::L0cT> &useC1Buf)
{
    if constexpr (Intf::conv3dConfig.kernelSplitMode == TPL_SPLIT_KERNEL_HW) {
        LoadL0c2OutForKernelSplitHW<Intf>(self, useC1Buf);
    } else if constexpr (Intf::conv3dConfig.kernelSplitMode == TPL_SPLIT_KERNEL_H) {
        LoadL0c2GmForKernelSplitH<Intf>(self, output, useC1Buf);
    } else {
        LoadL0c2GmForNz2Dn<Intf>(self, output, useC1Buf);
    }
}

template <class Intf>
static __aicore__ inline void LoadL0c2Gm(Intf *self, const GlobalTensor<typename Intf::DstT> &output,
                                         uint8_t enAtomic = 0, bool enSequentialWrite = false)
{
    if constexpr (Intf::conv3dConfig.kernelSplitMode != TPL_SPLIT_KERNEL_HW) {
        if ASCEND_IS_AIV {
            return;
        }
    }

    // 只有当参与了mmad运算时，才进行L0C的出队和拷贝
    if (!self->ctx.needComputeFlag_) {
        return;
    }

    LocalTensor<typename Intf::L0cT> useC1Buf;
    L0CDeQue<Intf>(self, useC1Buf);

    if constexpr(std::is_same<typename Intf::DstT, float>::value) {
        if (enAtomic == 1) {
            SetAtomicAdd<typename Intf::DstT>();
        }
    } else if constexpr(std::is_same<typename Intf::DstT, bfloat16_t>::value || std::is_same<typename Intf::DstT, half>::value) {
        if (self->ctx.enableSplitDk_) {
            // 满足此条件时当前Din不是第一次计算，开启累加
            enAtomic = (self->ctx.curDkIdx_ > 0) &&
                (self->ctx.curDinIdx_ + self->ctx.tiling_->padFront - self->ctx.tiling_->dout + 1 != self->ctx.curDkIdx_);
            if (enAtomic == 1) {
                SetAtomicAdd<float>();
            }
        }
    }
    if constexpr (Intf::Config::dType::format == Convolution3DBackprop::CubeFormat::NCDHW) {
        if (!enSequentialWrite) {
            LoadL0c2OutForNz2Dn<Intf>(self, output, useC1Buf);
        } else {
            return;
        }
    } else {
        if (!enSequentialWrite) {
            LoadL0c2GmForNz2Nd<Intf>(self, output, useC1Buf);
        } else {
            return;
        }
    }
    if constexpr(std::is_same<typename Intf::DstT, float>::value || std::is_same<typename Intf::DstT, bfloat16_t>::value ||
        std::is_same<typename Intf::DstT, half>::value) {
        if (enAtomic == 1) {
            SetAtomicNone();
        }
    }
    FreeTensorC1Buf(self, useC1Buf);
}
}  // namespace Convolution3DBackpropFunc

#endif
