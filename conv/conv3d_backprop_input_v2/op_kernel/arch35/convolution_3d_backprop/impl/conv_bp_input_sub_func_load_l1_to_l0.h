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
 * \file conv_bp_input_sub_func_load_l1_to_l0.h
 * \brief L1 to L0 load functions for conv3d backprop input
 */

#ifndef CONV3D_BP_INPUT_SUB_FUNC_LOAD_L1_TO_L0_H
#define CONV3D_BP_INPUT_SUB_FUNC_LOAD_L1_TO_L0_H

namespace Convolution3DBackpropFunc {

// 计算Load2A2的指令参数
template <class Intf>
static __aicore__ inline void InitLoadToA2Params(Intf* self)
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
    if constexpr (std::is_same<typename Intf::L0cT, int32_t>::value) {
        self->ctx.load3d_.padValue = self->ctx.tiling_->offsetX;
    }
}

template <class Intf>
static __aicore__ inline void UpdateLoadToA2ParamsM(Intf* self)
{
    // load3dStepM
    self->ctx.load3d_.mExtension = self->ctx.baseUseM_;
    // posM: 当前默认stepM = 1
#if defined(ASC_DEVKIT_VERSION_NUM) && (ASC_DEVKIT_VERSION_NUM >= 90000000)
    LoadDataRepeatParamWithStride repeatParam = {0, 1, 0, static_cast<uint16_t>(DivCeil16(self->ctx.baseUseM_))};
    SetLoadDataRepeatWithStride(repeatParam);
#else
    LoadDataRepeatParam repeatParam = {0, 1, 0, static_cast<uint16_t>(DivCeil16(self->ctx.baseUseM_))};
    SetLoadDataRepeat(repeatParam);
#endif
}

template <class Intf>
static __aicore__ inline void UpdateLoadToA2ParamsK(Intf* self, uint32_t kPos)
{
    // posK
    self->ctx.load3d_.kStartPt = kPos * self->ctx.tiling_->baseK;
}

// 计算B2参数
template <class Intf>
static __aicore__ inline void InitLoadToB2Params(Intf* self)
{
    self->ctx.load2dv2_.mStartPosition = 0;
    self->ctx.load2dv2_.kStartPosition = 0;
    self->ctx.load2dv2_.ifTranspose = 0;
    self->ctx.load2dv2_.mStep = 1;
    self->ctx.load2dv2_.dstStride = 1;
    if constexpr (
        Intf::conv3dConfig.loadB2Condition != TPL_REVERSE_ONLY &&
        Intf::conv3dConfig.loadB2Condition != TPL_NO_TRANSPOSE_NO_REVERSE) {
        self->ctx.load2dv2_.ifTranspose = 1;
    }
}

template <class Intf>
static __aicore__ inline void UpdateLoadToB2ParamsN(Intf* self)
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
                self->ctx.dstB2Stride_ =
                    self->ctx.blockBaseN_ * self->ctx.splitHkWkList_[self->ctx.splitIndex_] * blockSize;
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
static __aicore__ inline void UpdateLoadToB2ParamsK(Intf* self, uint32_t kPos)
{
    if constexpr (Intf::conv3dConfig.loadB2Condition != TPL_REVERSE_ONLY) {
        self->ctx.load2dv2_.srcStride = self->ctx.curLoadKbl1_ >> self->ctx.tiling_->c0BitsB;
    }
}

template <class Intf>
static __aicore__ inline void UpdateLoadToB2ParamsKForKernelSplit(Intf* self, uint32_t kPos)
{
    uint32_t kRepeat = self->ctx.tiling_->baseK / self->ctx.splitHkWkC0List_[self->ctx.splitIndex_]; // cout1
    self->ctx.startAddrOffset_ = self->ctx.hkWk_ * kPos * kRepeat;                                   // 跳过多少个cout
    self->ctx.load2dv2_.kStep = self->ctx.splitWkList_[self->ctx.splitIndex_]; // 当前kernel拆分场景只可以为1或者2
    self->ctx.load2dv2_.mStartPosition =
        ((self->ctx.splitHkList_[self->ctx.splitIndex_] - 1) * self->ctx.tiling_->strideH * self->ctx.tiling_->wk +
         (self->ctx.splitWkList_[self->ctx.splitIndex_] - 1) * self->ctx.tiling_->strideW +
         (self->ctx.splitHIndex_ * self->ctx.tiling_->wk + self->ctx.splitWIndex_)) *
        self->ctx.blockBaseN_;
}

template <class Intf>
static __aicore__ inline void LoadToB2NoTransposeNoReverse(
    Intf* self, const LocalTensor<typename Intf::SrcBT>& l1B1Matrix, uint32_t srcB1Offset, uint32_t kPos,
    LocalTensor<typename Intf::SrcBT>& l0b)
{
    self->ctx.load2dv2_.mStartPosition = 0;
    self->ctx.load2dv2_.kStartPosition = 0;
    self->ctx.load2dv2_.kStep = 1;
    self->ctx.load2dv2_.mStep = self->ctx.blockBaseN_ * DivCeilC0<Intf>(self, self->ctx.baseUseK_);
    srcB1Offset = kPos * AlignUp16(self->ctx.baseUseN_) * self->ctx.tiling_->baseK;
    LoadData(l0b, l1B1Matrix[srcB1Offset], self->ctx.load2dv2_);
}

template <class Intf>
static __aicore__ inline void LoadToB2ReverseOnlyCommon(
    Intf* self, const LocalTensor<typename Intf::SrcBT>& l1B1Matrix, uint32_t blockSize, uint32_t srcB1Offset,
    uint32_t dstB2Offset, const uint32_t kPos, LocalTensor<typename Intf::SrcBT>& l0b)
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
    uint32_t kEndPos = kStartPos + ((self->ctx.baseUseK_ + self->ctx.tiling_->c0 - 1) >> self->ctx.tiling_->c0BitsB);
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
static __aicore__ inline void LoadToB2ForKernelSplitB1FullLoad(
    Intf* self, const LocalTensor<typename Intf::SrcBT>& l1B1Matrix, uint32_t blockSize, uint32_t srcB1Offset,
    uint32_t dstB2Offset, LocalTensor<typename Intf::SrcBT>& l0b)
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
                           static_cast<uint64_t>(self->ctx.tiling_->strideH) *
                           static_cast<uint64_t>(self->ctx.blockBaseN_);
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
static __aicore__ inline void LoadToB2ReverseOnly(
    Intf* self, const LocalTensor<typename Intf::SrcBT>& l1B1Matrix, uint32_t blockSize, uint32_t srcB1Offset,
    uint32_t dstB2Offset, const uint32_t kPos, LocalTensor<typename Intf::SrcBT>& l0b)
{
    if constexpr (Intf::conv3dConfig.kernelSplitMode == TPL_SPLIT_KERNEL_HW) {
        if (self->ctx.kSCoutFullLoad_) { // B1全载时，在load2b2阶段完成子kernel重排
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
static __aicore__ inline void LoadToB2(
    Intf* self, const LocalTensor<typename Intf::SrcBT>& l1B1Matrix, uint32_t kPos,
    LocalTensor<typename Intf::SrcBT>& l0b)
{
    uint32_t blockSize = self->ctx.tiling_->c0 << 4;
    uint32_t srcB1Offset = 0;
    uint32_t dstB2Offset = 0;
    if constexpr (Intf::conv3dConfig.loadB2Condition == TPL_TRANSPOSE_ONLY) {
        uint32_t kBlockC0Num = kPos * self->ctx.tiling_->baseK >> self->ctx.tiling_->c0BitsB; // 转置逆序
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
static __aicore__ inline void LoadToA2(
    Intf* self, const LocalTensor<typename Intf::SrcAT>& l1A1Matrix, LocalTensor<typename Intf::SrcAT>& l0a)
{
#if defined(ASC_DEVKIT_VERSION_NUM) && (ASC_DEVKIT_VERSION_NUM >= 90000000)
    LoadDataWithStride(l0a, l1A1Matrix, self->ctx.load3d_);
#else
    LoadData(l0a, l1A1Matrix, self->ctx.load3d_);
#endif
}

} // namespace Convolution3DBackpropFunc

#endif
