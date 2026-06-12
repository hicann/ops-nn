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
 * \file conv_bp_input_sub_func.h
 * \brief
 */

#ifndef CONV3D_BP_INPUT_SUB_FUNC_ADVANCE_H
#define CONV3D_BP_INPUT_SUB_FUNC_ADVANCE_H

#include "conv_bp_input_sub_func_load_gm_to_l1.h"
#include "conv_bp_input_sub_func_load_l1_to_l0.h"
#include "conv_bp_input_sub_func_store_l0c.h"
#include "conv_bp_input_sub_func_store_l0c_dispatch.h"
#include "conv_bp_sub_func_load_gm_to_l1a.h"

using AscendC::DivCeil;
using AscendC::Fixpipe;
using AscendC::FixpipeParamsC310;
using AscendC::GlobalTensor;
using AscendC::LocalTensor;
using AscendC::SetAtomicAdd;
using AscendC::SetAtomicNone;

namespace Convolution3DBackpropFunc {
template <class Intf>
static __aicore__ inline void InitMmadParams(Intf* self)
{
    self->ctx.mmad_.m = self->ctx.baseUseM_;
    if (self->ctx.curMIdx_ == self->ctx.mIter_ - 1) {
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
static __aicore__ inline void CalcParamsMmad(Intf* self, uint32_t kPos, bool isFirstDk)
{
    self->ctx.mmad_.cmatrixInitVal = (kPos == 0 && isFirstDk);
}

template <class Intf, bool hasBias>
static __aicore__ inline void MmadLocal(
    Intf* self, const LocalTensor<typename Intf::SrcAT>& l0a, const LocalTensor<typename Intf::SrcBT>& l0b,
    LocalTensor<typename Intf::L0cT>& l0c)
{
    // eType is bias Class
    if (hasBias) {
        if (self->ctx.mmad_.cmatrixInitVal && !self->ctx.computeBiasOnce_) {
            // bias 通路，C矩阵初始值通过BT（C2）进行初始化
            self->ctx.mmad_.cmatrixInitVal = 0; // 不初始化，使用bias的值
            self->ctx.mmad_.cmatrixSource = 1;  // 第一次mmad，cmatrix值从BT buffer获取
            uint64_t biasOffset =
                self->ctx.tiling_->isBiasFullLoad ? (self->ctx.curNIdx_ * self->ctx.tiling_->baseN) : 0;
            Mmad(l0c, l0a, l0b, self->ctx.biasBTBuf_[biasOffset], self->ctx.mmad_);
            self->ctx.mmad_.cmatrixSource = 0; // 后续值从l0c读取
            self->ctx.computeBiasOnce_ = true;
        } else {
            Mmad(l0c, l0a, l0b, self->ctx.mmad_);
        }
    } else {
        Mmad(l0c, l0a, l0b, self->ctx.mmad_);
    }
}

template <class Intf>
static __aicore__ inline void UpdateL1KParams(Intf* self, const uint64_t kIdx, uint32_t& curStepKa, uint32_t& curStepKb)
{
    if (unlikely(kIdx == 0)) {
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
static __aicore__ inline void LoadL0Zero(
    Intf* self, const LocalTensor<typename Intf::SrcAT>& l0a, uint32_t baseM,
    const LocalTensor<typename Intf::SrcBT>& l0b, uint32_t baseN)
{
    LocalTensor<typename Intf::SrcAT> dummyL1A(TPosition::A1, 0, 0);
    LocalTensor<typename Intf::SrcBT> dummyL1B(TPosition::B1, 0, 0);
    constexpr uint8_t k0A = GetDataBlockSizeInBytes() / sizeof(typename Intf::SrcAT);
    constexpr uint8_t k0B = GetDataBlockSizeInBytes() / sizeof(typename Intf::SrcBT);
    LoadData3DParamsV2<typename Intf::SrcAT> load3dA;
    LoadData3DParamsV2<typename Intf::SrcBT> load3dB;
    load3dA.filterW = 1;
    load3dA.filterH = 1;

    // 清零10240*255区域
    load3dA.padList[0] = 0;
    load3dA.padList[1] = 0;
    load3dA.padList[2] = 255;
    load3dA.padList[3] = 0;

    load3dA.l1H = 1;
    load3dA.l1W = 32;

    load3dA.channelSize = k0A;
    load3dA.kStartPt = 0;
    load3dA.mStartPt = 0;
    load3dA.kExtension = k0A;

    load3dB.filterW = 1;
    load3dB.filterH = 1;

    // 清零10240*255区域
    load3dB.padList[0] = 0;
    load3dB.padList[1] = 0;
    load3dB.padList[2] = 255;
    load3dB.padList[3] = 0;

    load3dB.l1H = 1;
    load3dB.l1W = 32;

    load3dB.channelSize = k0B;
    load3dB.kStartPt = 0;
    load3dB.mStartPt = 0;
    load3dB.kExtension = k0B;

    load3dA.mExtension = AlignUp(baseM, BLOCK_CUBE);
    load3dB.mExtension = AlignUp(baseN, BLOCK_CUBE);
#if defined(ASC_DEVKIT_VERSION_NUM) && (ASC_DEVKIT_VERSION_NUM >= 90000000)
    LoadDataRepeatParamWithStride repeat = {};
    repeat.repeatTime = 1;
    // 重置repeat参数
    SetLoadDataRepeatWithStride(repeat);

    LoadDataWithStride(l0a, dummyL1A, load3dA);
    LoadDataWithStride(l0b, dummyL1B, load3dB);
#else
    LoadDataRepeatParam repeat = {};
    repeat.repeatTime = 1;
    SetLoadDataRepeat(repeat);

    LoadData(l0a, dummyL1A, load3dA);
    LoadData(l0b, dummyL1B, load3dB);
#endif
}

template <class Intf>
__aicore__ inline void FullLoadBias(Intf* self)
{
    if ASCEND_IS_AIV_SHOULD_RETURN {
        return;
    }
    // GM -> L1
    LocalTensor<typename Intf::BiasT> useBiasL1 = self->ctx.biasL1Que_.template AllocTensor<typename Intf::BiasT>();
    uint16_t blockBytes = self->ctx.singleShapeCin_ * sizeof(typename Intf::BiasT);
    // 如果搬运的字节数在pad补齐后，没有64字节对齐，则有可能出现脏数据污染BT，需要主动清零L1BiasBuf
    if (blockBytes == 0 || ((blockBytes - 1) / ONE_BLK_SIZE) % 2 == 0) {
        InitZeroValue<Intf, typename Intf::BiasT>(self, useBiasL1);
    }
    DataCopyParams dataCopyParams(1, blockBytes, 0, 0);
    uint8_t rightPadding =
        DivCeil(blockBytes, ONE_BLK_SIZE) * ONE_BLK_SIZE / sizeof(typename Intf::BiasT) - self->ctx.singleShapeCin_;
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
__aicore__ inline void LoadBiasToBT(Intf* self)
{
    if ASCEND_IS_AIV_SHOULD_RETURN {
        return;
    }
    uint32_t btCinSize =
        self->ctx.singleShapeCin_ < self->ctx.baseUseN_ ? self->ctx.singleShapeCin_ : self->ctx.baseUseN_;
    // GM -> L1
    LocalTensor<typename Intf::BiasT> useBiasL1 = self->ctx.biasL1Que_.template AllocTensor<typename Intf::BiasT>();
    uint16_t blockBytes = btCinSize * sizeof(typename Intf::BiasT);
    // 如果搬运的字节数在pad补齐后，没有64字节对齐，则有可能出现脏数据污染BT，需要主动清零L1BiasBuf
    if (blockBytes == 0 || ((blockBytes - 1) / ONE_BLK_SIZE) % 2 == 0) {
        InitZeroValue<Intf, typename Intf::BiasT>(self, useBiasL1);
    }
    DataCopyParams dataCopyParams(1, blockBytes, 0, 0);
    uint8_t rightPadding = DivCeil(blockBytes, ONE_BLK_SIZE) * ONE_BLK_SIZE / sizeof(typename Intf::BiasT) - btCinSize;
    DataCopyPadParams padParams(true, 0, rightPadding, 0);
    uint64_t biasOffset = self->ctx.curCinStartIdx_ + self->ctx.curNIdx_ * self->ctx.tiling_->baseN;
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
__aicore__ inline void FullLoadToScaleL1(Intf* self)
{
    if ASCEND_IS_AIV_SHOULD_RETURN {
        return;
    }
    LocalTensor<typename Intf::ScaleT> useScaleL1 = self->ctx.scaleL1Que_.template AllocTensor<typename Intf::ScaleT>();
    uint16_t blockLen = self->ctx.singleShapeCin_ * sizeof(typename Intf::ScaleT);
    DataCopyParams dataCopyParams(1, blockLen, 0, 0);
    // 4 is B64 data num per block, currently scale is always uint64
    uint8_t rightPadding = DivCeil(self->ctx.singleShapeCin_, 4) * 4 - self->ctx.singleShapeCin_;
    DataCopyPadParams padParams(true, 0, rightPadding, 0);
    DataCopyPad<typename Intf::ScaleT>(useScaleL1, self->ctx.scaleGlobal_, dataCopyParams, padParams);
    self->ctx.scaleL1Que_.EnQue(useScaleL1);
    self->ctx.scaleL1Buf_ = self->ctx.scaleL1Que_.template DeQue<typename Intf::ScaleT>();
}

} // namespace Convolution3DBackpropFunc

#endif
