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
 * \file conv_bp_input_sub_func_arch51.h
 * \brief
 */

#ifndef CONV3D_BP_INPUT_SUB_FUNC_ARCH51_H
#define CONV3D_BP_INPUT_SUB_FUNC_ARCH51_H

#include "conv_bp_input_sub_func_utils_arch51.h"
#include "conv_bp_input_sub_func_index_calc_arch51.h"
#include "conv_bp_input_sub_func_load_gm_to_l1_arch51.h"
#include "conv_bp_sub_func_load_gm_to_l1a_arch51.h"
#include "conv_bp_input_sub_func_load_l1_to_l0_arch51.h"
#include "../conv3d_bp_sub_func_arch51.h"
#include "../conv3d_bp_store_arch51.h"
#include "../../../inc/macro.h"

using AscendC::DivCeil;
using AscendC::GlobalTensor;
using AscendC::LocalTensor;

namespace Convolution3DBackpropFunc {

template <class Intf>
__aicore__ inline void FullLoadBias(Intf* self)
{
    // GM -> L1
    LocalTensor<typename Intf::BiasT> useBiasL1 = self->ctx.biasL1Que_.template AllocTensor<typename Intf::BiasT>();
    uint16_t blockBytes = self->ctx.singleShapeCin_ * sizeof(typename Intf::BiasT);
    // 如果搬运的字节数在pad补齐后，没有64字节对齐，则有可能出现脏数据污染BT，需要主动清零L1BiasBuf
    if (blockBytes == 0 || ((blockBytes - 1) / ONE_BLK_SIZE) % 2 == 0) {
        InitZeroValue<Intf, typename Intf::BiasT>(self, useBiasL1);
    }
    DataCopyParams dataCopyParams(1, blockBytes, 0, 0);
    uint8_t rightPadding = DivCeil(blockBytes, ONE_BLK_SIZE) * ONE_BLK_SIZE / sizeof(typename Intf::BiasT) -
                           self->ctx.singleShapeCin_;
    DataCopyPadParams padParams(true, 0, rightPadding, 0);
    DataCopyPad<typename Intf::BiasT>(useBiasL1, self->ctx.biasGlobal_, dataCopyParams, padParams);
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
    uint32_t btCinSize = self->ctx.singleShapeCin_ < self->ctx.baseUseN_ ? self->ctx.singleShapeCin_ :
                                                                           self->ctx.baseUseN_;
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
    DataCopyPad<typename Intf::BiasT>(useBiasL1, self->ctx.biasGlobal_[biasOffset], dataCopyParams, padParams);
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
    uint16_t blockLen = self->ctx.singleShapeCin_ * sizeof(typename Intf::ScaleT0);
    DataCopyParams dataCopyParams(1, blockLen, 0, 0);
    // 4 is B64 data num per block, currently scale is always uint64
    uint8_t rightPadding = DivCeil(self->ctx.singleShapeCin_, 4) * 4 - self->ctx.singleShapeCin_;
    DataCopyPadParams padParams(true, 0, rightPadding, 0);
    if constexpr (Intf::Config::fType::format != Convolution3DBackprop::CubeFormat::UNSUPPORT) {
        if (self->ctx.tiling_->quantMode0 == static_cast<uint8_t>(Convolution3DBackprop::QuantMode::VECTOR_QUANT)) {
            LocalTensor<typename Intf::ScaleT0> useScaleL1 = self->ctx.scale0L1Que_
                                                                 .template AllocTensor<typename Intf::ScaleT0>();
            DataCopyPad<typename Intf::ScaleT0>(useScaleL1, self->ctx.scale0Global_, dataCopyParams, padParams);
            self->ctx.scale0L1Que_.EnQue(useScaleL1);
            self->ctx.scale0L1Buf_ = self->ctx.scale0L1Que_.template DeQue<typename Intf::ScaleT0>();
        }
    }
}

} // namespace Convolution3DBackpropFunc

#endif
