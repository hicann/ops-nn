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
 * \file conv3d_bp_impl_base_arch51.h
 * \brief
 */

#ifndef CONV3D_BP_IMPL_BASE_ARCH51_H
#define CONV3D_BP_IMPL_BASE_ARCH51_H

#include "conv3d_bp_config_arch51.h"
#include "conv3d_bp_func_arch51.h"
#include "conv3d_bp_util_arch51.h"
#include "basic_api/kernel_basic_intf.h"
#include "kernel_common.h"

using AscendC::LoadData2DParamsV2;
using AscendC::LoadData3DParamsV2;
using AscendC::TPipe;
using AscendC::TQue;

namespace Convolution3DBackprop {

template <typename Intf, class Config_>
struct ConvBpImpl {
public:
    using Config = Config_;

public:
    __aicore__ inline ConvBpImpl() {}

    DECLARE_IMPL(Config_, Convolution3DBackpropFunc, Init, Intf);
    DECLARE_IMPL(Config_, Convolution3DBackpropFunc, SetWeight, Intf);
    DECLARE_IMPL(Config_, Convolution3DBackpropFunc, SetBias, Intf);
    DECLARE_IMPL(Config_, Convolution3DBackpropFunc, SetScale, Intf);
    DECLARE_IMPL(Config_, Convolution3DBackpropFunc, SetOutBackprop, Intf);
    DECLARE_IMPL(Config_, Convolution3DBackpropFunc, SetSingleShape, Intf);
    DECLARE_IMPL(Config_, Convolution3DBackpropFunc, SetStartIdx, Intf);
    DECLARE_IMPL(Config_, Convolution3DBackpropFunc, FreeB1Tensor, Intf);
    DECLARE_IMPL(Config_, Convolution3DBackpropFunc, FreeBiasTensor, Intf);
    DECLARE_SYNC_IMPL(Config_, Convolution3DBackpropFunc, Iterate, Intf);
    DECLARE_SYNC_IMPL(Config_, Convolution3DBackpropFunc, IterateAll, Intf);
    DECLARE_SYNC_IMPL(Config_, Convolution3DBackpropFunc, GetTensorC, Intf);
    DECLARE_IMPL(Config_, Convolution3DBackpropFunc, End, Intf);
    struct ContextData : public Config::ContextData {
        __aicore__ inline ContextData(){};
        DEFINE_STUCT_FIELD(TPipe, pipe_);
        DEFINE_STUCT_FIELD(const Conv3DBackpropInputArch35TilingData*, tiling_);
        DEFINE_STUCT_FIELD(uint8_t, curEnableFullLoad_);
        DEFINE_STUCT_FIELD(uint64_t, mIter_);
        DEFINE_STUCT_FIELD(uint32_t, nIter_);
        DEFINE_STUCT_FIELD(uint64_t, kIter_);
        DEFINE_STUCT_FIELD(uint32_t, tailM_);
        DEFINE_STUCT_FIELD(uint32_t, tailN_);
        DEFINE_STUCT_FIELD(uint32_t, tailK_);
        DEFINE_STUCT_TEMPLATE_FIELD(TQue, inQueL1A_, TPosition::A1, 2); // 有Preload, 队列深度2
        DEFINE_STUCT_TEMPLATE_FIELD(TQue, inQueL1B_, TPosition::B1, 2);
        DEFINE_STUCT_TEMPLATE_FIELD(TQue, biasBTQue_, TPosition::C2, 1);
        DEFINE_STUCT_TEMPLATE_FIELD(TBuf, l0bBuf_, TPosition::B2);
        DEFINE_STUCT_FIELD(uint32_t, curLoadKbl1_);
        DEFINE_STUCT_FIELD(uint32_t, curNIdx_);
        DEFINE_STUCT_FIELD(uint64_t, curMIdx_);
        DEFINE_STUCT_FIELD(uint32_t, curDinIdx_);
        DEFINE_STUCT_FIELD(uint32_t, curDkIdx_);
        DEFINE_STUCT_FIELD(uint32_t, curHoSize_);
        DEFINE_STUCT_FIELD(uint64_t, kIterStepKaTail);
        DEFINE_STUCT_FIELD(uint64_t, kIterStepKbTail);
        DEFINE_STUCT_FIELD(uint32_t, stepKaTail);
        DEFINE_STUCT_FIELD(uint32_t, stepKbTail);
        DEFINE_STUCT_FIELD(int32_t, curHoIdx_);
        DEFINE_STUCT_FIELD(int32_t, curHkIdx_);
        DEFINE_STUCT_FIELD(int32_t, curWkIdx_);
        DEFINE_STUCT_FIELD(uint32_t, baseB1Offset_);
        DEFINE_STUCT_FIELD(uint32_t, blockBaseN_);
        DEFINE_STUCT_FIELD(uint32_t, baseUseM_);
        DEFINE_STUCT_FIELD(uint32_t, baseUseN_);
        DEFINE_STUCT_FIELD(uint32_t, baseUseK_);
        DEFINE_STUCT_FIELD(uint64_t, singleShapeHWk_);
        DEFINE_STUCT_FIELD(uint64_t, singleShapeM_);
        DEFINE_STUCT_FIELD(uint32_t, singleShapeCin_);
        DEFINE_STUCT_FIELD(uint32_t, singleShapeCout_);
        DEFINE_STUCT_FIELD(uint32_t, singleShapeDin_);
        DEFINE_STUCT_FIELD(uint32_t, curDinStartIdx_);
        DEFINE_STUCT_FIELD(int32_t, curHoStartIdx_);
        DEFINE_STUCT_FIELD(int32_t, curCinStartIdx_);
        DEFINE_STUCT_FIELD(int32_t, hoExpand_);

        DEFINE_STUCT_FIELD(uint64_t, curMStartIdx_);
        DEFINE_STUCT_FIELD(bool, isA1FullLoadFlag_);
        DEFINE_STUCT_FIELD(bool, isB1FullLoadFlag_);
        DEFINE_STUCT_FIELD(bool, isLoadA1_);
        DEFINE_STUCT_FIELD(bool, isLoadB1_);
        DEFINE_STUCT_FIELD(bool, isFreeA1_);
        DEFINE_STUCT_FIELD(bool, isFreeB1_);
        DEFINE_STUCT_FIELD(bool, needComputeFlag_);
        DEFINE_STUCT_FIELD(bool, hasBias_);
        DEFINE_STUCT_FIELD(bool, computeBiasOnce_);
        DEFINE_STUCT_FIELD(uint8_t, isFirstIter_);
        DEFINE_STUCT_FIELD(uint8_t, l0PingPongFlag_);
        DEFINE_STUCT_FIELD(uint8_t, enableL0PingPong_);

        using LoadData3DParamsV2SrcT = LoadData3DParamsV2<typename Intf::SrcAT>;
        DEFINE_STUCT_FIELD(LoadData3DParamsV2SrcT, load3d_);
        using srcLocalTensorA = LocalTensor<typename Intf::SrcAT>;
        using srcLocalTensorB = LocalTensor<typename Intf::SrcBT>;
        DEFINE_STUCT_FIELD(srcLocalTensorA, cacheA1Buf_);
        DEFINE_STUCT_FIELD(srcLocalTensorB, cacheB1Buf_);
        using GlobalTnesorA = GlobalTensor<typename Intf::SrcAT>;
        using GlobalTnesorB = GlobalTensor<typename Intf::SrcBT>;
        DEFINE_STUCT_FIELD(GlobalTnesorA, outBackPropGlobal_);
        DEFINE_STUCT_FIELD(GlobalTnesorB, weightGlobal_);
        DEFINE_STUCT_FIELD(GlobalTensor<typename Intf::BiasT>, biasGlobal_);
        DEFINE_STUCT_FIELD(LocalTensor<typename Intf::BiasT>, biasL1Buf_);
        DEFINE_STUCT_FIELD(LocalTensor<typename Intf::L0cT>, biasBTBuf_);
        DEFINE_STUCT_FIELD(GlobalTensor<typename Intf::ScaleT0>, scale0Global_);
        DEFINE_STUCT_FIELD(LocalTensor<typename Intf::ScaleT0>, scale0L1Buf_);
        DEFINE_STUCT_TEMPLATE_FIELD(TQue, biasL1Que_, TPosition::B1, 1);
        DEFINE_STUCT_TEMPLATE_FIELD(TQue, scale0L1Que_, TPosition::B1, 1);
        DEFINE_STUCT_TEMPLATE_FIELD(TBuf, l0cBuf_, TPosition::CO1);
        DEFINE_STUCT_FIELD(LocalTensor<typename Intf::DstT>, wholeL1ResTensor_);
        DEFINE_STUCT_FIELD(LocalTensor<typename Intf::L0cT>, wholeCl0Tensor_);
        DEFINE_STUCT_FIELD(uint8_t, l1ResPingPongFlag_);
        DEFINE_STUCT_ARRAY_FIELD(bool, fixmEventSeen_, 2);
        DEFINE_STUCT_FIELD(uint32_t, l0bPingPongAddr_);
        DEFINE_STUCT_FIELD(LoadData2DParamsV2, load2dv2_);
        DEFINE_STUCT_FIELD(uint64_t, curStepKa_);
        DEFINE_STUCT_FIELD(uint64_t, curStepKb_);
        DEFINE_STUCT_FIELD(uint64_t, hiWi_);
        DEFINE_STUCT_FIELD(uint64_t, diHiWi_);
        DEFINE_STUCT_FIELD(uint64_t, hoWo_);
        DEFINE_STUCT_FIELD(uint64_t, doHoWo_);
        DEFINE_STUCT_FIELD(uint32_t, dstB2Stride_);
        DEFINE_STUCT_FIELD(uint32_t, realWoSize_);
        DEFINE_STUCT_FIELD(uint32_t, curKIdx_);
        DEFINE_STUCT_FIELD(ConvFixBitModeParams, convFixParams_);
        DEFINE_STUCT_FIELD(int32_t, scaleL1StagedCin_); // SetDequantScale 的 GM→L1 scale 装载标记
        DEFINE_STUCT_FIELD(int32_t, scaleFbStagedCin_); // LoadScalePPB 的 L1→FB 装载标记
        DEFINE_STUCT_TEMPLATE_FIELD(TBuf, scaleFBTBuf_, TPosition::C2PIPE2GM);
        DEFINE_STUCT_FIELD(LocalTensor<typename Intf::ScaleT0>, scaleFB_);
    };
};

template <typename Intf_, class Config_>
struct Conv3DBpInputImpl : public ConvBpImpl<Intf_, Config_> {
public:
    __aicore__ inline Conv3DBpInputImpl() {}
    struct ContextData : public ConvBpImpl<Intf_, Config_>::ContextData {
        __aicore__ inline ContextData() {}
    };
};

} // namespace Convolution3DBackprop

#endif
