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
 * \file conv3d_bp_impl_base.h
 * \brief
 */

#ifndef CONV3D_BP_IMPL_ADVANCE_H
#define CONV3D_BP_IMPL_ADVANCE_H

#include "conv3d_bp_config_base.h"
#include "conv3d_bp_func.h"
#include "conv3d_bp_util.h"
#include "kernel_operator.h"
#include "kernel_utils.h"
#include "../conv3d_backprop_input_v2/conv3d_backprop_input_v2_tiling_data.h"

using AscendC::TQue;
using AscendC::TPipe;
using AscendC::MmadParams;
using AscendC::LoadData2DParams;
using AscendC::LoadData3DParamsV2;
#if defined(__DAV_C310__) || defined(__DAV_310R6__)
using AscendC::LoadData2DParamsV2;
#endif

namespace Convolution3DBackprop {
constexpr uint8_t GROUP_NDDMA_DIM_NUM = 4;
constexpr uint8_t SUB_KERNEL_NUM = 4;

template <typename Intf, class Config_>
struct ConvBpImpl {
public:
    using Config = Config_;

public:
    __aicore__ inline ConvBpImpl() {}

    DECLARE_IMPL(Config_, Convolution3DBackpropFunc, Init, Intf);
    DECLARE_IMPL(Config_, Convolution3DBackpropFunc, SetWeight, Intf);
#if defined(__DAV_310R6__)
    DECLARE_IMPL(Config_, Convolution3DBackpropFunc, SetBias, Intf);
#endif
    DECLARE_IMPL(Config_, Convolution3DBackpropFunc, SetOutBackprop, Intf);
    DECLARE_IMPL(Config_, Convolution3DBackpropFunc, SetKernelSplitParams, Intf);
    DECLARE_IMPL(Config_, Convolution3DBackpropFunc, SetSingleShapeParams, Intf);
    DECLARE_IMPL(Config_, Convolution3DBackpropFunc, SetSingleShape, Intf);
    DECLARE_IMPL(Config_, Convolution3DBackpropFunc, SetStartIdx, Intf);
    DECLARE_IMPL(Config_, Convolution3DBackpropFunc, FreeB1Tensor, Intf);
    DECLARE_SYNC_IMPL(Config_, Convolution3DBackpropFunc, Iterate, Intf);
    DECLARE_SYNC_IMPL(Config_, Convolution3DBackpropFunc, IterateAll, Intf);
    DECLARE_SYNC_IMPL(Config_, Convolution3DBackpropFunc, IterateAllForKernelSplit, Intf);
    DECLARE_SYNC_IMPL(Config_, Convolution3DBackpropFunc, GetTensorC, Intf);
    DECLARE_SYNC_IMPL(Config_, Convolution3DBackpropFunc, VecPreProcess, Intf);
    DECLARE_SYNC_IMPL(Config_, Convolution3DBackpropFunc, VecPostProcess, Intf);
    DECLARE_IMPL(Config_, Convolution3DBackpropFunc, End, Intf);
    struct ContextData : public Config::ContextData {
        __aicore__ inline ContextData(){};
        DEFINE_STUCT_FIELD(TPipe, pipe_);
        DEFINE_STUCT_FIELD(const conv_bp_v2_kernel::TConv3DInputV2Tiling *, tiling_);
        DEFINE_STUCT_FIELD(uint64_t, mIter_);
        DEFINE_STUCT_FIELD(uint32_t, nIter_);
        DEFINE_STUCT_FIELD(uint64_t, kIter_);
        DEFINE_STUCT_FIELD(uint32_t, tailM_);
        DEFINE_STUCT_FIELD(uint32_t, tailN_);
        DEFINE_STUCT_FIELD(uint32_t, tailK_);
        DEFINE_STUCT_TEMPLATE_FIELD(TQue, inQueL1A_, TPosition::A1, 2); // 有Preload, 队列深度2
        DEFINE_STUCT_TEMPLATE_FIELD(TQue, inQueL1B_, TPosition::B1, 2);
        DEFINE_STUCT_TEMPLATE_FIELD(TQue, outQueL0C_, TPosition::CO1, 1); // 一般的队列深度1即可, 也可支持pingpong切换，编译器有额外优化
        DEFINE_STUCT_TEMPLATE_FIELD(TQue, biasL1Que_, TPosition::A1, 1);
        DEFINE_STUCT_TEMPLATE_FIELD(TQue, biasBTQue_, TPosition::C2, 1);
        DEFINE_STUCT_TEMPLATE_FIELD(TBuf, l0aBuf_, TPosition::A2);
        DEFINE_STUCT_TEMPLATE_FIELD(TBuf, l0bBuf_, TPosition::B2);
        DEFINE_STUCT_FIELD(uint32_t, curStepM_);
        DEFINE_STUCT_FIELD(uint32_t, curStepN_);
        DEFINE_STUCT_FIELD(uint32_t, curLoadKbl1_);
        DEFINE_STUCT_FIELD(uint32_t, curNL0Idx_);
        DEFINE_STUCT_FIELD(uint32_t, curNL1Idx_);
        DEFINE_STUCT_FIELD(uint64_t, curML0Idx_);
        DEFINE_STUCT_FIELD(uint64_t, curML1Idx_);
        DEFINE_STUCT_FIELD(uint32_t, curDinIdx_);
        DEFINE_STUCT_FIELD(uint32_t, curDkIdx_);
        DEFINE_STUCT_FIELD(uint32_t, channelSize_);
        DEFINE_STUCT_FIELD(uint32_t, curHoSize_);
        DEFINE_STUCT_FIELD(uint64_t, kIterStepKaTail);
        DEFINE_STUCT_FIELD(uint64_t, kIterStepKbTail);
        DEFINE_STUCT_FIELD(uint32_t, stepKaTail);
        DEFINE_STUCT_FIELD(uint32_t, stepKbTail);
        DEFINE_STUCT_FIELD(int32_t, curHoIdx_);
        DEFINE_STUCT_FIELD(int32_t, curHkIdx_);
        DEFINE_STUCT_FIELD(int32_t, curWkIdx_);
        DEFINE_STUCT_FIELD(int32_t, curWoLeftIdx_);
        DEFINE_STUCT_FIELD(int32_t, curWoRightIdx_);
        DEFINE_STUCT_FIELD(uint32_t, baseB1Offset_);
        DEFINE_STUCT_FIELD(uint32_t, blockBaseN_);
        DEFINE_STUCT_FIELD(uint32_t, baseUseM_);
        DEFINE_STUCT_FIELD(uint32_t, baseUseN_);
        DEFINE_STUCT_FIELD(uint32_t, baseUseK_);
        DEFINE_STUCT_FIELD(uint64_t, hkWk_);
        DEFINE_STUCT_FIELD(uint64_t, singleShapeHWk_);
        DEFINE_STUCT_FIELD(uint64_t, singleShapeM_);
        DEFINE_STUCT_FIELD(uint32_t, singleShapeCin_);
        DEFINE_STUCT_FIELD(uint32_t, singleShapeCout_);
        DEFINE_STUCT_FIELD(uint32_t, curEnlargeCin1_);
        DEFINE_STUCT_FIELD(uint32_t, curEnlargeCout1_);
        DEFINE_STUCT_FIELD(uint32_t, singleShapeDin_);
        DEFINE_STUCT_FIELD(uint32_t, curDinStartIdx_);
        DEFINE_STUCT_FIELD(int32_t, curHoStartIdx_);
        DEFINE_STUCT_FIELD(int32_t, curCinStartIdx_);
        DEFINE_STUCT_FIELD(int32_t, curCoutStartIdx_);
        DEFINE_STUCT_FIELD(int32_t, hoExpand_);
        DEFINE_STUCT_ARRAY_FIELD(uint32_t, splitHkList_, SUB_KERNEL_NUM);
        DEFINE_STUCT_ARRAY_FIELD(uint32_t, splitWkList_, SUB_KERNEL_NUM);
        DEFINE_STUCT_ARRAY_FIELD(uint32_t, splitHkWkList_, SUB_KERNEL_NUM);
        DEFINE_STUCT_ARRAY_FIELD(uint32_t, splitHkWkC0List_, SUB_KERNEL_NUM);
        DEFINE_STUCT_FIELD(uint32_t, splitWi_);
        DEFINE_STUCT_FIELD(uint32_t, splitHIndex_);
        DEFINE_STUCT_FIELD(uint32_t, splitWIndex_);
        DEFINE_STUCT_FIELD(uint32_t, splitHStartIndex_);
        DEFINE_STUCT_FIELD(uint32_t, splitWStartIndex_);
        DEFINE_STUCT_FIELD(uint32_t, rearrangeHIndex_);
        DEFINE_STUCT_FIELD(uint32_t, rearrangeWIndex_);
        DEFINE_STUCT_FIELD(uint32_t, splitIndex_);
        DEFINE_STUCT_ARRAY_FIELD(int32_t, subPadLeftList_, SUB_KERNEL_NUM);
        DEFINE_STUCT_ARRAY_FIELD(int32_t, subPadRightList_, SUB_KERNEL_NUM);
        DEFINE_STUCT_ARRAY_FIELD(int32_t, subPadUpList_, SUB_KERNEL_NUM);
        DEFINE_STUCT_ARRAY_FIELD(int32_t, subPadDownList_, SUB_KERNEL_NUM);
        DEFINE_STUCT_FIELD(uint64_t, curMStartIdx_);
        DEFINE_STUCT_FIELD(uint32_t, groupIterIdx_);
        DEFINE_STUCT_FIELD(uint32_t, curEnlarge);
        DEFINE_STUCT_FIELD(bool, isA1FullLoadFlag_);
        DEFINE_STUCT_FIELD(bool, isB1FullLoadFlag_);
        DEFINE_STUCT_FIELD(bool, isLoadA1_);
        DEFINE_STUCT_FIELD(bool, isLoadB1_);
        DEFINE_STUCT_FIELD(bool, isFreeA1_);
        DEFINE_STUCT_FIELD(bool, isFreeB1_);
        DEFINE_STUCT_FIELD(bool, enableSplitDk_);
        DEFINE_STUCT_FIELD(bool, isLastDk_);
        DEFINE_STUCT_FIELD(bool, needComputeFlag_);
        DEFINE_STUCT_FIELD(uint8_t, isFirstIter_);
        DEFINE_STUCT_FIELD(uint8_t, l0PingPongFlag_);
        DEFINE_STUCT_FIELD(uint8_t, enableL0PingPong_);
        DEFINE_STUCT_FIELD(MmadParams, mmad_);
        DEFINE_STUCT_FIELD(LoadData2DParams, load2d_);
        using NDDMACopyParams = MultiCopyParams<typename Intf::SrcT, GROUP_NDDMA_DIM_NUM>;
        DEFINE_STUCT_FIELD(NDDMACopyParams, groupCopyParams_);

        using LoadData3DParamsV2SrcT = LoadData3DParamsV2<typename Intf::SrcT>;
        DEFINE_STUCT_FIELD(LoadData3DParamsV2SrcT, load3d_);
        using srcLocalTensor = LocalTensor<typename Intf::SrcT>;
        DEFINE_STUCT_FIELD(srcLocalTensor, cacheA1Buf_);
        DEFINE_STUCT_FIELD(srcLocalTensor, cacheB1Buf_);
        using GlobalTnesor = GlobalTensor<typename Intf::SrcT>;
        DEFINE_STUCT_FIELD(GlobalTnesor, outBackPropGlobal_);
        DEFINE_STUCT_FIELD(GlobalTnesor, fmapGlobal_);
        DEFINE_STUCT_FIELD(GlobalTnesor, weightGlobal_);
        DEFINE_STUCT_FIELD(GlobalTensor<typename Intf::BiasT>, biasGlobal_);
        DEFINE_STUCT_FIELD(LocalTensor<typename Intf::BiasT>, biasL1Buf_);
        DEFINE_STUCT_FIELD(LocalTensor<typename Intf::L0cT>, biasBTBuf_);
        DEFINE_STUCT_FIELD(GlobalTensor<float>, l0cOutGm_); // tmp workspace to store result data with fp32
        DEFINE_STUCT_FIELD(GlobalTensor<typename Intf::DstT>, l0cOutWorkspace_); // tmp workspace for kernel split
#if defined(__DAV_C310__) || defined(__DAV_310R6__)
        DEFINE_STUCT_TEMPLATE_FIELD(TBuf, b1UbPing_, TPosition::B1);
        DEFINE_STUCT_TEMPLATE_FIELD(TBuf, b1UbPong_, TPosition::B1);
        DEFINE_STUCT_TEMPLATE_FIELD(TBuf, vecBuf_, TPosition::VECIN);
        DEFINE_STUCT_TEMPLATE_FIELD(TBuf, ndVecBuf_, TPosition::VECIN);
        DEFINE_STUCT_TEMPLATE_FIELD(TBuf, nzVecBuf_, TPosition::VECOUT);
        DEFINE_STUCT_TEMPLATE_FIELD(TBuf, idxVecBuf_, TPosition::VECCALC);
        DEFINE_STUCT_TEMPLATE_FIELD(TBuf, maskVecBuf_, TPosition::VECCALC);
        using ubLocalTensor = LocalTensor<typename Intf::SrcT>;
        DEFINE_STUCT_FIELD(ubLocalTensor, vecInBuf_);
        using ubDstLocalTensor = LocalTensor<typename Intf::DstT>;
        DEFINE_STUCT_FIELD(ubDstLocalTensor, vecOutBuf_);
        DEFINE_STUCT_FIELD(ubLocalTensor, ndVecTensor_);
        DEFINE_STUCT_FIELD(ubLocalTensor, nzVecTensor_);
        DEFINE_STUCT_FIELD(LocalTensor<float>, castVecTensor_);
        DEFINE_STUCT_FIELD(LocalTensor<typename Intf::IndexT>, idxVecTensor_);
        DEFINE_STUCT_FIELD(LocalTensor<uint32_t>, maskVecTensor_);
        DEFINE_STUCT_FIELD(LoadData2DParamsV2, load2dv2_);
        DEFINE_STUCT_FIELD(uint64_t, dkHkWk_);
        DEFINE_STUCT_FIELD(uint64_t, hiWi_);
        DEFINE_STUCT_FIELD(uint64_t, diHiWi_);
        DEFINE_STUCT_FIELD(uint64_t, hoWo_);
        DEFINE_STUCT_FIELD(uint64_t, doHoWo_);
        DEFINE_STUCT_FIELD(uint32_t, dstB2Stride_);
        DEFINE_STUCT_FIELD(uint64_t, startAddrOffset_);
        DEFINE_STUCT_FIELD(uint32_t, headWi_);
        DEFINE_STUCT_FIELD(uint32_t, midHi_);
        DEFINE_STUCT_FIELD(uint32_t, tailWi_);
        DEFINE_STUCT_FIELD(uint64_t, c04LoadToB1IterIdx_);
        DEFINE_STUCT_FIELD(uint32_t, vecBlockN_);
        DEFINE_STUCT_FIELD(int32_t, curBackPropPadUp_);
        DEFINE_STUCT_FIELD(uint32_t, coutChannelSize_);
        DEFINE_STUCT_FIELD(uint64_t, curStepKa_);
        DEFINE_STUCT_FIELD(uint64_t, curStepKb_);
        DEFINE_STUCT_FIELD(uint32_t, kSCoutFullLoad_);
        DEFINE_STUCT_FIELD(uint32_t, kSUseWorkSpace_);
        DEFINE_STUCT_FIELD(uint64_t, subKernelM_);
        DEFINE_STUCT_FIELD(uint32_t, l0aPingPongAddr_);
        DEFINE_STUCT_FIELD(uint32_t, l0bPingPongAddr_);
        DEFINE_STUCT_FIELD(uint32_t, realWoSize_);
#endif
    };
};

}  // namespace Convolution3DBackprop

#endif
