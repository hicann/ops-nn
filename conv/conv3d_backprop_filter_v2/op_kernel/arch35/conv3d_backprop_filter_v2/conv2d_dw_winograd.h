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
 * \file conv2d_dw_winograd.h
 * \brief
 */

#ifndef CONV2D_BACKPROP_FILTER_WINOGRAD_H
#define CONV2D_BACKPROP_FILTER_WINOGRAD_H

#include "conv3d_backprop_filter_v2_tiling_data.h"
#include "../conv3d_backprop/winograd/conv_bp_wino.h"
#include "../conv3d_backprop/winograd/conv_bp_wino_transdata.h"

using namespace AscendC;

template <typename SrcT, typename DstT, uint32_t WinoTilingFlag, bool WinoResidentFlag>
class Conv2dDwWinograd {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR dedy, GM_ADDR y, GM_ADDR workspace,
                                const conv_bp_v2_kernel::Conv3DBackpropFilterV2TilingData* tilingData)
    {
        x_ = reinterpret_cast<__gm__ SrcT*>(x);
        dy_ = reinterpret_cast<__gm__ SrcT*>(dedy);
        workspace_ = reinterpret_cast<__gm__ SrcT*>(workspace);
        y_ = reinterpret_cast<__gm__ DstT*>(y);

        batch_ = tilingData->dwTiling.batch;
        cout_ = tilingData->dwTiling.cout;
        cin_ = tilingData->dwTiling.cin;
        padH_ = tilingData->dwTiling.padUp;
        padW_ = tilingData->dwTiling.padLeft;
        fmapH_ = tilingData->dwTiling.hi;
        fmapW_ = tilingData->dwTiling.wi;
        dyH_ = tilingData->dwTiling.ho;
        dyW_ = tilingData->dwTiling.wo;
        hf32_ = tilingData->dwTiling.hf32Flag;
    }

    __aicore__ inline void Process()
    {
        uint32_t cin1 = Ops::Base::CeilDiv(cin_, C0<SrcT>());
        uint32_t cout1 = Ops::Base::CeilDiv(cout_, C0<SrcT>());

        __gm__ SrcT* transX = workspace_;
        __gm__ SrcT* transDy = transX + static_cast<uint64_t>(batch_) * cin1 * fmapH_ * fmapW_ * C0<SrcT>();
        __gm__ SrcT* nk1c1k0c0 = transDy + static_cast<uint64_t>(batch_) * cout1 * dyH_ * dyW_ * C0<SrcT>();

        WinoPreTransData<SrcT> transData;
        bool disableL2 = ShouldDisableTransDataL2();
        transData.Init();
        transData.TransData2NC1HWC0(x_, transX, batch_, cin_, fmapH_, fmapW_, disableL2);
        transData.TransData2NC1HWC0(dy_, transDy, batch_, cout_, dyH_, dyW_, disableL2);
        transData.End();

        using TilingT = decltype(BuildTilingType());
        WinoFmapFwdTransformer<SrcT, TilingT> fmapFwd(transX, fmapH_, fmapW_, cin_, padH_, padW_);
        WinoDyFwdTransformer<SrcT, TilingT> dyFwd(transDy, dyH_, dyW_, cout_, 0, 0);
        WinoMMAD<SrcT, TilingT> winoMmad(hf32_);

        uint32_t tileH = WinoDyFwdTransformer<SrcT, TilingT>::SlideWin::SrcLength2Tiles(dyH_);
        uint32_t tileW = WinoDyFwdTransformer<SrcT, TilingT>::SlideWin::SrcLength2Tiles(dyW_);

        NK1C1K0C0::Shape<SrcT> nk1c1k0c0Shape = NK1C1K0C0::Shape<SrcT>::template Create<TilingT>(
            BlockConfig::ResidentTarget<TilingT>() == BlockConfig::InputTensor::FMAP ? cin_ : cout_, tileH, tileW);

        __gm__ float* tailGm = reinterpret_cast<__gm__ float*>(nk1c1k0c0 + static_cast<uint64_t>(batch_) *
                                                                               nk1c1k0c0Shape.c1 * nk1c1k0c0Shape.c0 *
                                                                               nk1c1k0c0Shape.k0 * nk1c1k0c0Shape.k1);

        ConvBackpropFilterWinograd<SrcT, DstT, TilingT> winograd(fmapFwd, dyFwd, nk1c1k0c0, nk1c1k0c0Shape, y_, tailGm,
                                                                 winoMmad, tileH, tileW, batch_);

        winograd.Init();
        winograd.IterateAll();
        winograd.End();
    }

private:
    __aicore__ inline bool ShouldDisableTransDataL2() const
    {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
        constexpr uint32_t L2CacheBytes = 128 * 1024 * 1024;
#else
        constexpr uint32_t L2CacheBytes = 112 * 1024 * 1024;
#endif

        constexpr uint32_t L2CacheLimit = L2CacheBytes * 0.85f;
        uint64_t inputBytes = batch_ * sizeof(SrcT) *
                              (static_cast<uint64_t>(cin_) * fmapH_ * fmapW_ +
                               static_cast<uint64_t>(cout_) * dyH_ * dyW_);
        // 简单实现的判断，没太多考虑，不一定真的有价值
        // 当输出+输入的两倍(原始数据+转置数据) > L2cache的0.85倍(冗余一些，可能有其他的东西占用)就关掉原始数据的L2
        // 有问题在调
        uint64_t outputBytes = static_cast<uint64_t>(cin_) * cout_ * 3 * 3 * sizeof(DstT);
        constexpr uint32_t INPUT_DATA_COPIES = 2;
        return (outputBytes + inputBytes * INPUT_DATA_COPIES) > L2CacheLimit;
    }

    struct SingleShapeTile {
        uint16_t H, W;

        constexpr static __aicore__ inline SingleShapeTile Get()
        {
            // TileHTileW组合:
            // 1.fp16/bf16: H8W8  fp32:H4W8
            // 2.fp16/bf16: H4W16 fp32:H2W16
            constexpr bool isB32 = Std::is_same_v<SrcT, float>;
            if constexpr (WinoTilingFlag == TPL_WINOGRAD_SINGLE_SHAPE_TILE_1) {
                if constexpr (isB32) {
                    return {4, 8};
                } else {
                    return {8, 8};
                }
            } else if constexpr (WinoTilingFlag == TPL_WINOGRAD_SINGLE_SHAPE_TILE_2) {
                if constexpr (isB32) {
                    return {2, 16};
                } else {
                    return {4, 16};
                }
            }
        }
    };

    static __aicore__ inline constexpr auto BuildTilingType()
    {
        constexpr uint32_t singleShapeCout = 64;
        constexpr uint32_t singleShapeCin = 64;
        constexpr uint32_t singleShapeTransformC1 = 16 / C0<SrcT>();
        constexpr uint32_t singleShapeResidentC = 32;
        constexpr uint32_t fwdBufCnt = 4;
        constexpr uint32_t invTransBufCnt = 4;
        constexpr uint32_t invTransCout = 8;
        constexpr SingleShapeTile singleShapeTile = SingleShapeTile::Get();
        constexpr BlockConfig::InputTensor ResidentTensor = WinoResidentFlag == TPL_WINOGRAD_RESIDENT_FMAP ?
                                                                BlockConfig::InputTensor::FMAP :
                                                                BlockConfig::InputTensor::DY;

        return BlockConfig::Tiling<singleShapeCout, singleShapeCin, singleShapeTransformC1, singleShapeTile.H,
                                   singleShapeTile.W, fwdBufCnt, singleShapeResidentC, ResidentTensor, invTransBufCnt,
                                   invTransCout>{};
    }

    __gm__ SrcT* x_ = nullptr;
    __gm__ SrcT* dy_ = nullptr;
    __gm__ SrcT* workspace_ = nullptr;
    __gm__ DstT* y_ = nullptr;

    uint32_t batch_ = 0;
    uint32_t cout_ = 0;
    uint32_t cin_ = 0;
    uint32_t padH_ = 0;
    uint32_t padW_ = 0;
    uint32_t fmapH_ = 0;
    uint32_t fmapW_ = 0;
    uint32_t dyH_ = 0;
    uint32_t dyW_ = 0;
    bool hf32_ = false;
};

#endif
