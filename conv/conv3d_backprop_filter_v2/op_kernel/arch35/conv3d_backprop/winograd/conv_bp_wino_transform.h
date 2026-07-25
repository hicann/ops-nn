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
 * \file conv_bp_wino_transform.h
 * \brief
 */

#ifndef CONV_BP_WINO_TRANSFORM_H
#define CONV_BP_WINO_TRANSFORM_H

#include "conv_bp_wino_util.h"

// Transpose5HD做转置时按照16*16为最小单位的,所以搬运时hw轴要统一按照16元素对齐
static constexpr uint32_t HW_SRC_ALIGNED_16 = 16;
// tileBuf在满足tile空间的大小下需要pad 1列让宽变成奇数,防止行列变换时跨行读列时
// 一整列都在少数bank产生bank冲突
static constexpr uint8_t TILE_BUF_BANK_CONFLICT_PADDING = 1;

struct CSlice {
    uint32_t idx;
    uint32_t length;
    uint16_t c1;

    template <typename T>
    __aicore__ inline uint32_t C1Idx() const
    {
        return idx / C0<T>();
    }
};

struct TileBox {
    HWBox tile;
    HWBox src;
    HWPad pad;
    CSlice c;
};

namespace WinoTransformDetail {
constexpr inline uint32_t __aicore__ CalColUnfoldBufWidth(uint32_t th)
{
    // 要补一个pad到奇数
    return TileUnfoldSize(th) | TILE_BUF_BANK_CONFLICT_PADDING;
}

constexpr inline uint32_t __aicore__ Cal16TileHWBufWidth(uint32_t tileHW)
{
    // 补一个pad到奇数
    return tileHW | TILE_BUF_BANK_CONFLICT_PADDING;
}

template <typename T>
constexpr inline __aicore__ uint32_t GetTransformBufSizeC0(uint32_t tileHW)
{
    return F23_TRANSFORM_TILE_ELEMENTS_16 * Cal16TileHWBufWidth(tileHW) * C0<T>();
}

template <typename TransformConfig>
constexpr __aicore__ static inline uint32_t GetInputBufSizeC0()
{
    constexpr uint32_t STRIDE = TransformConfig::STRIDE;
    constexpr uint32_t WINDOW_SIZE = TransformConfig::WINDOW_SIZE;
    using SlideWin = SlideWindows<STRIDE, WINDOW_SIZE>;
    using TilingConfigT = typename TransformConfig::TilingT;
    using T = typename TransformConfig::T;

    constexpr uint32_t srcHW = SlideWin::Tiles2SrcLength(BlockConfig::SingleShapeTileH<TilingConfigT>()) *
                               SlideWin::Tiles2SrcLength(BlockConfig::SingleShapeTileW<TilingConfigT>());
    return srcHW * C0<T>();
}

template <typename TransformConfig>
constexpr inline __aicore__ uint32_t GetTransformBufSize()
{
    using TilingConfigT = typename TransformConfig::TilingT;
    using T = typename TransformConfig::T;
    constexpr uint32_t tileHW = BlockConfig::SingleShapeTileH<TilingConfigT>() *
                                BlockConfig::SingleShapeTileW<TilingConfigT>();
    return GetTransformBufSizeC0<T>(tileHW) * BlockConfig::SingleTransformC1<TilingConfigT>();
}

template <typename TransformConfig>
constexpr __aicore__ static inline uint32_t GetInputBufSize()
{
    using TilingConfigT = typename TransformConfig::TilingT;
    return GetInputBufSizeC0<TransformConfig>() * BlockConfig::SingleTransformC1<TilingConfigT>();
}

template <typename TransformConfig>
constexpr __aicore__ static inline uint32_t GetTmpBufLength()
{
    constexpr uint32_t STRIDE = TransformConfig::STRIDE;
    constexpr uint32_t WINDOW_SIZE = TransformConfig::WINDOW_SIZE;
    using TilingConfigT = typename TransformConfig::TilingT;
    using T = typename TransformConfig::T;

    constexpr uint32_t tileH = BlockConfig::SingleShapeTileH<TilingConfigT>();
    constexpr uint32_t tileW = BlockConfig::SingleShapeTileW<TilingConfigT>();
    constexpr uint32_t srcW = SlideWindows<STRIDE, WINDOW_SIZE>::Tiles2SrcLength(tileW);
    return srcW * CalColUnfoldBufWidth(tileH) * C0<T>();
}
} // namespace WinoTransformDetail

namespace WinoTransformDetail {
template <typename T, typename Impl>
struct UnfoldIntf {
    using UnfoldColParamsT = typename Impl::UnfoldColParamsT;
    using UnfoldRowParamsT = typename Impl::UnfoldRowParamsT;

    static __aicore__ AscendC::Std::tuple<UnfoldColParamsT, UnfoldRowParamsT> InitUnfoldParams(const TileBox& box)
    {
        return Impl::InitUnfoldParams(box);
    }

    template <bool isTailTile>
    static __simd_callee__ inline void UnfoldColsVf(__ubuf__ T* unfoldColBuf, __ubuf__ T* srcBuf,
                                                    const UnfoldColParamsT& params)
    {
        Impl::template UnfoldColsVf<isTailTile>(unfoldColBuf, srcBuf, params);
    }

    static __simd_callee__ inline void UnfoldRowsVf(__ubuf__ T* outBuf, __ubuf__ T* srcBuf,
                                                    const UnfoldRowParamsT& params)
    {
        Impl::UnfoldRowsVf(outBuf, srcBuf, params);
    }
};
} // namespace WinoTransformDetail

template <typename Type, uint32_t STRIDE_VAL, uint32_t WINDOWS_SIZE_VAL, typename UnfoldImplType, typename TilingType>
struct TransformConfig {
    using T = Type;
    using UnfoldImpl = UnfoldImplType;
    using TilingT = TilingType;

    static constexpr uint32_t STRIDE = STRIDE_VAL;
    static constexpr uint32_t WINDOW_SIZE = WINDOWS_SIZE_VAL;
};

template <typename Config>
class WinoTransformer {
public:
    static constexpr uint32_t STRIDE = Config::STRIDE;
    static constexpr uint32_t WINDOW_SIZE = Config::WINDOW_SIZE;
    using T = typename Config::T;
    using TilingConfigT = typename Config::TilingT;
    using SlideWin = SlideWindows<STRIDE, WINDOW_SIZE>;
    using UnfoldPolicy = WinoTransformDetail::UnfoldIntf<T, typename Config::UnfoldImpl>;

    __aicore__ inline WinoTransformer(__gm__ T* in5HD, const uint32_t srcH, const uint32_t srcW, const uint32_t srcC,
                                      const uint16_t padH, const uint16_t padW)
        : srcH_(srcH), srcW_(srcW), srcC_(srcC), padH_(padH), padW_(padW)
    {
        gm_.SetGlobalBuffer(in5HD);
    }

    __aicore__ inline TileBox CalculateSrcBox(const HWBox& tile, uint32_t cIdx, uint32_t cLength) const
    {
        TileBox box = {tile, {}, {}, {}};
        SlideWin::CalculateSrcBox(box.tile, srcH_, srcW_, padH_, padW_, box.src, box.pad);
        box.c.idx = cIdx;
        box.c.length = cLength;
        box.c.c1 = Ops::Base::CeilDiv(cLength, C0<T>());
        return box;
    }

    __aicore__ inline uint32_t SrcH() const { return srcH_; }

    __aicore__ inline uint32_t SrcW() const { return srcW_; }

    __aicore__ inline uint32_t SrcC() const { return srcC_; }

    __aicore__ inline void CopyIn(const AscendC::LocalTensor<T>& srcBuf, const TileBox& box,
                                  const uint32_t batchIdx) const
    {
        const HWBox& src = box.src;
        if (unlikely(src.elements == 0)) {
            return;
        }

        uint32_t srcC1 = Ops::Base::CeilDiv(srcC_, C0<T>());
        uint64_t srcWC0 = srcW_ * C0<T>();
        uint64_t srcHWC0 = srcH_ * srcWC0;
        // nc1hwc0搬入
        uint64_t gmOffset = static_cast<uint64_t>(batchIdx) * srcC1 * srcHWC0 +
                            static_cast<uint64_t>(box.c.C1Idx<T>()) * srcHWC0 +
                            static_cast<uint64_t>(src.hIdx) * srcWC0 + static_cast<uint64_t>(src.wIdx) * C0<T>();

        if constexpr (BlockConfig::SingleTransformC1<TilingConfigT>() > 1) {
            AscendC::LoopModeParams loop;
            loop.loop1Size = box.c.c1;
            loop.loop1SrcStride = srcHWC0 * sizeof(T);
            loop.loop1DstStride = WinoTransformDetail::GetInputBufSizeC0<Config>() * sizeof(T);
            loop.loop2Size = 1;
            loop.loop2SrcStride = 0;
            loop.loop2DstStride = 0;
            SetLoopModePara(loop, AscendC::DataCopyMVType::OUT_TO_UB);
        }

        uint32_t srcFullLenW = SlideWin::Tiles2SrcLength(box.tile.wLength);
        AscendC::DataCopyParams params;
        params.blockCount = src.hLength;
        params.blockLen = src.wLength;
        params.srcGap = srcW_ - src.wLength;
        params.dstGap = srcFullLenW - src.wLength;
        // 留出位置给pad补0
        uint32_t hPadOffset = (box.pad.hTop * srcFullLenW + box.pad.wLeft) * C0<T>();
        AscendC::DataCopy(srcBuf[hPadOffset], gm_[gmOffset], params);

        if constexpr (BlockConfig::SingleTransformC1<TilingConfigT>() > 1) {
            AscendC::ResetLoopModePara(AscendC::DataCopyMVType::OUT_TO_UB);
        }
    }

    __aicore__ inline void Compute(AscendC::LocalTensor<T>& srcBuf, AscendC::LocalTensor<T>& outBuf,
                                   AscendC::LocalTensor<T>& tmpBuf, const TileBox& box) const
    {
        constexpr uint32_t srcBufSizeC0 = WinoTransformDetail::GetInputBufSizeC0<Config>();
        uint32_t outBufSizeC0 = WinoTransformDetail::GetTransformBufSizeC0<T>(box.tile.elements);
        const HWBox& src = box.src;

        if (unlikely(src.elements == 0)) {
            // 整个tile都由padding区域产生,不做计算直接置0,
            AscendC::Duplicate(outBuf, static_cast<T>(0), outBufSizeC0 * box.c.c1);
            return;
        }

        const auto params = UnfoldPolicy::InitUnfoldParams(box);
        const typename UnfoldPolicy::UnfoldColParamsT& ucp = AscendC::Std::get<0>(params);
        const typename UnfoldPolicy::UnfoldRowParamsT& urp = AscendC::Std::get<1>(params);

        __ubuf__ T* tmpBufAddr = reinterpret_cast<__ubuf__ T*>(tmpBuf.GetPhyAddr());
        __ubuf__ T* srcBufAddr = reinterpret_cast<__ubuf__ T*>(srcBuf.GetPhyAddr());
        __ubuf__ T* outBufAddr = reinterpret_cast<__ubuf__ T*>(outBuf.GetPhyAddr());

        // 当前需要优化的点主要集中在列变换，列变换是不是可以不管尾块统一按标准块处理？
        const bool isTail = box.tile.wLength < BlockConfig::SingleShapeTileW<TilingConfigT>() ||
                            box.tile.hLength < BlockConfig::SingleShapeTileH<TilingConfigT>();

        uint16_t c1Len = box.c.c1;
        if constexpr (BlockConfig::SingleTransformC1<TilingConfigT>() == 1) {
            c1Len = 1;
        }
        for (uint16_t c1Idx = 0; c1Idx < c1Len; c1Idx++) {
            if (isTail) {
                UnfoldVf<true>(outBufAddr, tmpBufAddr, srcBufAddr, ucp, urp, box.pad, src.hLength, src.wLength);
            } else {
                UnfoldVf<false>(outBufAddr, tmpBufAddr, srcBufAddr, ucp, urp, box.pad, src.hLength, src.wLength);
            }
            outBufAddr += outBufSizeC0;
            srcBufAddr += srcBufSizeC0;
        }
    }

    __aicore__ inline void SetNK1C1K0C0CopyParams(NK1C1K0C0::CopyK0Params& copyParams, const TileBox& box) const
    {
        copyParams.tiles = box.tile.elements;
        copyParams.srcBufWidthBlockStride = WinoTransformDetail::Cal16TileHWBufWidth(box.tile.elements);
        copyParams.c1Idx = box.c.C1Idx<T>();
        copyParams.c1Length = box.c.c1;
    }

private:
    template <bool IsTailTile>
    __simd_vf__ static inline void UnfoldVf(__ubuf__ T* outBuf, __ubuf__ T* colUnfoldBuf, __ubuf__ T* srcBuf,
                                            const typename UnfoldPolicy::UnfoldColParamsT ucp,
                                            const typename UnfoldPolicy::UnfoldRowParamsT urp, HWPad pad, uint16_t srcH,
                                            uint16_t srcW)
    {
        Padding(srcBuf, pad, srcH, srcW);

        AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();

        UnfoldPolicy::template UnfoldColsVf<IsTailTile>(colUnfoldBuf, srcBuf, ucp);

        AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();

        UnfoldPolicy::UnfoldRowsVf(outBuf, colUnfoldBuf, urp);
    }

    __simd_callee__ static inline void Padding(__ubuf__ T* srcBuf, const HWPad& pad, uint16_t srcH, uint16_t srcW)
    {
        using namespace Reg;
        RegTensor<T> paddingValue;
        Duplicate(paddingValue, 0);

        const uint16_t padHTop = pad.hTop;
        const uint16_t padHButton = pad.hBottom;
        const uint16_t padWLeft = pad.wLeft;
        const uint16_t padWRight = pad.wRight;

        const uint16_t wBlocks = srcW + padWLeft + padWRight;
        const uint32_t wElements = wBlocks * C0<T>();

        const uint32_t hTopElements = wElements * padHTop;
        const uint16_t hTopRepeatTimes = CeilDivision(hTopElements, VL<T>());

        __ubuf__ T* src = srcBuf;
        uint32_t hTopMaskValue = hTopElements;
        for (uint16_t i = 0; i < hTopRepeatTimes; i++) {
            MaskReg mask = Reg::UpdateMask<T>(hTopMaskValue);
            StoreAlign<T, PostLiteral::POST_MODE_UPDATE>(src, paddingValue, VL<T>(), mask);
        }

        const uint32_t hBtnElements = wElements * padHButton;
        const uint16_t hBtnRepeatTimes = CeilDivision(hBtnElements, VL<T>());

        src = srcBuf + (padHTop + srcH) * wElements;
        uint32_t hBtnMaskValue = hBtnElements;
        for (uint16_t i = 0; i < hBtnRepeatTimes; i++) {
            MaskReg mask = Reg::UpdateMask<T>(hBtnMaskValue);
            StoreAlign<T, PostLiteral::POST_MODE_UPDATE>(src, paddingValue, VL<T>(), mask);
        }

        const uint32_t hElements = srcH * C0<T>();
        const uint16_t hRepeatTimes = CeilDivision(hElements, VL<T>());
        const uint16_t wPadStride = (VL<T>() / C0<T>()) * wBlocks;

        src = srcBuf + padHTop * wElements;
        for (uint16_t i = 0; i < padWLeft; i++) {
            uint32_t maskValue = hElements;
            __ubuf__ T* src0 = src + C0<T>() * i;
            for (uint16_t h = 0; h < hRepeatTimes; h++) {
                MaskReg mask = Reg::UpdateMask<T>(maskValue);
                StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(src0, paddingValue, wBlocks,
                                                                                            wPadStride, mask);
            }
        }

        src = srcBuf + padHTop * wElements + (srcW + padWLeft) * C0<T>();
        for (uint16_t i = 0; i < padWRight; i++) {
            uint32_t maskValue = hElements;
            __ubuf__ T* src0 = src + C0<T>() * i;
            for (uint16_t h = 0; h < hRepeatTimes; h++) {
                MaskReg mask = Reg::UpdateMask<T>(maskValue);
                StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(src0, paddingValue, wBlocks,
                                                                                            wPadStride, mask);
            }
        }
    }

    AscendC::GlobalTensor<T> gm_;
    const uint32_t srcH_;
    const uint32_t srcW_;
    const uint32_t srcC_;
    const uint16_t padH_;
    const uint16_t padW_;
};

#endif // CONV_BP_WINO_TRANSFORM_H
