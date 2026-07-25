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
 * \file conv_bp_wino_transform_fmap.h
 * \brief
 */
#ifndef CONV_BP_WINO_TRANSFORM_FMAP_H
#define CONV_BP_WINO_TRANSFORM_FMAP_H

#include "conv_bp_wino_transform_dy.h"

namespace WinoTransformDetail {
constexpr uint32_t F23_FMAP_STRIDE = 2;
constexpr uint32_t F23_FMAP_WINDOWS = 4;

using namespace AscendC::Reg;

template <typename T, typename TilingT>
struct Fmap {
    // fmap一个循环里展开2个tile,所以额外添加首位轮参数
    struct UnfoldFmapRowParams : DefaultUnfoldRowParams {
        uint16_t tileWMainRepeatTimes;
        uint16_t tileWTailRepeatTimes;
    };

    struct UnfoldFmapColParams : DefaultUnfoldColParams {
        uint16_t tileHMainRepeatTimes;
        uint16_t tileHTailRepeatTimes;
    };

    using UnfoldRowParamsT = UnfoldFmapRowParams;
    using UnfoldColParamsT = UnfoldFmapColParams;

    using ParamsTuple = AscendC::Std::tuple<UnfoldColParamsT, UnfoldRowParamsT>;

    static ParamsTuple inline __aicore__ InitUnfoldParams(const TileBox& box)
    {
        UnfoldFmapColParams ucp = {};
        UnfoldFmapRowParams urp = {};

        InitDefaultUnfoldParams<T, F23_FMAP_STRIDE, F23_FMAP_WINDOWS>(box, ucp, urp);

        ucp.tileHMainRepeatTimes = ucp.tileH >> 1;
        ucp.tileHTailRepeatTimes = ucp.tileH & 1;

        urp.tileWMainRepeatTimes = urp.tileW >> 1;
        urp.tileWTailRepeatTimes = urp.tileW & 1;

        return AscendC::Std::make_tuple(ucp, urp);
    }

    template <bool IsTailTile>
    static __simd_callee__ inline void UnfoldColsVf(__ubuf__ T* tileBuf, __ubuf__ T* fmapBuf,
                                                    const UnfoldFmapColParams& params)
    {
        if constexpr (IsTailTile) {
            UnfoldColsDefaultVf(tileBuf, fmapBuf, params);
        } else {
            constexpr uint32_t TileH = BlockConfig::SingleShapeTileH<TilingT>();
            constexpr uint32_t TileW = BlockConfig::SingleShapeTileW<TilingT>();
            if constexpr (TileH <= 4) {
                // fmap的正变换为了复用Tile间的重叠数据，是W循环在外，H循环在内
                // 如果H太小则W循环带来的scalar占比会比较大，所以TileH较小的情况下将TileH进行展开
                UnfoldColsUnRollTileHVf<TileH, TileW>(tileBuf, fmapBuf);
            } else {
                UnfoldColsDefaultVf(tileBuf, fmapBuf, params);
            }
        }
    }

    template <uint16_t TileH, uint16_t SrcH, uint32_t WElements, uint32_t TileBufWidthBlocks, uint32_t DstStride>
    struct TileHUnRollHelper {
        __ubuf__ T* src[SrcH];
        __ubuf__ T* dst[TileH];
        RegTensor<T> s[SrcH];

        template <uint16_t Idx>
        static __simd_callee__ inline void InitSrcAddrImpl(TileHUnRollHelper& helper, __ubuf__ T* fmapBuf)
        {
            helper.src[Idx] = fmapBuf + WElements * Idx;
        }

        template <uint16_t N = SrcH>
        static __simd_callee__ inline void InitSrcAddr(TileHUnRollHelper& helper, __ubuf__ T* fmapBuf)
        {
            if constexpr (N > 0) {
                // 🌟 技巧 2：先调 N-1 的递归，再执行 N-1 的动作。
                // 这样就能保证是从 0, 1, 2... 顺序执行的！
                InitSrcAddr<N - 1>(helper, fmapBuf);
                InitSrcAddrImpl<N - 1>(helper, fmapBuf);
            }
        }

        template <uint16_t Idx>
        static __simd_callee__ inline void InitDstAddrImpl(TileHUnRollHelper& helper, __ubuf__ T* tileBuf)
        {
            helper.dst[Idx] = tileBuf + F23_TRANSFORM_TILE_SIZE_4 * C0<T>() * Idx;
        }

        template <uint16_t N = TileH>
        static __simd_callee__ inline void InitDstAddr(TileHUnRollHelper& helper, __ubuf__ T* tileBuf)
        {
            if constexpr (N > 0) {
                InitDstAddr<N - 1>(helper, tileBuf);
                InitDstAddrImpl<N - 1>(helper, tileBuf);
            }
        }

        template <uint16_t Idx>
        static __simd_callee__ inline void LoadSrcImpl(TileHUnRollHelper& helper)
        {
            LoadAlign<T, PostLiteral::POST_MODE_UPDATE>(helper.s[Idx], helper.src[Idx], VL<T>());
        }

        template <uint16_t N = SrcH>
        static __simd_callee__ inline void LoadHSrc(TileHUnRollHelper& helper)
        {
            if constexpr (N > 0) {
                LoadHSrc<N - 1>(helper);
                LoadSrcImpl<N - 1>(helper);
            }
        }

        template <uint16_t Idx>
        static __simd_callee__ inline void TransformAndStoreImpl(TileHUnRollHelper& helper, MaskReg& mask)
        {
            constexpr uint16_t sIdx = Idx * F23_FMAP_STRIDE;
            RegTensor<T>& s0 = helper.s[sIdx];
            RegTensor<T>& s1 = helper.s[sIdx + 1];
            RegTensor<T>& s2 = helper.s[sIdx + 2];
            RegTensor<T>& s3 = helper.s[sIdx + 3];

            RegTensor<T> d0;
            RegTensor<T> d1;
            RegTensor<T> d2;
            RegTensor<T> d3;

            TransformVf(s0, s1, s2, s3, d0, d1, d2, d3, mask);

            StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(helper.dst[Idx], d0,
                                                                                        TileBufWidthBlocks, 1, mask);
            StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(helper.dst[Idx], d1,
                                                                                        TileBufWidthBlocks, 1, mask);
            StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(helper.dst[Idx], d2,
                                                                                        TileBufWidthBlocks, 1, mask);
            StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                helper.dst[Idx], d3, TileBufWidthBlocks, DstStride, mask);
        }

        template <uint16_t N = TileH>
        static __simd_callee__ inline void TransformAndStore(TileHUnRollHelper& helper, MaskReg& mask)
        {
            if constexpr (N > 0) {
                TransformAndStore<N - 1>(helper, mask);
                TransformAndStoreImpl<N - 1>(helper, mask);
            }
        }
    };

    template <uint16_t TileH, uint16_t TileW>
    static __simd_callee__ inline void UnfoldColsUnRollTileHVf(__ubuf__ T* tileBuf, __ubuf__ T* fmapBuf)
    {
        using SlideWin = SlideWindows<F23_FMAP_STRIDE, F23_FMAP_WINDOWS>;
        constexpr uint16_t SrcH = SlideWin::Tiles2SrcLength(TileH);
        constexpr uint32_t WElements = SlideWin::Tiles2SrcLength(TileW) * C0<T>();
        constexpr uint16_t WRepeatTimes = ConstexprMaths::CeilDiv(WElements, VL<T>());
        constexpr uint16_t tileBufWidthBlocks = CalColUnfoldBufWidth(TileH);
        constexpr uint32_t DstStride = tileBufWidthBlocks * (VL<T>() / C0<T>()) - F23_TRANSFORM_TILE_SIZE_4 + 1;

        // vf里面直接用pragma roll发现会有编译失败，不展开循环似乎没法用数组，所以手动用模板把代码展开
        using HelperT = TileHUnRollHelper<TileH, SrcH, WElements, tileBufWidthBlocks, DstStride>;
        HelperT helper;

        HelperT::InitSrcAddr(helper, fmapBuf);
        HelperT::InitDstAddr(helper, tileBuf);

        uint32_t maskValue = WElements;
        for (uint16_t i = 0; i < WRepeatTimes; i++) {
            MaskReg mask = UpdateMask<T>(maskValue);
            HelperT::LoadHSrc(helper);
            HelperT::TransformAndStore(helper, mask);
        }
    }

    static __simd_callee__ inline void UnfoldColsDefaultVf(__ubuf__ T* tileBuf, __ubuf__ T* fmapBuf,
                                                           const UnfoldFmapColParams& params)
    {
        const uint32_t wValidElements = params.wValidElements;
        const uint32_t tileBufWidthBlocks = params.tileBufWidthBlocks;
        const uint16_t wRepeatTimes = params.wRepeatTimes;
        const uint16_t tileHMainRepeatTimes = params.tileHMainRepeatTimes;
        const uint16_t tileHTailRepeatTimes = params.tileHTailRepeatTimes;

        uint32_t maskValue = wValidElements;
        for (uint16_t i = 0; i < wRepeatTimes; i++) {
            MaskReg mask = UpdateMask<T>(maskValue);
            RegTensor<T> s0;
            RegTensor<T> s1;
            RegTensor<T> s2;
            RegTensor<T> s3;

            RegTensor<T> d0;
            RegTensor<T> d1;
            RegTensor<T> d2;
            RegTensor<T> d3;

            // 从最上方的tile开始滑窗
            // 先读取fmap首2行,每次循环往下读2行凑成4行执行变换
            // 但若一个滑窗在fmap的1-4行分别读入s0,s1,s2,s3
            // 在下一个滑窗s2,s3就变成1-2行,不考虑重新读取的话2-3行就只能读入s0,s1,滑窗1-4行就变成s2,s3,s0,s1
            // 如果将s0,s1的数据拷贝到s2,s3可能会产生多余的mov指令
            // 所以这里按照最朴素的方式展开循环一个循环内处理2个连续滑窗,
            // 如果滑窗为奇数,则通过tileHTailRepeatTimes额外执行一次滑窗

            // 循环fmapW
            const uint32_t wOffset = i * VL<T>();

            __ubuf__ T* src = fmapBuf + wOffset;

            LoadAlign<T, PostLiteral::POST_MODE_UPDATE>(s0, src, wValidElements);
            LoadAlign<T, PostLiteral::POST_MODE_UPDATE>(s1, src, wValidElements);

            __ubuf__ T* dst = tileBuf + tileBufWidthBlocks * wOffset;

            for (uint16_t th = 0; th < tileHMainRepeatTimes; th++) {
                LoadAlign<T, PostLiteral::POST_MODE_UPDATE>(s2, src, wValidElements);
                LoadAlign<T, PostLiteral::POST_MODE_UPDATE>(s3, src, wValidElements);

                TransformVf(s0, s1, s2, s3, d0, d1, d2, d3, mask);
                UnfoldColsDefaultStore(dst, d0, d1, d2, d3, tileBufWidthBlocks, mask);

                LoadAlign<T, PostLiteral::POST_MODE_UPDATE>(s0, src, wValidElements);
                LoadAlign<T, PostLiteral::POST_MODE_UPDATE>(s1, src, wValidElements);

                TransformVf(s2, s3, s0, s1, d0, d1, d2, d3, mask);
                UnfoldColsDefaultStore(dst, d0, d1, d2, d3, tileBufWidthBlocks, mask);
            }

            for (uint16_t th = 0; th < tileHTailRepeatTimes; th++) {
                LoadAlign<T, PostLiteral::POST_MODE_UPDATE>(s2, src, wValidElements);
                LoadAlign<T, PostLiteral::POST_MODE_UPDATE>(s3, src, wValidElements);
                TransformVf(s0, s1, s2, s3, d0, d1, d2, d3, mask);
                UnfoldColsDefaultStore(dst, d0, d1, d2, d3, tileBufWidthBlocks, mask);
            }
        }
    }

    static __simd_callee__ inline void UnfoldColsDefaultStore(__ubuf__ T*& dst, RegTensor<T>& d0, RegTensor<T>& d1,
                                                              RegTensor<T>& d2, RegTensor<T>& d3,
                                                              uint32_t tileBufWidthBlocks, MaskReg& mask)
    {
        StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(dst, d0, tileBufWidthBlocks, 1,
                                                                                    mask);
        StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(dst, d1, tileBufWidthBlocks, 1,
                                                                                    mask);
        StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(dst, d2, tileBufWidthBlocks, 1,
                                                                                    mask);
        StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(dst, d3, tileBufWidthBlocks, 1,
                                                                                    mask);
    }

    static __simd_callee__ inline void UnfoldRowsVf(__ubuf__ T* out, __ubuf__ T* buf, const UnfoldFmapRowParams& params)
    {
        const uint32_t srcTileBufWidth = params.srcTileBufWidth;
        const uint16_t dstTileBufWidthBlocks = params.dstTileBufWidthBlocks;
        const uint16_t hRepeatTimes = params.hRepeatTimes;
        const uint16_t tileWMainRepeatTimes = params.tileWMainRepeatTimes;
        const uint16_t tileWTailRepeatTimes = params.tileWTailRepeatTimes;
        const uint16_t tileW = params.tileW;

        Unfold16TileHWStorer::StoreInfo<T> s;
        Unfold16TileHWStorer::CreateStoreInfo(s, out, tileW, dstTileBufWidthBlocks);

        uint32_t maskValue = params.hValidElements;
        for (uint16_t i = 0; i < hRepeatTimes; i++) {
            MaskReg mask = UpdateMask<T>(maskValue);
            MaskReg storeMask;
            Unfold16TileHWStorer::GetHighHalfPartMask(s, storeMask, mask);

            RegTensor<T> s0;
            RegTensor<T> s1;
            RegTensor<T> s2;
            RegTensor<T> s3;

            RegTensor<T> d0;
            RegTensor<T> d1;
            RegTensor<T> d2;
            RegTensor<T> d3;

            __ubuf__ T* src = buf + VL<T>() * i;

            LoadAlign<T, PostLiteral::POST_MODE_UPDATE>(s0, src, srcTileBufWidth);
            LoadAlign<T, PostLiteral::POST_MODE_UPDATE>(s1, src, srcTileBufWidth);

            for (uint16_t tw = 0; tw < tileWMainRepeatTimes; tw++) {
                LoadAlign<T, PostLiteral::POST_MODE_UPDATE>(s2, src, srcTileBufWidth);
                LoadAlign<T, PostLiteral::POST_MODE_UPDATE>(s3, src, srcTileBufWidth);

                TransformVf(s0, s1, s2, s3, d0, d1, d2, d3, mask);

                Unfold16TileHWStorer::store(s, d0, d1, d2, d3, storeMask);

                LoadAlign<T, PostLiteral::POST_MODE_UPDATE>(s0, src, srcTileBufWidth);
                LoadAlign<T, PostLiteral::POST_MODE_UPDATE>(s1, src, srcTileBufWidth);

                TransformVf(s2, s3, s0, s1, d0, d1, d2, d3, mask);

                Unfold16TileHWStorer::store(s, d0, d1, d2, d3, storeMask);
            }

            for (uint16_t th = 0; th < tileWTailRepeatTimes; th++) {
                LoadAlign<T, PostLiteral::POST_MODE_UPDATE>(s2, src, srcTileBufWidth);
                LoadAlign<T, PostLiteral::POST_MODE_UPDATE>(s3, src, srcTileBufWidth);

                TransformVf(s0, s1, s2, s3, d0, d1, d2, d3, mask);

                Unfold16TileHWStorer::store(s, d0, d1, d2, d3, storeMask);
            }

            Unfold16TileHWStorer::UpdateStoreInfo(s, tileW);
        }
    }

    static __simd_callee__ inline void TransformVf(RegTensor<T>& s0, RegTensor<T>& s1, RegTensor<T>& s2,
                                                   RegTensor<T>& s3, RegTensor<T>& d0, RegTensor<T>& d1,
                                                   RegTensor<T>& d2, RegTensor<T>& d3, MaskReg& mask)
    {
        Sub(d0, s0, s2, mask);
        Add(d1, s1, s2, mask);
        Sub(d2, s2, s1, mask);
        Sub(d3, s1, s3, mask);
    }
};

template <typename T, typename TilingT>
using FmapConfig = TransformConfig<T, F23_FMAP_STRIDE, F23_FMAP_WINDOWS, Fmap<T, TilingT>, TilingT>;

} // namespace WinoTransformDetail

template <typename T, typename TilingT>
using WinoFmapFwdTransformer = WinoTransformer<WinoTransformDetail::FmapConfig<T, TilingT> >;

#endif // CONV_BP_WINO_TRANSFORM_FMAP_H
