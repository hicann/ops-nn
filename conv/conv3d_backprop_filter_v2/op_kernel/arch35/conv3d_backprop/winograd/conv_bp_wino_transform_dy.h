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
 * \file conv_bp_wino_transform_dy.h
 * \brief
 */

#ifndef CONV_BP_WINO_TRANSFORM_DY_H
#define CONV_BP_WINO_TRANSFORM_DY_H

#include "conv_bp_wino_transform.h"

namespace WinoTransformDetail {
constexpr uint32_t F23_DY_STRIDE = 2;
constexpr uint32_t F23_DY_WINDOWS = 2;

using namespace AscendC::Reg;
using namespace AscendC;

struct DefaultUnfoldColParams {
    uint32_t wValidElements;
    uint32_t tileBufWidthBlocks;
    uint16_t wRepeatTimes;
    uint16_t tileH;
};

struct DefaultUnfoldRowParams {
    uint32_t hValidElements;
    uint32_t srcTileBufWidth;
    uint32_t dstTileBufWidthBlocks;
    uint16_t hRepeatTimes;
    uint16_t tileW;
};

template <typename T, uint32_t F23_STRIDE, uint32_t F23_WINDOW>
static inline __aicore__ void InitDefaultUnfoldParams(const TileBox& box, DefaultUnfoldColParams& ucp,
                                                      DefaultUnfoldRowParams& urp)
{
    ucp.wValidElements = SlideWindows<F23_STRIDE, F23_WINDOW>::Tiles2SrcLength(box.tile.wLength) * C0<T>();
    ucp.tileBufWidthBlocks = CalColUnfoldBufWidth(box.tile.hLength);
    ucp.wRepeatTimes = Ops::Base::CeilDiv(ucp.wValidElements, VL<T>());
    ucp.tileH = box.tile.hLength;

    urp.hValidElements = TileUnfoldSize(box.tile.hLength) * C0<T>();
    urp.srcTileBufWidth = ucp.tileBufWidthBlocks * C0<T>();
    urp.dstTileBufWidthBlocks = Cal16TileHWBufWidth(box.tile.elements);
    urp.hRepeatTimes = Ops::Base::CeilDiv(urp.hValidElements, VL<T>());
    urp.tileW = box.tile.wLength;
}

struct Unfold16TileHWStorer {
    template <typename T>
    struct StoreInfo {
        __ubuf__ T* dst0;
        __ubuf__ T* dst1;
        __ubuf__ T* dst2;
        __ubuf__ T* dst3;
        __ubuf__ T* dst4;
        __ubuf__ T* dst5;
        __ubuf__ T* dst6;
        __ubuf__ T* dst7;

        uint16_t dstTileBufWidthBlocks;

        MaskReg maskAll;
        MaskReg lowHalfPartMask;
        MaskReg highHalfPartMask;
    };

    template <typename T>
    static __simd_callee__ inline void CreateStoreInfo(StoreInfo<T>& p, __ubuf__ T* out, uint16_t tileW,
                                                       uint16_t dstTileBufWidthBlocks)
    {
        uint32_t dstStride = C0<T>() * F23_TRANSFORM_TILE_SIZE_4 * dstTileBufWidthBlocks;

        p.dstTileBufWidthBlocks = dstTileBufWidthBlocks;
        p.dst0 = out;
        p.dst1 = p.dst0 + dstStride;
        p.dst2 = p.dst1 + dstStride;
        p.dst3 = p.dst2 + dstStride;

        // 这些地址用于reg的后半部分写入，需要减掉前半部分的地址偏移
        // 测试地址减掉后越界
        p.dst4 = p.dst0 + tileW * C0<T>() - C0<T>() * 4 * dstTileBufWidthBlocks;
        p.dst5 = p.dst4 + dstStride;
        p.dst6 = p.dst5 + dstStride;
        p.dst7 = p.dst6 + dstStride;

        p.lowHalfPartMask = CreateMask<T, MaskPattern::H>();
        p.maskAll = CreateMask<T, MaskPattern::ALL>();
        Not(p.highHalfPartMask, p.lowHalfPartMask, p.maskAll);
    }

    template <typename T>
    static __simd_callee__ inline void UpdateStoreInfo(StoreInfo<T>& p, uint16_t tileW)
    {
        uint32_t step = C0<T>() * tileW;
        p.dst0 += step;
        p.dst1 += step;
        p.dst2 += step;
        p.dst3 += step;
        p.dst4 += step;
        p.dst5 += step;
        p.dst6 += step;
        p.dst7 += step;
    }

    template <typename T>
    static __simd_callee__ inline void GetHighHalfPartMask(StoreInfo<T>& p, MaskReg& highHalfPartMask,
                                                           MaskReg& tileHMask)
    {
        And(highHalfPartMask, tileHMask, p.highHalfPartMask, p.maskAll);
    }

    template <typename T>
    static __simd_callee__ inline void store(StoreInfo<T>& p, RegTensor<T>& r0, RegTensor<T>& r1, RegTensor<T>& r2,
                                             RegTensor<T>& r3, MaskReg& highHalfPartMask)
    {
        // 尝试先gather在select完成block级别的交织

        StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(p.dst0, r0, p.dstTileBufWidthBlocks,
                                                                                    1, p.lowHalfPartMask);
        StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(p.dst1, r1, p.dstTileBufWidthBlocks,
                                                                                    1, p.lowHalfPartMask);

        StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(p.dst2, r2, p.dstTileBufWidthBlocks,
                                                                                    1, p.lowHalfPartMask);
        StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(p.dst3, r3, p.dstTileBufWidthBlocks,
                                                                                    1, p.lowHalfPartMask);

        StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(p.dst4, r0, p.dstTileBufWidthBlocks,
                                                                                    1, highHalfPartMask);
        StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(p.dst5, r1, p.dstTileBufWidthBlocks,
                                                                                    1, highHalfPartMask);

        StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(p.dst6, r2, p.dstTileBufWidthBlocks,
                                                                                    1, highHalfPartMask);
        StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(p.dst7, r3, p.dstTileBufWidthBlocks,
                                                                                    1, highHalfPartMask);
    }
};

template <typename T, typename TilingT>
struct Dy {
    using UnfoldRowParamsT = DefaultUnfoldRowParams;
    using UnfoldColParamsT = DefaultUnfoldColParams;

    using ParamsTuple = AscendC::Std::tuple<UnfoldColParamsT, UnfoldRowParamsT>;

    static ParamsTuple inline __aicore__ InitUnfoldParams(const TileBox& box)
    {
        DefaultUnfoldRowParams urp = {};
        DefaultUnfoldColParams ucp = {};

        InitDefaultUnfoldParams<T, F23_DY_STRIDE, F23_DY_WINDOWS>(box, ucp, urp);

        return AscendC::Std::make_tuple(ucp, urp);
    }

    template <bool isTailTile>
    static __simd_callee__ inline void UnfoldColsVf(__ubuf__ T* tileBuf, __ubuf__ T* dyBuf,
                                                    const DefaultUnfoldColParams& params)
    {
        constexpr uint32_t TileH = BlockConfig::SingleShapeTileH<TilingT>();
        constexpr uint32_t TileW = BlockConfig::SingleShapeTileW<TilingT>();

        using sw = SlideWindows<F23_DY_STRIDE, F23_DY_WINDOWS>;
        constexpr uint32_t SrcW = sw::Tiles2SrcLength(TileW);

        // 大轴在内，小轴在外，scalar都在第一层循环，所以外循环越小scalar占比越低
        constexpr uint32_t LoopH = TileH;
        constexpr uint32_t LoopW = ConstexprMaths::CeilDiv(SrcW * C0<T>(), VL<T>());

        if constexpr (LoopH > LoopW) {
            UnfoldColsHFirstVf(tileBuf, dyBuf, params);
        } else {
            UnfoldColsWFirstVf(tileBuf, dyBuf, params);
        }
    }

    static __simd_callee__ inline void UnfoldColsWFirstVf(__ubuf__ T* tileBuf, __ubuf__ T* dyBuf,
                                                          const DefaultUnfoldColParams& params)
    {
        const uint32_t wValidElements = params.wValidElements;
        const uint32_t tileBufWidthBlocks = params.tileBufWidthBlocks;
        const uint16_t wRepeatTimes = params.wRepeatTimes;
        const uint16_t tileH = params.tileH;
        const uint16_t dstStride = tileBufWidthBlocks * (VL<T>() / C0<T>()) - F23_TRANSFORM_TILE_SIZE_4 + 1;

        RegTensor<bfloat16_t> bf16NegativeOne;
        if constexpr (Std::is_same_v<T, bfloat16_t>) {
            Duplicate(bf16NegativeOne, -1);
        }

        for (uint16_t th = 0; th < tileH; th++) {
            constexpr uint32_t thStride = F23_TRANSFORM_TILE_SIZE_4 * C0<T>();

            __ubuf__ T* dst = tileBuf + thStride * th;
            __ubuf__ T* src0 = dyBuf + wValidElements * th * F23_DY_STRIDE;
            __ubuf__ T* src1 = src0 + wValidElements;

            uint32_t maskValue = wValidElements;
            for (uint16_t i = 0; i < wRepeatTimes; i++) {
                MaskReg mask = UpdateMask<T>(maskValue);

                RegTensor<T> s0;
                RegTensor<T> s1;
                LoadAlign<T, PostLiteral::POST_MODE_UPDATE>(s0, src0, VL<T>());
                LoadAlign<T, PostLiteral::POST_MODE_UPDATE>(s1, src1, VL<T>());

                RegTensor<T> d0;
                RegTensor<T> d1;
                RegTensor<T> d2;
                TransformVf(bf16NegativeOne, s0, s1, d0, d1, d2, mask);

                StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(dst, s0, tileBufWidthBlocks,
                                                                                            1, mask);
                StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(dst, d0, tileBufWidthBlocks,
                                                                                            1, mask);
                StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(dst, d1, tileBufWidthBlocks,
                                                                                            1, mask);
                StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(dst, d2, tileBufWidthBlocks,
                                                                                            dstStride, mask);
            }
        }
    }

    static __simd_callee__ inline void UnfoldColsHFirstVf(__ubuf__ T* tileBuf, __ubuf__ T* dyBuf,
                                                          const DefaultUnfoldColParams& params)
    {
        const uint32_t wValidElements = params.wValidElements;
        const uint32_t tileBufWidthBlocks = params.tileBufWidthBlocks;
        const uint16_t wRepeatTimes = params.wRepeatTimes;
        const uint16_t tileH = params.tileH;

        RegTensor<bfloat16_t> bf16NegativeOne;
        if constexpr (Std::is_same_v<T, bfloat16_t>) {
            Duplicate(bf16NegativeOne, -1);
        }

        uint32_t maskValue = wValidElements;
        for (uint16_t w = 0; w < wRepeatTimes; w++) {
            MaskReg mask = UpdateMask<T>(maskValue);

            __ubuf__ T* src = dyBuf + w * VL<T>();
            __ubuf__ T* dst = tileBuf + tileBufWidthBlocks * VL<T>() * w;

            for (uint16_t i = 0; i < tileH; i++) {
                RegTensor<T> s0, s1, d0, d1, d2;

                LoadAlign<T, PostLiteral::POST_MODE_UPDATE>(s0, src, wValidElements);
                LoadAlign<T, PostLiteral::POST_MODE_UPDATE>(s1, src, wValidElements);

                TransformVf(bf16NegativeOne, s0, s1, d0, d1, d2, mask);

                StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(dst, s0, tileBufWidthBlocks,
                                                                                            1, mask);
                StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(dst, d0, tileBufWidthBlocks,
                                                                                            1, mask);
                StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(dst, d1, tileBufWidthBlocks,
                                                                                            1, mask);
                StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(dst, d2, tileBufWidthBlocks,
                                                                                            1, mask);
            }
        }
    }

    static __simd_callee__ inline void UnfoldRowsVf(__ubuf__ T* out, __ubuf__ T* buf,
                                                    const DefaultUnfoldRowParams& params)
    {
        const uint32_t srcTileBufWidth = params.srcTileBufWidth;
        const uint16_t dstTileBufWidthBlocks = params.dstTileBufWidthBlocks;
        const uint16_t hRepeatTimes = params.hRepeatTimes;
        const uint16_t tileW = params.tileW;

        RegTensor<bfloat16_t> bf16NegativeOne;
        if constexpr (Std::is_same_v<T, bfloat16_t>) {
            Duplicate(bf16NegativeOne, -1);
        }

        Unfold16TileHWStorer::StoreInfo<T> s;
        Unfold16TileHWStorer::CreateStoreInfo(s, out, tileW, dstTileBufWidthBlocks);

        uint32_t maskValue = params.hValidElements;
        for (uint16_t i = 0; i < hRepeatTimes; i++) {
            __ubuf__ T* src = buf + i * VL<T>();
            MaskReg mask = UpdateMask<T>(maskValue);
            MaskReg storeMask;
            Unfold16TileHWStorer::GetHighHalfPartMask(s, storeMask, mask);

            for (uint16_t th = 0; th < tileW; th++) {
                RegTensor<T> s0;
                RegTensor<T> s1;

                LoadAlign<T, PostLiteral::POST_MODE_UPDATE>(s0, src, srcTileBufWidth);
                LoadAlign<T, PostLiteral::POST_MODE_UPDATE>(s1, src, srcTileBufWidth);

                RegTensor<T> d0;
                RegTensor<T> d1;
                RegTensor<T> d2;
                TransformVf(bf16NegativeOne, s0, s1, d0, d1, d2, mask);

                Unfold16TileHWStorer::store(s, s0, d0, d1, d2, storeMask);
            }
            Unfold16TileHWStorer::UpdateStoreInfo(s, tileW);
        }
    }

    static __simd_callee__ inline void TransformVf(RegTensor<bfloat16_t>& negativeOne, RegTensor<T>& s0,
                                                   RegTensor<T>& s1, RegTensor<T>& d0, RegTensor<T>& d1,
                                                   RegTensor<T>& d2, MaskReg& mask)
    {
        Add(d0, s0, s1, mask);
        Sub(d1, s0, s1, mask);
        if constexpr (Std::is_same_v<bfloat16_t, T>) {
            // bf16不支持Neg指令，用乘-1替代
            Mul(d2, s1, negativeOne, mask);
        } else {
            Neg(d2, s1, mask);
        }
    }
};

template <typename T, typename TilingT>
using DyConfig = TransformConfig<T, F23_DY_STRIDE, F23_DY_WINDOWS, Dy<T, TilingT>, TilingT>;
} // namespace WinoTransformDetail

template <typename T, typename TilingT>
using WinoDyFwdTransformer = WinoTransformer<WinoTransformDetail::DyConfig<T, TilingT> >;

#endif // CONV_BP_WINO_TRANSFORM_DY_H
