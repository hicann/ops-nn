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
 * \file full_mn_kernel_tile_ops.h
 * \brief Tile-layer: independent stateless generic structs, each exposed via operator().
 */

#ifndef FULL_MN_KERNEL_TILE_OPS_H
#define FULL_MN_KERNEL_TILE_OPS_H

#include "full_mn_kernel_common.h"
#include "full_mn_kernel_block_ctx.h"

namespace full_mn {

// ---- 1. LoadAL1SegTool: fmap K-segment NCHW → L1 NZ (Dn2Nz) ----
template <typename BlockCtx, typename TilingData>
struct LoadAL1SegTool {
    using FmapT = typename BlockCtx::FmapType;

    __aicore__ inline void operator()(const BlockCtx& bc, const TilingData& tiling, uint32_t curHiLoad,
                                      uint32_t hiLoadStart, const GlobalTensor<FmapT>& fmapGm, uint32_t kAL1Iter,
                                      LocalTensor<FmapT>& al1) const
    {
        uint32_t cinAligned = static_cast<uint32_t>(AlignB(tiling.singleCoreCi, BlockCtx::GK0));
        uint32_t kTotal = cinAligned * tiling.kh * tiling.kw;
        uint32_t maxKAL1Iter = static_cast<uint32_t>(CeilDiv(kTotal, tiling.kAL1) - 1);
        bool isTail = (kAL1Iter == maxKAL1Iter);
        uint32_t dValue = isTail ? tiling.cinATailInCore : tiling.cinAInCore;
        uint32_t orgWin = static_cast<uint32_t>(tiling.win);
        uint64_t khxkw = tiling.kh * tiling.kw;
        uint64_t orgHixWi = tiling.hin * tiling.win;

        Dn2NzParams p;
        p.dnNum = 1;
        p.nValue = curHiLoad * orgWin;
        p.dValue = dValue;
        p.srcDnMatrixStride = 0;
        p.srcDValue = static_cast<uint32_t>(tiling.hin * tiling.win);
        p.dstNzC0Stride = curHiLoad * orgWin;
        p.dstNzNStride = 1;

        uint64_t cinOffsetBlockInGM = tiling.kAL1 / khxkw * orgHixWi;
        uint64_t gmOffset = static_cast<uint64_t>(kAL1Iter) * cinOffsetBlockInGM;
        DataCopy<FmapT>(al1, fmapGm[gmOffset], p);
    }
};

template <typename BlockCtx, typename TilingData>
struct LoadBL1Seg5HDTool {
    using WeightT = typename BlockCtx::WeightType;

    __aicore__ inline void operator()(const BlockCtx& bc, const TilingData& tiling,
                                      const GlobalTensor<WeightT>& weight5HDGm, uint32_t kBL1Iter,
                                      LocalTensor<WeightT>& bl1) const
    {
        constexpr uint32_t k0 = BlockCtx::GK0;
        uint64_t hw = static_cast<uint64_t>(tiling.kh * tiling.kw);
        uint64_t c1Total = CeilDiv(static_cast<uint64_t>(tiling.cin), static_cast<uint64_t>(k0));
        uint64_t totalK = c1Total * hw * k0;
        uint64_t gmOffset = bc.nIdxStart * totalK + kBL1Iter * tiling.kBL1;
        uint64_t c1PerChunk = static_cast<uint64_t>(tiling.kBL1) / (hw * k0);
        uint64_t c1Base = kBL1Iter * c1PerChunk;
        uint64_t c1Cnt = c1PerChunk < (c1Total - c1Base) ? c1PerChunk : (c1Total - c1Base);

        Nd2NzParams nd2nzParams;
        nd2nzParams.dValue = c1Cnt * hw * k0;
        nd2nzParams.dstNzC0Stride = tiling.nBL1;
        if (totalK <= ND2NZ_SRC_DVALUE_MAX) {
            nd2nzParams.ndNum = 1;
            nd2nzParams.nValue = bc.actualCo;
            nd2nzParams.srcNdMatrixStride = 0;
            nd2nzParams.srcDValue = totalK;
            nd2nzParams.dstNzNStride = 1;
            nd2nzParams.dstNzMatrixStride = 0;
        } else {
            nd2nzParams.ndNum = bc.actualCo;
            nd2nzParams.srcNdMatrixStride = totalK;
            nd2nzParams.dstNzMatrixStride = k0;
            nd2nzParams.nValue = 1;
            nd2nzParams.srcDValue = nd2nzParams.dValue;
            nd2nzParams.dstNzNStride = 1;
        }
        DataCopy<WeightT>(bl1, weight5HDGm[gmOffset], nd2nzParams);
    }
};

// ---- 3. LoadBiasL1Tool: bias GM → L1 full load (InitConstValue + DataCopyPad) ----
template <typename BlockCtx, typename TilingData>
struct LoadBiasL1Tool {
    using BiasT = typename BlockCtx::BiasType;

    __aicore__ inline void operator()(const BlockCtx& bc, const TilingData& tiling, const GlobalTensor<BiasT>& biasGm,
                                      uint32_t biasL1Offset) const
    {
        LocalTensor<BiasT> biasL1(TPosition::A1, biasL1Offset, bc.actualCo);
        InitConstValueParams<BiasT> initParams(
            1, static_cast<uint16_t>(AlignB(bc.actualCo, BLOCK_L0_N) * CONST_VALUE_4 / C0_SIZE), 0, 0);
        InitConstValue<BiasT>(biasL1, initParams);

        uint32_t byteNum = sizeof(BiasT);
        DataCopyParams dataCopyParams(1, bc.actualCo * byteNum, 0, 0);
        uint8_t rightPadding = static_cast<uint8_t>(AlignB(bc.actualCo * byteNum, PADDING_ALIGN_SIZE) / byteNum -
                                                    bc.actualCo);
        DataCopyPadParams padParams(true, 0, rightPadding, 0);
        DataCopyPad<BiasT>(biasL1, biasGm[0], dataCopyParams, padParams);
    }
};

// ---- 4. LoadBiasToBTTool: bias L1 → BT (DataCopy) ----
template <typename BlockCtx, typename TilingData>
struct LoadBiasToBTTool {
    using BiasT = typename BlockCtx::BiasType;
    using L0cT = typename BlockCtx::L0cType;

    __aicore__ inline void operator()(const BlockCtx& bc, const TilingData& tiling, uint32_t biasL1Offset) const
    {
        constexpr uint32_t BT_ALIGN = 64;
        uint32_t btElemNum = static_cast<uint32_t>(AlignB(tiling.singleCoreCo * sizeof(L0cT), BT_ALIGN)) / sizeof(L0cT);
        LocalTensor<L0cT> biasBT(TPosition::C2, 0, btElemNum);
        LocalTensor<BiasT> biasL1src(TPosition::A1, biasL1Offset, bc.actualCo);
        uint32_t blkCnt = static_cast<uint32_t>(AlignB(bc.actualCo * sizeof(BiasT), BT_ALIGN)) / 32;
        DataCopyParams cp(1, static_cast<uint16_t>(blkCnt), 0, 0);
        DataCopy(biasBT, biasL1src[0], cp);
    }
};

// ---- 5. SetupLoad3DTool: Load3D invariant setup ----
template <typename BlockCtx, typename TilingData>
struct SetupLoad3DTool {
    __aicore__ inline void operator()(const BlockCtx& bc, const TilingData& tiling, uint32_t curHiLoad,
                                      int32_t padTopL1, uint64_t& load3dXtBase, uint64_t& load3dXmTmp) const
    {
        uint8_t padList[4] = {static_cast<uint8_t>(tiling.padLeft), static_cast<uint8_t>(tiling.padRight),
                              static_cast<uint8_t>(padTopL1), static_cast<uint8_t>(255)};
        uint32_t orgWin = static_cast<uint32_t>(tiling.win);
        Load3DSetFMatrixCal(curHiLoad, orgWin, padList);
        Load3DSetPaddingCal(tiling.offsetx);

        load3dXtBase = ((static_cast<uint64_t>(tiling.strideW) & MASK_6) << 0) |
                       ((static_cast<uint64_t>(tiling.kw) & MASK_8) << KERNELW_OFFSET) |
                       ((static_cast<uint64_t>(tiling.kh) & MASK_8) << KERNELH_OFFSET) |
                       ((static_cast<uint64_t>(tiling.strideH) & MASK_6) << STRIDEH_OFFSET) |
                       ((static_cast<uint64_t>(tiling.kw) & NINTH_BIT_MASK) << KERNELW_HIGHEST_BIT_OFFSET) |
                       ((static_cast<uint64_t>(tiling.kh) & NINTH_BIT_MASK) << KERNELH_HIGHEST_BIT_OFFSET) |
                       ((static_cast<uint64_t>(tiling.dilationW) & MASK_8) << DILATIONW_OFFSET) |
                       ((static_cast<uint64_t>(tiling.dilationH) & MASK_8) << DILATIONH_OFFSET);

        uint32_t currentML0Align = static_cast<uint32_t>(AlignB(bc.actualM, M0));
        uint32_t posM = bc.mIdxStart % static_cast<uint32_t>(tiling.wout);
        load3dXmTmp = ((static_cast<uint64_t>(bc.actualM) & MASK_16) << MSTEP_OFFSET) |
                      ((static_cast<uint64_t>(posM) & MASK_16) << POSM_OFFSET);

#if defined(ASC_DEVKIT_VERSION_NUM) && (ASC_DEVKIT_VERSION_NUM >= 90000000)
        LoadDataRepeatParamWithStride repeatParams(0, 1, 0, static_cast<uint16_t>(currentML0Align / M0));
        SetLoadDataRepeatWithStride(repeatParams);
#else
        LoadDataRepeatParam repeatParams(0, 1, 0, static_cast<uint16_t>(currentML0Align / M0));
        SetLoadDataRepeat(repeatParams);
#endif
    }
};

// ---- 6. LoadAL0Tool: Load3D AL1 → AL0 ----
template <typename BlockCtx>
struct LoadAL0Tool {
    using FmapT = typename BlockCtx::FmapType;

    __aicore__ inline void operator()(const LocalTensor<FmapT>& al1, LocalTensor<FmapT>& al0, uint32_t posK,
                                      uint32_t currentKL0, uint32_t channelSize, uint64_t load3dXtBase,
                                      uint64_t load3dXmTmp) const
    {
        uint64_t posKVal = static_cast<uint64_t>(posK);
        uint64_t xm = static_cast<uint64_t>(currentKL0) | (posKVal << POSK_OFFSET) | load3dXmTmp;
        uint64_t xt = load3dXtBase | ((static_cast<uint64_t>(channelSize) & MASK_16) << CIN_OFFSET);

        Load3DBitModeParam param;
        param.SetConfig0(xm);
        param.SetConfig1(xt);
        LoadData<TPosition::A2, TPosition::A1, FmapT>(al0, al1, param);
    }
};

// ---- 7. LoadBL0Tool: Load2D BL1 → BL0 ----
template <typename BlockCtx>
struct LoadBL0Tool {
    using WeightT = typename BlockCtx::WeightType;

    __aicore__ inline void operator()(const BlockCtx& bc, const LocalTensor<WeightT>& bl1, LocalTensor<WeightT>& bl0,
                                      uint32_t KStartPosition, uint32_t currentKL0, uint32_t srcStride) const
    {
        uint32_t kStep = static_cast<uint32_t>(CeilDiv(currentKL0, BlockCtx::GK0));
        uint32_t mStep = bc.n1PerCore;
        Load2DBitModeParam param;
        param.SetMStartPosition(0);
        param.SetKStartPosition(KStartPosition);
        param.SetMStep(static_cast<uint16_t>(mStep));
        param.SetKStep(static_cast<uint16_t>(kStep));
        param.SetSrcStride(srcStride);
        param.SetDstStride(static_cast<uint16_t>(mStep));
        param.SetIfTranspose(false);
        LoadData<TPosition::B2, TPosition::B1, WeightT>(bl0, bl1, param);
    }
};

// ---- 8. MmadTool: MMAD ----
template <typename BlockCtx>
struct MmadTool {
    using FmapT = typename BlockCtx::FmapType;
    using WeightT = typename BlockCtx::WeightType;
    using L0cT = typename BlockCtx::L0cType;

    __aicore__ inline void operator()(LocalTensor<L0cT>& cl0, LocalTensor<FmapT>& al0, LocalTensor<WeightT>& bl0,
                                      MmadParams& mp) const
    {
        Mmad(cl0, al0, bl0, mp);
    }
};

// ---- 9. CopyOutTool: Fixpipe + CopyOut (L0C → GM) ----
template <typename BlockCtx, typename TilingData>
struct CopyOutTool {
    using OutputT = typename BlockCtx::OutputType;
    using L0cT = typename BlockCtx::L0cType;

    __aicore__ inline void operator()(const BlockCtx& bc, const TilingData& tiling, GlobalTensor<OutputT>& outputGm,
                                      LocalTensor<L0cT>& cl0) const
    {
        uint32_t currentML0Align = static_cast<uint32_t>(AlignB(bc.actualM, M0));
        uint64_t hwOut = static_cast<uint64_t>(tiling.hout) * tiling.wout;
        FixpipeParamsC310<CO2Layout::COLUMN_MAJOR> fp;
        fp.nSize = static_cast<uint16_t>(bc.actualCo);
        fp.mSize = static_cast<uint16_t>(bc.actualM);
        fp.srcStride = static_cast<uint16_t>(currentML0Align);
        fp.dstStride = static_cast<uint32_t>(hwOut);
        fp.unitFlag = UNIT_FLAG_ENABLE_WITH_FLIP;
        fp.params.dnNum = 1;
        fp.params.srcNzMatrixStride = currentML0Align * CeilDiv(bc.actualCo, M0);
        fp.params.dstDnMatrixStride = hwOut;
        fp.params.srcNzC0Stride = 1;

        if constexpr (AscendC::IsSameType<OutputT, half>::value) {
            fp.quantPre = QuantMode_t::F322F16;
            fp.deqScalar = DEQ_SCALAR_ONE;
            Fixpipe<OutputT, L0cT, CFG_COLUMN_MAJOR>(outputGm, cl0, fp);
        } else if constexpr (AscendC::IsSameType<OutputT, bfloat16_t>::value) {
            fp.quantPre = QuantMode_t::F322BF16;
            fp.deqScalar = DEQ_SCALAR_ONE;
            Fixpipe<OutputT, L0cT, CFG_COLUMN_MAJOR>(outputGm, cl0, fp);
        } else if constexpr (AscendC::IsSameType<OutputT, hifloat8_t>::value) {
            fp.quantPre = QuantMode_t::QF322HIF8_PRE;
            fp.deqScalar = DEQ_SCALAR_ONE;
            Fixpipe<OutputT, L0cT, CFG_COLUMN_MAJOR>(outputGm, cl0, fp);
        } else if constexpr (AscendC::IsSameType<OutputT, float>::value) {
            fp.quantPre = QuantMode_t::NoQuant;
            fp.deqScalar = DEQ_SCALAR_ONE;
            Fixpipe<OutputT, L0cT, CFG_COLUMN_MAJOR>(outputGm, cl0, fp);
        }
    }
};

} // namespace full_mn

#endif // FULL_MN_KERNEL_TILE_OPS_H
