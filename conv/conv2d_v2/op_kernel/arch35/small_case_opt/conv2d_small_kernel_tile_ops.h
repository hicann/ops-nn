/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * Stateless tile-layer operations shared by all conv2d small-kernel pipelines.
 * Each struct is an independent functor exposed via operator(); no state is
 * retained between calls.
 *
 * Mode-agnostic: every M-mode / HW-mode decision (block windows, chunk
 * geometry, GM->L1 fmap copy shapes, output base offsets) lives in the spatial
 * schemes (spatial/); these tools only consume scheme-produced values.
 *
 * Ported from ops-nn conv2d_small_kernel{,_parallelism,_fm_partload}.h, keeping
 * the FULL feature set: NCHW/NHWC in/out, FRACTAL_Z/NCHW weight, groups>1,
 * dual-output, inner-batch, per-channel scale and vector relu.
 */
#ifndef CONV2D_SMALL_KERNEL_TILE_OPS_H
#define CONV2D_SMALL_KERNEL_TILE_OPS_H

#include "conv2d_small_kernel_common.h"
#include "conv2d_small_kernel_block_ctx.h"
#include "spatial/spatial_scheme.h"

namespace small_conv {

struct Load3DCtx {
    uint64_t xtBase;
    uint64_t xmTmp;
};

// ----------------------------------------------------------------- L1 layout
// Full dual-output offset chain (mirrors ops-nn SetupL1Layout / SetupL1SplitLayout).
template <typename BlockCtx, typename TilingData>
struct ComputeL1LayoutTool {
    __aicore__ inline L1Layout operator()(const TilingData& tiling, uint32_t al1BufCount, uint32_t al1ElemPerBuf,
                                          uint32_t bl1ElemCount) const
    {
        L1Layout l1;
        uint32_t al1BufBytes = al1ElemPerBuf * sizeof(typename BlockCtx::FmapT);
        l1.al1ElemCount = al1BufCount * al1ElemPerBuf;
        l1.al1BufBytes = al1BufBytes;
        l1.bl1ElemCount = bl1ElemCount;
        l1.bl1OffBytes = al1BufCount * al1BufBytes;

        uint32_t afterBl1 = l1.bl1OffBytes + l1.bl1ElemCount * sizeof(typename BlockCtx::WeightT);
        l1.biasL1OffBytes = static_cast<uint32_t>(AlignB(afterBl1, ADDR_ALIGN_SIZE));
        uint32_t afterBias = tiling.hasBias ?
                                 l1.biasL1OffBytes + tiling.singleCoreCo * sizeof(typename BlockCtx::L0cT) :
                                 afterBl1;
        l1.scale0L1OffBytes = static_cast<uint32_t>(AlignB(afterBias, ADDR_ALIGN_SIZE));
        uint32_t afterScale0 = tiling.quantMode0 == static_cast<uint8_t>(QuantModeType::VECTOR_QUANT) ?
                                   l1.scale0L1OffBytes + tiling.singleCoreCo * sizeof(uint64_t) :
                                   afterBias;
        l1.reluWeight0L1OffBytes = static_cast<uint32_t>(AlignB(afterScale0, ADDR_ALIGN_SIZE));
        uint32_t afterReluWeight0 = tiling.reluMode0 == static_cast<uint8_t>(ReluMode::VECTOR_RELU) ?
                                        l1.reluWeight0L1OffBytes + tiling.singleCoreCo * sizeof(float) :
                                        afterScale0;
        l1.scale1L1OffBytes = static_cast<uint32_t>(AlignB(afterReluWeight0, ADDR_ALIGN_SIZE));
        uint32_t afterScale1 = tiling.quantMode1 == static_cast<uint8_t>(QuantModeType::VECTOR_QUANT) ?
                                   l1.scale1L1OffBytes + tiling.singleCoreCo * sizeof(uint64_t) :
                                   afterReluWeight0;
        l1.reluWeight1L1OffBytes = static_cast<uint32_t>(AlignB(afterScale1, ADDR_ALIGN_SIZE));
        l1.endBytes = tiling.reluMode1 == static_cast<uint8_t>(ReluMode::VECTOR_RELU) ?
                          l1.reluWeight1L1OffBytes + tiling.singleCoreCo * sizeof(float) :
                          afterScale1;
        return l1;
    }
};

// -------------------------------------------------------------- Global setup
// outBaseOff is the mode-specific output offset within the batch/channel slab
// (M-mode: mIdxStart; HW-mode: hoIdxStart*wout + woIdxStart), provided by the
// spatial scheme.
template <typename BlockCtx, typename TilingData>
struct SetupGmTool {
    __aicore__ inline void operator()(
        const BlockCtx& blockCtx, const TilingData& tiling, GM_ADDR x, GM_ADDR filter, GM_ADDR bias, GM_ADDR y,
        const ExtendParams* ep, uint32_t groupIter, uint32_t groupCoutOff, uint32_t groupChanOff, uint32_t outBaseOff,
        GlobalTensor<typename BlockCtx::FmapT>& fmapGm, GlobalTensor<typename BlockCtx::WeightT>& filterGm,
        GlobalTensor<typename BlockCtx::BiasT>& biasGm, GlobalTensor<typename BlockCtx::OutputT>& outputGm,
        GlobalTensor<typename BlockCtx::Output1T>& output1Gm, GlobalTensor<uint64_t>& scale0Gm,
        GlobalTensor<uint64_t>& scale1Gm, GlobalTensor<float>& reluWeight0Gm, GlobalTensor<float>& reluWeight1Gm) const
    {
        using FmapT = typename BlockCtx::FmapT;
        using WeightT = typename BlockCtx::WeightT;
        using BiasT = typename BlockCtx::BiasT;
        using OutputT = typename BlockCtx::OutputT;
        using Output1T = typename BlockCtx::Output1T;

        uint64_t hwIn = static_cast<uint64_t>(tiling.hin) * tiling.win;
        uint64_t hwOut = static_cast<uint64_t>(tiling.hout) * tiling.wout;

        // fmap: batch + group channel offset (hi/wi offsets applied by Load tools).
        uint64_t groupFmapOff;
        if constexpr (BlockCtx::isNHWCin) {
            groupFmapOff = groupChanOff;
        } else {
            groupFmapOff = static_cast<uint64_t>(groupChanOff) * hwIn;
        }
        uint64_t batchFmapOff = static_cast<uint64_t>(blockCtx.batchStart) * tiling.cin * hwIn;
        fmapGm.SetGlobalBuffer(reinterpret_cast<__gm__ FmapT*>(x) + batchFmapOff + groupFmapOff, tiling.cin * hwIn);

        // weight: FRACTAL_Z uses [k1, n1, GN0, GK0]; NCHW uses nIdx * cin * khkw.
        if constexpr (BlockCtx::weightFmt == ConvFormat::NCHW) {
            uint32_t khkw = tiling.kh * tiling.kw;
            uint64_t wtOff = static_cast<uint64_t>(groupCoutOff + blockCtx.nIdxStart) * tiling.singleCoreCi * khkw;
            filterGm.SetGlobalBuffer(reinterpret_cast<__gm__ WeightT*>(filter) + wtOff,
                                     tiling.singleCoreCo * tiling.singleCoreCi * khkw);
        } else {
            uint32_t n1PerGroup = static_cast<uint32_t>(AlignB(blockCtx.groupCoutStep, GN0)) / GN0;
            uint64_t weightOneIterNz = static_cast<uint64_t>(blockCtx.k1Total) * n1PerGroup * GN0 * BlockCtx::GK0;
            uint64_t weightOffset = (static_cast<uint64_t>(blockCtx.groupIdx) * blockCtx.groupBlockStride + groupIter) *
                                    weightOneIterNz;
            filterGm.SetGlobalBuffer(reinterpret_cast<__gm__ WeightT*>(filter) + weightOffset,
                                     static_cast<uint64_t>(blockCtx.k1Total) * n1PerGroup * GN0 * BlockCtx::GK0);
        }

        // bias: nIdx + group cout offset.
        if (tiling.hasBias) {
            biasGm.SetGlobalBuffer(reinterpret_cast<__gm__ BiasT*>(bias) + groupCoutOff + blockCtx.nIdxStart,
                                   blockCtx.actualCo);
        }
        // scale / relu GM for the fixpipe (scalar quant / relu read from GM).
        if (tiling.quantMode0 == static_cast<uint8_t>(QuantModeType::VECTOR_QUANT)) {
            scale0Gm.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(ep->scale0) + groupCoutOff + blockCtx.nIdxStart,
                                     blockCtx.actualCo);
        } else if (tiling.quantMode0 == static_cast<uint8_t>(QuantModeType::SCALAR_QUANT)) {
            scale0Gm.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(ep->scale0));
        }
        if (tiling.reluMode0 == static_cast<uint8_t>(ReluMode::VECTOR_RELU)) {
            reluWeight0Gm.SetGlobalBuffer(
                reinterpret_cast<__gm__ float*>(ep->reluWeight0) + groupCoutOff + blockCtx.nIdxStart,
                blockCtx.actualCo);
        } else if (tiling.reluMode0 == static_cast<uint8_t>(ReluMode::SCALAR_RELU)) {
            reluWeight0Gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(ep->reluWeight0));
        }
        if (tiling.dualOutput) {
            if (tiling.quantMode1 == static_cast<uint8_t>(QuantModeType::VECTOR_QUANT)) {
                scale1Gm.SetGlobalBuffer(
                    reinterpret_cast<__gm__ uint64_t*>(ep->scale1) + groupCoutOff + blockCtx.nIdxStart,
                    blockCtx.actualCo);
            } else if (tiling.quantMode1 == static_cast<uint8_t>(QuantModeType::SCALAR_QUANT)) {
                scale1Gm.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(ep->scale1));
            }
            if (tiling.reluMode1 == static_cast<uint8_t>(ReluMode::VECTOR_RELU)) {
                reluWeight1Gm.SetGlobalBuffer(
                    reinterpret_cast<__gm__ float*>(ep->reluWeight1) + groupCoutOff + blockCtx.nIdxStart,
                    blockCtx.actualCo);
            } else if (tiling.reluMode1 == static_cast<uint8_t>(ReluMode::SCALAR_RELU)) {
                reluWeight1Gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(ep->reluWeight1));
            }
        }

        // output: batch + nIdx + m/ho/wo.
        uint64_t effectiveCout = (tiling.strideWriteDstStride != 0) ? tiling.total_shape_cout : tiling.cout;
        uint64_t batchOutOff = static_cast<uint64_t>(blockCtx.curBatchIdx) * hwOut * effectiveCout;
        if constexpr (BlockCtx::isNHWCout) {
            uint64_t nOutOff = groupCoutOff + blockCtx.nIdxStart;
            uint64_t outOff = static_cast<uint64_t>(outBaseOff);
            outputGm.SetGlobalBuffer(
                reinterpret_cast<__gm__ OutputT*>(y) + batchOutOff + nOutOff + outOff * effectiveCout,
                blockCtx.actualCo);
        } else {
            uint64_t nOutOff = (groupCoutOff + blockCtx.nIdxStart) * hwOut;
            uint64_t outOff = static_cast<uint64_t>(outBaseOff);
            outputGm.SetGlobalBuffer(reinterpret_cast<__gm__ OutputT*>(y) + batchOutOff + nOutOff + outOff,
                                     blockCtx.actualCo * hwOut);
        }
        if (tiling.dualOutput) {
            if constexpr (BlockCtx::isNHWCout) {
                uint64_t nOutOff = groupCoutOff + blockCtx.nIdxStart;
                uint64_t outOff = static_cast<uint64_t>(outBaseOff);
                output1Gm.SetGlobalBuffer(
                    reinterpret_cast<__gm__ Output1T*>(ep->y1) + batchOutOff + nOutOff + outOff * effectiveCout,
                    blockCtx.actualCo);
            } else {
                uint64_t nOutOff = (groupCoutOff + blockCtx.nIdxStart) * hwOut;
                uint64_t outOff = static_cast<uint64_t>(outBaseOff);
                output1Gm.SetGlobalBuffer(reinterpret_cast<__gm__ Output1T*>(ep->y1) + batchOutOff + nOutOff + outOff,
                                          blockCtx.actualCo * hwOut);
            }
        }
    }
};

// ------------------------------------------------------------ weight L1 load
template <typename BlockCtx, typename TilingData>
struct LoadWeightL1Tool {
    __aicore__ inline void operator()(const BlockCtx& blockCtx, const TilingData& tiling,
                                      const GlobalTensor<typename BlockCtx::WeightT>& filterGm,
                                      LocalTensor<typename BlockCtx::WeightT>& bl1) const
    {
        using WeightT = typename BlockCtx::WeightT;
        uint32_t bl1ElemCount = blockCtx.k1Total * blockCtx.n1PerCore * GN0 * BlockCtx::GK0;
        if constexpr (BlockCtx::weightFmt == ConvFormat::NCHW) {
            uint32_t khkw = tiling.kh * tiling.kw;
            if (khkw == 1) {
                Nd2NzParams copyParams;
                copyParams.ndNum = 1;
                copyParams.nValue = blockCtx.actualCo;
                copyParams.dValue = tiling.singleCoreCi;
                copyParams.srcNdMatrixStride = 0;
                copyParams.srcDValue = tiling.singleCoreCi;
                copyParams.dstNzC0Stride = blockCtx.n1PerCore * GN0;
                copyParams.dstNzNStride = 1;
                copyParams.dstNzMatrixStride = 0;
                DataCopy(bl1, filterGm[0], copyParams);
            } else {
                Dn2NzParams copyParams;
                copyParams.dnNum = blockCtx.actualCo;
                copyParams.nValue = khkw;
                copyParams.dValue = tiling.singleCoreCi;
                copyParams.srcDnMatrixStride = tiling.coutOffsetBlock;
                copyParams.srcDValue = khkw;
                copyParams.dstNzC0Stride = khkw * blockCtx.n1PerCore * GN0;
                copyParams.dstNzNStride = blockCtx.n1PerCore * GN0;
                copyParams.dstNzMatrixStride = BlockCtx::GK0;
                DataCopy(bl1, filterGm[0], copyParams);
            }
        } else {
            if (tiling.nDim == 1 &&
                blockCtx.n1PerCore == static_cast<uint32_t>(AlignB(blockCtx.groupCoutStep, GN0)) / GN0) {
                DataCopy(bl1, filterGm[0], static_cast<int32_t>(bl1ElemCount));
            } else {
                uint32_t n1Start = blockCtx.nIdx * tiling.singleCoreCo / GN0;
                uint32_t tileBytes = GN0 * BlockCtx::GK0 * sizeof(WeightT);
                uint32_t srcGmOff = n1Start * GN0 * BlockCtx::GK0;
                uint32_t n1PerGroup = static_cast<uint32_t>(AlignB(blockCtx.groupCoutStep, GN0)) / GN0;
                uint16_t blkLen = static_cast<uint16_t>((blockCtx.n1PerCore * tileBytes) / 32);
                uint16_t srcGap = static_cast<uint16_t>(((n1PerGroup - blockCtx.n1PerCore) * tileBytes) / 32);
                DataCopyParams cp(static_cast<uint16_t>(blockCtx.k1Total), blkLen, srcGap, 0);
                DataCopy(bl1, filterGm[srcGmOff], cp);
            }
        }
    }
};

// ------------------------------------------------------- bias/scale L1 load
template <typename BlockCtx, typename TilingData>
struct LoadBiasScaleL1Tool {
    __aicore__ inline void operator()(const BlockCtx& blockCtx, const TilingData& tiling,
                                      const GlobalTensor<typename BlockCtx::BiasT>& biasGm,
                                      const GlobalTensor<uint64_t>& scale0Gm, const GlobalTensor<uint64_t>& scale1Gm,
                                      const GlobalTensor<float>& reluWeight0Gm,
                                      const GlobalTensor<float>& reluWeight1Gm, const L1Layout& l1) const
    {
        using BiasT = typename BlockCtx::BiasT;
        if (tiling.hasBias && blockCtx.actualCo > 0) {
            LocalTensor<BiasT> biasL1(TPosition::A1, l1.biasL1OffBytes, tiling.singleCoreCo);
            LoadChannelWiseL1FullLoad<BiasT>(biasL1, biasGm[0], blockCtx.actualCo);
            SetFlag<HardEvent::MTE2_MTE1>(EVT_BIAS_DONE);
        }
        if (tiling.quantMode0 == static_cast<uint8_t>(QuantModeType::VECTOR_QUANT)) {
            LocalTensor<uint64_t> scale0L1(TPosition::A1, l1.scale0L1OffBytes, tiling.singleCoreCo);
            LoadChannelWiseL1FullLoad<uint64_t>(scale0L1, scale0Gm[0], blockCtx.actualCo);
        }
        if (tiling.reluMode0 == static_cast<uint8_t>(ReluMode::VECTOR_RELU)) {
            LocalTensor<float> reluWeight0L1(TPosition::A1, l1.reluWeight0L1OffBytes, tiling.singleCoreCo);
            LoadChannelWiseL1FullLoad<float>(reluWeight0L1, reluWeight0Gm[0], blockCtx.actualCo);
        }
        if (tiling.dualOutput) {
            if (tiling.quantMode1 == static_cast<uint8_t>(QuantModeType::VECTOR_QUANT)) {
                LocalTensor<uint64_t> scale1L1(TPosition::A1, l1.scale1L1OffBytes, tiling.singleCoreCo);
                LoadChannelWiseL1FullLoad<uint64_t>(scale1L1, scale1Gm[0], blockCtx.actualCo);
            }
            if (tiling.reluMode1 == static_cast<uint8_t>(ReluMode::VECTOR_RELU)) {
                LocalTensor<float> reluWeight1L1(TPosition::A1, l1.reluWeight1L1OffBytes, tiling.singleCoreCo);
                LoadChannelWiseL1FullLoad<float>(reluWeight1L1, reluWeight1Gm[0], blockCtx.actualCo);
            }
        }
    }
};

template <typename BlockCtx, typename TilingData>
struct LoadBiasToBTTool {
    using L0cT = typename BlockCtx::L0cT;
    using BiasT = typename BlockCtx::BiasT;
    __aicore__ inline void operator()(const TilingData& tiling, const L1Layout& l1, uint32_t nOff, uint32_t curN) const
    {
        constexpr uint32_t BT_ALIGN = 64;
        // N tiles reuse BT address zero; finish the preceding MMAD before
        // replacing the bias it reads as cmatrix source.
        if (tiling.nL0 < tiling.nBL1) {
            SetFlag<HardEvent::M_MTE1>(EVT_BIAS_DONE);
            WaitFlag<HardEvent::M_MTE1>(EVT_BIAS_DONE);
        }
        uint32_t btElemNum = static_cast<uint32_t>(AlignB(curN * sizeof(L0cT), BT_ALIGN)) / sizeof(L0cT);
        LocalTensor<L0cT> biasBT(TPosition::C2, 0, btElemNum);
        LocalTensor<BiasT> biasL1src(TPosition::A1, l1.biasL1OffBytes, tiling.singleCoreCo);
        uint32_t blkCnt = static_cast<uint32_t>(AlignB(curN * sizeof(BiasT), BT_ALIGN)) / 32;
        DataCopyParams cp(1, static_cast<uint16_t>(blkCnt), 0, 0);
#if defined(__DAV_35_FAMILY__)
        if constexpr (AscendC::IsSameType<typename BlockCtx::WeightT, half>::value) {
            cp.fixShiftVal = FIX_SHIFT_VAL_LEN_A16W16 - tiling.fixedShiftValue;
        }
#endif
        DataCopy(biasBT, biasL1src[nOff], cp);
    }
};

// ------------------------------------------------------------- Load3D setup
// Block-level Load3D context (full-load keeps the whole block window in L1).
// posM is the mode-specific block start position (M-mode: mIdxStart % wout;
// HW-mode: woIdxStart), provided by the spatial scheme; the block window pads
// already carry the mode-specific values.
template <typename BlockCtx, typename TilingData>
struct SetupLoad3DBaseTool {
    __aicore__ inline Load3DCtx operator()(const BlockCtx& blockCtx, const TilingData& tiling, const FmapRange& r,
                                           uint32_t posM) const
    {
        uint8_t padList[PAD_LIST_NUM] = {static_cast<uint8_t>(r.padLeft), static_cast<uint8_t>(r.padRight),
                                         static_cast<uint8_t>(r.padTop), static_cast<uint8_t>(PAD_BOTTOM_VALUE)};
        Load3DSetFMatrixCal(r.curHi, r.orgWin, padList);
        Load3DSetPaddingCal(tiling.offsetx);

        Load3DCtx c;
        c.xtBase = ((static_cast<uint64_t>(tiling.strideW) & MASK_6) << 0) |
                   ((static_cast<uint64_t>(tiling.kw) & MASK_8) << KERNELW_OFFSET) |
                   ((static_cast<uint64_t>(tiling.kh) & MASK_8) << KERNELH_OFFSET) |
                   ((static_cast<uint64_t>(tiling.strideH) & MASK_6) << STRIDEH_OFFSET) |
                   ((static_cast<uint64_t>(tiling.kw) & NINTH_BIT_MASK) << KERNELW_HIGHEST_BIT_OFFSET) |
                   ((static_cast<uint64_t>(tiling.kh) & NINTH_BIT_MASK) << KERNELH_HIGHEST_BIT_OFFSET) |
                   ((static_cast<uint64_t>(tiling.dilationW) & MASK_8) << DILATIONW_OFFSET) |
                   ((static_cast<uint64_t>(tiling.dilationH) & MASK_8) << DILATIONH_OFFSET) |
                   ((static_cast<uint64_t>(blockCtx.cinAligned) & MASK_16) << CIN_OFFSET);
        c.xmTmp = ((static_cast<uint64_t>(AlignB(blockCtx.actualM, GM0)) & MASK_16) << MSTEP_OFFSET) |
                  ((static_cast<uint64_t>(posM) & MASK_16) << POSM_OFFSET);
        return c;
    }
};

// ------------------------------------------------------------------ L0 loads
template <typename BlockCtx>
struct LoadAL0Tool {
    __aicore__ inline void operator()(const LocalTensor<typename BlockCtx::FmapT>& al1,
                                      LocalTensor<typename BlockCtx::FmapT>& al0, uint32_t kOff, uint32_t curK,
                                      const Load3DCtx& c) const
    {
        using FmapT = typename BlockCtx::FmapT;
        uint64_t xm = static_cast<uint64_t>(curK) | (static_cast<uint64_t>(kOff) << POSK_OFFSET) | c.xmTmp;
        Load3DBitModeParam param;
        param.SetConfig0(xm);
        param.SetConfig1(c.xtBase);
        LoadData<TPosition::A2, TPosition::A1, FmapT>(al0, al1, param);
    }
};

template <typename BlockCtx>
struct LoadBL0Tool {
    __aicore__ inline void operator()(const BlockCtx& blockCtx, LocalTensor<typename BlockCtx::WeightT>& bl0,
                                      LocalTensor<typename BlockCtx::WeightT>& bl1, uint32_t kOff, uint32_t curK,
                                      uint32_t nOff, uint32_t curN) const
    {
        using WeightT = typename BlockCtx::WeightT;
        uint32_t kStep = static_cast<uint32_t>(CeilDiv(curK, BlockCtx::GK0));
        uint32_t nLoad = static_cast<uint32_t>(AlignB(curN, GN0));
        uint32_t mStep = nLoad / GN0;
        Load2DBitModeParam param;
        param.SetMStartPosition(nOff / GN0);
        param.SetKStartPosition(kOff / BlockCtx::GK0);
        param.SetMStep(static_cast<uint16_t>(mStep));
        param.SetKStep(static_cast<uint16_t>(kStep));
        param.SetSrcStride(static_cast<int32_t>(blockCtx.n1PerCore));
        param.SetDstStride(static_cast<uint16_t>(mStep));
        param.SetIfTranspose(false);
        LoadData<TPosition::B2, TPosition::B1, WeightT>(bl0, bl1, param);
    }
};

// ------------------------------------------------------------------ Mmad
// fixShiftVal applies to every small-kernel variant under __DAV_35_FAMILY__
// (the legacy fm-partload used a separate arch id that is equivalent to this
// family macro).
template <typename BlockCtx, typename TilingData>
struct InitMmadParamsTool {
    __aicore__ inline void operator()(const TilingData& tiling, MmadParams& mmadParams, uint32_t m, uint32_t n) const
    {
#if defined(__DAV_35_FAMILY__)
        if constexpr (AscendC::IsSameType<typename BlockCtx::FmapT, half>::value) {
            mmadParams.fixShiftVal = tiling.fixedShiftValue;
        }
#endif
        mmadParams.m = m;
        mmadParams.n = n;
        mmadParams.cmatrixInitVal = !(tiling.hasBias);
        mmadParams.cmatrixSource = (tiling.hasBias != 0);
    }
};

// ------------------------------------------------------------------ CopyOut
template <typename BlockCtx, typename TilingData>
struct CopyOutTool {
    template <typename OutputT, uint64_t FixpipeIdx>
    __aicore__ inline QuantMode_t GetQuantPreInt32(const TilingData& tiling) const
    {
        if constexpr (AscendC::IsSameType<OutputT, half>::value) {
            if constexpr (AscendC::IsSameType<typename BlockCtx::WeightT, int8_t>::value) {
                uint8_t quantMode = (FixpipeIdx == 0) ? tiling.quantMode0 : tiling.quantMode1;
                bool isVector = (quantMode == static_cast<uint8_t>(QuantModeType::VECTOR_QUANT));
                return isVector ? QuantMode_t::VDEQF16 : QuantMode_t::DEQF16;
            } else if constexpr (AscendC::IsSameType<typename BlockCtx::WeightT, half>::value) {
                return QuantMode_t::DEQF16;
            }
        } else if constexpr (AscendC::IsSameType<OutputT, int8_t>::value) {
            if constexpr (AscendC::IsSameType<typename BlockCtx::WeightT, int8_t>::value) {
                uint8_t quantMode = (FixpipeIdx == 0) ? tiling.quantMode0 : tiling.quantMode1;
                bool isVector = (quantMode == static_cast<uint8_t>(QuantModeType::VECTOR_QUANT));
                return isVector ? QuantMode_t::VREQ8 : QuantMode_t::REQ8;
            } else if constexpr (AscendC::IsSameType<typename BlockCtx::WeightT, half>::value) {
                return QuantMode_t::REQ8;
            }
        }
        return QuantMode_t::DEQF16;
    }

    template <typename OutputT, uint64_t FixpipeIdx>
    __aicore__ inline QuantMode_t GetQuantPreFp32() const
    {
        if constexpr (AscendC::IsSameType<OutputT, float>::value) {
            return QuantMode_t::NoQuant;
        } else if constexpr (AscendC::IsSameType<OutputT, bfloat16_t>::value) {
            return QuantMode_t::F322BF16;
        }
        return QuantMode_t::F322F16;
    }

    template <typename OutputT, uint64_t FixpipeIdx>
    __aicore__ inline void DoCopyOut(const BlockCtx& blockCtx, const TilingData& tiling, GM_ADDR yAddr,
                                     LocalTensor<typename BlockCtx::L0cT>& cl0, const L1Layout& l1, uint32_t outOffset,
                                     uint32_t curM, uint32_t curMAlign, uint32_t actualCo, uint32_t fpDnNum,
                                     uint32_t fpDstDnStride, uint32_t nOff, uint32_t groupCoutOff,
                                     const GlobalTensor<uint64_t>& scale0Gm, const GlobalTensor<uint64_t>& scale1Gm,
                                     const GlobalTensor<float>& reluWeight0Gm,
                                     const GlobalTensor<float>& reluWeight1Gm) const
    {
        using L0cT = typename BlockCtx::L0cT;
        constexpr bool IsOutput0 = (FixpipeIdx == 0);
        uint8_t reluMode = IsOutput0 ? tiling.reluMode0 : tiling.reluMode1;
        uint8_t quantMode = IsOutput0 ? tiling.quantMode0 : tiling.quantMode1;
        uint32_t scaleL1OffBytes = IsOutput0 ? l1.scale0L1OffBytes : l1.scale1L1OffBytes;
        uint32_t reluWeightL1OffBytes = IsOutput0 ? l1.reluWeight0L1OffBytes : l1.reluWeight1L1OffBytes;
        // Unit-flag protocol: only the LAST fixpipe of a unit may flip; with
        // dual output that is the output-1 fixpipe (legacy behaviour).
        uint32_t unitFlag;
        if constexpr (IsOutput0) {
            unitFlag = (tiling.dualOutput == 1) ? UNIT_FLAG_ENABLE_ONLY : UNIT_FLAG_ENABLE_WITH_FLIP;
        } else {
            unitFlag = UNIT_FLAG_ENABLE_WITH_FLIP;
        }

        constexpr CO2Layout Layout = BlockCtx::isNHWCout ? CO2Layout::ROW_MAJOR : CO2Layout::COLUMN_MAJOR;
        uint64_t hwOut = static_cast<uint64_t>(tiling.hout) * tiling.wout;
        uint64_t effectiveCout = (tiling.strideWriteDstStride != 0) ? tiling.total_shape_cout : tiling.cout;
        uint64_t batchOutOff = static_cast<uint64_t>(blockCtx.curBatchIdx) * hwOut * effectiveCout;

        GlobalTensor<OutputT> outputGm;
        FixpipeParamsArch3510<Layout> fixpipeParams;
#if defined(__DAV_35_FAMILY__)
        if constexpr (AscendC::IsSameType<typename BlockCtx::WeightT, half>::value) {
            fixpipeParams.fixShiftVal = FIX_SHIFT_VAL_LEN_A16W16 - tiling.fixedShiftValue;
        }
#endif
        fixpipeParams.nSize = actualCo;
        fixpipeParams.mSize = curM;
        fixpipeParams.srcStride = curMAlign;
        fixpipeParams.reluEn = reluMode != 0;
        if constexpr (AscendC::IsSameType<L0cT, float>::value) {
            fixpipeParams.quantPre = GetQuantPreFp32<OutputT, FixpipeIdx>();
        } else {
            fixpipeParams.quantPre = GetQuantPreInt32<OutputT, FixpipeIdx>(tiling);
        }
        fixpipeParams.unitFlag = unitFlag;

        uint64_t outOff;
        if constexpr (BlockCtx::isNHWCout) {
            uint64_t nOutOff = groupCoutOff + blockCtx.nIdxStart + nOff;
            outOff = static_cast<uint64_t>(outOffset) * effectiveCout;
            outputGm.SetGlobalBuffer(reinterpret_cast<__gm__ OutputT*>(yAddr) + batchOutOff + nOutOff);
            fixpipeParams.dstStride = static_cast<uint32_t>(effectiveCout);
            fixpipeParams.params.ndNum = fpDnNum;
            fixpipeParams.params.srcNdStride = (fpDnNum == 1) ? 0 : AlignB(curM, GM0);
            fixpipeParams.params.dstNdStride = fpDstDnStride * effectiveCout;
        } else {
            uint64_t nOutOff = (groupCoutOff + blockCtx.nIdxStart + nOff) * hwOut;
            outOff = static_cast<uint64_t>(outOffset);
            outputGm.SetGlobalBuffer(reinterpret_cast<__gm__ OutputT*>(yAddr) + batchOutOff + nOutOff);
            fixpipeParams.dstStride = static_cast<uint32_t>(hwOut);
            fixpipeParams.params.dnNum = fpDnNum;
            fixpipeParams.params.srcNzMatrixStride = (fpDnNum == 1) ? (curMAlign * CeilDiv(actualCo, GM0)) :
                                                                      AlignB(curM, GM0);
            fixpipeParams.params.dstDnMatrixStride = fpDstDnStride;
            fixpipeParams.params.srcNzC0Stride = 1;
        }

#if defined(__DAV_35_FAMILY__)
        fixpipeParams.preReluMode = static_cast<ReluMode>(reluMode);
        if (reluMode == static_cast<uint8_t>(ReluMode::SCALAR_RELU)) {
            float m2 = IsOutput0 ? reluWeight0Gm.GetValue(0) : reluWeight1Gm.GetValue(0);
            fixpipeParams.reluScalar = reinterpret_cast<uint64_t&>(m2);
        } else if (reluMode == static_cast<uint8_t>(ReluMode::VECTOR_RELU)) {
            LocalTensor<float> reluWeightL1(TPosition::A1, reluWeightL1OffBytes, tiling.singleCoreCo);
            fixpipeParams.vectorRelu = reluWeightL1[nOff].GetPhyAddr();
        }

        if (quantMode == static_cast<uint8_t>(QuantModeType::VECTOR_QUANT)) {
            LocalTensor<uint64_t> scaleL1(TPosition::A1, scaleL1OffBytes, tiling.singleCoreCo);
            if constexpr (Layout == CO2Layout::ROW_MAJOR) {
                Fixpipe<OutputT, L0cT, CFG_ROW_MAJOR>(outputGm[outOff], cl0, scaleL1[nOff], fixpipeParams);
            } else {
                Fixpipe<OutputT, L0cT, CFG_COLUMN_MAJOR>(outputGm[outOff], cl0, scaleL1[nOff], fixpipeParams);
            }
        } else if (quantMode == static_cast<uint8_t>(QuantModeType::SCALAR_QUANT)) {
            fixpipeParams.deqScalar = IsOutput0 ? scale0Gm.GetValue(0) : scale1Gm.GetValue(0);
            if constexpr (AscendC::IsSameType<typename BlockCtx::WeightT, half>::value) {
                if constexpr (Layout == CO2Layout::ROW_MAJOR) {
                    Fixpipe<OutputT, L0cT, CFG_ROW_MAJOR_FIXED_POINT>(outputGm[outOff], cl0, fixpipeParams);
                } else {
                    Fixpipe<OutputT, L0cT, CFG_COLUMN_MAJOR_FIXED_POINT>(outputGm[outOff], cl0, fixpipeParams);
                }
            } else {
                if constexpr (Layout == CO2Layout::ROW_MAJOR) {
                    Fixpipe<OutputT, L0cT, CFG_ROW_MAJOR>(outputGm[outOff], cl0, fixpipeParams);
                } else {
                    Fixpipe<OutputT, L0cT, CFG_COLUMN_MAJOR>(outputGm[outOff], cl0, fixpipeParams);
                }
            }
        } else {
            fixpipeParams.deqScalar = FLOAT_ONE_FIXED_POINT;
            if constexpr (Layout == CO2Layout::ROW_MAJOR) {
                Fixpipe<OutputT, L0cT, CFG_ROW_MAJOR_FIXED_POINT>(outputGm[outOff], cl0, fixpipeParams);
            } else {
                Fixpipe<OutputT, L0cT, CFG_COLUMN_MAJOR_FIXED_POINT>(outputGm[outOff], cl0, fixpipeParams);
            }
        }
#else
        fixpipeParams.deqScalar = FLOAT_ONE_FIXED_POINT;
        if constexpr (Layout == CO2Layout::ROW_MAJOR) {
            Fixpipe<OutputT, L0cT, CFG_ROW_MAJOR>(outputGm[outOff], cl0, fixpipeParams);
        } else {
            Fixpipe<OutputT, L0cT, CFG_COLUMN_MAJOR>(outputGm[outOff], cl0, fixpipeParams);
        }
#endif
    }
};

} // namespace small_conv

#endif // CONV2D_SMALL_KERNEL_TILE_OPS_H
