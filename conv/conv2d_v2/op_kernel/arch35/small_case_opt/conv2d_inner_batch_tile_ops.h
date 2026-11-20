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
 * Inner-batch tile ops: the fold-group variants that differ substantially from
 * the shared tools (fmap L1 load and the folded fixpipe); the small-delta tools
 * (SetupLoad3DBase / LoadAL0 / InitMmadParams) live in the shared tile_ops and
 * are reused from here. The folded flow keeps the L0C unit-flag state machine
 * disabled: both Mmad and Fixpipe set unitFlag = UNIT_FLAG_DISABLE explicitly
 * (legacy parity).
 */
#ifndef CONV2D_INNER_BATCH_TILE_OPS_H
#define CONV2D_INNER_BATCH_TILE_OPS_H

#include "conv2d_small_kernel_common.h"
#include "conv2d_small_kernel_block_ctx.h"
#include "conv2d_small_kernel_tile_ops.h"

namespace small_conv {

// GM->L1 load of one fold group in a single instruction: 1x1 merges the curIb
// maps into ONE big NZ matrix via the destination strides (the source stays
// innerBatch separate matrices — NCHW rows are contiguous only inside one channel
// plane, NHWC rows are cin apart); MULTI keeps per-map matrices.
template <typename BlockCtx, typename TilingData, int8_t InnerBatchMode>
struct LoadAL1InnerBatchTool {
    static constexpr bool isKernel1x1Ib = InnerBatchMode == static_cast<int8_t>(ConvInnerBatch::KERNEL_1X1_MULTI_BATCH);

    __aicore__ inline void operator()(const BlockCtx& blockCtx, const TilingData& tiling,
                                      const GlobalTensor<typename BlockCtx::FmapT>& fmapGm,
                                      LocalTensor<typename BlockCtx::FmapT>& al1, uint32_t curHi, uint32_t hiLoadOff,
                                      uint32_t curWi) const
    {
        const uint32_t innerBatch = blockCtx.curIb;
        const uint32_t rows = curHi * curWi;
        const uint32_t curCin = blockCtx.groupCinStep;
        if constexpr (BlockCtx::isNHWCin) {
            Nd2NzParams copyParams;
            copyParams.ndNum = innerBatch;
            copyParams.nValue = rows;
            copyParams.dValue = curCin;
            copyParams.srcDValue = static_cast<uint32_t>(tiling.cin);
            copyParams.srcNdMatrixStride = static_cast<uint32_t>(tiling.cin) * tiling.hin * tiling.win;
            copyParams.dstNzNStride = 1;
            if constexpr (isKernel1x1Ib) {
                copyParams.dstNzC0Stride = innerBatch * rows;
                copyParams.dstNzMatrixStride = rows * BlockCtx::GK0Fmap;
            } else {
                copyParams.dstNzC0Stride = rows;
                copyParams.dstNzMatrixStride = blockCtx.cinAlignedFmap * rows;
            }
            uint64_t gmOff = static_cast<uint64_t>(hiLoadOff) * curWi * tiling.cin;
            DataCopy(al1, fmapGm[gmOff], copyParams);
        } else {
            Dn2NzParams copyParams;
            copyParams.dnNum = innerBatch;
            copyParams.nValue = rows;
            copyParams.dValue = curCin;
            copyParams.srcDValue = static_cast<uint32_t>(tiling.hin * tiling.win);
            copyParams.srcDnMatrixStride = static_cast<uint32_t>(tiling.cin) * tiling.hin * tiling.win;
            copyParams.dstNzNStride = 1;
            if constexpr (isKernel1x1Ib) {
                copyParams.dstNzC0Stride = innerBatch * rows;
                copyParams.dstNzMatrixStride = rows * BlockCtx::GK0Fmap;
            } else {
                copyParams.dstNzC0Stride = rows;
                copyParams.dstNzMatrixStride = blockCtx.cinAlignedFmap * rows;
            }
            uint64_t gmOff = static_cast<uint64_t>(hiLoadOff) * curWi;
            DataCopy(al1, fmapGm[gmOff], copyParams);
        }
    }
};

// Batch-folded Fixpipe: one instruction writes all maps of the group (dn/nd
// dim iterates the maps); strides mirror legacy CopyOutInnerBatch
// (InnerBatchParamsCHW / InnerBatchParamsHWC).
// Inherits the shared CopyOutTool: the quant/relu helpers (GetQuantPreInt32 /
// GetQuantPreFp32) are reused as-is; only DoCopyOut is the folded variant.
template <typename BlockCtx, typename TilingData, int8_t InnerBatchMode>
struct CopyOutInnerBatchTool : CopyOutTool<BlockCtx, TilingData> {
    static constexpr bool isKernel1x1Ib = InnerBatchMode == static_cast<int8_t>(ConvInnerBatch::KERNEL_1X1_MULTI_BATCH);

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

        constexpr CO2Layout Layout = BlockCtx::isNHWCout ? CO2Layout::ROW_MAJOR : CO2Layout::COLUMN_MAJOR;
        uint64_t hwOut = static_cast<uint64_t>(tiling.hout) * tiling.wout;
        uint64_t effectiveCout = (tiling.strideWriteDstStride != 0) ? tiling.total_shape_cout : tiling.cout;
        uint64_t batchOutOff = static_cast<uint64_t>(blockCtx.curBatchIdx) * hwOut * effectiveCout;

        GlobalTensor<OutputT> outputGm;
        FixpipeParamsArch3510<Layout> fixpipeParams;
        fixpipeParams.unitFlag = UNIT_FLAG_DISABLE;
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
            fixpipeParams.quantPre = this->template GetQuantPreFp32<OutputT, FixpipeIdx>();
        } else {
            fixpipeParams.quantPre = this->template GetQuantPreInt32<OutputT, FixpipeIdx>(tiling);
        }

        uint64_t outOff;
        if constexpr (BlockCtx::isNHWCout) {
            uint64_t nOutOff = groupCoutOff + blockCtx.nIdxStart + nOff;
            outOff = static_cast<uint64_t>(outOffset) * effectiveCout;
            outputGm.SetGlobalBuffer(reinterpret_cast<__gm__ OutputT*>(yAddr) + batchOutOff + nOutOff);
            fixpipeParams.dstStride = static_cast<uint32_t>(effectiveCout);
            fixpipeParams.params.ndNum = fpDnNum;
            if constexpr (isKernel1x1Ib) {
                // Merged L0C matrix: srcNzMatrixStride deliberately NOT
                // M0-aligned (legacy InnerBatchParamsHWC); srcStride carries
                // the M0-aligned whole-matrix extent.
                fixpipeParams.srcStride = AlignB(fpDnNum * curM, GM0);
                fixpipeParams.params.srcNdStride = curM;
            } else {
                fixpipeParams.srcStride = curMAlign;
                fixpipeParams.params.srcNdStride = curMAlign * CeilDiv(actualCo, GM0);
            }
            fixpipeParams.params.dstNdStride = fpDstDnStride * effectiveCout;
        } else {
            uint64_t nOutOff = (groupCoutOff + blockCtx.nIdxStart + nOff) * hwOut;
            outOff = static_cast<uint64_t>(outOffset);
            outputGm.SetGlobalBuffer(reinterpret_cast<__gm__ OutputT*>(yAddr) + batchOutOff + nOutOff);
            fixpipeParams.dstStride = static_cast<uint32_t>(hwOut);
            fixpipeParams.params.dnNum = fpDnNum;
            if constexpr (isKernel1x1Ib) {
                // Merged L0C matrix: srcNzMatrixStride deliberately NOT
                // M0-aligned (legacy InnerBatchParamsCHW); srcStride carries
                // the M0-aligned whole-matrix extent.
                fixpipeParams.srcStride = AlignB(fpDnNum * curM, GM0);
                fixpipeParams.params.srcNzMatrixStride = curM;
            } else {
                fixpipeParams.srcStride = curMAlign;
                fixpipeParams.params.srcNzMatrixStride = curMAlign * CeilDiv(actualCo, GM0);
            }
            fixpipeParams.params.srcNzC0Stride = 1;
            fixpipeParams.params.dstDnMatrixStride = fpDstDnStride;
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

#endif // CONV2D_INNER_BATCH_TILE_OPS_H
