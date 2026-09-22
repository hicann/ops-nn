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
 * \file cla_gate_quant_common.h
 * \brief Common building blocks for the ClaGate MX quantization kernel family
 *
 * Naming convention (keep new code consistent with it):
 *   - `LocalTensor<T>` variables are named `<what>Tensor` (e.g. rowScaleTensor);
 *   - `__ubuf__ T *` values are named `<what>Addr`; a pointer kept at a
 *     streaming region's base is `<what>StartAddr`, while the cursors over that
 *     region are `<what>WriteAddr` / `<what>ReadAddr`;
 *   - always name the quantity, never a bare `maxAddr`/`tmpAddr`: use
 *     `absMaxAddr` (amax scratch) vs `maxExpAddr` (max-exponent scratch);
 *   - sizes and counts use `...Num` / `...Count`, never `Addr`;
 *   - a few early helper locals (e.g. `activationInput`, `globalHead0`,
 *     `mergedResult`) predate this rule - rename them only when touching them.
 *
 * The operator fuses a CLA gate with MX quantization:
 *   merged = global_attn * sigmoid(global_gate) + local_attn * sigmoid(local_gate)
 * where global_attn/local_attn have shape [T, N, D] and gate logits [T, N].
 *
 * After the gate merge, the result is quantized into:
 *   - dual_axis_flag=false (single-axis): row_data [T, N*D] and row_scale.
 *   - dual_axis_flag=true (dual-axis): also col_data [T, N*D] and col_scale.
 *
 * In the kernel each AIV core processes disjoint row/column tiles. A tile is a
 * rowCount-by-256-column chunk in UB; rows are processed in tileRowCount-row
 * batches.
 */

#ifndef OPS_NN_CLA_GATE_QUANT_COMMON_H
#define OPS_NN_CLA_GATE_QUANT_COMMON_H

#define FLOAT_OVERFLOW_MODE_CTRL 60

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../inc/platform.h"
#include "../inc/kernel_utils.h"
#include "cla_gate_quant_tiling_key.h"
#include "cla_gate_quant_tilingdata.h"

namespace ClaGateQuant {

using namespace AscendC;

constexpr int64_t DB_BUFFER = 2;
constexpr int64_t DIGIT_TWO = 2;
constexpr int64_t DIGIT_THREE = 3;
constexpr int64_t OUT_ELE_NUM_ONE_BLK = 64;
constexpr uint16_t NAN_CUSTOMIZATION = 0x7f81;

constexpr uint32_t MAX_EXP_FOR_FP32 = 0x7f800000;
// FP4 E2M1 grid alignment keeps the exponent field in place, where a power of
// two 2^k is the bit pattern (k + 127) << 23.  In that form the two scale
// factors of the 2^{e-1} grid are single integer subtracts off the masked
// exponent field.
constexpr uint32_t FP32_EXP_UNIT_BITS = 1u << 23;   // 0x00800000: one exponent step
constexpr uint32_t FP32_EXP_ZERO_BITS = 127u << 23; // 0x3f800000: e == 0, the subnormal clamp
constexpr uint16_t NAN_FOR_FP8_E8M0 = 0x00ff;
constexpr uint16_t SPECIAL_VALUE_E2M1 = 0x00ff;
constexpr uint16_t SPECIAL_VALUE_E1M2 = 0x007f;
constexpr uint16_t SPECIAL_EXP_THRESHOLD = 0x0040;
constexpr int16_t SHR_NUM_FOR_BF16 = 7;
constexpr int16_t SHR_NUM_FOR_FP32 = 23;
constexpr uint16_t FP4_E2M1_BF16_MAX_EXP = 0x0100;
constexpr uint16_t BF16_EXP_BIAS = 0x7f00;
constexpr uint16_t FP8_E4M3_MAX_EXP = 0x0400;
constexpr uint16_t FP8_E5M2_MAX_EXP = 0x0780;
constexpr int32_t FP32_BIAS = 127;
constexpr int32_t FP32_BIAS_NEG = -127;
constexpr int32_t NEG_ONE = -1;
constexpr float FOUR = 4.0;
constexpr float ONE_FOURTH = 0.25;
constexpr int32_t NEG_ZERO = 0x80000000;
constexpr uint32_t FP8_E5M2_INV_MAX = 0x37924925; // 1 / max finite FP8_E5M2 value
constexpr uint32_t FP8_E4M3_INV_MAX = 0x3b124925; // 1 / max finite FP8_E4M3 value
constexpr uint16_t EXP_MASK_BF16 = 0x7f80;
constexpr uint16_t EXP_MASK_FP16 = 0x7c00;

// CuBLAS-like scale algorithm (scaleAlg=1) constants
constexpr uint16_t ABS_MASK_FOR_16BIT = 0x7fff;
constexpr uint32_t MAN_MASK_FLOAT = 0x007fffff;
constexpr uint32_t FP32_EXP_BIAS_CUBLAS = 0x00007f00;
constexpr uint32_t MAX_EXP_FOR_FP8_IN_FP32 = 0x000000ff;
constexpr uint32_t EXP_254 = 0x000000fe;
constexpr uint32_t HALF_FOR_MAN = 0x00400000;
constexpr uint32_t VF_LEN_FP32 = platform::GetVRegSize() / sizeof(float);
constexpr uint32_t VF_LEN_B16 = platform::GetVRegSize() / sizeof(half);
constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t DOUBLE_BLOCK_SIZE = 64;
constexpr int64_t ONCE_ROW_LEN = 256;
// D=128: one ONCE_ROW_LEN (256 element) tile/segment carries two heads, so a
// single head spans ONCE_ROW_LEN / 2 = 128 elements, which is exactly one b16
// vector register on this platform.
constexpr int64_t D128_HEAD_ELEMS = ONCE_ROW_LEN / DIGIT_TWO;
constexpr int64_t UB_BLOCK_SIZE = platform::GetUbBlockSize();
constexpr uint32_t ROW_SCALE_RECIPROCAL_ROW_ELEMS = UB_BLOCK_SIZE / sizeof(uint16_t);
// DAV_3510 BrcbCommonImpl advances a DIST_E2B_B16 source by this many elements.
constexpr uint32_t E2B_B16_SOURCE_ELEMS = BRCB_BROADCAST_NUMBER;
constexpr uint32_t COL_SCALE_STORE_COUNT = ONCE_ROW_LEN / VF_LEN_FP32;
constexpr uint32_t COL_SCALE_STORE_STRIDE_BYTES = DIGIT_TWO * VF_LEN_FP32;
constexpr uint32_t COL_SCALE_ONE_STORE_BYTES = DIGIT_TWO * platform::GetVRegSize();
constexpr uint32_t COL_SCALE_BUFFER_BYTES = COL_SCALE_ONE_STORE_BYTES +
                                            (COL_SCALE_STORE_COUNT - 1U) * COL_SCALE_STORE_STRIDE_BYTES;

static_assert(ONCE_ROW_LEN % VF_LEN_FP32 == 0, "A full row must contain an integral number of vector registers");
static_assert(ROW_SCALE_RECIPROCAL_ROW_ELEMS * sizeof(uint16_t) == UB_BLOCK_SIZE,
              "Each row-scale reciprocal row must occupy exactly one UB block");
static_assert(E2B_B16_SOURCE_ELEMS * sizeof(uint16_t) <= UB_BLOCK_SIZE,
              "DIST_E2B_B16 must not read beyond one row-scale reciprocal row");
static_assert(COL_SCALE_BUFFER_BYTES == 896U, "Four overlapping DIST_INTLV_B8 stores must fit in the col-scale buffer");

// Coordinates and valid extents of one tile in the row-block/column-block grid.
struct TileDesc {
    int64_t rowBlockIdx;
    int64_t colBlockIdx;
    int64_t absRowStart;
    int64_t colOffset;
    int64_t calcRow;
    int64_t calcCol;
    int64_t headsPerRowInBlock;
};

static constexpr Reg::CastTrait CAST_X_TO_FP32_ZERO = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                       Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
static constexpr Reg::CastTrait CAST_X_TO_FP32_ONE = {Reg::RegLayout::ONE, Reg::SatMode::UNKNOWN,
                                                      Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
static constexpr Reg::CastTrait CAST_HALF_TO_BF16 = {Reg::RegLayout::UNKNOWN, Reg::SatMode::UNKNOWN,
                                                     Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_TRUNC};

static constexpr Reg::CastTrait CAST_FP32_TO_FP16_BF16 = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                          Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
static constexpr AscendC::Reg::CastTrait CAST_32_TO_80 = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::SAT,
                                                          AscendC::Reg::MaskMergeMode::ZEROING,
                                                          AscendC::RoundMode::CAST_RINT};
static constexpr AscendC::Reg::CastTrait CAST_32_TO_81 = {AscendC::Reg::RegLayout::ONE, AscendC::Reg::SatMode::SAT,
                                                          AscendC::Reg::MaskMergeMode::ZEROING,
                                                          AscendC::RoundMode::CAST_RINT};
static constexpr AscendC::Reg::CastTrait CAST_32_TO_82 = {AscendC::Reg::RegLayout::TWO, AscendC::Reg::SatMode::SAT,
                                                          AscendC::Reg::MaskMergeMode::ZEROING,
                                                          AscendC::RoundMode::CAST_RINT};
static constexpr AscendC::Reg::CastTrait CAST_32_TO_83 = {AscendC::Reg::RegLayout::THREE, AscendC::Reg::SatMode::SAT,
                                                          AscendC::Reg::MaskMergeMode::ZEROING,
                                                          AscendC::RoundMode::CAST_RINT};

} // namespace ClaGateQuant

// VF definitions depend on the constants and traits declared above.
#include "vf/compute.h"

namespace ClaGateQuant {

template <typename xDtype, typename rowDataDtype, AscendC::RoundMode roundMode, uint64_t scaleAlg>
class ClaGateQuantCommonBase {
public:
    __aicore__ inline ClaGateQuantCommonBase(const ClaGateQuantTilingData* tilingData, TPipe* pipe)
        : tilingData_(tilingData), pipe_(pipe){};
    template <uint64_t dualAxisFlag>
    __aicore__ inline void Init(GM_ADDR globalOut, GM_ADDR localOut, GM_ADDR globalGate, GM_ADDR localGate,
                                GM_ADDR rowData, GM_ADDR rowScale, GM_ADDR colData, GM_ADDR colScale);

protected:
    static constexpr Reg::CastTrait castTraitBF16toFp4 = {Reg::RegLayout::ZERO, Reg::SatMode::SAT,
                                                          Reg::MaskMergeMode::ZEROING, roundMode};
    static constexpr Reg::CastTrait castTraitFp32toBF16 = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                           Reg::MaskMergeMode::ZEROING, roundMode};
    static constexpr Reg::CastTrait castTraitFp32toYdtype = {Reg::RegLayout::ZERO, Reg::SatMode::SAT,
                                                             Reg::MaskMergeMode::ZEROING, roundMode};

    template <uint64_t dualAxisFlag>
    __aicore__ inline void InitParams();
    __aicore__ inline void CopyInClaGate(const TileDesc& tile);
    __aicore__ inline void ComputeClaGateFullTile(__ubuf__ xDtype* globalAddr, __ubuf__ xDtype* localAddr,
                                                  __ubuf__ xDtype* globalGateAddr, __ubuf__ xDtype* localGateAddr,
                                                  __ubuf__ xDtype* outputAddr, uint16_t calcRows);
    __aicore__ inline void ComputeClaGateD128TwoHeads(__ubuf__ xDtype* globalAddr, __ubuf__ xDtype* localAddr,
                                                      __ubuf__ xDtype* globalGateAddr, __ubuf__ xDtype* localGateAddr,
                                                      __ubuf__ xDtype* outputAddr, uint16_t calcRows);
    __aicore__ inline void FillZero(__ubuf__ xDtype* outputAddr, uint32_t elementCount);
    __aicore__ inline void ComputeRowDataToFp8(uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr,
                                               __ubuf__ uint16_t* rowScaleReciprocalAddr,
                                               __ubuf__ uint8_t* rowDataAddr);
    __aicore__ inline void ComputeRowDataToFp4(uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr,
                                               __ubuf__ uint16_t* rowScaleReciprocalAddr, __ubuf__ uint8_t* rowDataAddr,
                                               uint16_t reciprocalStride = ROW_SCALE_RECIPROCAL_ROW_ELEMS);

    // Tiling
    const ClaGateQuantTilingData* tilingData_;

    // UB queues and buffers
    TPipe* pipe_;
    // Double-buffered input queues: activation halves and gate scalars are split
    // so their lifetimes/synchronization can be managed independently.
    TQue<QuePosition::VECIN, DB_BUFFER> activationInQueue_;
    TQue<QuePosition::VECIN, DB_BUFFER> gateInQueue_;
    TBuf<TPosition::VECCALC> mergedClaGateBuf_;
    // One full FP32 register slot per row. Sigmoid is calculated only in lane 0;
    // StoreAlign + LoadAlign(DIST_BRC_B32) expands that scalar for the D lanes.
    TBuf<TPosition::VECCALC> sigmoidBroadcastBuf_;
    // Output queues are shared by both axes. Dual-axis column results follow
    // the row results in the same queue frames:
    //   rowDataQueue_  : row_data bytes, then col_data bytes (dual axis)
    //   rowScaleQueue_ : row_scale bytes, then col_scale bytes (dual axis)
    // This keeps the two double-buffered input and output queues within the
    // eight available event IDs.
    TQue<QuePosition::VECOUT, DB_BUFFER> rowDataQueue_;
    TQue<QuePosition::VECOUT, DB_BUFFER> rowScaleQueue_;
    TBuf<TPosition::VECCALC> rowScaleReciprocalBuf_;
    TBuf<TPosition::VECCALC> colScaleReciprocalBuf_;

    // Global memory
    GlobalTensor<xDtype> globalOutGm_;
    GlobalTensor<xDtype> localOutGm_;
    GlobalTensor<xDtype> globalGateGm_;
    GlobalTensor<xDtype> localGateGm_;
    GlobalTensor<uint8_t> rowDataGm_;
    GlobalTensor<uint8_t> rowScaleGm_;
    GlobalTensor<uint8_t> colDataGm_;
    GlobalTensor<uint8_t> colScaleGm_;

    // Derived geometry
    int64_t blockIdx_ = 0;
    int64_t tileRowLength_ = 0;          // Fixed 256-element tile width.
    int64_t colTailSize_ = 0;            // Valid width of the final dual-axis column tile.
    int64_t tileRowCount_ = 0;           // 64 rows for dual-axis; batch capacity for single-axis.
    int64_t rowScaleElementsPerRow_ = 0; // Row-scale elements per input row.
    uint32_t outputDtypeInvMax_ = 0;
    uint16_t outputDtypeMaxExp_ = 0;
    int64_t halfBufferElemCount_ = 0; // Elements in each global/local input half.
    int64_t rowLength_ = 0;
    int64_t headCount_ = 0;
    int64_t headSize_ = 0;
    int64_t headsPerTileRow_ = 0; // Heads per row in a 256-element tile.
    int64_t gateSideOffset_ = 0;  // Offset between global and local gate regions.

    // Byte offsets of the column outputs inside the merged VECOUT queues.
    // Only used by the dual axis; zero for the single axis.
    int64_t colDataOffsetInOutQueue_ = 0;
    int64_t colScaleOffsetInOutQueue_ = 0;

    int64_t ubBlockElemCountB16_ = UB_BLOCK_SIZE / sizeof(xDtype);
    int64_t ubBlockElemCountB8_ = UB_BLOCK_SIZE / sizeof(uint8_t);
};

template <typename xDtype, typename rowDataDtype, AscendC::RoundMode roundMode, uint64_t scaleAlg>
template <uint64_t dualAxisFlag>
__aicore__ inline void ClaGateQuantCommonBase<xDtype, rowDataDtype, roundMode, scaleAlg>::InitParams()
{
    blockIdx_ = GetBlockIdx();
    tileRowLength_ = ONCE_ROW_LEN;
    colTailSize_ = tilingData_->colTailSize;
    // Dual-axis keeps its 64-row algorithmic tile.  Single-axis receives the
    // UB-derived batch capacity from host tiling instead of assuming 64.
    if constexpr (dualAxisFlag == TPL_SINGLE_AXIS) {
        tileRowCount_ = tilingData_->batchSegmentCapacity > 0 ? tilingData_->batchSegmentCapacity : DOUBLE_BLOCK_SIZE;
    } else {
        tileRowCount_ = DOUBLE_BLOCK_SIZE;
    }

    rowLength_ = tilingData_->rowLength;
    headCount_ = tilingData_->headCount;
    headSize_ = tilingData_->headDim;
    headsPerTileRow_ = tileRowLength_ / headSize_;
    int64_t logicalGateCount = headsPerTileRow_ * tileRowCount_;
    if constexpr (dualAxisFlag == TPL_SINGLE_AXIS) {
        // Single-axis batch sigmoid is generated a full FP32 VF (64 scalars)
        // at a time.  Keep each global/local side on a complete-VF boundary:
        // this satisfies the MTE alignment requirement and prevents the final
        // unmasked VF store from overwriting the opposite side.
        gateSideOffset_ = ops::CeilDiv(logicalGateCount, static_cast<int64_t>(VF_LEN_FP32)) * VF_LEN_FP32;
    } else {
        // Dual-axis uses its fixed 64-row layout without single-axis VF padding.
        gateSideOffset_ = logicalGateCount;
    }

    // Set dtype-specific constants for MX quantization
    if constexpr (IsSameType<rowDataDtype, fp8_e4m3fn_t>::value) {
        outputDtypeMaxExp_ = FP8_E4M3_MAX_EXP;
        outputDtypeInvMax_ = FP8_E4M3_INV_MAX;
    } else if constexpr (IsSameType<rowDataDtype, fp8_e5m2_t>::value) {
        outputDtypeMaxExp_ = FP8_E5M2_MAX_EXP;
        outputDtypeInvMax_ = FP8_E5M2_INV_MAX;
    } else if constexpr (IsSameType<rowDataDtype, fp4x2_e2m1_t>::value) {
        outputDtypeMaxExp_ = FP4_E2M1_BF16_MAX_EXP;
    }
}

template <typename xDtype, typename rowDataDtype, AscendC::RoundMode roundMode, uint64_t scaleAlg>
template <uint64_t dualAxisFlag>
__aicore__ inline void ClaGateQuantCommonBase<xDtype, rowDataDtype, roundMode, scaleAlg>::Init(
    GM_ADDR globalOut, GM_ADDR localOut, GM_ADDR globalGate, GM_ADDR localGate, GM_ADDR rowData, GM_ADDR rowScale,
    GM_ADDR colData, GM_ADDR colScale)
{
#if (__NPU_ARCH__ == 3510)
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
#endif

    InitParams<dualAxisFlag>();

    // Set up Global Memory tensors
    globalOutGm_.SetGlobalBuffer((__gm__ xDtype*)(globalOut));
    localOutGm_.SetGlobalBuffer((__gm__ xDtype*)(localOut));
    globalGateGm_.SetGlobalBuffer((__gm__ xDtype*)(globalGate));
    localGateGm_.SetGlobalBuffer((__gm__ xDtype*)(localGate));
    globalOutGm_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    localOutGm_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    rowDataGm_.SetGlobalBuffer((__gm__ uint8_t*)(rowData));
    rowScaleGm_.SetGlobalBuffer((__gm__ uint8_t*)(rowScale));
    if constexpr (dualAxisFlag == TPL_DUAL_AXIS) {
        colDataGm_.SetGlobalBuffer((__gm__ uint8_t*)(colData));
        colScaleGm_.SetGlobalBuffer((__gm__ uint8_t*)(colScale));
    }

    halfBufferElemCount_ = tileRowLength_ * tileRowCount_;
    int64_t inBufferSize = halfBufferElemCount_ * static_cast<int64_t>(sizeof(xDtype));

    // axis=-1 scale buffer
    int64_t mxScale1BufferSize = tileRowCount_ * UB_BLOCK_SIZE;

    // axis=-2 1/scale (xDtype sized for bf16 reciprocal storage)
    int64_t tmpScale2BufferSize = tileRowLength_ * DIGIT_TWO * static_cast<int64_t>(sizeof(xDtype));
    if constexpr (dualAxisFlag == TPL_SINGLE_AXIS) {
        // Single-axis reuses this buffer for the compact max-exp stream:
        // eight uint16 values are produced for every 1x256 tile.
        int64_t compactScaleCount = tileRowCount_ * (tileRowLength_ / BLOCK_SIZE);
        // Phase 2 loads up to one complete B16 VF (128 maxima), so reserve the
        // aligned tail as well even when the logical batch is not a multiple
        // of 16 rows.
        int64_t compactMaxExpSize = ops::CeilDiv(compactScaleCount, static_cast<int64_t>(VF_LEN_B16)) * VF_LEN_B16 *
                                    sizeof(uint16_t);
        tmpScale2BufferSize = tmpScale2BufferSize > compactMaxExpSize ? tmpScale2BufferSize : compactMaxExpSize;
    }

    // Double-buffered global/local activation and gate input frames.
    int64_t actFrameBytes = inBufferSize * DIGIT_TWO;
    int64_t gateFrameBytes = DIGIT_TWO * gateSideOffset_ * static_cast<int64_t>(sizeof(xDtype));
    pipe_->InitBuffer(activationInQueue_, DB_BUFFER, actFrameBytes);
    pipe_->InitBuffer(gateInQueue_, DB_BUFFER, gateFrameBytes);
    pipe_->InitBuffer(mergedClaGateBuf_, inBufferSize);
    // Sigmoid scratch: one gateSideOffset_-element FP32 region per global/local side.
    pipe_->InitBuffer(sigmoidBroadcastBuf_, DIGIT_TWO * gateSideOffset_ * sizeof(float));
    // Merged output queues: the single axis only needs the row regions, the dual
    // axis appends the column regions to the very same double buffer.
    int64_t rowDataFrameBytes = halfBufferElemCount_;
    int64_t rowScaleFrameBytes = mxScale1BufferSize;
    if constexpr (dualAxisFlag == TPL_DUAL_AXIS) {
        int64_t mxScale2BufferSize = tileRowLength_ * DIGIT_THREE;
        if constexpr (scaleAlg != TPL_SCALE_ALG_0) {
            mxScale2BufferSize = COL_SCALE_BUFFER_BYTES;
        }
        colDataOffsetInOutQueue_ = rowDataFrameBytes;
        colScaleOffsetInOutQueue_ = rowScaleFrameBytes;
        rowDataFrameBytes += halfBufferElemCount_; // row_data + col_data
        rowScaleFrameBytes += mxScale2BufferSize;  // row_scale + col_scale
    }
    pipe_->InitBuffer(rowDataQueue_, DB_BUFFER, rowDataFrameBytes);
    pipe_->InitBuffer(rowScaleQueue_, DB_BUFFER, rowScaleFrameBytes);
    pipe_->InitBuffer(rowScaleReciprocalBuf_, mxScale1BufferSize);
    pipe_->InitBuffer(colScaleReciprocalBuf_, tmpScale2BufferSize);
}

template <typename xDtype, typename rowDataDtype, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__aicore__ inline void ClaGateQuantCommonBase<xDtype, rowDataDtype, roundMode, scaleAlg>::CopyInClaGate(
    const TileDesc& tile)
{
    DataCopyPadExtParams<xDtype> pad = {false, 0, 0, 0};
    int64_t outOffset = tile.absRowStart * rowLength_ + tile.colOffset;
    int64_t rowStride = rowLength_;
    int64_t headIdx = tile.colOffset / headSize_;

    // Enqueue gates independently so sigmoid can overlap the activation transfer.
    LocalTensor<xDtype> gateInput = gateInQueue_.template AllocTensor<xDtype>();
    // Copy all heads covered by each row in row-major order. The D=128 VF
    // consumes this interleaved two-head layout directly.
    int64_t gateOffset = tile.absRowStart * headCount_ + headIdx;
    uint32_t headsBytes = static_cast<uint32_t>(tile.headsPerRowInBlock * sizeof(xDtype));
    DataCopyExtParams gateCopy;
    if (headCount_ == tile.headsPerRowInBlock) {
        // The tile covers every head, hence all selected rows are contiguous.
        gateCopy = {1, static_cast<uint32_t>(tile.calcRow) * headsBytes, 0, 0, 0};
    } else {
        gateCopy = {static_cast<uint16_t>(tile.calcRow), headsBytes,
                    static_cast<uint32_t>((headCount_ - tile.headsPerRowInBlock) * sizeof(xDtype)), 0, 0};
    }
    DataCopyPad<xDtype, AscendC::PaddingMode::Compact>(gateInput, globalGateGm_[gateOffset], gateCopy, pad);
    DataCopyPad<xDtype, AscendC::PaddingMode::Compact>(gateInput[gateSideOffset_], localGateGm_[gateOffset], gateCopy,
                                                       pad);
    // Do not wait here. The DeQue() in Process is the consumer-side sync point;
    // with depth=DB_BUFFER, the next MTE2 can overlap the current vector compute.
    gateInQueue_.template EnQue(gateInput);

    LocalTensor<xDtype> activationInput = activationInQueue_.template AllocTensor<xDtype>();
    // DataCopyPad strides: srcStride is in bytes for GM, dstStride is in 32B UB
    // blocks. The UB tile always stores rows at tileRowLength_ (256) elements, so a
    // partial-width tile must leave a destination gap after every copied row.
    uint32_t dstStride = 0;
    if (tile.calcCol < tileRowLength_) {
        dstStride = static_cast<uint32_t>((tileRowLength_ - tile.calcCol) / ubBlockElemCountB16_);
    }
    DataCopyExtParams outCopy = {static_cast<uint16_t>(tile.calcRow),
                                 static_cast<uint32_t>(tile.calcCol * sizeof(xDtype)),
                                 static_cast<uint32_t>((rowStride - tile.calcCol) * sizeof(xDtype)), dstStride, 0};
    DataCopyPad(activationInput, globalOutGm_[outOffset], outCopy, pad);
    DataCopyPad(activationInput[halfBufferElemCount_], localOutGm_[outOffset], outCopy, pad);
    activationInQueue_.template EnQue(activationInput);
}

template <typename xDtype, typename rowDataDtype, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__aicore__ inline void ClaGateQuantCommonBase<xDtype, rowDataDtype, roundMode, scaleAlg>::ComputeClaGateFullTile(
    __ubuf__ xDtype* globalAddr, __ubuf__ xDtype* localAddr, __ubuf__ xDtype* globalGateAddr,
    __ubuf__ xDtype* localGateAddr, __ubuf__ xDtype* outputAddr, uint16_t calcRows)
{
    asc_vf_call<ComputeClaGateFullTileVF<xDtype, rowDataDtype, roundMode, scaleAlg>>(
        globalAddr, localAddr, globalGateAddr, localGateAddr, outputAddr, calcRows, headSize_,
        (__ubuf__ uint8_t*)sigmoidBroadcastBuf_.Get<float>().GetPhyAddr(), gateSideOffset_);
}

template <typename xDtype, typename rowDataDtype, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__aicore__ inline void ClaGateQuantCommonBase<xDtype, rowDataDtype, roundMode, scaleAlg>::ComputeClaGateD128TwoHeads(
    __ubuf__ xDtype* globalAddr, __ubuf__ xDtype* localAddr, __ubuf__ xDtype* globalGateAddr,
    __ubuf__ xDtype* localGateAddr, __ubuf__ xDtype* outputAddr, uint16_t calcRows)
{
    asc_vf_call<ComputeClaGateD128TwoHeadsVF<xDtype, rowDataDtype, roundMode, scaleAlg>>(
        globalAddr, localAddr, globalGateAddr, localGateAddr, outputAddr, calcRows,
        (__ubuf__ uint8_t*)sigmoidBroadcastBuf_.Get<float>().GetPhyAddr(), gateSideOffset_);
}

template <typename xDtype, typename rowDataDtype, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__aicore__ inline void ClaGateQuantCommonBase<xDtype, rowDataDtype, roundMode, scaleAlg>::FillZero(
    __ubuf__ xDtype* outputAddr, uint32_t elementCount)
{
    asc_vf_call<FillZeroVF<xDtype, rowDataDtype, roundMode, scaleAlg>>(outputAddr, elementCount);
}

template <typename xDtype, typename rowDataDtype, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__aicore__ inline void ClaGateQuantCommonBase<xDtype, rowDataDtype, roundMode, scaleAlg>::ComputeRowDataToFp8(
    uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr, __ubuf__ uint16_t* rowScaleReciprocalAddr,
    __ubuf__ uint8_t* rowDataAddr)
{
    asc_vf_call<ComputeRowDataToFp8VF<xDtype, rowDataDtype, roundMode, scaleAlg>>(dataLen, blockCount, xAddr,
                                                                                  rowScaleReciprocalAddr, rowDataAddr);
}

template <typename xDtype, typename rowDataDtype, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__aicore__ inline void ClaGateQuantCommonBase<xDtype, rowDataDtype, roundMode, scaleAlg>::ComputeRowDataToFp4(
    uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr, __ubuf__ uint16_t* rowScaleReciprocalAddr,
    __ubuf__ uint8_t* rowDataAddr, uint16_t reciprocalStride)
{
    asc_vf_call<ComputeRowDataToFp4VF<xDtype, rowDataDtype, roundMode, scaleAlg>>(
        dataLen, blockCount, xAddr, rowScaleReciprocalAddr, rowDataAddr, reciprocalStride);
}

} // namespace ClaGateQuant

#endif // OPS_NN_CLA_GATE_QUANT_COMMON_H
