/* *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file swiglu_mx_quant_with_dual_axis_regbase.h
 * \brief Fused SwiGLU + grouped dual-axis MX quantization kernel for Ascend950 (regbase)
 *
 * Input x is [M, 2N] (left half [M, N] and right half [M, N] for SwiGLU).
 * group_index is [numGroups] cumsum int64 defining group boundaries.
 * Outputs: y1, mx_scale1 (axis=-1 quantization), y2, mx_scale2 (axis=-2 quantization).
 *
 * Each task = (group, columnBlock). Within a task, iterate over splitBlockH-row chunks.
 */

#ifndef OPS_NN_SWIGLU_MX_QUANT_WITH_DUAL_AXIS_REGBASE_H
#define OPS_NN_SWIGLU_MX_QUANT_WITH_DUAL_AXIS_REGBASE_H

#define FLOAT_OVERFLOW_MODE_CTRL 60

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../inc/platform.h"
#include "../inc/kernel_utils.h"
#include "swiglu_mx_quant_with_dual_axis_tiling_key.h"
#include "swiglu_mx_quant_with_dual_axis_tilingdata.h"

namespace SwigluMxQuantWithDualAxis {
using namespace AscendC;

constexpr int64_t DB_BUFFER = 2;
constexpr int64_t DIGIT_TWO = 2;
constexpr int64_t DIGIT_THREE = 3;
constexpr int64_t OUT_ELE_NUM_ONE_BLK = 64;
constexpr uint16_t NAN_CUSTOMIZATION = 0x7f81;

constexpr uint32_t MAX_EXP_FOR_FP32 = 0x7f800000;
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
constexpr uint32_t FP8_E5M2_MAX = 0x37924925;
constexpr uint32_t FP8_E4M3_MAX = 0x3b124925;
constexpr uint16_t EXP_MASK_BF16 = 0x7f80;
constexpr uint16_t EXP_MASK_FP16 = 0x7c00;

// CLUB (scaleAlg=1) specific constants
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
constexpr int64_t UB_BLOCK_SIZE = platform::GetUbBlockSize();
constexpr uint32_t SCALE1_RECIPROCAL_ROW_ELEMS = UB_BLOCK_SIZE / sizeof(uint16_t);
// DAV_3510 BrcbCommonImpl advances a DIST_E2B_B16 source by this many elements.
constexpr uint32_t E2B_B16_SOURCE_ELEMS = BRCB_BROADCAST_NUMBER;
constexpr uint32_t CUBLAS_SCALE2_STORE_COUNT = ONCE_ROW_LEN / VF_LEN_FP32;
constexpr uint32_t CUBLAS_SCALE2_STORE_STRIDE_BYTES = DIGIT_TWO * VF_LEN_FP32;
constexpr uint32_t CUBLAS_SCALE2_ONE_STORE_BYTES = DIGIT_TWO * platform::GetVRegSize();
constexpr uint32_t CUBLAS_SCALE2_BUFFER_BYTES = CUBLAS_SCALE2_ONE_STORE_BYTES +
                                                (CUBLAS_SCALE2_STORE_COUNT - 1U) * CUBLAS_SCALE2_STORE_STRIDE_BYTES;

static_assert(ONCE_ROW_LEN % VF_LEN_FP32 == 0, "A full row must contain an integral number of vector registers");
static_assert(SCALE1_RECIPROCAL_ROW_ELEMS * sizeof(uint16_t) == UB_BLOCK_SIZE,
              "Each scale1 reciprocal row must occupy exactly one UB block");
static_assert(E2B_B16_SOURCE_ELEMS * sizeof(uint16_t) <= UB_BLOCK_SIZE,
              "DIST_E2B_B16 must not read beyond one scale1 reciprocal row");
static_assert(CUBLAS_SCALE2_BUFFER_BYTES == 896U,
              "Four overlapping DIST_INTLV_B8 stores must fit in the scale2 buffer");

static constexpr MicroAPI::CastTrait CAST_X_TO_FP32_ZERO = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                            MicroAPI::MaskMergeMode::ZEROING,
                                                            AscendC::RoundMode::UNKNOWN};
static constexpr MicroAPI::CastTrait CAST_X_TO_FP32_ONE = {MicroAPI::RegLayout::ONE, MicroAPI::SatMode::UNKNOWN,
                                                           MicroAPI::MaskMergeMode::ZEROING,
                                                           AscendC::RoundMode::UNKNOWN};
static constexpr MicroAPI::CastTrait CAST_HALF_TO_BF16 = {MicroAPI::RegLayout::UNKNOWN, MicroAPI::SatMode::UNKNOWN,
                                                          MicroAPI::MaskMergeMode::ZEROING,
                                                          AscendC::RoundMode::CAST_TRUNC};

static constexpr MicroAPI::CastTrait CAST_FP32_TO_FP16_BF16 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                                                               MicroAPI::MaskMergeMode::ZEROING,
                                                               AscendC::RoundMode::CAST_RINT};
static constexpr AscendC::MicroAPI::CastTrait CAST_32_TO_80 = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};
static constexpr AscendC::MicroAPI::CastTrait CAST_32_TO_81 = {
    AscendC::MicroAPI::RegLayout::ONE, AscendC::MicroAPI::SatMode::SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};
static constexpr AscendC::MicroAPI::CastTrait CAST_32_TO_82 = {
    AscendC::MicroAPI::RegLayout::TWO, AscendC::MicroAPI::SatMode::SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};
static constexpr AscendC::MicroAPI::CastTrait CAST_32_TO_83 = {
    AscendC::MicroAPI::RegLayout::THREE, AscendC::MicroAPI::SatMode::SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};

template <typename xDtype, typename y1Dtype, uint64_t mode, AscendC::RoundMode roundMode, uint64_t scaleAlg,
          uint64_t isGroupIdx>
class SwigluMxQuantWithDualAxisBase {
public:
    __aicore__ inline SwigluMxQuantWithDualAxisBase(const SwigluMxQuantWithDualAxisTilingData* tilingData, TPipe* pipe)
        : tilingData_(tilingData), pipe_(pipe){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR groupIndex, GM_ADDR y1, GM_ADDR mxScale1, GM_ADDR y2,
                                GM_ADDR mxScale2);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitParams();
    __aicore__ inline void CopyInSwiglu(int64_t absRowStart, int64_t colOffset, int64_t calcRow, int64_t calcCol);
    __aicore__ inline void ComputeSwiglu(uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* actAddr,
                                         __ubuf__ xDtype* gateAddr, __ubuf__ xDtype* swigluOutAddr,
                                         uint32_t alignDim1Out);
    __aicore__ inline void ComputeSwigluFullTile(__ubuf__ xDtype* actAddr, __ubuf__ xDtype* gateAddr,
                                                 __ubuf__ xDtype* swigluOutAddr);
    __aicore__ inline void PadZeroM(__ubuf__ xDtype* swigluOutAddr, uint32_t num);
    __aicore__ inline void ComputeScaleOcp(uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr,
                                           __ubuf__ uint8_t* mxScale1Addr, __ubuf__ uint16_t* mxScale1ReciprocalAddr,
                                           __ubuf__ uint8_t* mxScale2Addr, __ubuf__ uint16_t* mxScale2ReciprocalAddr);
    __aicore__ inline void ComputeScaleCuBLAS(uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr,
                                              __ubuf__ uint8_t* y1Addr, __ubuf__ uint8_t* mxScale1Addr,
                                              __ubuf__ uint16_t* mxScale1ReciprocalAddr, __ubuf__ uint8_t* mxScale2Addr,
                                              __ubuf__ uint16_t* mxScale2ReciprocalAddr);
    __aicore__ inline void ComputeScaleCuBLASSecondLast(uint16_t dataLen, uint32_t localInvDtypeMax,
                                                        __ubuf__ uint16_t* mxScale2ReciprocalAddr,
                                                        __ubuf__ uint8_t* mxScale2Addr);
    // DAV_3510 MicroAPI models source RegTensor/MaskReg operands as mutable references;
    // these parameters are semantically read-only but cannot be const-qualified.
    __aicore__ inline void ComputeScaleCuBLASForSlot(
        __ubuf__ uint16_t* maxReadAddr, __ubuf__ uint16_t* reciprocalWriteAddr, MicroAPI::RegTensor<uint8_t>& scale8,
        MicroAPI::RegTensor<uint32_t>& invMax, MicroAPI::RegTensor<uint32_t>& manMaskReg,
        MicroAPI::RegTensor<uint32_t>& expMaskReg, MicroAPI::RegTensor<uint32_t>& zero32Reg,
        MicroAPI::RegTensor<uint32_t>& scaleBiasReg, MicroAPI::RegTensor<uint32_t>& nan32Reg,
        MicroAPI::RegTensor<uint32_t>& fp8Nan32Reg, MicroAPI::MaskReg& maskAll, MicroAPI::MaskReg& maskAll32,
        MicroAPI::MaskReg& maskB16);
    __aicore__ inline void ComputeY1ToFP8(uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr,
                                          __ubuf__ uint16_t* mxScale1ReciprocalAddr, __ubuf__ uint8_t* y1Addr);
    __aicore__ inline void ComputeY1ToFP4(uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr,
                                          __ubuf__ uint16_t* mxScale1ReciprocalAddr, __ubuf__ uint8_t* y1Addr);
    __aicore__ inline void ComputeY2ToFP8(uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr,
                                          __ubuf__ uint16_t* mxScale2ReciprocalAddr, __ubuf__ uint8_t* y2Addr);
    __aicore__ inline void ComputeY2ToFP4(uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr,
                                          __ubuf__ uint16_t* mxScale2ReciprocalAddr, __ubuf__ uint8_t* y2Addr);
    __aicore__ inline void ComputeFP4FromHalf(MicroAPI::RegTensor<float>& Reg);
    __aicore__ inline void CopyOut(int64_t yOffset, int64_t scale1OutOffset, int64_t scale2OutOffset,
                                   int64_t blockCount, int64_t blockCountAlign, int64_t dataLen, int64_t dataLenAlign);
    __aicore__ inline void ComputeInterleave(__ubuf__ uint8_t* dstAddr, __ubuf__ uint8_t* src0Addr,
                                             __ubuf__ uint8_t* src1Addr);

protected:
    static constexpr MicroAPI::CastTrait castTraitBF16toFp4 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT,
                                                               MicroAPI::MaskMergeMode::ZEROING, roundMode};
    static constexpr MicroAPI::CastTrait castTraitFp32toBF16 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                                                                MicroAPI::MaskMergeMode::ZEROING, roundMode};
    static constexpr MicroAPI::CastTrait castTraitFp32toYdtype = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT,
                                                                  MicroAPI::MaskMergeMode::ZEROING, roundMode};

private:
    // tiling data
    const SwigluMxQuantWithDualAxisTilingData* tilingData_;

    // pipe & queue & buf
    TPipe* pipe_;
    TQue<QuePosition::VECIN, 1> inQueue_;
    TBuf<TPosition::VECCALC> swigluBuf_;
    TQue<QuePosition::VECOUT, 1> outQueue1_;
    TQue<QuePosition::VECOUT, 1> outQueue2_;
    TQue<QuePosition::VECOUT, 1> mxScaleQueue1_;
    TQue<QuePosition::VECOUT, 1> mxScaleQueue2_;
    TBuf<TPosition::VECCALC> mxScale1ReciprocalBuf_;
    TBuf<TPosition::VECCALC> mxScale2ReciprocalBuf_;

    // gm
    GlobalTensor<xDtype> xGm_;
    GlobalTensor<int64_t> groupIndexGm_;
    GlobalTensor<uint8_t> yGm1_;
    GlobalTensor<uint8_t> mxScaleGm1_;
    GlobalTensor<uint8_t> yGm2_;
    GlobalTensor<uint8_t> mxScaleGm2_;

    // base variables
    int64_t blockIdx_ = 0;
    int64_t ubRowLen_ = 0;        // blockW = 256
    int64_t ubRowLenTail_ = 0;    // dimNTail
    int64_t ubRowCount_ = 0;      // splitBlockH = 64
                                  // MX block size, fixed
    int64_t dimNeg1ScaleNum_ = 0; // ceil(dimN / blockSize)
    uint32_t invDtypeMax_ = 0;
    uint16_t dtypeYMaxExp_ = 0;
    int64_t activateLeft_ = 1;
    int64_t inHalfSize_ = 0; // per-half buffer size in elements (splitBlockH * blockW)
    int64_t dimN_ = 0;

    int64_t oneBlockCountB16_ = UB_BLOCK_SIZE / sizeof(xDtype);
    int64_t oneBlockCountB8_ = UB_BLOCK_SIZE / sizeof(uint8_t);
};

// ============================================================================
// InitParams — cache tiling parameters, set dtype constants
// ============================================================================
template <typename xDtype, typename y1Dtype, uint64_t mode, AscendC::RoundMode roundMode, uint64_t scaleAlg,
          uint64_t isGroupIdx>
__aicore__ inline void
SwigluMxQuantWithDualAxisBase<xDtype, y1Dtype, mode, roundMode, scaleAlg, isGroupIdx>::InitParams()
{
    blockIdx_ = GetBlockIdx();
    ubRowLen_ = ONCE_ROW_LEN; // 固定256
    ubRowLenTail_ = tilingData_->dimNTail;
    ubRowCount_ = DOUBLE_BLOCK_SIZE; // 固定64
    activateLeft_ = tilingData_->activateLeft;
    dimN_ = tilingData_->dimN;

    // Set dtype-specific constants for MX quantization
    if constexpr (IsSameType<y1Dtype, fp8_e4m3fn_t>::value) {
        dtypeYMaxExp_ = FP8_E4M3_MAX_EXP;
        invDtypeMax_ = FP8_E4M3_MAX;
    } else if constexpr (IsSameType<y1Dtype, fp8_e5m2_t>::value) {
        dtypeYMaxExp_ = FP8_E5M2_MAX_EXP;
        invDtypeMax_ = FP8_E5M2_MAX;
    } else if constexpr (IsSameType<y1Dtype, fp4x2_e2m1_t>::value) {
        dtypeYMaxExp_ = FP4_E2M1_BF16_MAX_EXP;
    }
}

// ============================================================================
// Init — set up GlobalTensors, allocate UB buffers
// ============================================================================
template <typename xDtype, typename y1Dtype, uint64_t mode, AscendC::RoundMode roundMode, uint64_t scaleAlg,
          uint64_t isGroupIdx>
__aicore__ inline void SwigluMxQuantWithDualAxisBase<xDtype, y1Dtype, mode, roundMode, scaleAlg, isGroupIdx>::Init(
    GM_ADDR x, GM_ADDR groupIndex, GM_ADDR y1, GM_ADDR mxScale1, GM_ADDR y2, GM_ADDR mxScale2)
{
#if (__NPU_ARCH__ == 3510)
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
#endif

    InitParams();

    // Set up Global Memory tensors
    xGm_.SetGlobalBuffer((__gm__ xDtype*)(x));
    if constexpr (isGroupIdx == static_cast<uint64_t>(1)) {
        groupIndexGm_.SetGlobalBuffer((__gm__ int64_t*)(groupIndex));
        groupIndexGm_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    }
    xGm_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    yGm1_.SetGlobalBuffer((__gm__ uint8_t*)(y1));
    mxScaleGm1_.SetGlobalBuffer((__gm__ uint8_t*)(mxScale1));
    yGm2_.SetGlobalBuffer((__gm__ uint8_t*)(y2));
    mxScaleGm2_.SetGlobalBuffer((__gm__ uint8_t*)(mxScale2));

    inHalfSize_ = ubRowLen_ * ubRowCount_;
    int64_t inBufferSize = inHalfSize_ * static_cast<int64_t>(sizeof(xDtype));

    int64_t mxScale2BufferSize = ubRowLen_ * DIGIT_THREE;
    if constexpr (scaleAlg != TPL_SCALE_ALG_0) {
        mxScale2BufferSize = CUBLAS_SCALE2_BUFFER_BYTES;
    }

    // axis=-1 scale buffer
    int64_t mxScale1BufferSize = ubRowCount_ * UB_BLOCK_SIZE;

    // axis=-2 1/scale (xDtype sized for bf16 reciprocal storage)
    int64_t tmpScale2BufferSize = ubRowLen_ * DIGIT_TWO * static_cast<int64_t>(sizeof(xDtype));

    // Allocate buffers with double buffering (DB_BUFFER=2)
    // inQueue_ holds left + right halves in a single buffer (2 * inBufferSize_)
    pipe_->InitBuffer(inQueue_, DB_BUFFER, inBufferSize * DIGIT_TWO);
    pipe_->InitBuffer(swigluBuf_, inBufferSize);
    pipe_->InitBuffer(outQueue1_, DB_BUFFER, inHalfSize_);
    pipe_->InitBuffer(outQueue2_, DB_BUFFER, inHalfSize_);
    pipe_->InitBuffer(mxScaleQueue1_, DB_BUFFER, mxScale1BufferSize);
    pipe_->InitBuffer(mxScaleQueue2_, DB_BUFFER, mxScale2BufferSize);
    pipe_->InitBuffer(mxScale1ReciprocalBuf_, mxScale1BufferSize);
    pipe_->InitBuffer(mxScale2ReciprocalBuf_, tmpScale2BufferSize); // 后续检查下这些ub大小是否正确
}

// ============================================================================
// Process — outer loop over groups; two core-distribution scenarios:
//   Scenario 1 (nSplitNum < usedCoreNum): groups rotate across core ranges.
//     Each group's blocks are assigned starting from a rotating core offset.
//     e.g. group0: core 0..31, group1: core 32..63, group2: core 0..31, ...
//   Scenario 2 (nSplitNum >= usedCoreNum): each group distributes all its
//     blocks across all usedCoreNum cores (same as dynamic_mx_quant_with_dual_axis).
// ============================================================================
template <typename xDtype, typename y1Dtype, uint64_t mode, AscendC::RoundMode roundMode, uint64_t scaleAlg,
          uint64_t isGroupIdx>
__aicore__ inline void SwigluMxQuantWithDualAxisBase<xDtype, y1Dtype, mode, roundMode, scaleAlg, isGroupIdx>::Process()
{
    int64_t numGroups = 1;
    if constexpr (isGroupIdx == static_cast<uint64_t>(1)) {
        numGroups = tilingData_->numGroups;
    }
    int64_t dimM = tilingData_->dimM;
    // Scale column count per row: ceil(dimN / blockSize)
    dimNeg1ScaleNum_ = (ops::CeilDiv(dimN_, DOUBLE_BLOCK_SIZE)) * DIGIT_TWO;
    int64_t dimNBlockNum = tilingData_->dimNBlockNum;
    int64_t totalCoreNum = tilingData_->usedCoreNum; // kernel 侧 usedCoreNum 即 totalCoreNum
    if (blockIdx_ >= totalCoreNum) {
        return;
    }

    // scale2 row offset (computed per-group, matching grouped_dynamic_mx_quant formula)
    // Rotating core offset for mode=ROTATE
    [[maybe_unused]] int64_t coreRotateOffset = 0; // 每个group从哪个物理核开始用

    for (int64_t g = 0; g < numGroups; g++) {
        // Read group boundaries from cumsum group_index
        int64_t groupStart = 0;
        int64_t groupEnd = dimM;
        if constexpr (isGroupIdx == static_cast<uint64_t>(1)) {
            groupStart = (g > 0) ? groupIndexGm_.GetValue(g - 1) : 0;
            groupEnd = groupIndexGm_.GetValue(g);
        }
        int64_t groupRows = groupEnd - groupStart; // 每个group处理行数
        if (groupRows <= 0) {
            continue;
        }
        // Compute scale2 row offset: each group's scale2 occupies 2 rows in GM
        int64_t scale2GmRowOffset = (groupStart / DOUBLE_BLOCK_SIZE + g) * DIGIT_TWO;

        // Per-group block count
        int64_t dimMSplitG = ops::CeilDiv(groupRows, ubRowCount_);
        int64_t blockCountG = dimMSplitG * dimNBlockNum; // 每个group的所有块个数

        // Determine this core's work for this group
        int64_t loopPerCoreG = 0;
        int64_t blockOffsetG = 0;

        if constexpr (mode == TPL_MODE_ROTATE) { // N方向块数 < totalCoreNum
            // mode=0: dimNBlockNum < totalCoreNum — groups rotate across core ranges
            int64_t usedCoreNumG = (blockCountG < totalCoreNum) ? blockCountG : totalCoreNum; // 这次group用多少个核
            int64_t myCoreInGroup = blockIdx_ - coreRotateOffset;
            if (myCoreInGroup < 0) {
                myCoreInGroup += totalCoreNum; // wrap around
            }

            if (myCoreInGroup < usedCoreNumG) {
                int64_t headCoreNumG = blockCountG % usedCoreNumG;
                int64_t blockPerHeadCoreG = ops::CeilDiv(blockCountG, usedCoreNumG);
                int64_t blockPerTailCoreG = blockCountG / usedCoreNumG;
                if (myCoreInGroup < headCoreNumG) {
                    loopPerCoreG = blockPerHeadCoreG;
                    blockOffsetG = myCoreInGroup * loopPerCoreG;
                } else {
                    loopPerCoreG = blockPerTailCoreG;
                    blockOffsetG = headCoreNumG * blockPerHeadCoreG + (myCoreInGroup - headCoreNumG) * loopPerCoreG;
                }
            }
            // Advance rotating offset for next group
            coreRotateOffset = (coreRotateOffset + blockCountG) % totalCoreNum;
            if (loopPerCoreG == 0) {
                continue;
            }
        } else {
            // mode=1: dimNBlockNum >= totalCoreNum — all cores share each group's blocks
            int64_t usedCoreNumG = totalCoreNum;
            int64_t headCoreNumG = blockCountG % usedCoreNumG;
            int64_t blockPerHeadCoreG = ops::CeilDiv(blockCountG, usedCoreNumG);
            int64_t blockPerTailCoreG = blockCountG / usedCoreNumG;
            if (blockIdx_ < headCoreNumG) {
                loopPerCoreG = blockPerHeadCoreG;
                blockOffsetG = blockIdx_ * loopPerCoreG;
            } else {
                loopPerCoreG = blockPerTailCoreG;
                blockOffsetG = headCoreNumG * blockPerHeadCoreG + (blockIdx_ - headCoreNumG) * loopPerCoreG;
            }
        }

        // Tail sizes for this group
        int64_t dimMTailG = groupRows % ubRowCount_ == 0 ? ubRowCount_ : groupRows % ubRowCount_;

        // Process assigned blocks (same as dynamic_mx_quant_with_dual_axis::Process)
        for (int64_t i = 0; i < loopPerCoreG; i++) { // -2轴循环多少次
            int64_t blockInGroup = blockOffsetG + i;
            int64_t rowBlockIdx = blockInGroup / dimNBlockNum; // M方向
            int64_t colBlockIdx = blockInGroup % dimNBlockNum; // N方向

            int64_t calcCol = (colBlockIdx == dimNBlockNum - 1) ? ubRowLenTail_ : ubRowLen_;
            int64_t calcRow = (rowBlockIdx == dimMSplitG - 1) ? dimMTailG : ubRowCount_;
            int64_t absRowStart = groupStart + rowBlockIdx * ubRowCount_;
            int64_t colOffset = colBlockIdx * ubRowLen_;
            // ---- CopyIn + SwiGLU ----
            CopyInSwiglu(absRowStart, colOffset, calcRow, calcCol);

            LocalTensor<xDtype> xLocal = inQueue_.template DeQue<xDtype>();
            LocalTensor<xDtype> swigluLocal = swigluBuf_.template Get<xDtype>();

            auto actAddr = (__ubuf__ xDtype*)xLocal.GetPhyAddr();
            auto gateAddr = (__ubuf__ xDtype*)xLocal[inHalfSize_].GetPhyAddr();
            if (activateLeft_ == 0) {
                actAddr = (__ubuf__ xDtype*)xLocal[inHalfSize_].GetPhyAddr();
                gateAddr = (__ubuf__ xDtype*)xLocal.GetPhyAddr();
            }
            auto swigluAddr = (__ubuf__ xDtype*)swigluLocal.GetPhyAddr();
            uint32_t alignDim1OutAlgin = ops::CeilDiv(calcCol, DOUBLE_BLOCK_SIZE) * DOUBLE_BLOCK_SIZE;
            uint32_t calcPadRowAlgin = ops::CeilDiv(calcRow, DOUBLE_BLOCK_SIZE) * DOUBLE_BLOCK_SIZE;
            if (calcCol == ubRowLen_ && calcRow == ubRowCount_) {
                ComputeSwigluFullTile(actAddr, gateAddr, swigluAddr);
            } else {
                ComputeSwiglu(static_cast<uint16_t>(calcCol), static_cast<uint16_t>(calcRow), actAddr, gateAddr,
                              swigluAddr, alignDim1OutAlgin);
            }
            inQueue_.template FreeTensor(xLocal);
            if (calcRow % DOUBLE_BLOCK_SIZE != 0) {
                uint32_t calcPadRow = calcPadRowAlgin - calcRow;
                uint32_t allNumZero = calcPadRow * ubRowLen_;
                auto swigluAddrPadZero = (__ubuf__ xDtype*)swigluLocal[calcRow * ubRowLen_].GetPhyAddr();
                PadZeroM(swigluAddrPadZero, allNumZero); // M 方向补0
            }
            // ---- ComputeAll (scale + quantize) — same as dynamic_mx_quant_with_dual_axis ----
            LocalTensor<uint8_t> mxScale1 = mxScaleQueue1_.template AllocTensor<uint8_t>();
            LocalTensor<uint8_t> mxScale2 = mxScaleQueue2_.template AllocTensor<uint8_t>();
            LocalTensor<uint8_t> y1 = outQueue1_.template AllocTensor<uint8_t>();
            LocalTensor<uint8_t> y2 = outQueue2_.template AllocTensor<uint8_t>();
            LocalTensor<uint16_t> mxScale1Reciprocal = mxScale1ReciprocalBuf_.template Get<uint16_t>();
            LocalTensor<uint16_t> mxScale2Reciprocal = mxScale2ReciprocalBuf_.template Get<uint16_t>();

            auto y1Addr = (__ubuf__ uint8_t*)y1.GetPhyAddr();
            auto y2Addr = (__ubuf__ uint8_t*)y2.GetPhyAddr();
            auto ms1Addr = (__ubuf__ uint8_t*)mxScale1.GetPhyAddr();
            auto ms2Addr = (__ubuf__ uint8_t*)mxScale2.GetPhyAddr();
            auto ms1RecipAddr = (__ubuf__ uint16_t*)mxScale1Reciprocal.GetPhyAddr();
            auto ms2RecipAddr = (__ubuf__ uint16_t*)mxScale2Reciprocal.GetPhyAddr();

            int64_t calcBlockLoop = calcPadRowAlgin / BLOCK_SIZE;

            if constexpr (scaleAlg != TPL_SCALE_ALG_0) {
                ComputeScaleCuBLAS(static_cast<uint16_t>(ubRowLen_), static_cast<uint16_t>(calcPadRowAlgin), swigluAddr,
                                   y1Addr, ms1Addr, ms1RecipAddr, ms2Addr, ms2RecipAddr);
            }

            for (int64_t blk = 0; blk < calcBlockLoop; blk++) { // 多少个32，这里一次只计算32行
                int64_t sOff = blk * BLOCK_SIZE * ubRowLen_;
                int64_t yOff = sOff;
                if constexpr (IsSameType<y1Dtype, fp4x2_e2m1_t>::value || IsSameType<y1Dtype, fp4x2_e1m2_t>::value) {
                    yOff = sOff / DIGIT_TWO;
                }
                int64_t s1Off = blk * BLOCK_SIZE * ops::CeilAlign(ubRowLen_ / BLOCK_SIZE, oneBlockCountB8_);
                int64_t s2Off = blk * ubRowLen_;
                int64_t r1Off = blk * BLOCK_SIZE * ops::CeilAlign(ubRowLen_ / BLOCK_SIZE, oneBlockCountB16_);
                int64_t r2Off = blk * ubRowLen_;

                if constexpr (scaleAlg == TPL_SCALE_ALG_0) {
                    ComputeScaleOcp(static_cast<uint16_t>(ubRowLen_), static_cast<uint16_t>(BLOCK_SIZE),
                                    swigluAddr + sOff, ms1Addr + s1Off, ms1RecipAddr + r1Off, ms2Addr + s2Off,
                                    ms2RecipAddr + r2Off);
                }
                if constexpr (IsSameType<y1Dtype, fp4x2_e2m1_t>::value || IsSameType<y1Dtype, fp4x2_e1m2_t>::value) {
                    // 算Y1是交织处理
                    ComputeY1ToFP4(static_cast<uint16_t>(ubRowLen_), static_cast<uint16_t>(BLOCK_SIZE),
                                   swigluAddr + sOff, ms1RecipAddr + r1Off, y1Addr + yOff);
                    // 算y2是按单VF处理，基本块是两个VF长度，所以需要算两次
                    ComputeY2ToFP4(static_cast<uint16_t>(ubRowLen_), static_cast<uint16_t>(BLOCK_SIZE),
                                   swigluAddr + sOff, ms2RecipAddr + r2Off, y2Addr + yOff);
                    ComputeY2ToFP4(static_cast<uint16_t>(ubRowLen_), static_cast<uint16_t>(BLOCK_SIZE),
                                   swigluAddr + sOff + VF_LEN_B16, ms2RecipAddr + r2Off + VF_LEN_B16,
                                   y2Addr + yOff + VF_LEN_B16 / DIGIT_TWO);
                } else {
                    if constexpr (scaleAlg == TPL_SCALE_ALG_0) {
                        ComputeY1ToFP8(static_cast<uint16_t>(ubRowLen_), static_cast<uint16_t>(BLOCK_SIZE),
                                       swigluAddr + sOff, ms1RecipAddr + r1Off, y1Addr + yOff);
                    }
                    ComputeY2ToFP8(static_cast<uint16_t>(ubRowLen_), static_cast<uint16_t>(BLOCK_SIZE),
                                   swigluAddr + sOff, ms2RecipAddr + r2Off, y2Addr + yOff);
                }
            }
            if constexpr (scaleAlg == TPL_SCALE_ALG_0) {
                // Scale2 interleave
                for (int64_t blk = 1; blk < calcBlockLoop; blk += 2) {
                    auto src0Addr = (__ubuf__ uint8_t*)mxScale2[(blk - 1) * ubRowLen_].GetPhyAddr();
                    auto src1Addr = (__ubuf__ uint8_t*)mxScale2[blk * ubRowLen_].GetPhyAddr();
                    auto dstAddr = (__ubuf__ uint8_t*)mxScale2[(blk - 1) * ubRowLen_].GetPhyAddr();
                    ComputeInterleave(dstAddr, src0Addr, src1Addr);
                }
            }
            mxScaleQueue1_.template EnQue(mxScale1);
            outQueue1_.template EnQue(y1);
            mxScaleQueue2_.template EnQue(mxScale2);
            outQueue2_.template EnQue(y2);

            // ---- CopyOut with GM offsets ----
            int64_t yGmOffset = absRowStart * dimN_ + colOffset;
            int64_t scale1GmOffset = absRowStart * dimNeg1ScaleNum_ + colOffset / BLOCK_SIZE;
            int64_t scale2RowIdx = scale2GmRowOffset + rowBlockIdx * ubRowCount_ / BLOCK_SIZE;
            int64_t scale2GmOffset = scale2RowIdx * dimN_ + colOffset * DIGIT_TWO;

            CopyOut(yGmOffset, scale1GmOffset, scale2GmOffset, calcRow, calcPadRowAlgin, calcCol, alignDim1OutAlgin);
        }
    }
}

template <typename xDtype, typename y1Dtype, uint64_t mode, AscendC::RoundMode roundMode, uint64_t scaleAlg,
          uint64_t isGroupIdx>
__aicore__ inline void SwigluMxQuantWithDualAxisBase<xDtype, y1Dtype, mode, roundMode, scaleAlg,
                                                     isGroupIdx>::ComputeInterleave(__ubuf__ uint8_t* dstAddr,
                                                                                    __ubuf__ uint8_t* src0Addr,
                                                                                    __ubuf__ uint8_t* src1Addr)
{
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<uint8_t> src0Reg;
        MicroAPI::RegTensor<uint8_t> src1Reg;
        MicroAPI::MaskReg maskB8 = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::LoadAlign(src0Reg, src0Addr);
        MicroAPI::LoadAlign(src1Reg, src1Addr);
        MicroAPI::StoreAlign<uint8_t, MicroAPI::StoreDist::DIST_INTLV_B8>(dstAddr, src0Reg, src1Reg, maskB8);
    }
}
// ============================================================================
// CopyInSwiglu — load left and right halves of x into a single inQueue_ buffer
// x layout: [M, 2N] where left = x[row, 0..N-1], right = x[row, N..2N-1]
// Buffer layout: [0, inHalfSize_) = left half, [inHalfSize_, 2*inHalfSize_) = right half
// NO padding applied here — zero-padding is done in ComputeSwiglu via masking.
// ============================================================================
template <typename xDtype, typename y1Dtype, uint64_t mode, AscendC::RoundMode roundMode, uint64_t scaleAlg,
          uint64_t isGroupIdx>
__aicore__ inline void SwigluMxQuantWithDualAxisBase<xDtype, y1Dtype, mode, roundMode, scaleAlg,
                                                     isGroupIdx>::CopyInSwiglu(int64_t absRowStart, int64_t colOffset,
                                                                               int64_t calcRow, int64_t calcCol)
{
    int64_t xRowStride = DIGIT_TWO * dimN_; // total columns in x (2N)

    LocalTensor<xDtype> xLocal = inQueue_.template AllocTensor<xDtype>();

    DataCopyExtParams copyParams = {0, 0, 0, 0, 0};
    DataCopyPadExtParams<xDtype> padParams = {false, 0, 0, 0};
    copyParams.blockCount = static_cast<uint16_t>(calcRow);
    copyParams.blockLen = static_cast<uint32_t>(calcCol * static_cast<int64_t>(sizeof(xDtype)));
    copyParams.srcStride = static_cast<uint32_t>((xRowStride - calcCol) * static_cast<int64_t>(sizeof(xDtype)));

    // Left half: x[absRowStart, colOffset .. colOffset + calcCol - 1] → xLocal[0..]
    int64_t leftGmOffset = absRowStart * xRowStride + colOffset;
    DataCopyPad(xLocal, xGm_[leftGmOffset], copyParams, padParams);

    // Right half: x[absRowStart, dimN + colOffset .. dimN + colOffset + calcCol - 1] → xLocal[inHalfSize_..]
    int64_t rightGmOffset = leftGmOffset + dimN_;
    DataCopyPad(xLocal[inHalfSize_], xGm_[rightGmOffset], copyParams, padParams);

    inQueue_.template EnQue(xLocal);
}

// ============================================================================
// ComputeSwiglu — SwiGLU activation: output = SiLU(act) * gate
// SiLU(x) = x / (1 + exp(-x))
//
// actAddr:      activation input (left or right half based on activateLeft)
// gateAddr:     gate input (the other half)
// swigluOutAddr: output buffer (aligned to ubRowLen_ = blockW = 256)
//
// Zero-padding of tail columns is done via masking (zero-mode), NOT in CopyIn.
// ============================================================================
template <typename xDtype, typename y1Dtype, uint64_t mode, AscendC::RoundMode roundMode, uint64_t scaleAlg,
          uint64_t isGroupIdx>
__aicore__ inline void
SwigluMxQuantWithDualAxisBase<xDtype, y1Dtype, mode, roundMode, scaleAlg, isGroupIdx>::ComputeSwiglu(
    uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* actAddr, __ubuf__ xDtype* gateAddr,
    __ubuf__ xDtype* swigluOutAddr, uint32_t alignDim1Out)
{
    uint32_t localOneBlockNum = oneBlockCountB16_;
    uint32_t outAllNum = ubRowLen_;

    // Tail handling: compute masks for zero-padding
    uint16_t dim0VfTimes = blockCount;
    uint32_t localVfLenFp32 = VF_LEN_FP32;
    uint16_t dim1VfTimes = static_cast<uint16_t>(dataLen / VF_LEN_FP32);
    uint32_t dim1Tail = dataLen % localVfLenFp32;
    uint16_t dim1TailTimes = 0;
    uint16_t dim1Tail2 = 0;
    uint32_t mask1Num = 0;
    uint32_t mask2Num = 0;
    uint32_t mask3Num = 0;
    uint32_t alignDim1In = ((dataLen + localOneBlockNum - 1) / localOneBlockNum) * localOneBlockNum;

    __ubuf__ xDtype* actAddr1 = actAddr;
    __ubuf__ xDtype* gateAddr1 = gateAddr;
    __ubuf__ xDtype* swigluAddr1 = swigluOutAddr;
    __ubuf__ xDtype* swigluAddr2 = swigluOutAddr;

    xDtype numZero = 0;
    if (dim1Tail > 0) {
        mask1Num = dim1Tail;
        dim1TailTimes = 1;
        uint32_t padNum = alignDim1Out - dim1VfTimes * localVfLenFp32;
        if (padNum <= localVfLenFp32) {
            mask2Num = padNum;
        } else {
            dim1Tail2 = 1;
            mask2Num = localVfLenFp32;
            mask3Num = padNum - localVfLenFp32;
        }
        int32_t offsetAlign = dim1VfTimes * localVfLenFp32;
        actAddr1 = actAddr + offsetAlign;
        gateAddr1 = gateAddr + offsetAlign;
        swigluAddr1 = swigluOutAddr + offsetAlign;
        swigluAddr2 = swigluOutAddr + offsetAlign + dim1TailTimes * localVfLenFp32;
    }
    float scalarOne = 1.0f;
    float negScalarOne = -1.0f;

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<xDtype> vregAct;
        MicroAPI::RegTensor<xDtype> vregGate;
        MicroAPI::RegTensor<float> vregActF;
        MicroAPI::RegTensor<float> vregGateF;
        MicroAPI::RegTensor<float> negReg;
        MicroAPI::RegTensor<float> expReg;
        MicroAPI::RegTensor<float> addsReg;
        MicroAPI::RegTensor<float> outFReg;
        MicroAPI::RegTensor<xDtype> outTReg;

        MicroAPI::MaskReg mask = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg mask1 = MicroAPI::UpdateMask<float>(mask1Num);
        MicroAPI::MaskReg mask2 = MicroAPI::UpdateMask<float>(mask2Num);
        MicroAPI::MaskReg mask3 = MicroAPI::UpdateMask<xDtype>(mask3Num);

        for (uint16_t dim0vfLoopIdx = 0; dim0vfLoopIdx < dim0VfTimes; dim0vfLoopIdx++) {
            // Full VF iterations (no tail)
            for (uint16_t dim1vfLoopIdx = 0; dim1vfLoopIdx < dim1VfTimes; dim1vfLoopIdx++) {
                MicroAPI::AddrReg srcIdxOffset = MicroAPI::CreateAddrReg<xDtype>(dim0vfLoopIdx, alignDim1In,
                                                                                 dim1vfLoopIdx, VF_LEN_FP32);
                MicroAPI::LoadAlign<xDtype, MicroAPI::LoadDist::DIST_UNPACK_B16>(vregAct, actAddr, srcIdxOffset);
                MicroAPI::LoadAlign<xDtype, MicroAPI::LoadDist::DIST_UNPACK_B16>(vregGate, gateAddr, srcIdxOffset);

                MicroAPI::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(vregActF, vregAct, mask);
                MicroAPI::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(vregGateF, vregGate, mask);
                MicroAPI::Mul(outFReg, vregActF, vregGateF, mask);

                MicroAPI::Muls(negReg, vregActF, negScalarOne, mask);
                MicroAPI::Exp(expReg, negReg, mask);
                MicroAPI::Adds(addsReg, expReg, scalarOne, mask);
                MicroAPI::Div(outFReg, outFReg, addsReg, mask);

                MicroAPI::Cast<xDtype, float, CAST_FP32_TO_FP16_BF16>(outTReg, outFReg, mask);
                MicroAPI::AddrReg outOffset = MicroAPI::CreateAddrReg<xDtype>(dim0vfLoopIdx, outAllNum, dim1vfLoopIdx,
                                                                              VF_LEN_FP32);
                MicroAPI::StoreAlign<xDtype, MicroAPI::StoreDist::DIST_PACK_B32>(swigluOutAddr, outTReg, outOffset,
                                                                                 mask);
            }

            // Tail VF iteration with mask-based zero-padding
            MicroAPI::AddrReg srcIdxOffset1 = MicroAPI::CreateAddrReg<xDtype>(dim0vfLoopIdx, alignDim1In);
            MicroAPI::AddrReg outOffset1 = MicroAPI::CreateAddrReg<xDtype>(dim0vfLoopIdx, outAllNum);

            for (uint16_t aa = 0; aa < dim1TailTimes; aa++) {
                MicroAPI::LoadAlign<xDtype, MicroAPI::LoadDist::DIST_UNPACK_B16>(vregAct, actAddr1, srcIdxOffset1);
                MicroAPI::LoadAlign<xDtype, MicroAPI::LoadDist::DIST_UNPACK_B16>(vregGate, gateAddr1, srcIdxOffset1);

                MicroAPI::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(vregActF, vregAct, mask1);
                MicroAPI::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(vregGateF, vregGate, mask1);
                MicroAPI::Mul(outFReg, vregActF, vregGateF, mask1);

                MicroAPI::Muls(negReg, vregActF, negScalarOne, mask1);
                MicroAPI::Exp(expReg, negReg, mask1);
                MicroAPI::Adds(addsReg, expReg, scalarOne, mask1);
                MicroAPI::Div(outFReg, outFReg, addsReg, mask1);

                // mask2 writes zeros for positions beyond valid data (zero-mode mask)
                MicroAPI::Cast<xDtype, float, CAST_FP32_TO_FP16_BF16>(outTReg, outFReg, mask1);
                MicroAPI::StoreAlign<xDtype, MicroAPI::StoreDist::DIST_PACK_B32>(swigluAddr1, outTReg, outOffset1,
                                                                                 mask2);
            }
            // Additional zero-fill for extra padding positions
            for (uint16_t cc = 0; cc < dim1Tail2; cc++) {
                Duplicate<xDtype>(vregAct, numZero);
                MicroAPI::StoreAlign<xDtype>(swigluAddr2, vregAct, outOffset1, mask3);
            }
        }
    }
}

// 64 x 256 full-tile path: fixed bounds remove runtime tail-mask setup while
// preserving the common path's SwiGLU arithmetic and output layout.
template <typename xDtype, typename y1Dtype, uint64_t mode, AscendC::RoundMode roundMode, uint64_t scaleAlg,
          uint64_t isGroupIdx>
__aicore__ inline void SwigluMxQuantWithDualAxisBase<xDtype, y1Dtype, mode, roundMode, scaleAlg,
                                                     isGroupIdx>::ComputeSwigluFullTile(__ubuf__ xDtype* actAddr,
                                                                                        __ubuf__ xDtype* gateAddr,
                                                                                        __ubuf__ xDtype* swigluOutAddr)
{
    constexpr uint16_t fullTileRows = static_cast<uint16_t>(DOUBLE_BLOCK_SIZE);
    constexpr uint32_t fullTileCols = static_cast<uint32_t>(ONCE_ROW_LEN);
    static_assert(fullTileCols % VF_LEN_FP32 == 0,
                  "The full-tile column count must contain an integral number of vector registers");
    constexpr uint16_t fullTileVfs = static_cast<uint16_t>(fullTileCols / VF_LEN_FP32);
    float scalarOne = 1.0f;
    float negScalarOne = -1.0f;

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<xDtype> vregAct;
        MicroAPI::RegTensor<xDtype> vregGate;
        MicroAPI::RegTensor<float> vregActF;
        MicroAPI::RegTensor<float> vregGateF;
        MicroAPI::RegTensor<float> negReg;
        MicroAPI::RegTensor<float> expReg;
        MicroAPI::RegTensor<float> addsReg;
        MicroAPI::RegTensor<float> outFReg;
        MicroAPI::RegTensor<xDtype> outTReg;
        MicroAPI::MaskReg mask = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();

        for (uint16_t row = 0; row < fullTileRows; row++) {
            for (uint16_t vf = 0; vf < fullTileVfs; vf++) {
                MicroAPI::AddrReg srcOffset = MicroAPI::CreateAddrReg<xDtype>(row, fullTileCols, vf, VF_LEN_FP32);
                MicroAPI::LoadAlign<xDtype, MicroAPI::LoadDist::DIST_UNPACK_B16>(vregAct, actAddr, srcOffset);
                MicroAPI::LoadAlign<xDtype, MicroAPI::LoadDist::DIST_UNPACK_B16>(vregGate, gateAddr, srcOffset);
                MicroAPI::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(vregActF, vregAct, mask);
                MicroAPI::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(vregGateF, vregGate, mask);
                MicroAPI::Mul(outFReg, vregActF, vregGateF, mask);
                MicroAPI::Muls(negReg, vregActF, negScalarOne, mask);
                MicroAPI::Exp(expReg, negReg, mask);
                MicroAPI::Adds(addsReg, expReg, scalarOne, mask);
                MicroAPI::Div(outFReg, outFReg, addsReg, mask);
                MicroAPI::Cast<xDtype, float, CAST_FP32_TO_FP16_BF16>(outTReg, outFReg, mask);
                MicroAPI::AddrReg outOffset = MicroAPI::CreateAddrReg<xDtype>(row, fullTileCols, vf, VF_LEN_FP32);
                MicroAPI::StoreAlign<xDtype, MicroAPI::StoreDist::DIST_PACK_B32>(swigluOutAddr, outTReg, outOffset,
                                                                                 mask);
            }
        }
    }
}

template <typename xDtype, typename y1Dtype, uint64_t mode, AscendC::RoundMode roundMode, uint64_t scaleAlg,
          uint64_t isGroupIdx>
__aicore__ inline void SwigluMxQuantWithDualAxisBase<xDtype, y1Dtype, mode, roundMode, scaleAlg, isGroupIdx>::PadZeroM(
    __ubuf__ xDtype* swigluOutAddr, uint32_t num)
{
    uint16_t times = CeilDivision(num, VF_LEN_B16);
    uint32_t size = num;
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<xDtype> zeroReg;
        AscendC::MicroAPI::MaskReg mask;
        MicroAPI::Duplicate(zeroReg, 0);
        for (uint16_t i = 0; i < times; i++) {
            mask = AscendC::MicroAPI::UpdateMask<xDtype>(size);
            MicroAPI::AddrReg offset = MicroAPI::CreateAddrReg<xDtype>(i, VF_LEN_B16);
            AscendC::MicroAPI::StoreAlign(swigluOutAddr, zeroReg, offset, mask);
        }
    }
}

// ============================================================================
// ComputeScaleOcp — compute MX scales for both axis=-1 and axis=-2
// Identical to dynamic_mx_quant_with_dual_axis ComputeScaleOcp
// ============================================================================
template <typename xDtype, typename y1Dtype, uint64_t mode, AscendC::RoundMode roundMode, uint64_t scaleAlg,
          uint64_t isGroupIdx>
__aicore__ inline void
SwigluMxQuantWithDualAxisBase<xDtype, y1Dtype, mode, roundMode, scaleAlg, isGroupIdx>::ComputeScaleOcp(
    uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr, __ubuf__ uint8_t* mxScale1Addr,
    __ubuf__ uint16_t* mxScale1ReciprocalAddr, __ubuf__ uint8_t* mxScale2Addr,
    __ubuf__ uint16_t* mxScale2ReciprocalAddr)
{
    int64_t localVlForHalfNumber = VF_LEN_B16;
    int64_t localUBBlockSize = UB_BLOCK_SIZE;
    uint16_t localDtypeYMaxExp = dtypeYMaxExp_;

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<xDtype> x0;
        MicroAPI::RegTensor<xDtype> x1;
        MicroAPI::RegTensor<uint16_t> x0ExpFP16;
        MicroAPI::RegTensor<uint16_t> x1ExpFP16;
        MicroAPI::RegTensor<bfloat16_t> x0BF16;
        MicroAPI::RegTensor<bfloat16_t> x1BF16;
        MicroAPI::RegTensor<uint16_t> x0ExpBF16;
        MicroAPI::RegTensor<uint16_t> x1ExpBF16;
        MicroAPI::RegTensor<uint16_t> expMaskBF16;
        MicroAPI::RegTensor<uint16_t> expMaskFP16;
        MicroAPI::RegTensor<uint16_t> expMaxDim1;
        MicroAPI::RegTensor<uint16_t> expMax1Dim2;
        MicroAPI::RegTensor<uint16_t> expMax2Dim2;
        MicroAPI::RegTensor<uint16_t> yMaxExp;
        MicroAPI::RegTensor<uint16_t> nanE8M0;
        MicroAPI::RegTensor<uint16_t> biasE8M0;
        MicroAPI::RegTensor<uint16_t> zero;
        MicroAPI::RegTensor<uint16_t> nanBF16;
        MicroAPI::RegTensor<uint16_t> specialExp;
        MicroAPI::RegTensor<uint16_t> mxScale1B16;
        MicroAPI::RegTensor<uint8_t> mxScale1B8;
        MicroAPI::RegTensor<uint16_t> reversedShareExp1;

        MicroAPI::RegTensor<uint16_t> mxScale2ZeroB16;
        MicroAPI::RegTensor<uint8_t> mxScale2ZeroB8;
        MicroAPI::RegTensor<uint16_t> reversedShareExp2Zero;
        MicroAPI::RegTensor<uint16_t> mxScale2OneB16;
        MicroAPI::RegTensor<uint8_t> mxScale2OneB8;
        MicroAPI::RegTensor<uint16_t> reversedShareExp2One;

        MicroAPI::MaskReg infMask;
        MicroAPI::MaskReg zeroMask;
        MicroAPI::MaskReg invalidDataMask;
        MicroAPI::MaskReg infNanDataMask0;
        MicroAPI::MaskReg infNanDataMask1;
        MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<xDtype, MicroAPI::MaskPattern::ALL>();

        MicroAPI::MaskReg maskB8 = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg maskReduceB8 = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::VL8>();
        MicroAPI::MaskReg maskReduceB16 = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::VL16>();

        MicroAPI::Duplicate(expMaskBF16, EXP_MASK_BF16);
        MicroAPI::Duplicate(expMaskFP16, EXP_MASK_FP16);
        MicroAPI::Duplicate(expMax1Dim2, static_cast<uint16_t>(0));
        MicroAPI::Duplicate(expMax2Dim2, static_cast<uint16_t>(0));
        MicroAPI::Duplicate(yMaxExp, localDtypeYMaxExp);
        MicroAPI::Duplicate(nanE8M0, NAN_FOR_FP8_E8M0);
        MicroAPI::Duplicate(biasE8M0, BF16_EXP_BIAS);
        MicroAPI::Duplicate(zero, static_cast<uint16_t>(0));
        MicroAPI::Duplicate(nanBF16, NAN_CUSTOMIZATION);
        MicroAPI::Duplicate(specialExp, SPECIAL_EXP_THRESHOLD);

        for (uint16_t i = 0; i < blockCount; i++) {
            // Interleaved load: splits blockW bf16/fp16 elements into even/odd halves
            MicroAPI::LoadAlign<xDtype, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_DINTLV_B16>(
                x0, x1, xAddr, ONCE_ROW_LEN);
            if constexpr (IsSameType<xDtype, half>::value) {
                // FP16 path: extract exponent, check INF/NaN, cast to BF16
                MicroAPI::And(x0ExpFP16, (MicroAPI::RegTensor<uint16_t>&)x0, expMaskFP16, maskAll);
                MicroAPI::And(x1ExpFP16, (MicroAPI::RegTensor<uint16_t>&)x1, expMaskFP16, maskAll);
                MicroAPI::Compare<uint16_t, CMPMODE::NE>(infNanDataMask0, x0ExpFP16, expMaskFP16, maskAll);
                MicroAPI::Compare<uint16_t, CMPMODE::NE>(infNanDataMask1, x1ExpFP16, expMaskFP16, maskAll);
                MicroAPI::Cast<bfloat16_t, xDtype, CAST_HALF_TO_BF16>(x0BF16, x0, maskAll);
                MicroAPI::Cast<bfloat16_t, xDtype, CAST_HALF_TO_BF16>(x1BF16, x1, maskAll);
                MicroAPI::And(x0ExpBF16, (MicroAPI::RegTensor<uint16_t>&)x0BF16, expMaskBF16, maskAll);
                MicroAPI::And(x1ExpBF16, (MicroAPI::RegTensor<uint16_t>&)x1BF16, expMaskBF16, maskAll);
                MicroAPI::Select<uint16_t>(x0ExpBF16, x0ExpBF16, expMaskBF16, infNanDataMask0);
                MicroAPI::Select<uint16_t>(x1ExpBF16, x1ExpBF16, expMaskBF16, infNanDataMask1);
            } else {
                // BF16 path: extract exponent bits directly
                MicroAPI::And(x0ExpBF16, (MicroAPI::RegTensor<uint16_t>&)x0, expMaskBF16, maskAll);
                MicroAPI::And(x1ExpBF16, (MicroAPI::RegTensor<uint16_t>&)x1, expMaskBF16, maskAll);
            }

            // axis=-1: max of adjacent pair exponents
            MicroAPI::Max(expMaxDim1, x0ExpBF16, x1ExpBF16, maskAll);
            MicroAPI::ReduceDataBlock<AscendC::Reg::ReduceType::MAX>(expMaxDim1, expMaxDim1, maskAll);

            // axis=-2: accumulate column-wise max exponents across rows
            MicroAPI::Max(expMax1Dim2, expMax1Dim2, x0ExpBF16, maskAll);
            MicroAPI::Max(expMax2Dim2, expMax2Dim2, x1ExpBF16, maskAll);

            // ---- axis=-1 scale computation ----
            MicroAPI::Compare<uint16_t, CMPMODE::NE>(infMask, expMaxDim1, expMaskBF16, maskAll);
            MicroAPI::Compare<uint16_t, CMPMODE::NE>(zeroMask, expMaxDim1, zero, maskAll);
            MicroAPI::Compare<uint16_t, CMPMODE::LE>(invalidDataMask, expMaxDim1, yMaxExp, maskAll);
            MicroAPI::Select<uint16_t>(expMaxDim1, yMaxExp, expMaxDim1, invalidDataMask);

            MicroAPI::Sub(expMaxDim1, expMaxDim1, yMaxExp, maskAll);
            MicroAPI::ShiftRights(mxScale1B16, expMaxDim1, SHR_NUM_FOR_BF16, maskAll);
            MicroAPI::Select<uint16_t>(mxScale1B16, mxScale1B16, nanE8M0, infMask);
            MicroAPI::Select<uint16_t>(mxScale1B16, mxScale1B16, zero, zeroMask);

            MicroAPI::Pack<uint8_t, uint16_t, MicroAPI::HighLowPart::LOWEST>(mxScale1B8, mxScale1B16);
            MicroAPI::StoreAlign<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(mxScale1Addr, mxScale1B8,
                                                                                   UB_BLOCK_SIZE, maskReduceB8);
            // Compute 1/scale
            MicroAPI::Compare<uint16_t, CMPMODE::EQ>(invalidDataMask, expMaxDim1, biasE8M0, maskAll);

            MicroAPI::Sub(reversedShareExp1, biasE8M0, expMaxDim1, maskAll);
            MicroAPI::Select<uint16_t>(reversedShareExp1, reversedShareExp1, nanBF16, infMask);
            MicroAPI::Select<uint16_t>(reversedShareExp1, reversedShareExp1, zero, zeroMask);
            MicroAPI::Select<uint16_t>(reversedShareExp1, specialExp, reversedShareExp1, invalidDataMask);
            MicroAPI::StoreAlign<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                mxScale1ReciprocalAddr, reversedShareExp1, SCALE1_RECIPROCAL_ROW_ELEMS, maskReduceB16);
        }

        // ---- axis=-2 scale computation (interleaved part 1: even rows) ----
        MicroAPI::Compare<uint16_t, CMPMODE::NE>(infMask, expMax1Dim2, expMaskBF16, maskAll);
        MicroAPI::Compare<uint16_t, CMPMODE::NE>(zeroMask, expMax1Dim2, zero, maskAll);
        MicroAPI::Compare<uint16_t, CMPMODE::LE>(invalidDataMask, expMax1Dim2, yMaxExp, maskAll);
        MicroAPI::Select<uint16_t>(expMax1Dim2, yMaxExp, expMax1Dim2, invalidDataMask);
        MicroAPI::Sub(expMax1Dim2, expMax1Dim2, yMaxExp, maskAll);
        MicroAPI::ShiftRights(mxScale2ZeroB16, expMax1Dim2, SHR_NUM_FOR_BF16, maskAll);
        MicroAPI::Select<uint16_t>(mxScale2ZeroB16, mxScale2ZeroB16, nanE8M0, infMask);
        MicroAPI::Select<uint16_t>(mxScale2ZeroB16, mxScale2ZeroB16, zero, zeroMask);

        MicroAPI::Pack<uint8_t, uint16_t, MicroAPI::HighLowPart::LOWEST>(mxScale2ZeroB8, mxScale2ZeroB16);

        MicroAPI::Compare<uint16_t, CMPMODE::EQ>(invalidDataMask, expMax1Dim2, biasE8M0, maskAll); // scale计算结束

        MicroAPI::Sub(reversedShareExp2Zero, biasE8M0, expMax1Dim2, maskAll);
        MicroAPI::Select<uint16_t>(reversedShareExp2Zero, reversedShareExp2Zero, nanBF16, infMask);
        MicroAPI::Select<uint16_t>(reversedShareExp2Zero, reversedShareExp2Zero, zero, zeroMask);
        MicroAPI::Select<uint16_t>(reversedShareExp2Zero, specialExp, reversedShareExp2Zero,
                                   invalidDataMask); // int16 scale 结束

        // ---- axis=-2 scale computation (interleaved part 2: odd rows) ----
        MicroAPI::Compare<uint16_t, CMPMODE::NE>(infMask, expMax2Dim2, expMaskBF16, maskAll);
        MicroAPI::Compare<uint16_t, CMPMODE::NE>(zeroMask, expMax2Dim2, zero, maskAll);
        MicroAPI::Compare<uint16_t, CMPMODE::LE>(invalidDataMask, expMax2Dim2, yMaxExp, maskAll);
        MicroAPI::Select<uint16_t>(expMax2Dim2, yMaxExp, expMax2Dim2, invalidDataMask);
        MicroAPI::Sub(expMax2Dim2, expMax2Dim2, yMaxExp, maskAll);
        MicroAPI::ShiftRights(mxScale2OneB16, expMax2Dim2, SHR_NUM_FOR_BF16, maskAll);
        MicroAPI::Select<uint16_t>(mxScale2OneB16, mxScale2OneB16, nanE8M0, infMask);
        MicroAPI::Select<uint16_t>(mxScale2OneB16, mxScale2OneB16, zero, zeroMask);

        MicroAPI::Pack<uint8_t, uint16_t, MicroAPI::HighLowPart::LOWEST>(mxScale2OneB8, mxScale2OneB16);

        MicroAPI::Compare<uint16_t, CMPMODE::EQ>(invalidDataMask, expMax2Dim2, biasE8M0, maskAll);
        MicroAPI::Sub(reversedShareExp2One, biasE8M0, expMax2Dim2, maskAll);
        MicroAPI::Select<uint16_t>(reversedShareExp2One, reversedShareExp2One, nanBF16, infMask);
        MicroAPI::Select<uint16_t>(reversedShareExp2One, reversedShareExp2One, zero, zeroMask);
        MicroAPI::Select<uint16_t>(reversedShareExp2One, specialExp, reversedShareExp2One, invalidDataMask);

        // Interleaved store: merge even/odd scale and 1/scale for axis=-2
        MicroAPI::StoreAlign<uint8_t, MicroAPI::StoreDist::DIST_INTLV_B8>(mxScale2Addr, mxScale2ZeroB8, mxScale2OneB8,
                                                                          maskB8);
        MicroAPI::StoreAlign<uint16_t, MicroAPI::StoreDist::DIST_INTLV_B16>(
            mxScale2ReciprocalAddr, reversedShareExp2Zero, reversedShareExp2One, maskAll);
    }
}

template <typename xDtype, typename y1Dtype, uint64_t mode, AscendC::RoundMode roundMode, uint64_t scaleAlg,
          uint64_t isGroupIdx>
__aicore__ inline void
SwigluMxQuantWithDualAxisBase<xDtype, y1Dtype, mode, roundMode, scaleAlg, isGroupIdx>::ComputeScaleCuBLASForSlot(
    __ubuf__ uint16_t* maxReadAddr, __ubuf__ uint16_t* reciprocalWriteAddr, MicroAPI::RegTensor<uint8_t>& scale8,
    MicroAPI::RegTensor<uint32_t>& invMax, MicroAPI::RegTensor<uint32_t>& manMaskReg,
    MicroAPI::RegTensor<uint32_t>& expMaskReg, MicroAPI::RegTensor<uint32_t>& zero32Reg,
    MicroAPI::RegTensor<uint32_t>& scaleBiasReg, MicroAPI::RegTensor<uint32_t>& nan32Reg,
    MicroAPI::RegTensor<uint32_t>& fp8Nan32Reg, MicroAPI::MaskReg& maskAll, MicroAPI::MaskReg& maskAll32,
    MicroAPI::MaskReg& maskB16)
{
    MicroAPI::RegTensor<uint16_t> max16Reg;
    MicroAPI::RegTensor<uint32_t> max32Reg;
    MicroAPI::RegTensor<uint32_t> exp32Reg;
    MicroAPI::RegTensor<uint32_t> man32Reg;
    MicroAPI::RegTensor<uint32_t> expOne32Reg;
    MicroAPI::RegTensor<uint32_t> extractExp;
    MicroAPI::RegTensor<uint32_t> halfScale;
    MicroAPI::RegTensor<uint16_t> scale16;
    MicroAPI::RegTensor<uint16_t> recip16;
    MicroAPI::MaskReg cmpResult;
    MicroAPI::MaskReg zeroMask;
    MicroAPI::MaskReg p0;
    MicroAPI::MaskReg p1;

    MicroAPI::LoadAlign<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_UNPACK_B16>(
        max16Reg, maxReadAddr, VF_LEN_FP32);
    MicroAPI::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>((MicroAPI::RegTensor<float>&)max32Reg,
                                                       (MicroAPI::RegTensor<xDtype>&)max16Reg, maskAll);
    MicroAPI::Compare<uint32_t, CMPMODE::LT>(cmpResult, max32Reg, expMaskReg, maskAll32);
    MicroAPI::Compare<uint32_t, CMPMODE::NE>(zeroMask, max32Reg, zero32Reg, maskAll32);
    MicroAPI::Mul((MicroAPI::RegTensor<float>&)max32Reg, (MicroAPI::RegTensor<float>&)max32Reg,
                  (MicroAPI::RegTensor<float>&)invMax, maskAll32);
    MicroAPI::ShiftRights(exp32Reg, max32Reg, SHR_NUM_FOR_FP32, maskAll32);
    MicroAPI::And(man32Reg, max32Reg, manMaskReg, maskAll32);
    MicroAPI::Compares<uint32_t, CMPMODE::GT>(p0, exp32Reg, static_cast<uint32_t>(0), maskAll32);
    MicroAPI::Compares<uint32_t, CMPMODE::LT>(p0, exp32Reg, EXP_254, p0);
    MicroAPI::Compares<uint32_t, CMPMODE::GT>(p0, man32Reg, static_cast<uint32_t>(0), p0);
    MicroAPI::Compares<uint32_t, CMPMODE::EQ>(p1, exp32Reg, static_cast<uint32_t>(0), maskAll32);
    MicroAPI::Compares<uint32_t, CMPMODE::GT>(p1, man32Reg, HALF_FOR_MAN, p1);
    MicroAPI::Or(p0, p0, p1, maskAll32);
    MicroAPI::Adds(expOne32Reg, exp32Reg, 1, maskAll32);
    MicroAPI::Select(extractExp, expOne32Reg, exp32Reg, p0);
    MicroAPI::Select<uint32_t>(extractExp, extractExp, fp8Nan32Reg, cmpResult);
    MicroAPI::Select<uint32_t>(extractExp, extractExp, zero32Reg, zeroMask);
    MicroAPI::Pack<uint16_t, uint32_t, MicroAPI::HighLowPart::LOWEST>(scale16, extractExp);
    MicroAPI::Pack<uint8_t, uint16_t, MicroAPI::HighLowPart::LOWEST>(scale8, scale16);

    MicroAPI::ShiftLefts(extractExp, extractExp, SHR_NUM_FOR_BF16, maskAll32);
    MicroAPI::Sub(halfScale, scaleBiasReg, extractExp, maskAll32);
    MicroAPI::Select<uint32_t>(halfScale, halfScale, nan32Reg, cmpResult);
    MicroAPI::Select<uint32_t>(halfScale, halfScale, zero32Reg, zeroMask);
    MicroAPI::Pack<uint16_t, uint32_t, MicroAPI::HighLowPart::LOWEST>(recip16, halfScale);
    MicroAPI::StoreAlign<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(reciprocalWriteAddr, recip16, VF_LEN_FP32,
                                                                            maskB16);
}

template <typename xDtype, typename y1Dtype, uint64_t mode, AscendC::RoundMode roundMode, uint64_t scaleAlg,
          uint64_t isGroupIdx>
__aicore__ inline void
SwigluMxQuantWithDualAxisBase<xDtype, y1Dtype, mode, roundMode, scaleAlg, isGroupIdx>::ComputeScaleCuBLASSecondLast(
    uint16_t dataLen, uint32_t localInvDtypeMax, __ubuf__ uint16_t* mxScale2ReciprocalAddr,
    __ubuf__ uint8_t* mxScale2Addr)
{
    uint16_t times = dataLen / VF_LEN_FP32;
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<uint32_t> invMax;
        MicroAPI::RegTensor<uint32_t> manMaskReg;
        MicroAPI::RegTensor<uint32_t> expMaskReg;
        MicroAPI::RegTensor<uint32_t> zero32Reg;
        MicroAPI::RegTensor<uint32_t> scaleBiasReg;
        MicroAPI::RegTensor<uint32_t> nan32Reg;
        MicroAPI::RegTensor<uint32_t> fp8Nan32Reg;
        MicroAPI::RegTensor<uint8_t> scale8Slot0;
        MicroAPI::RegTensor<uint8_t> scale8Slot1;
        MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<xDtype, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg maskAll32 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg maskB16 = MicroAPI::CreateMask<uint16_t, MicroAPI::MaskPattern::VL64>();
        MicroAPI::MaskReg interleaveMask = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::Duplicate(scaleBiasReg, FP32_EXP_BIAS_CUBLAS);
        MicroAPI::Duplicate(expMaskReg, MAX_EXP_FOR_FP32);
        MicroAPI::Duplicate(zero32Reg, static_cast<uint32_t>(0));
        MicroAPI::Duplicate(invMax, localInvDtypeMax);
        MicroAPI::Duplicate(manMaskReg, MAN_MASK_FLOAT);
        MicroAPI::Duplicate(fp8Nan32Reg, MAX_EXP_FOR_FP8_IN_FP32);
        MicroAPI::Duplicate(nan32Reg, static_cast<uint32_t>(NAN_CUSTOMIZATION));
        for (uint16_t i = 0; i < times; i++) {
            uint16_t colOffset = i * VF_LEN_FP32;
            __ubuf__ uint16_t* slot0Addr = mxScale2ReciprocalAddr + colOffset;
            __ubuf__ uint16_t* slot1Addr = mxScale2ReciprocalAddr + dataLen + colOffset;
            ComputeScaleCuBLASForSlot(slot0Addr, slot0Addr, scale8Slot0, invMax, manMaskReg, expMaskReg, zero32Reg,
                                      scaleBiasReg, nan32Reg, fp8Nan32Reg, maskAll, maskAll32, maskB16);
            ComputeScaleCuBLASForSlot(slot1Addr, slot1Addr, scale8Slot1, invMax, manMaskReg, expMaskReg, zero32Reg,
                                      scaleBiasReg, nan32Reg, fp8Nan32Reg, maskAll, maskAll32, maskB16);
            MicroAPI::StoreAlign<uint8_t, MicroAPI::StoreDist::DIST_INTLV_B8>(mxScale2Addr + DIGIT_TWO * colOffset,
                                                                              scale8Slot0, scale8Slot1, interleaveMask);
        }
    }
}

// ============================================================================
// ComputeScaleCuBLAS — CUBLAS (scaleAlg=1) scale computation for both axes
// Uses absolute value max + FP32 multiply by 1/dtype_max + mantissa rounding
// Only supported for FP8 output types
// ============================================================================
template <typename xDtype, typename y1Dtype, uint64_t mode, AscendC::RoundMode roundMode, uint64_t scaleAlg,
          uint64_t isGroupIdx>
__aicore__ inline void
SwigluMxQuantWithDualAxisBase<xDtype, y1Dtype, mode, roundMode, scaleAlg, isGroupIdx>::ComputeScaleCuBLAS(
    uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr, __ubuf__ uint8_t* y1Addr,
    __ubuf__ uint8_t* mxScale1Addr, __ubuf__ uint16_t* mxScale1ReciprocalAddr, __ubuf__ uint8_t* mxScale2Addr,
    __ubuf__ uint16_t* mxScale2ReciprocalAddr)
{
    uint32_t localInvDtypeMax = invDtypeMax_;
    __ubuf__ uint16_t* tempMaxAddr = mxScale2ReciprocalAddr;
    __ubuf__ xDtype* xAddrBase = xAddr;
    __ubuf__ uint16_t* reciprocalBase = mxScale1ReciprocalAddr;

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<xDtype> x0;
        MicroAPI::RegTensor<xDtype> x1;
        MicroAPI::RegTensor<uint16_t> x0Abs;
        MicroAPI::RegTensor<uint16_t> x1Abs;
        MicroAPI::RegTensor<uint16_t> absMaxDim1;
        MicroAPI::RegTensor<uint16_t> scale2Slot0Part0;
        MicroAPI::RegTensor<uint16_t> scale2Slot0Part1;
        MicroAPI::RegTensor<uint16_t> scale2Slot1Part0;
        MicroAPI::RegTensor<uint16_t> scale2Slot1Part1;
        MicroAPI::RegTensor<uint32_t> max32;
        MicroAPI::RegTensor<uint32_t> exp32;
        MicroAPI::RegTensor<uint32_t> man32;
        MicroAPI::RegTensor<uint32_t> expAddOne32;
        MicroAPI::RegTensor<uint32_t> extractExp;
        MicroAPI::RegTensor<uint32_t> halfScale;
        MicroAPI::RegTensor<uint16_t> scale16;
        MicroAPI::RegTensor<uint8_t> scale8Reg;
        MicroAPI::RegTensor<uint8_t> scale8Row;
        MicroAPI::RegTensor<uint16_t> recip16Reg;
        MicroAPI::RegTensor<uint16_t> recip16Row;
        MicroAPI::RegTensor<int8_t> extractIdx;
        MicroAPI::RegTensor<uint16_t> absMask;
        MicroAPI::RegTensor<uint32_t> invMax;
        MicroAPI::RegTensor<uint32_t> manMaskReg;
        MicroAPI::RegTensor<uint32_t> expMaskReg;
        MicroAPI::RegTensor<uint32_t> zeroReg32;
        MicroAPI::RegTensor<uint32_t> scaleBiasReg;
        MicroAPI::RegTensor<uint32_t> nanReg32;
        MicroAPI::RegTensor<uint32_t> fp8NanReg32;
        MicroAPI::MaskReg cmpResult;
        MicroAPI::MaskReg zeroMask;
        MicroAPI::MaskReg p0;
        MicroAPI::MaskReg p1;
        MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<xDtype, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg maskAll32 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg maskReduceB8 = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::VL8>();
        MicroAPI::MaskReg maskReduceB16 = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::VL16>();
        MicroAPI::UnalignReg ureg;

        MicroAPI::Duplicate(absMask, ABS_MASK_FOR_16BIT);
        MicroAPI::Duplicate(scale2Slot0Part0, static_cast<uint16_t>(0));
        MicroAPI::Duplicate(scale2Slot0Part1, static_cast<uint16_t>(0));
        MicroAPI::Duplicate(scale2Slot1Part0, static_cast<uint16_t>(0));
        MicroAPI::Duplicate(scale2Slot1Part1, static_cast<uint16_t>(0));

        // Phase 1: collect scale1 maxima compactly while preserving perf/all's
        // two independent 32-row scale2 slots.
        uint16_t slot0Rows = blockCount < BLOCK_SIZE ? blockCount : BLOCK_SIZE;
        for (uint16_t i = 0; i < slot0Rows; i++) {
            MicroAPI::LoadAlign<xDtype, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_DINTLV_B16>(
                x0, x1, xAddr, dataLen);
            MicroAPI::And(x0Abs, (MicroAPI::RegTensor<uint16_t>&)x0, absMask, maskAll);
            MicroAPI::And(x1Abs, (MicroAPI::RegTensor<uint16_t>&)x1, absMask, maskAll);
            MicroAPI::Max(absMaxDim1, x0Abs, x1Abs, maskAll);
            MicroAPI::ReduceDataBlock<AscendC::Reg::ReduceType::MAX>(absMaxDim1, absMaxDim1, maskAll);
            MicroAPI::StoreUnAlign<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(tempMaxAddr, absMaxDim1, ureg,
                                                                                      dataLen / BLOCK_SIZE);
            MicroAPI::Max(scale2Slot0Part0, scale2Slot0Part0, x0Abs, maskAll);
            MicroAPI::Max(scale2Slot0Part1, scale2Slot0Part1, x1Abs, maskAll);
        }
        for (uint16_t i = slot0Rows; i < blockCount; i++) {
            MicroAPI::LoadAlign<xDtype, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_DINTLV_B16>(
                x0, x1, xAddr, dataLen);
            MicroAPI::And(x0Abs, (MicroAPI::RegTensor<uint16_t>&)x0, absMask, maskAll);
            MicroAPI::And(x1Abs, (MicroAPI::RegTensor<uint16_t>&)x1, absMask, maskAll);
            MicroAPI::Max(absMaxDim1, x0Abs, x1Abs, maskAll);
            MicroAPI::ReduceDataBlock<AscendC::Reg::ReduceType::MAX>(absMaxDim1, absMaxDim1, maskAll);
            MicroAPI::StoreUnAlign<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(tempMaxAddr, absMaxDim1, ureg,
                                                                                      dataLen / BLOCK_SIZE);
            MicroAPI::Max(scale2Slot1Part0, scale2Slot1Part0, x0Abs, maskAll);
            MicroAPI::Max(scale2Slot1Part1, scale2Slot1Part1, x1Abs, maskAll);
        }
        MicroAPI::StoreUnAlignPost(tempMaxAddr, ureg, 0);

        MicroAPI::Duplicate(invMax, localInvDtypeMax);
        MicroAPI::Duplicate(manMaskReg, MAN_MASK_FLOAT);
        MicroAPI::Duplicate(expMaskReg, MAX_EXP_FOR_FP32);
        MicroAPI::Duplicate(zeroReg32, static_cast<uint32_t>(0));
        MicroAPI::Duplicate(scaleBiasReg, FP32_EXP_BIAS_CUBLAS);
        MicroAPI::Duplicate(nanReg32, static_cast<uint32_t>(NAN_CUSTOMIZATION));
        MicroAPI::Duplicate(fp8NanReg32, MAX_EXP_FOR_FP8_IN_FP32);

        // Phase 2: convert 64 cached maxima per VF, then scatter eight rows
        // back to the original 32-byte-aligned scale/reciprocal layout.
        __ubuf__ uint16_t* readMaxAddr = mxScale2ReciprocalAddr;
        uint16_t scaleCount = dataLen / BLOCK_SIZE;
        uint16_t batchCount = ops::CeilDiv(static_cast<uint16_t>(blockCount * scaleCount),
                                           static_cast<uint16_t>(VF_LEN_FP32));
        for (uint16_t j = 0; j < batchCount; j++) {
            MicroAPI::LoadAlign<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_UNPACK_B16>(
                absMaxDim1, readMaxAddr, VF_LEN_FP32);
            MicroAPI::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>((MicroAPI::RegTensor<float>&)max32,
                                                               (MicroAPI::RegTensor<xDtype>&)absMaxDim1, maskAll);
            MicroAPI::Compare<uint32_t, CMPMODE::LT>(cmpResult, max32, expMaskReg, maskAll32);
            MicroAPI::Compare<uint32_t, CMPMODE::NE>(zeroMask, max32, zeroReg32, maskAll32);
            MicroAPI::Mul((MicroAPI::RegTensor<float>&)max32, (MicroAPI::RegTensor<float>&)max32,
                          (MicroAPI::RegTensor<float>&)invMax, maskAll32);
            MicroAPI::ShiftRights(exp32, max32, SHR_NUM_FOR_FP32, maskAll32);
            MicroAPI::And(man32, max32, manMaskReg, maskAll32);
            MicroAPI::Compares<uint32_t, CMPMODE::GT>(p0, exp32, static_cast<uint32_t>(0), maskAll32);
            MicroAPI::Compares<uint32_t, CMPMODE::LT>(p0, exp32, EXP_254, p0);
            MicroAPI::Compares<uint32_t, CMPMODE::GT>(p0, man32, static_cast<uint32_t>(0), p0);
            MicroAPI::Compares<uint32_t, CMPMODE::EQ>(p1, exp32, static_cast<uint32_t>(0), maskAll32);
            MicroAPI::Compares<uint32_t, CMPMODE::GT>(p1, man32, HALF_FOR_MAN, p1);
            MicroAPI::Or(p0, p0, p1, maskAll32);
            MicroAPI::Adds(expAddOne32, exp32, 1, maskAll32);
            MicroAPI::Select(extractExp, expAddOne32, exp32, p0);
            MicroAPI::Select<uint32_t>(extractExp, extractExp, fp8NanReg32, cmpResult);
            MicroAPI::Select<uint32_t>(extractExp, extractExp, zeroReg32, zeroMask);
            MicroAPI::Pack<uint16_t, uint32_t, MicroAPI::HighLowPart::LOWEST>(scale16, extractExp);
            MicroAPI::Pack<uint8_t, uint16_t, MicroAPI::HighLowPart::LOWEST>(scale8Reg, scale16);
            MicroAPI::ShiftLefts(extractExp, extractExp, SHR_NUM_FOR_BF16, maskAll32);
            MicroAPI::Sub(halfScale, scaleBiasReg, extractExp, maskAll32);
            MicroAPI::Select<uint32_t>(halfScale, halfScale, nanReg32, cmpResult);
            MicroAPI::Select<uint32_t>(halfScale, halfScale, zeroReg32, zeroMask);
            MicroAPI::Pack<uint16_t, uint32_t, MicroAPI::HighLowPart::LOWEST>(recip16Reg, halfScale);

            for (uint16_t k = 0; k < 8; k++) {
                MicroAPI::Arange(extractIdx, static_cast<int8_t>(k * 8));
                MicroAPI::Gather(scale8Row, scale8Reg, (MicroAPI::RegTensor<uint8_t>&)extractIdx);
                MicroAPI::StoreAlign<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(mxScale1Addr, scale8Row,
                                                                                       UB_BLOCK_SIZE, maskReduceB8);
                MicroAPI::Arange(extractIdx, static_cast<int8_t>(k * 16));
                MicroAPI::Gather((MicroAPI::RegTensor<uint8_t>&)recip16Row, (MicroAPI::RegTensor<uint8_t>&)recip16Reg,
                                 (MicroAPI::RegTensor<uint8_t>&)extractIdx);
                MicroAPI::StoreAlign<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    mxScale1ReciprocalAddr, recip16Row, SCALE1_RECIPROCAL_ROW_ELEMS, maskReduceB16);
            }
        }

        // Generate y1 immediately after scale1 while staying in the same vector scope.
        MicroAPI::MaskReg maskAllB8 = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::RegTensor<uint16_t> scaleForMulFP16;
        MicroAPI::RegTensor<float> scaleForMulFP32;
        MicroAPI::RegTensor<float> x0ZeroFP32;
        MicroAPI::RegTensor<float> x0OneFP32;
        MicroAPI::RegTensor<float> x1ZeroFP32;
        MicroAPI::RegTensor<float> x1OneFP32;
        MicroAPI::RegTensor<y1Dtype> x0ZeroFP8;
        MicroAPI::RegTensor<y1Dtype> x0OneFP8;
        MicroAPI::RegTensor<y1Dtype> x1ZeroFP8;
        MicroAPI::RegTensor<y1Dtype> x1OneFP8;
        __ubuf__ xDtype* xReadAddr = xAddrBase;
        __ubuf__ uint16_t* recipReadAddr = reciprocalBase;

        for (uint16_t i = 0; i < blockCount; i++) {
            MicroAPI::LoadAlign<xDtype, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_DINTLV_B16>(
                x0, x1, xReadAddr, dataLen);
            MicroAPI::LoadAlign<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_E2B_B16>(
                scaleForMulFP16, recipReadAddr, SCALE1_RECIPROCAL_ROW_ELEMS);

            if constexpr (IsSameType<xDtype, half>::value) {
                MicroAPI::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(x0ZeroFP32, x0, maskAll);
                MicroAPI::Cast<float, xDtype, CAST_X_TO_FP32_ONE>(x0OneFP32, x0, maskAll);
                MicroAPI::Cast<float, bfloat16_t, CAST_X_TO_FP32_ZERO>(
                    scaleForMulFP32, (MicroAPI::RegTensor<bfloat16_t>&)scaleForMulFP16, maskAll);
                MicroAPI::Mul(x0ZeroFP32, x0ZeroFP32, scaleForMulFP32, maskAll);
                MicroAPI::Mul(x0OneFP32, x0OneFP32, scaleForMulFP32, maskAll);
                MicroAPI::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(x1ZeroFP32, x1, maskAll);
                MicroAPI::Cast<float, xDtype, CAST_X_TO_FP32_ONE>(x1OneFP32, x1, maskAll);
                MicroAPI::Mul(x1ZeroFP32, x1ZeroFP32, scaleForMulFP32, maskAll);
                MicroAPI::Mul(x1OneFP32, x1OneFP32, scaleForMulFP32, maskAll);
            } else {
                MicroAPI::Mul(x0, x0, (MicroAPI::RegTensor<xDtype>&)scaleForMulFP16, maskAll);
                MicroAPI::Mul(x1, x1, (MicroAPI::RegTensor<xDtype>&)scaleForMulFP16, maskAll);
                MicroAPI::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(x0ZeroFP32, x0, maskAll);
                MicroAPI::Cast<float, xDtype, CAST_X_TO_FP32_ONE>(x0OneFP32, x0, maskAll);
                MicroAPI::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(x1ZeroFP32, x1, maskAll);
                MicroAPI::Cast<float, xDtype, CAST_X_TO_FP32_ONE>(x1OneFP32, x1, maskAll);
            }
            MicroAPI::Cast<y1Dtype, float, CAST_32_TO_80>(x0ZeroFP8, x0ZeroFP32, maskAll);
            MicroAPI::Cast<y1Dtype, float, CAST_32_TO_82>(x0OneFP8, x0OneFP32, maskAll);
            MicroAPI::Cast<y1Dtype, float, CAST_32_TO_81>(x1ZeroFP8, x1ZeroFP32, maskAll);
            MicroAPI::Cast<y1Dtype, float, CAST_32_TO_83>(x1OneFP8, x1OneFP32, maskAll);
            MicroAPI::Add((MicroAPI::RegTensor<uint8_t>&)x0ZeroFP8, (MicroAPI::RegTensor<uint8_t>&)x0ZeroFP8,
                          (MicroAPI::RegTensor<uint8_t>&)x0OneFP8, maskAllB8);
            MicroAPI::Add((MicroAPI::RegTensor<uint8_t>&)x0ZeroFP8, (MicroAPI::RegTensor<uint8_t>&)x0ZeroFP8,
                          (MicroAPI::RegTensor<uint8_t>&)x1ZeroFP8, maskAllB8);
            MicroAPI::Add((MicroAPI::RegTensor<uint8_t>&)x0ZeroFP8, (MicroAPI::RegTensor<uint8_t>&)x0ZeroFP8,
                          (MicroAPI::RegTensor<uint8_t>&)x1OneFP8, maskAllB8);
            MicroAPI::StoreAlign<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::StoreDist::DIST_NORM_B8>(
                y1Addr, (MicroAPI::RegTensor<uint8_t>&)x0ZeroFP8, dataLen, maskAllB8);
        }

        // Reuse the temporary area for the normal axis=-2 maxima output.
        MicroAPI::StoreAlign<uint16_t, MicroAPI::StoreDist::DIST_INTLV_B16>(mxScale2ReciprocalAddr, scale2Slot0Part0,
                                                                            scale2Slot0Part1, maskAll);
        if (blockCount > BLOCK_SIZE) {
            MicroAPI::StoreAlign<uint16_t, MicroAPI::StoreDist::DIST_INTLV_B16>(
                mxScale2ReciprocalAddr + dataLen, scale2Slot1Part0, scale2Slot1Part1, maskAll);
        }
    }
    ComputeScaleCuBLASSecondLast(dataLen, localInvDtypeMax, mxScale2ReciprocalAddr, mxScale2Addr);
}
// ============================================================================
// ComputeY1ToFP8 — quantize SwiGLU result to FP8 along axis=-1
// ============================================================================
template <typename xDtype, typename y1Dtype, uint64_t mode, AscendC::RoundMode roundMode, uint64_t scaleAlg,
          uint64_t isGroupIdx>
__aicore__ inline void
SwigluMxQuantWithDualAxisBase<xDtype, y1Dtype, mode, roundMode, scaleAlg, isGroupIdx>::ComputeY1ToFP8(
    uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr, __ubuf__ uint16_t* mxScale1ReciprocalAddr,
    __ubuf__ uint8_t* y1Addr)
{
    int64_t localVlForHalfNumber = VF_LEN_B16;
    int64_t localUBBlockSize = UB_BLOCK_SIZE;

    __VEC_SCOPE__
    {
        MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<uint16_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg maskAllB8 = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::RegTensor<uint16_t> scaleForMulFP16;
        MicroAPI::RegTensor<float> scaleForMulFP32;
        MicroAPI::RegTensor<xDtype> x0;
        MicroAPI::RegTensor<xDtype> x1;
        MicroAPI::RegTensor<bfloat16_t> x0BF16;
        MicroAPI::RegTensor<bfloat16_t> x1BF16;
        MicroAPI::RegTensor<float> x0ZeroFP32;
        MicroAPI::RegTensor<float> x0OneFP32;
        MicroAPI::RegTensor<float> x1ZeroFP32;
        MicroAPI::RegTensor<float> x1OneFP32;
        MicroAPI::RegTensor<y1Dtype> x0ZeroFP8;
        MicroAPI::RegTensor<y1Dtype> x0OneFP8;
        MicroAPI::RegTensor<y1Dtype> x1ZeroFP8;
        MicroAPI::RegTensor<y1Dtype> x1OneFP8;

        for (uint16_t i = 0; i < blockCount; i++) {
            MicroAPI::LoadAlign<xDtype, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_DINTLV_B16>(
                x0, x1, xAddr, ONCE_ROW_LEN);
            MicroAPI::LoadAlign<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_E2B_B16>(
                scaleForMulFP16, mxScale1ReciprocalAddr, SCALE1_RECIPROCAL_ROW_ELEMS);

            if constexpr (IsSameType<xDtype, half>::value) {
                MicroAPI::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(x0ZeroFP32, x0, maskAll);
                MicroAPI::Cast<float, xDtype, CAST_X_TO_FP32_ONE>(x0OneFP32, x0, maskAll);
                MicroAPI::Cast<float, bfloat16_t, CAST_X_TO_FP32_ZERO>(
                    scaleForMulFP32, (MicroAPI::RegTensor<bfloat16_t>&)scaleForMulFP16, maskAll);

                MicroAPI::Mul(x0ZeroFP32, x0ZeroFP32, scaleForMulFP32, maskAll);
                MicroAPI::Mul(x0OneFP32, x0OneFP32, scaleForMulFP32, maskAll);

                MicroAPI::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(x1ZeroFP32, x1, maskAll);
                MicroAPI::Cast<float, xDtype, CAST_X_TO_FP32_ONE>(x1OneFP32, x1, maskAll);
                MicroAPI::Mul(x1ZeroFP32, x1ZeroFP32, scaleForMulFP32, maskAll);
                MicroAPI::Mul(x1OneFP32, x1OneFP32, scaleForMulFP32, maskAll);
            } else {
                MicroAPI::Mul(x0, x0, (MicroAPI::RegTensor<xDtype>&)scaleForMulFP16, maskAll);
                MicroAPI::Mul(x1, x1, (MicroAPI::RegTensor<xDtype>&)scaleForMulFP16, maskAll);

                AscendC::MicroAPI::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(x0ZeroFP32, x0, maskAll);
                AscendC::MicroAPI::Cast<float, xDtype, CAST_X_TO_FP32_ONE>(x0OneFP32, x0, maskAll);
                AscendC::MicroAPI::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(x1ZeroFP32, x1, maskAll);
                AscendC::MicroAPI::Cast<float, xDtype, CAST_X_TO_FP32_ONE>(x1OneFP32, x1, maskAll);
            }
            AscendC::MicroAPI::Cast<y1Dtype, float, CAST_32_TO_80>(x0ZeroFP8, x0ZeroFP32, maskAll);
            AscendC::MicroAPI::Cast<y1Dtype, float, CAST_32_TO_82>(x0OneFP8, x0OneFP32, maskAll);
            AscendC::MicroAPI::Cast<y1Dtype, float, CAST_32_TO_81>(x1ZeroFP8, x1ZeroFP32, maskAll);
            AscendC::MicroAPI::Cast<y1Dtype, float, CAST_32_TO_83>(x1OneFP8, x1OneFP32, maskAll);

            AscendC::MicroAPI::Add((AscendC::MicroAPI::RegTensor<uint8_t>&)x0ZeroFP8,
                                   (AscendC::MicroAPI::RegTensor<uint8_t>&)x0ZeroFP8,
                                   (AscendC::MicroAPI::RegTensor<uint8_t>&)x0OneFP8, maskAllB8);
            AscendC::MicroAPI::Add((AscendC::MicroAPI::RegTensor<uint8_t>&)x0ZeroFP8,
                                   (AscendC::MicroAPI::RegTensor<uint8_t>&)x0ZeroFP8,
                                   (AscendC::MicroAPI::RegTensor<uint8_t>&)x1ZeroFP8, maskAllB8);
            AscendC::MicroAPI::Add((AscendC::MicroAPI::RegTensor<uint8_t>&)x0ZeroFP8,
                                   (AscendC::MicroAPI::RegTensor<uint8_t>&)x0ZeroFP8,
                                   (AscendC::MicroAPI::RegTensor<uint8_t>&)x1OneFP8, maskAllB8);
            AscendC::MicroAPI::StoreAlign<uint8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                          AscendC::MicroAPI::StoreDist::DIST_NORM_B8>(
                y1Addr, (AscendC::MicroAPI::RegTensor<uint8_t>&)x0ZeroFP8, ONCE_ROW_LEN, maskAllB8);
        }
    }
}

// ============================================================================
// ComputeY1ToFP4 — quantize SwiGLU result to FP4 along axis=-1
// ============================================================================
template <typename xDtype, typename y1Dtype, uint64_t mode, AscendC::RoundMode roundMode, uint64_t scaleAlg,
          uint64_t isGroupIdx>
__aicore__ inline void
SwigluMxQuantWithDualAxisBase<xDtype, y1Dtype, mode, roundMode, scaleAlg, isGroupIdx>::ComputeY1ToFP4(
    uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr, __ubuf__ uint16_t* mxScale1ReciprocalAddr,
    __ubuf__ uint8_t* y1Addr)
{
    int64_t localVlForHalfNumber = VF_LEN_B16;
    int64_t localUBBlockSize = UB_BLOCK_SIZE;

    __VEC_SCOPE__
    {
        MicroAPI::MaskReg dataMaskB8 = MicroAPI::CreateMask<uint8_t>();
        MicroAPI::MaskReg dataMaskB16 = MicroAPI::CreateMask<half>();
        MicroAPI::MaskReg dataMaskB32 = MicroAPI::CreateMask<float>();
        MicroAPI::RegTensor<uint16_t> scaleForMulFP16;
        MicroAPI::RegTensor<xDtype> x0;
        MicroAPI::RegTensor<xDtype> x1;

        MicroAPI::RegTensor<float> x0ZeroFP32;
        MicroAPI::RegTensor<float> x0OneFP32;
        MicroAPI::RegTensor<float> x1ZeroFP32;
        MicroAPI::RegTensor<float> x1OneFP32;
        MicroAPI::RegTensor<float> scaleForMulZeroFP32;
        MicroAPI::RegTensor<float> scaleForMulOneFP32;

        MicroAPI::RegTensor<bfloat16_t> x0ZeroBF16;
        MicroAPI::RegTensor<bfloat16_t> x0OneBF16;
        MicroAPI::RegTensor<bfloat16_t> x1ZeroBF16;
        MicroAPI::RegTensor<bfloat16_t> x1OneBF16;

        MicroAPI::RegTensor<y1Dtype> x0FP4;
        MicroAPI::RegTensor<y1Dtype> x1FP4;

        for (uint16_t i = 0; i < blockCount; i++) {
            MicroAPI::LoadAlign<xDtype, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_DINTLV_B16>(
                x0, x1, xAddr, ONCE_ROW_LEN);
            MicroAPI::LoadAlign<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_E2B_B16>(
                scaleForMulFP16, mxScale1ReciprocalAddr, SCALE1_RECIPROCAL_ROW_ELEMS);

            if constexpr (IsSameType<xDtype, half>::value) {
                MicroAPI::Cast<float, bfloat16_t, CAST_X_TO_FP32_ZERO>(
                    scaleForMulZeroFP32, (MicroAPI::RegTensor<bfloat16_t>&)scaleForMulFP16, dataMaskB16);

                // x0 cast to fp32 and multiply by 1/scale
                MicroAPI::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(x0ZeroFP32, x0, dataMaskB16);
                MicroAPI::Cast<float, xDtype, CAST_X_TO_FP32_ONE>(x0OneFP32, x0, dataMaskB16);

                MicroAPI::Mul(x0ZeroFP32, scaleForMulZeroFP32, x0ZeroFP32, dataMaskB32);
                MicroAPI::Mul(x0OneFP32, scaleForMulZeroFP32, x0OneFP32, dataMaskB32);
                ComputeFP4FromHalf(x0ZeroFP32);
                ComputeFP4FromHalf(x0OneFP32);
                MicroAPI::Cast<bfloat16_t, float, castTraitFp32toBF16>(x0ZeroBF16, x0ZeroFP32, dataMaskB32);
                MicroAPI::Cast<bfloat16_t, float, castTraitFp32toBF16>(x0OneBF16, x0OneFP32, dataMaskB32);
                MicroAPI::Pack<uint16_t, uint32_t, MicroAPI::HighLowPart::LOWEST>(
                    (MicroAPI::RegTensor<uint16_t>&)x0ZeroBF16, (MicroAPI::RegTensor<uint32_t>&)x0ZeroBF16);
                MicroAPI::Pack<uint16_t, uint32_t, MicroAPI::HighLowPart::LOWEST>(
                    (MicroAPI::RegTensor<uint16_t>&)x0OneBF16, (MicroAPI::RegTensor<uint32_t>&)x0OneBF16);
                MicroAPI::Interleave(x0ZeroBF16, x0OneBF16, x0ZeroBF16, x0OneBF16);

                // x1 cast to fp32 and multiply by 1/scale
                MicroAPI::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(x1ZeroFP32, x1, dataMaskB16);
                MicroAPI::Cast<float, xDtype, CAST_X_TO_FP32_ONE>(x1OneFP32, x1, dataMaskB16);

                MicroAPI::Mul(x1ZeroFP32, scaleForMulZeroFP32, x1ZeroFP32, dataMaskB32);
                MicroAPI::Mul(x1OneFP32, scaleForMulZeroFP32, x1OneFP32, dataMaskB32);
                ComputeFP4FromHalf(x1ZeroFP32);
                ComputeFP4FromHalf(x1OneFP32);
                MicroAPI::Cast<bfloat16_t, float, castTraitFp32toBF16>(x1ZeroBF16, x1ZeroFP32, dataMaskB32);
                MicroAPI::Cast<bfloat16_t, float, castTraitFp32toBF16>(x1OneBF16, x1OneFP32, dataMaskB32);
                MicroAPI::Pack<uint16_t, uint32_t, MicroAPI::HighLowPart::LOWEST>(
                    (MicroAPI::RegTensor<uint16_t>&)x1ZeroBF16, (MicroAPI::RegTensor<uint32_t>&)x1ZeroBF16);
                MicroAPI::Pack<uint16_t, uint32_t, MicroAPI::HighLowPart::LOWEST>(
                    (MicroAPI::RegTensor<uint16_t>&)x1OneBF16, (MicroAPI::RegTensor<uint32_t>&)x1OneBF16);
                MicroAPI::Interleave(x1ZeroBF16, x1OneBF16, x1ZeroBF16, x1OneBF16);

                // Interleave x0 and x1, then cast to FP4
                MicroAPI::Interleave(x0ZeroBF16, x1ZeroBF16, x0ZeroBF16, x1ZeroBF16);
                MicroAPI::Cast<y1Dtype, bfloat16_t, castTraitBF16toFp4>(x0FP4, x0ZeroBF16, dataMaskB16);
                MicroAPI::Cast<y1Dtype, bfloat16_t, castTraitBF16toFp4>(x1FP4, x1ZeroBF16, dataMaskB16);
            } else {
                // BF16 input path
                MicroAPI::Mul(x0, x0, (MicroAPI::RegTensor<xDtype>&)scaleForMulFP16, dataMaskB16);
                MicroAPI::Mul(x1, x1, (MicroAPI::RegTensor<xDtype>&)scaleForMulFP16, dataMaskB16);
                MicroAPI::Interleave(x0, x1, x0, x1);
                MicroAPI::Cast<y1Dtype, xDtype, castTraitBF16toFp4>(x0FP4, x0, dataMaskB16);
                MicroAPI::Cast<y1Dtype, xDtype, castTraitBF16toFp4>(x1FP4, x1, dataMaskB16);
            }

            MicroAPI::StoreAlign<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::StoreDist::DIST_PACK4_B32>(
                y1Addr, (MicroAPI::RegTensor<uint8_t>&)x0FP4, OUT_ELE_NUM_ONE_BLK, dataMaskB8);
            MicroAPI::StoreAlign<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::StoreDist::DIST_PACK4_B32>(
                y1Addr, (MicroAPI::RegTensor<uint8_t>&)x1FP4, OUT_ELE_NUM_ONE_BLK, dataMaskB8);
        }
    }
    return;
}

// ============================================================================
// ComputeY2ToFP8 — quantize SwiGLU result to FP8 along axis=-2
// ============================================================================
template <typename xDtype, typename y1Dtype, uint64_t mode, AscendC::RoundMode roundMode, uint64_t scaleAlg,
          uint64_t isGroupIdx>
__aicore__ inline void
SwigluMxQuantWithDualAxisBase<xDtype, y1Dtype, mode, roundMode, scaleAlg, isGroupIdx>::ComputeY2ToFP8(
    uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr, __ubuf__ uint16_t* mxScale2ReciprocalAddr,
    __ubuf__ uint8_t* y2Addr)
{
    int64_t localUbRowLen = ubRowLen_;
    constexpr uint32_t dualLoadLen = VF_LEN_B16 * DIGIT_TWO;

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<xDtype> xEven;
        MicroAPI::RegTensor<xDtype> xOdd;
        MicroAPI::RegTensor<uint16_t> reciprocalEven;
        MicroAPI::RegTensor<uint16_t> reciprocalOdd;
        MicroAPI::RegTensor<float> reciprocalEvenFP32Layout0;
        MicroAPI::RegTensor<float> reciprocalEvenFP32Layout1;
        MicroAPI::RegTensor<float> reciprocalOddFP32Layout0;
        MicroAPI::RegTensor<float> reciprocalOddFP32Layout1;
        MicroAPI::RegTensor<float> xEvenFP32Layout0;
        MicroAPI::RegTensor<float> xEvenFP32Layout1;
        MicroAPI::RegTensor<float> xOddFP32Layout0;
        MicroAPI::RegTensor<float> xOddFP32Layout1;
        MicroAPI::RegTensor<y1Dtype> yEvenFP8Layout0;
        MicroAPI::RegTensor<y1Dtype> yEvenFP8Layout2;
        MicroAPI::RegTensor<y1Dtype> yOddFP8Layout1;
        MicroAPI::RegTensor<y1Dtype> yOddFP8Layout3;

        MicroAPI::MaskReg maskB8 = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg maskB16 = MicroAPI::CreateMask<uint16_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg maskB32 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();

        // Load all 256 reciprocal values once and reuse them for all 32 rows in this scale2 slot.
        MicroAPI::LoadAlign<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_DINTLV_B16>(
            reciprocalEven, reciprocalOdd, mxScale2ReciprocalAddr, dualLoadLen);
        if constexpr (IsSameType<xDtype, half>::value) {
            MicroAPI::Cast<float, bfloat16_t, CAST_X_TO_FP32_ZERO>(
                reciprocalEvenFP32Layout0, (MicroAPI::RegTensor<bfloat16_t>&)reciprocalEven, maskB16);
            MicroAPI::Cast<float, bfloat16_t, CAST_X_TO_FP32_ONE>(
                reciprocalEvenFP32Layout1, (MicroAPI::RegTensor<bfloat16_t>&)reciprocalEven, maskB16);
            MicroAPI::Cast<float, bfloat16_t, CAST_X_TO_FP32_ZERO>(
                reciprocalOddFP32Layout0, (MicroAPI::RegTensor<bfloat16_t>&)reciprocalOdd, maskB16);
            MicroAPI::Cast<float, bfloat16_t, CAST_X_TO_FP32_ONE>(
                reciprocalOddFP32Layout1, (MicroAPI::RegTensor<bfloat16_t>&)reciprocalOdd, maskB16);
        }

        for (uint16_t row = 0; row < blockCount; row++) {
            __ubuf__ xDtype* xCursor = xAddr + row * localUbRowLen;
            MicroAPI::LoadAlign<xDtype, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_DINTLV_B16>(
                xEven, xOdd, xCursor, dualLoadLen);
            if constexpr (IsSameType<xDtype, half>::value) {
                MicroAPI::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(xEvenFP32Layout0, xEven, maskB16);
                MicroAPI::Cast<float, xDtype, CAST_X_TO_FP32_ONE>(xEvenFP32Layout1, xEven, maskB16);
                MicroAPI::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(xOddFP32Layout0, xOdd, maskB16);
                MicroAPI::Cast<float, xDtype, CAST_X_TO_FP32_ONE>(xOddFP32Layout1, xOdd, maskB16);

                MicroAPI::Mul(xEvenFP32Layout0, xEvenFP32Layout0, reciprocalEvenFP32Layout0, maskB32);
                MicroAPI::Mul(xEvenFP32Layout1, xEvenFP32Layout1, reciprocalEvenFP32Layout1, maskB32);
                MicroAPI::Mul(xOddFP32Layout0, xOddFP32Layout0, reciprocalOddFP32Layout0, maskB32);
                MicroAPI::Mul(xOddFP32Layout1, xOddFP32Layout1, reciprocalOddFP32Layout1, maskB32);
            } else {
                MicroAPI::Mul(xEven, xEven, (MicroAPI::RegTensor<xDtype>&)reciprocalEven, maskB16);
                MicroAPI::Mul(xOdd, xOdd, (MicroAPI::RegTensor<xDtype>&)reciprocalOdd, maskB16);
                MicroAPI::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(xEvenFP32Layout0, xEven, maskB16);
                MicroAPI::Cast<float, xDtype, CAST_X_TO_FP32_ONE>(xEvenFP32Layout1, xEven, maskB16);
                MicroAPI::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(xOddFP32Layout0, xOdd, maskB16);
                MicroAPI::Cast<float, xDtype, CAST_X_TO_FP32_ONE>(xOddFP32Layout1, xOdd, maskB16);
            }

            MicroAPI::Cast<y1Dtype, float, CAST_32_TO_80>(yEvenFP8Layout0, xEvenFP32Layout0, maskB32);
            MicroAPI::Cast<y1Dtype, float, CAST_32_TO_82>(yEvenFP8Layout2, xEvenFP32Layout1, maskB32);
            MicroAPI::Cast<y1Dtype, float, CAST_32_TO_81>(yOddFP8Layout1, xOddFP32Layout0, maskB32);
            MicroAPI::Cast<y1Dtype, float, CAST_32_TO_83>(yOddFP8Layout3, xOddFP32Layout1, maskB32);

            MicroAPI::Add((MicroAPI::RegTensor<uint8_t>&)yEvenFP8Layout0,
                          (MicroAPI::RegTensor<uint8_t>&)yEvenFP8Layout0,
                          (MicroAPI::RegTensor<uint8_t>&)yEvenFP8Layout2, maskB8);
            MicroAPI::Add((MicroAPI::RegTensor<uint8_t>&)yOddFP8Layout1, (MicroAPI::RegTensor<uint8_t>&)yOddFP8Layout1,
                          (MicroAPI::RegTensor<uint8_t>&)yOddFP8Layout3, maskB8);
            MicroAPI::Add((MicroAPI::RegTensor<uint8_t>&)yEvenFP8Layout0,
                          (MicroAPI::RegTensor<uint8_t>&)yEvenFP8Layout0, (MicroAPI::RegTensor<uint8_t>&)yOddFP8Layout1,
                          maskB8);

            __ubuf__ uint8_t* yCursor = y2Addr + row * localUbRowLen;
            MicroAPI::StoreAlign<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::StoreDist::DIST_NORM_B8>(
                yCursor, (MicroAPI::RegTensor<uint8_t>&)yEvenFP8Layout0, dualLoadLen, maskB8);
        }
    }
}

// ============================================================================
// ComputeY2ToFP4 — quantize SwiGLU result to FP4 along axis=-2
// ============================================================================
template <typename xDtype, typename y1Dtype, uint64_t mode, AscendC::RoundMode roundMode, uint64_t scaleAlg,
          uint64_t isGroupIdx>
__aicore__ inline void
SwigluMxQuantWithDualAxisBase<xDtype, y1Dtype, mode, roundMode, scaleAlg, isGroupIdx>::ComputeY2ToFP4(
    uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr, __ubuf__ uint16_t* mxScale2ReciprocalAddr,
    __ubuf__ uint8_t* y2Addr)
{
    int64_t localUbRowLen = ubRowLen_;

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<xDtype> x;
        MicroAPI::RegTensor<bfloat16_t> x0BF16;
        MicroAPI::RegTensor<bfloat16_t> x1BF16;
        MicroAPI::RegTensor<bfloat16_t> xBF16;
        MicroAPI::RegTensor<float> x0FP32;
        MicroAPI::RegTensor<float> x1FP32;
        MicroAPI::RegTensor<uint16_t> reversedShareExp;
        MicroAPI::RegTensor<float> reversedShareExp0FP32;
        MicroAPI::RegTensor<float> reversedShareExp1FP32;
        MicroAPI::RegTensor<y1Dtype> yZeroFP4;

        MicroAPI::MaskReg pregAll8 = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg pregAll16 = MicroAPI::CreateMask<uint16_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg pregAll32 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();

        // Load per-column 1/scale
        MicroAPI::LoadAlign<uint16_t, MicroAPI::LoadDist::DIST_NORM>(reversedShareExp, mxScale2ReciprocalAddr);

        for (uint16_t j = 0; j < blockCount; j++) {
            MicroAPI::LoadAlign<xDtype, MicroAPI::LoadDist::DIST_NORM>(x, xAddr + j * localUbRowLen);

            if constexpr (IsSameType<xDtype, half>::value) {
                MicroAPI::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(x0FP32, x, pregAll16);
                MicroAPI::Cast<float, xDtype, CAST_X_TO_FP32_ONE>(x1FP32, x, pregAll16);

                MicroAPI::Cast<float, bfloat16_t, CAST_X_TO_FP32_ZERO>(
                    reversedShareExp0FP32, (MicroAPI::RegTensor<bfloat16_t>&)reversedShareExp, pregAll16);

                MicroAPI::Cast<float, bfloat16_t, CAST_X_TO_FP32_ONE>(
                    reversedShareExp1FP32, (MicroAPI::RegTensor<bfloat16_t>&)reversedShareExp, pregAll16);

                MicroAPI::Mul(x0FP32, x0FP32, reversedShareExp0FP32, pregAll32);
                MicroAPI::Mul(x1FP32, x1FP32, reversedShareExp1FP32, pregAll32);

                ComputeFP4FromHalf(x0FP32);
                ComputeFP4FromHalf(x1FP32);

                MicroAPI::Cast<bfloat16_t, float, castTraitFp32toBF16>((MicroAPI::RegTensor<bfloat16_t>&)x0BF16, x0FP32,
                                                                       pregAll32);
                MicroAPI::Pack<uint16_t, uint32_t, MicroAPI::HighLowPart::LOWEST>(
                    (MicroAPI::RegTensor<uint16_t>&)x0BF16, (MicroAPI::RegTensor<uint32_t>&)x0BF16);

                MicroAPI::Cast<bfloat16_t, float, castTraitFp32toBF16>((MicroAPI::RegTensor<bfloat16_t>&)x1BF16, x1FP32,
                                                                       pregAll32);

                MicroAPI::Pack<uint16_t, uint32_t, MicroAPI::HighLowPart::LOWEST>(
                    (MicroAPI::RegTensor<uint16_t>&)x1BF16, (MicroAPI::RegTensor<uint32_t>&)x1BF16);

                MicroAPI::Interleave(x0BF16, x1BF16, x0BF16, x1BF16);
                MicroAPI::Cast<y1Dtype, bfloat16_t, castTraitBF16toFp4>(
                    yZeroFP4, (MicroAPI::RegTensor<bfloat16_t>&)x0BF16, pregAll16);
            } else {
                MicroAPI::Mul(xBF16, x, (MicroAPI::RegTensor<bfloat16_t>&)reversedShareExp, pregAll16);
                MicroAPI::Cast<y1Dtype, bfloat16_t, castTraitBF16toFp4>(yZeroFP4, xBF16, pregAll16);
            }

            MicroAPI::StoreAlign<uint8_t, MicroAPI::StoreDist::DIST_PACK4_B32>(
                y2Addr + (j * 128), (MicroAPI::RegTensor<uint8_t>&)yZeroFP4, pregAll8);
        }
    }
}

// ============================================================================
// ComputeFP4FromHalf — FP4 quantization rounding logic (E2M1 / E1M2)
// ============================================================================
template <typename xDtype, typename y1Dtype, uint64_t mode, AscendC::RoundMode roundMode, uint64_t scaleAlg,
          uint64_t isGroupIdx>
__aicore__ inline void SwigluMxQuantWithDualAxisBase<xDtype, y1Dtype, mode, roundMode, scaleAlg,
                                                     isGroupIdx>::ComputeFP4FromHalf(MicroAPI::RegTensor<float>& Reg)
{
    MicroAPI::MaskReg pregAll32 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg zeroMask;
    MicroAPI::MaskReg specialMask;
    MicroAPI::MaskReg negInfMask;

    MicroAPI::RegTensor<int32_t> negZero;
    MicroAPI::RegTensor<int32_t> maxExpFP32;
    MicroAPI::RegTensor<int32_t> exp0FP32;
    MicroAPI::RegTensor<int32_t> exp1FP32;

    MicroAPI::Duplicate(negZero, NEG_ZERO);

    MicroAPI::Compare<int32_t, CMPMODE::EQ>(negInfMask, (MicroAPI::RegTensor<int32_t>&)Reg, negZero, pregAll32);
    if constexpr (IsSameType<y1Dtype, fp4x2_e1m2_t>::value) {
        // E1M2: multiply by 4, truncate, divide by 4
        MicroAPI::Muls(Reg, Reg, FOUR, pregAll32);
        MicroAPI::Compares<float, CMPMODE::LT>(specialMask, Reg, 0, pregAll32);
        MicroAPI::Truncate<float, roundMode>(Reg, Reg, pregAll32);
        MicroAPI::Muls(Reg, Reg, ONE_FOURTH, pregAll32);
    } else {
        // E2M1: bit-level exponent manipulation for correct rounding
        MicroAPI::Duplicate(maxExpFP32, MAX_EXP_FOR_FP32);
        MicroAPI::And(exp0FP32, (MicroAPI::RegTensor<int32_t>&)Reg, maxExpFP32, pregAll32);
        MicroAPI::ShiftRights(exp0FP32, exp0FP32, SHR_NUM_FOR_FP32, pregAll32);
        MicroAPI::Adds(exp0FP32, exp0FP32, FP32_BIAS_NEG, pregAll32);
        MicroAPI::Maxs(exp0FP32, exp0FP32, 0, pregAll32);
        MicroAPI::Adds(exp0FP32, exp0FP32, NEG_ONE, pregAll32);
        MicroAPI::Muls(exp1FP32, exp0FP32, NEG_ONE, pregAll32);
        MicroAPI::Adds(exp1FP32, exp1FP32, FP32_BIAS, pregAll32);
        MicroAPI::ShiftLefts(exp1FP32, exp1FP32, SHR_NUM_FOR_FP32, pregAll32);

        MicroAPI::Mul(Reg, Reg, (MicroAPI::RegTensor<float>&)exp1FP32, pregAll32);
        MicroAPI::Adds(exp0FP32, exp0FP32, FP32_BIAS, pregAll32);
        MicroAPI::ShiftLefts(exp0FP32, exp0FP32, SHR_NUM_FOR_FP32, pregAll32);
        MicroAPI::Compares<float, CMPMODE::LT>(specialMask, Reg, 0, pregAll32);
        MicroAPI::Truncate<float, roundMode>(Reg, Reg, pregAll32);
        MicroAPI::Mul(Reg, Reg, (MicroAPI::RegTensor<float>&)exp0FP32, pregAll32);
    }
    // Handle negative zero
    MicroAPI::Compares<float, CMPMODE::EQ>(zeroMask, Reg, 0, pregAll32);
    MicroAPI::And(zeroMask, specialMask, zeroMask, pregAll32);
    MicroAPI::Or(zeroMask, negInfMask, zeroMask, pregAll32);
    MicroAPI::Select<int32_t>((MicroAPI::RegTensor<int32_t>&)Reg, negZero, (MicroAPI::RegTensor<int32_t>&)Reg,
                              zeroMask);
}

// ============================================================================
// CopyOut — transfer y1, y2, scale1, scale2 from UB back to GM
// ============================================================================
template <typename xDtype, typename y1Dtype, uint64_t mode, AscendC::RoundMode roundMode, uint64_t scaleAlg,
          uint64_t isGroupIdx>
__aicore__ inline void SwigluMxQuantWithDualAxisBase<xDtype, y1Dtype, mode, roundMode, scaleAlg, isGroupIdx>::CopyOut(
    int64_t yOffset, int64_t scale1OutOffset, int64_t scale2OutOffset, int64_t blockCount, int64_t blockCountAlign,
    int64_t dataLen, int64_t dataLenAlign)
{
    uint16_t outBurst = static_cast<uint16_t>(blockCount);
    uint32_t outBlockLen = 0;
    uint32_t srcStride = 0;
    uint32_t dstStride = 0;

    int64_t yOffsetNow = yOffset;

    // axis=-2 two rows interleaved, accounting for 32B alignment
    uint32_t scaleSrcStride = DIGIT_TWO * ops::CeilDiv(dataLen, UB_BLOCK_SIZE) -
                              ops::CeilDiv(DIGIT_TWO * dataLen, UB_BLOCK_SIZE);

    if constexpr (IsSameType<y1Dtype, fp4x2_e2m1_t>::value || IsSameType<y1Dtype, fp4x2_e1m2_t>::value) {
        // FP4: two fp4 values packed into one byte
        outBlockLen = static_cast<uint32_t>(dataLen / DIGIT_TWO * sizeof(uint8_t));
        srcStride = static_cast<uint32_t>((ubRowLen_ - dataLen) / DIGIT_TWO * sizeof(uint8_t) / UB_BLOCK_SIZE);
        dstStride = static_cast<uint32_t>((dimN_ - dataLen) / DIGIT_TWO * sizeof(uint8_t));
        yOffsetNow = yOffset / DIGIT_TWO;
    } else {
        // FP8: one byte per element
        outBlockLen = static_cast<uint32_t>(dataLen * sizeof(uint8_t));
        srcStride = static_cast<uint32_t>((ubRowLen_ - dataLen) * sizeof(y1Dtype) / UB_BLOCK_SIZE);
        dstStride = static_cast<uint32_t>((dimN_ - dataLen) * sizeof(uint8_t));
    }

    DataCopyExtParams yCopyOutParams = {outBurst, outBlockLen, srcStride, dstStride, 0};

    // axis=-1 scale output: shape [M, ceil(N/blockSize)]
    uint32_t scale1OutLen = dataLenAlign / BLOCK_SIZE;

    DataCopyExtParams scale1CopyOutParams = {
        outBurst, static_cast<uint32_t>(scale1OutLen * sizeof(uint8_t)), static_cast<uint32_t>(0),
        static_cast<uint32_t>(ops::CeilAlign(dimN_, DOUBLE_BLOCK_SIZE) / BLOCK_SIZE - scale1OutLen),
        static_cast<uint32_t>(0)};

    // axis=-2 scale output: two rows interleaved per blockSize group
    DataCopyExtParams scale2CopyOutParams = {
        static_cast<uint16_t>(blockCountAlign / DOUBLE_BLOCK_SIZE),
        static_cast<uint32_t>(dataLen * DIGIT_TWO * sizeof(uint8_t)), static_cast<uint32_t>(scaleSrcStride),
        static_cast<uint32_t>(DIGIT_TWO * (dimN_ - dataLen) * sizeof(uint8_t)), static_cast<uint32_t>(0)};

    // Dequeue and copy y1
    LocalTensor<uint8_t> y1Local = outQueue1_.template DeQue<uint8_t>();
    DataCopyPad(yGm1_[yOffsetNow], y1Local, yCopyOutParams);
    outQueue1_.FreeTensor(y1Local);

    // Dequeue and copy y2
    LocalTensor<uint8_t> y2Local = outQueue2_.template DeQue<uint8_t>();
    DataCopyPad(yGm2_[yOffsetNow], y2Local, yCopyOutParams);
    outQueue2_.FreeTensor(y2Local);

    // Dequeue and copy scale1 (axis=-1)
    LocalTensor<uint8_t> mxScale1Local = mxScaleQueue1_.template DeQue<uint8_t>();
    DataCopyPad(mxScaleGm1_[scale1OutOffset], mxScale1Local, scale1CopyOutParams);
    mxScaleQueue1_.FreeTensor(mxScale1Local);

    // Dequeue and copy scale2 (axis=-2)
    LocalTensor<uint8_t> mxScale2Local = mxScaleQueue2_.template DeQue<uint8_t>();
    DataCopyPad(mxScaleGm2_[scale2OutOffset], mxScale2Local, scale2CopyOutParams);
    mxScaleQueue2_.FreeTensor(mxScale2Local);
}
} // namespace SwigluMxQuantWithDualAxis

#endif // OPS_NN_SWIGLU_MX_QUANT_WITH_DUAL_AXIS_REGBASE_H
