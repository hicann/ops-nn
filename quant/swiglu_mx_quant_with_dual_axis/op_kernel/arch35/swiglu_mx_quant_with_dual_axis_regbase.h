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
    // DAV_3510 Reg models source RegTensor/MaskReg operands as mutable references;
    // these parameters are semantically read-only but cannot be const-qualified.
    __aicore__ inline void ComputeScaleCuBLASForSlot(
        __ubuf__ uint16_t* maxReadAddr, __ubuf__ uint16_t* reciprocalWriteAddr, Reg::RegTensor<uint8_t>& scale8,
        Reg::RegTensor<uint32_t>& invMax, Reg::RegTensor<uint32_t>& manMaskReg, Reg::RegTensor<uint32_t>& expMaskReg,
        Reg::RegTensor<uint32_t>& zero32Reg, Reg::RegTensor<uint32_t>& scaleBiasReg, Reg::RegTensor<uint32_t>& nan32Reg,
        Reg::RegTensor<uint32_t>& fp8Nan32Reg, Reg::MaskReg& maskAll, Reg::MaskReg& maskAll32, Reg::MaskReg& maskB16);
    __aicore__ inline void ComputeY1ToFP8(uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr,
                                          __ubuf__ uint16_t* mxScale1ReciprocalAddr, __ubuf__ uint8_t* y1Addr);
    __aicore__ inline void ComputeY1ToFP4(uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr,
                                          __ubuf__ uint16_t* mxScale1ReciprocalAddr, __ubuf__ uint8_t* y1Addr);
    __aicore__ inline void ComputeY2ToFP8(uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr,
                                          __ubuf__ uint16_t* mxScale2ReciprocalAddr, __ubuf__ uint8_t* y2Addr);
    __aicore__ inline void ComputeY2ToFP4(uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr,
                                          __ubuf__ uint16_t* mxScale2ReciprocalAddr, __ubuf__ uint8_t* y2Addr);
    __aicore__ inline void ComputeFP4FromHalf(Reg::RegTensor<float>& Reg);
    __aicore__ inline void CopyOut(int64_t yOffset, int64_t scale1OutOffset, int64_t scale2OutOffset,
                                   int64_t blockCount, int64_t blockCountAlign, int64_t dataLen, int64_t dataLenAlign);
    __aicore__ inline void ComputeInterleave(__ubuf__ uint8_t* dstAddr, __ubuf__ uint8_t* src0Addr,
                                             __ubuf__ uint8_t* src1Addr);

protected:
    static constexpr Reg::CastTrait castTraitBF16toFp4 = {Reg::RegLayout::ZERO, Reg::SatMode::SAT,
                                                          Reg::MaskMergeMode::ZEROING, roundMode};
    static constexpr Reg::CastTrait castTraitFp32toBF16 = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                           Reg::MaskMergeMode::ZEROING, roundMode};
    static constexpr Reg::CastTrait castTraitFp32toYdtype = {Reg::RegLayout::ZERO, Reg::SatMode::SAT,
                                                             Reg::MaskMergeMode::ZEROING, roundMode};

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
        Reg::RegTensor<uint8_t> src0Reg;
        Reg::RegTensor<uint8_t> src1Reg;
        Reg::MaskReg maskB8 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
        Reg::LoadAlign(src0Reg, src0Addr);
        Reg::LoadAlign(src1Reg, src1Addr);
        Reg::StoreAlign<uint8_t, Reg::StoreDist::DIST_INTLV_B8>(dstAddr, src0Reg, src1Reg, maskB8);
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
        Reg::RegTensor<xDtype> vregAct;
        Reg::RegTensor<xDtype> vregGate;
        Reg::RegTensor<float> vregActF;
        Reg::RegTensor<float> vregGateF;
        Reg::RegTensor<float> negReg;
        Reg::RegTensor<float> expReg;
        Reg::RegTensor<float> addsReg;
        Reg::RegTensor<float> sigmoidReg;
        Reg::RegTensor<float> outFReg;
        Reg::RegTensor<xDtype> outTReg;

        Reg::MaskReg mask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
        Reg::MaskReg mask1 = Reg::UpdateMask<float>(mask1Num);
        Reg::MaskReg mask2 = Reg::UpdateMask<float>(mask2Num);
        Reg::MaskReg mask3 = Reg::UpdateMask<xDtype>(mask3Num);

        for (uint16_t dim0vfLoopIdx = 0; dim0vfLoopIdx < dim0VfTimes; dim0vfLoopIdx++) {
            // Full VF iterations (no tail)
            for (uint16_t dim1vfLoopIdx = 0; dim1vfLoopIdx < dim1VfTimes; dim1vfLoopIdx++) {
                Reg::AddrReg srcIdxOffset = Reg::CreateAddrReg<xDtype>(dim0vfLoopIdx, alignDim1In, dim1vfLoopIdx,
                                                                       VF_LEN_FP32);
                Reg::LoadAlign<xDtype, Reg::LoadDist::DIST_UNPACK_B16>(vregAct, actAddr, srcIdxOffset);
                Reg::LoadAlign<xDtype, Reg::LoadDist::DIST_UNPACK_B16>(vregGate, gateAddr, srcIdxOffset);

                Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(vregActF, vregAct, mask);
                Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(vregGateF, vregGate, mask);

                Reg::Muls(negReg, vregActF, negScalarOne, mask);
                Reg::Exp(expReg, negReg, mask);
                Reg::Adds(addsReg, expReg, scalarOne, mask);
                Reg::Div(sigmoidReg, vregActF, addsReg, mask);
                Reg::Mul(outFReg, sigmoidReg, vregGateF, mask);

                Reg::Cast<xDtype, float, CAST_FP32_TO_FP16_BF16>(outTReg, outFReg, mask);
                Reg::AddrReg outOffset = Reg::CreateAddrReg<xDtype>(dim0vfLoopIdx, outAllNum, dim1vfLoopIdx,
                                                                    VF_LEN_FP32);
                Reg::StoreAlign<xDtype, Reg::StoreDist::DIST_PACK_B32>(swigluOutAddr, outTReg, outOffset, mask);
            }

            // Tail VF iteration with mask-based zero-padding
            Reg::AddrReg srcIdxOffset1 = Reg::CreateAddrReg<xDtype>(dim0vfLoopIdx, alignDim1In);
            Reg::AddrReg outOffset1 = Reg::CreateAddrReg<xDtype>(dim0vfLoopIdx, outAllNum);

            for (uint16_t aa = 0; aa < dim1TailTimes; aa++) {
                Reg::LoadAlign<xDtype, Reg::LoadDist::DIST_UNPACK_B16>(vregAct, actAddr1, srcIdxOffset1);
                Reg::LoadAlign<xDtype, Reg::LoadDist::DIST_UNPACK_B16>(vregGate, gateAddr1, srcIdxOffset1);

                Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(vregActF, vregAct, mask1);
                Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(vregGateF, vregGate, mask1);

                Reg::Muls(negReg, vregActF, negScalarOne, mask1);
                Reg::Exp(expReg, negReg, mask1);
                Reg::Adds(addsReg, expReg, scalarOne, mask1);
                Reg::Div(sigmoidReg, vregActF, addsReg, mask1);
                Reg::Mul(outFReg, sigmoidReg, vregGateF, mask1);

                // mask2 writes zeros for positions beyond valid data (zero-mode mask)
                Reg::Cast<xDtype, float, CAST_FP32_TO_FP16_BF16>(outTReg, outFReg, mask1);
                Reg::StoreAlign<xDtype, Reg::StoreDist::DIST_PACK_B32>(swigluAddr1, outTReg, outOffset1, mask2);
            }
            // Additional zero-fill for extra padding positions
            for (uint16_t cc = 0; cc < dim1Tail2; cc++) {
                Duplicate<xDtype>(vregAct, numZero);
                Reg::StoreAlign<xDtype>(swigluAddr2, vregAct, outOffset1, mask3);
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
        Reg::RegTensor<xDtype> vregAct;
        Reg::RegTensor<xDtype> vregGate;
        Reg::RegTensor<float> vregActF;
        Reg::RegTensor<float> vregGateF;
        Reg::RegTensor<float> negReg;
        Reg::RegTensor<float> expReg;
        Reg::RegTensor<float> addsReg;
        Reg::RegTensor<float> sigmoidReg;
        Reg::RegTensor<float> outFReg;
        Reg::RegTensor<xDtype> outTReg;
        Reg::MaskReg mask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();

        for (uint16_t row = 0; row < fullTileRows; row++) {
            for (uint16_t vf = 0; vf < fullTileVfs; vf++) {
                Reg::AddrReg srcOffset = Reg::CreateAddrReg<xDtype>(row, fullTileCols, vf, VF_LEN_FP32);
                Reg::LoadAlign<xDtype, Reg::LoadDist::DIST_UNPACK_B16>(vregAct, actAddr, srcOffset);
                Reg::LoadAlign<xDtype, Reg::LoadDist::DIST_UNPACK_B16>(vregGate, gateAddr, srcOffset);
                Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(vregActF, vregAct, mask);
                Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(vregGateF, vregGate, mask);
                Reg::Muls(negReg, vregActF, negScalarOne, mask);
                Reg::Exp(expReg, negReg, mask);
                Reg::Adds(addsReg, expReg, scalarOne, mask);
                Reg::Div(sigmoidReg, vregActF, addsReg, mask);
                Reg::Mul(outFReg, sigmoidReg, vregGateF, mask);
                Reg::Cast<xDtype, float, CAST_FP32_TO_FP16_BF16>(outTReg, outFReg, mask);
                Reg::AddrReg outOffset = Reg::CreateAddrReg<xDtype>(row, fullTileCols, vf, VF_LEN_FP32);
                Reg::StoreAlign<xDtype, Reg::StoreDist::DIST_PACK_B32>(swigluOutAddr, outTReg, outOffset, mask);
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
        Reg::RegTensor<xDtype> zeroReg;
        AscendC::Reg::MaskReg mask;
        Reg::Duplicate(zeroReg, 0);
        for (uint16_t i = 0; i < times; i++) {
            mask = AscendC::Reg::UpdateMask<xDtype>(size);
            Reg::AddrReg offset = Reg::CreateAddrReg<xDtype>(i, VF_LEN_B16);
            AscendC::Reg::StoreAlign(swigluOutAddr, zeroReg, offset, mask);
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
        Reg::RegTensor<xDtype> x0;
        Reg::RegTensor<xDtype> x1;
        Reg::RegTensor<uint16_t> x0ExpFP16;
        Reg::RegTensor<uint16_t> x1ExpFP16;
        Reg::RegTensor<bfloat16_t> x0BF16;
        Reg::RegTensor<bfloat16_t> x1BF16;
        Reg::RegTensor<uint16_t> x0ExpBF16;
        Reg::RegTensor<uint16_t> x1ExpBF16;
        Reg::RegTensor<uint16_t> expMaskBF16;
        Reg::RegTensor<uint16_t> expMaskFP16;
        Reg::RegTensor<uint16_t> expMaxDim1;
        Reg::RegTensor<uint16_t> expMax1Dim2;
        Reg::RegTensor<uint16_t> expMax2Dim2;
        Reg::RegTensor<uint16_t> yMaxExp;
        Reg::RegTensor<uint16_t> nanE8M0;
        Reg::RegTensor<uint16_t> biasE8M0;
        Reg::RegTensor<uint16_t> zero;
        Reg::RegTensor<uint16_t> nanBF16;
        Reg::RegTensor<uint16_t> specialExp;
        Reg::RegTensor<uint16_t> mxScale1B16;
        Reg::RegTensor<uint8_t> mxScale1B8;
        Reg::RegTensor<uint16_t> reversedShareExp1;

        Reg::RegTensor<uint16_t> mxScale2ZeroB16;
        Reg::RegTensor<uint8_t> mxScale2ZeroB8;
        Reg::RegTensor<uint16_t> reversedShareExp2Zero;
        Reg::RegTensor<uint16_t> mxScale2OneB16;
        Reg::RegTensor<uint8_t> mxScale2OneB8;
        Reg::RegTensor<uint16_t> reversedShareExp2One;

        Reg::MaskReg infMask;
        Reg::MaskReg zeroMask;
        Reg::MaskReg invalidDataMask;
        Reg::MaskReg infNanDataMask0;
        Reg::MaskReg infNanDataMask1;
        Reg::MaskReg maskAll = Reg::CreateMask<xDtype, Reg::MaskPattern::ALL>();

        Reg::MaskReg maskB8 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg maskReduceB8 = Reg::CreateMask<uint8_t, Reg::MaskPattern::VL8>();
        Reg::MaskReg maskReduceB16 = Reg::CreateMask<uint8_t, Reg::MaskPattern::VL16>();

        Reg::Duplicate(expMaskBF16, EXP_MASK_BF16);
        Reg::Duplicate(expMaskFP16, EXP_MASK_FP16);
        Reg::Duplicate(expMax1Dim2, static_cast<uint16_t>(0));
        Reg::Duplicate(expMax2Dim2, static_cast<uint16_t>(0));
        Reg::Duplicate(yMaxExp, localDtypeYMaxExp);
        Reg::Duplicate(nanE8M0, NAN_FOR_FP8_E8M0);
        Reg::Duplicate(biasE8M0, BF16_EXP_BIAS);
        Reg::Duplicate(zero, static_cast<uint16_t>(0));
        Reg::Duplicate(nanBF16, NAN_CUSTOMIZATION);
        Reg::Duplicate(specialExp, SPECIAL_EXP_THRESHOLD);

        for (uint16_t i = 0; i < blockCount; i++) {
            // Interleaved load: splits blockW bf16/fp16 elements into even/odd halves
            Reg::LoadAlign<xDtype, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B16>(x0, x1, xAddr,
                                                                                                       ONCE_ROW_LEN);
            if constexpr (IsSameType<xDtype, half>::value) {
                // FP16 path: extract exponent, check INF/NaN, cast to BF16
                Reg::And(x0ExpFP16, (Reg::RegTensor<uint16_t>&)x0, expMaskFP16, maskAll);
                Reg::And(x1ExpFP16, (Reg::RegTensor<uint16_t>&)x1, expMaskFP16, maskAll);
                Reg::Compare<uint16_t, CMPMODE::NE>(infNanDataMask0, x0ExpFP16, expMaskFP16, maskAll);
                Reg::Compare<uint16_t, CMPMODE::NE>(infNanDataMask1, x1ExpFP16, expMaskFP16, maskAll);
                Reg::Cast<bfloat16_t, xDtype, CAST_HALF_TO_BF16>(x0BF16, x0, maskAll);
                Reg::Cast<bfloat16_t, xDtype, CAST_HALF_TO_BF16>(x1BF16, x1, maskAll);
                Reg::And(x0ExpBF16, (Reg::RegTensor<uint16_t>&)x0BF16, expMaskBF16, maskAll);
                Reg::And(x1ExpBF16, (Reg::RegTensor<uint16_t>&)x1BF16, expMaskBF16, maskAll);
                Reg::Select<uint16_t>(x0ExpBF16, x0ExpBF16, expMaskBF16, infNanDataMask0);
                Reg::Select<uint16_t>(x1ExpBF16, x1ExpBF16, expMaskBF16, infNanDataMask1);
            } else {
                // BF16 path: extract exponent bits directly
                Reg::And(x0ExpBF16, (Reg::RegTensor<uint16_t>&)x0, expMaskBF16, maskAll);
                Reg::And(x1ExpBF16, (Reg::RegTensor<uint16_t>&)x1, expMaskBF16, maskAll);
            }

            // axis=-1: max of adjacent pair exponents
            Reg::Max(expMaxDim1, x0ExpBF16, x1ExpBF16, maskAll);
            Reg::ReduceDataBlock<AscendC::Reg::ReduceType::MAX>(expMaxDim1, expMaxDim1, maskAll);

            // axis=-2: accumulate column-wise max exponents across rows
            Reg::Max(expMax1Dim2, expMax1Dim2, x0ExpBF16, maskAll);
            Reg::Max(expMax2Dim2, expMax2Dim2, x1ExpBF16, maskAll);

            // ---- axis=-1 scale computation ----
            Reg::Compare<uint16_t, CMPMODE::NE>(infMask, expMaxDim1, expMaskBF16, maskAll);
            Reg::Compare<uint16_t, CMPMODE::NE>(zeroMask, expMaxDim1, zero, maskAll);
            Reg::Compare<uint16_t, CMPMODE::LE>(invalidDataMask, expMaxDim1, yMaxExp, maskAll);
            Reg::Select<uint16_t>(expMaxDim1, yMaxExp, expMaxDim1, invalidDataMask);

            Reg::Sub(expMaxDim1, expMaxDim1, yMaxExp, maskAll);
            Reg::ShiftRights(mxScale1B16, expMaxDim1, SHR_NUM_FOR_BF16, maskAll);
            Reg::Select<uint16_t>(mxScale1B16, mxScale1B16, nanE8M0, infMask);
            Reg::Select<uint16_t>(mxScale1B16, mxScale1B16, zero, zeroMask);

            Reg::Pack<uint8_t, uint16_t, Reg::HighLowPart::LOWEST>(mxScale1B8, mxScale1B16);
            Reg::StoreAlign<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE>(mxScale1Addr, mxScale1B8, UB_BLOCK_SIZE,
                                                                         maskReduceB8);
            // Compute 1/scale
            Reg::Compare<uint16_t, CMPMODE::EQ>(invalidDataMask, expMaxDim1, biasE8M0, maskAll);

            Reg::Sub(reversedShareExp1, biasE8M0, expMaxDim1, maskAll);
            Reg::Select<uint16_t>(reversedShareExp1, reversedShareExp1, nanBF16, infMask);
            Reg::Select<uint16_t>(reversedShareExp1, reversedShareExp1, zero, zeroMask);
            Reg::Select<uint16_t>(reversedShareExp1, specialExp, reversedShareExp1, invalidDataMask);
            Reg::StoreAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(mxScale1ReciprocalAddr, reversedShareExp1,
                                                                          SCALE1_RECIPROCAL_ROW_ELEMS, maskReduceB16);
        }

        // ---- axis=-2 scale computation (interleaved part 1: even rows) ----
        Reg::Compare<uint16_t, CMPMODE::NE>(infMask, expMax1Dim2, expMaskBF16, maskAll);
        Reg::Compare<uint16_t, CMPMODE::NE>(zeroMask, expMax1Dim2, zero, maskAll);
        Reg::Compare<uint16_t, CMPMODE::LE>(invalidDataMask, expMax1Dim2, yMaxExp, maskAll);
        Reg::Select<uint16_t>(expMax1Dim2, yMaxExp, expMax1Dim2, invalidDataMask);
        Reg::Sub(expMax1Dim2, expMax1Dim2, yMaxExp, maskAll);
        Reg::ShiftRights(mxScale2ZeroB16, expMax1Dim2, SHR_NUM_FOR_BF16, maskAll);
        Reg::Select<uint16_t>(mxScale2ZeroB16, mxScale2ZeroB16, nanE8M0, infMask);
        Reg::Select<uint16_t>(mxScale2ZeroB16, mxScale2ZeroB16, zero, zeroMask);

        Reg::Pack<uint8_t, uint16_t, Reg::HighLowPart::LOWEST>(mxScale2ZeroB8, mxScale2ZeroB16);

        Reg::Compare<uint16_t, CMPMODE::EQ>(invalidDataMask, expMax1Dim2, biasE8M0, maskAll); // scale计算结束

        Reg::Sub(reversedShareExp2Zero, biasE8M0, expMax1Dim2, maskAll);
        Reg::Select<uint16_t>(reversedShareExp2Zero, reversedShareExp2Zero, nanBF16, infMask);
        Reg::Select<uint16_t>(reversedShareExp2Zero, reversedShareExp2Zero, zero, zeroMask);
        Reg::Select<uint16_t>(reversedShareExp2Zero, specialExp, reversedShareExp2Zero,
                              invalidDataMask); // int16 scale 结束

        // ---- axis=-2 scale computation (interleaved part 2: odd rows) ----
        Reg::Compare<uint16_t, CMPMODE::NE>(infMask, expMax2Dim2, expMaskBF16, maskAll);
        Reg::Compare<uint16_t, CMPMODE::NE>(zeroMask, expMax2Dim2, zero, maskAll);
        Reg::Compare<uint16_t, CMPMODE::LE>(invalidDataMask, expMax2Dim2, yMaxExp, maskAll);
        Reg::Select<uint16_t>(expMax2Dim2, yMaxExp, expMax2Dim2, invalidDataMask);
        Reg::Sub(expMax2Dim2, expMax2Dim2, yMaxExp, maskAll);
        Reg::ShiftRights(mxScale2OneB16, expMax2Dim2, SHR_NUM_FOR_BF16, maskAll);
        Reg::Select<uint16_t>(mxScale2OneB16, mxScale2OneB16, nanE8M0, infMask);
        Reg::Select<uint16_t>(mxScale2OneB16, mxScale2OneB16, zero, zeroMask);

        Reg::Pack<uint8_t, uint16_t, Reg::HighLowPart::LOWEST>(mxScale2OneB8, mxScale2OneB16);

        Reg::Compare<uint16_t, CMPMODE::EQ>(invalidDataMask, expMax2Dim2, biasE8M0, maskAll);
        Reg::Sub(reversedShareExp2One, biasE8M0, expMax2Dim2, maskAll);
        Reg::Select<uint16_t>(reversedShareExp2One, reversedShareExp2One, nanBF16, infMask);
        Reg::Select<uint16_t>(reversedShareExp2One, reversedShareExp2One, zero, zeroMask);
        Reg::Select<uint16_t>(reversedShareExp2One, specialExp, reversedShareExp2One, invalidDataMask);

        // Interleaved store: merge even/odd scale and 1/scale for axis=-2
        Reg::StoreAlign<uint8_t, Reg::StoreDist::DIST_INTLV_B8>(mxScale2Addr, mxScale2ZeroB8, mxScale2OneB8, maskB8);
        Reg::StoreAlign<uint16_t, Reg::StoreDist::DIST_INTLV_B16>(mxScale2ReciprocalAddr, reversedShareExp2Zero,
                                                                  reversedShareExp2One, maskAll);
    }
}

template <typename xDtype, typename y1Dtype, uint64_t mode, AscendC::RoundMode roundMode, uint64_t scaleAlg,
          uint64_t isGroupIdx>
__aicore__ inline void
SwigluMxQuantWithDualAxisBase<xDtype, y1Dtype, mode, roundMode, scaleAlg, isGroupIdx>::ComputeScaleCuBLASForSlot(
    __ubuf__ uint16_t* maxReadAddr, __ubuf__ uint16_t* reciprocalWriteAddr, Reg::RegTensor<uint8_t>& scale8,
    Reg::RegTensor<uint32_t>& invMax, Reg::RegTensor<uint32_t>& manMaskReg, Reg::RegTensor<uint32_t>& expMaskReg,
    Reg::RegTensor<uint32_t>& zero32Reg, Reg::RegTensor<uint32_t>& scaleBiasReg, Reg::RegTensor<uint32_t>& nan32Reg,
    Reg::RegTensor<uint32_t>& fp8Nan32Reg, Reg::MaskReg& maskAll, Reg::MaskReg& maskAll32, Reg::MaskReg& maskB16)
{
    Reg::RegTensor<uint16_t> max16Reg;
    Reg::RegTensor<uint32_t> max32Reg;
    Reg::RegTensor<uint32_t> exp32Reg;
    Reg::RegTensor<uint32_t> man32Reg;
    Reg::RegTensor<uint32_t> expOne32Reg;
    Reg::RegTensor<uint32_t> extractExp;
    Reg::RegTensor<uint32_t> halfScale;
    Reg::RegTensor<uint16_t> scale16;
    Reg::RegTensor<uint16_t> recip16;
    Reg::MaskReg cmpResult;
    Reg::MaskReg zeroMask;
    Reg::MaskReg p0;
    Reg::MaskReg p1;

    Reg::LoadAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_UNPACK_B16>(max16Reg, maxReadAddr,
                                                                                                 VF_LEN_FP32);
    Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>((Reg::RegTensor<float>&)max32Reg, (Reg::RegTensor<xDtype>&)max16Reg,
                                                  maskAll);
    Reg::Compare<uint32_t, CMPMODE::LT>(cmpResult, max32Reg, expMaskReg, maskAll32);
    Reg::Compare<uint32_t, CMPMODE::NE>(zeroMask, max32Reg, zero32Reg, maskAll32);
    Reg::Mul((Reg::RegTensor<float>&)max32Reg, (Reg::RegTensor<float>&)max32Reg, (Reg::RegTensor<float>&)invMax,
             maskAll32);
    Reg::ShiftRights(exp32Reg, max32Reg, SHR_NUM_FOR_FP32, maskAll32);
    Reg::And(man32Reg, max32Reg, manMaskReg, maskAll32);
    Reg::Compares<uint32_t, CMPMODE::GT>(p0, exp32Reg, static_cast<uint32_t>(0), maskAll32);
    Reg::Compares<uint32_t, CMPMODE::LT>(p0, exp32Reg, EXP_254, p0);
    Reg::Compares<uint32_t, CMPMODE::GT>(p0, man32Reg, static_cast<uint32_t>(0), p0);
    Reg::Compares<uint32_t, CMPMODE::EQ>(p1, exp32Reg, static_cast<uint32_t>(0), maskAll32);
    Reg::Compares<uint32_t, CMPMODE::GT>(p1, man32Reg, HALF_FOR_MAN, p1);
    Reg::Or(p0, p0, p1, maskAll32);
    Reg::Adds(expOne32Reg, exp32Reg, 1, maskAll32);
    Reg::Select(extractExp, expOne32Reg, exp32Reg, p0);
    Reg::Select<uint32_t>(extractExp, extractExp, fp8Nan32Reg, cmpResult);
    Reg::Select<uint32_t>(extractExp, extractExp, zero32Reg, zeroMask);
    Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>(scale16, extractExp);
    Reg::Pack<uint8_t, uint16_t, Reg::HighLowPart::LOWEST>(scale8, scale16);

    Reg::ShiftLefts(extractExp, extractExp, SHR_NUM_FOR_BF16, maskAll32);
    Reg::Sub(halfScale, scaleBiasReg, extractExp, maskAll32);
    Reg::Select<uint32_t>(halfScale, halfScale, nan32Reg, cmpResult);
    Reg::Select<uint32_t>(halfScale, halfScale, zero32Reg, zeroMask);
    Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>(recip16, halfScale);
    Reg::StoreAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(reciprocalWriteAddr, recip16, VF_LEN_FP32, maskB16);
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
        Reg::RegTensor<uint32_t> invMax;
        Reg::RegTensor<uint32_t> manMaskReg;
        Reg::RegTensor<uint32_t> expMaskReg;
        Reg::RegTensor<uint32_t> zero32Reg;
        Reg::RegTensor<uint32_t> scaleBiasReg;
        Reg::RegTensor<uint32_t> nan32Reg;
        Reg::RegTensor<uint32_t> fp8Nan32Reg;
        Reg::RegTensor<uint8_t> scale8Slot0;
        Reg::RegTensor<uint8_t> scale8Slot1;
        Reg::MaskReg maskAll = Reg::CreateMask<xDtype, Reg::MaskPattern::ALL>();
        Reg::MaskReg maskAll32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg maskB16 = Reg::CreateMask<uint16_t, Reg::MaskPattern::VL64>();
        Reg::MaskReg interleaveMask = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
        Reg::Duplicate(scaleBiasReg, FP32_EXP_BIAS_CUBLAS);
        Reg::Duplicate(expMaskReg, MAX_EXP_FOR_FP32);
        Reg::Duplicate(zero32Reg, static_cast<uint32_t>(0));
        Reg::Duplicate(invMax, localInvDtypeMax);
        Reg::Duplicate(manMaskReg, MAN_MASK_FLOAT);
        Reg::Duplicate(fp8Nan32Reg, MAX_EXP_FOR_FP8_IN_FP32);
        Reg::Duplicate(nan32Reg, static_cast<uint32_t>(NAN_CUSTOMIZATION));
        for (uint16_t i = 0; i < times; i++) {
            uint16_t colOffset = i * VF_LEN_FP32;
            __ubuf__ uint16_t* slot0Addr = mxScale2ReciprocalAddr + colOffset;
            __ubuf__ uint16_t* slot1Addr = mxScale2ReciprocalAddr + dataLen + colOffset;
            ComputeScaleCuBLASForSlot(slot0Addr, slot0Addr, scale8Slot0, invMax, manMaskReg, expMaskReg, zero32Reg,
                                      scaleBiasReg, nan32Reg, fp8Nan32Reg, maskAll, maskAll32, maskB16);
            ComputeScaleCuBLASForSlot(slot1Addr, slot1Addr, scale8Slot1, invMax, manMaskReg, expMaskReg, zero32Reg,
                                      scaleBiasReg, nan32Reg, fp8Nan32Reg, maskAll, maskAll32, maskB16);
            Reg::StoreAlign<uint8_t, Reg::StoreDist::DIST_INTLV_B8>(mxScale2Addr + DIGIT_TWO * colOffset, scale8Slot0,
                                                                    scale8Slot1, interleaveMask);
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
        Reg::RegTensor<xDtype> x0;
        Reg::RegTensor<xDtype> x1;
        Reg::RegTensor<uint16_t> x0Abs;
        Reg::RegTensor<uint16_t> x1Abs;
        Reg::RegTensor<uint16_t> absMaxDim1;
        Reg::RegTensor<uint16_t> scale2Slot0Part0;
        Reg::RegTensor<uint16_t> scale2Slot0Part1;
        Reg::RegTensor<uint16_t> scale2Slot1Part0;
        Reg::RegTensor<uint16_t> scale2Slot1Part1;
        Reg::RegTensor<uint32_t> max32;
        Reg::RegTensor<uint32_t> exp32;
        Reg::RegTensor<uint32_t> man32;
        Reg::RegTensor<uint32_t> expAddOne32;
        Reg::RegTensor<uint32_t> extractExp;
        Reg::RegTensor<uint32_t> halfScale;
        Reg::RegTensor<uint16_t> scale16;
        Reg::RegTensor<uint8_t> scale8Reg;
        Reg::RegTensor<uint8_t> scale8Row;
        Reg::RegTensor<uint16_t> recip16Reg;
        Reg::RegTensor<uint16_t> recip16Row;
        Reg::RegTensor<int8_t> extractIdx;
        Reg::RegTensor<uint16_t> absMask;
        Reg::RegTensor<uint32_t> invMax;
        Reg::RegTensor<uint32_t> manMaskReg;
        Reg::RegTensor<uint32_t> expMaskReg;
        Reg::RegTensor<uint32_t> zeroReg32;
        Reg::RegTensor<uint32_t> scaleBiasReg;
        Reg::RegTensor<uint32_t> nanReg32;
        Reg::RegTensor<uint32_t> fp8NanReg32;
        Reg::MaskReg cmpResult;
        Reg::MaskReg zeroMask;
        Reg::MaskReg p0;
        Reg::MaskReg p1;
        Reg::MaskReg maskAll = Reg::CreateMask<xDtype, Reg::MaskPattern::ALL>();
        Reg::MaskReg maskAll32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg maskReduceB8 = Reg::CreateMask<uint8_t, Reg::MaskPattern::VL8>();
        Reg::MaskReg maskReduceB16 = Reg::CreateMask<uint8_t, Reg::MaskPattern::VL16>();
        Reg::UnalignReg ureg;

        Reg::Duplicate(absMask, ABS_MASK_FOR_16BIT);
        Reg::Duplicate(scale2Slot0Part0, static_cast<uint16_t>(0));
        Reg::Duplicate(scale2Slot0Part1, static_cast<uint16_t>(0));
        Reg::Duplicate(scale2Slot1Part0, static_cast<uint16_t>(0));
        Reg::Duplicate(scale2Slot1Part1, static_cast<uint16_t>(0));

        // Phase 1: collect scale1 maxima compactly while preserving perf/all's
        // two independent 32-row scale2 slots.
        uint16_t slot0Rows = blockCount < BLOCK_SIZE ? blockCount : BLOCK_SIZE;
        for (uint16_t i = 0; i < slot0Rows; i++) {
            Reg::LoadAlign<xDtype, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B16>(x0, x1, xAddr,
                                                                                                       dataLen);
            Reg::And(x0Abs, (Reg::RegTensor<uint16_t>&)x0, absMask, maskAll);
            Reg::And(x1Abs, (Reg::RegTensor<uint16_t>&)x1, absMask, maskAll);
            Reg::Max(absMaxDim1, x0Abs, x1Abs, maskAll);
            Reg::ReduceDataBlock<AscendC::Reg::ReduceType::MAX>(absMaxDim1, absMaxDim1, maskAll);
            Reg::StoreUnAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(tempMaxAddr, absMaxDim1, ureg,
                                                                            dataLen / BLOCK_SIZE);
            Reg::Max(scale2Slot0Part0, scale2Slot0Part0, x0Abs, maskAll);
            Reg::Max(scale2Slot0Part1, scale2Slot0Part1, x1Abs, maskAll);
        }
        for (uint16_t i = slot0Rows; i < blockCount; i++) {
            Reg::LoadAlign<xDtype, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B16>(x0, x1, xAddr,
                                                                                                       dataLen);
            Reg::And(x0Abs, (Reg::RegTensor<uint16_t>&)x0, absMask, maskAll);
            Reg::And(x1Abs, (Reg::RegTensor<uint16_t>&)x1, absMask, maskAll);
            Reg::Max(absMaxDim1, x0Abs, x1Abs, maskAll);
            Reg::ReduceDataBlock<AscendC::Reg::ReduceType::MAX>(absMaxDim1, absMaxDim1, maskAll);
            Reg::StoreUnAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(tempMaxAddr, absMaxDim1, ureg,
                                                                            dataLen / BLOCK_SIZE);
            Reg::Max(scale2Slot1Part0, scale2Slot1Part0, x0Abs, maskAll);
            Reg::Max(scale2Slot1Part1, scale2Slot1Part1, x1Abs, maskAll);
        }
        Reg::StoreUnAlignPost(tempMaxAddr, ureg, 0);

        Reg::Duplicate(invMax, localInvDtypeMax);
        Reg::Duplicate(manMaskReg, MAN_MASK_FLOAT);
        Reg::Duplicate(expMaskReg, MAX_EXP_FOR_FP32);
        Reg::Duplicate(zeroReg32, static_cast<uint32_t>(0));
        Reg::Duplicate(scaleBiasReg, FP32_EXP_BIAS_CUBLAS);
        Reg::Duplicate(nanReg32, static_cast<uint32_t>(NAN_CUSTOMIZATION));
        Reg::Duplicate(fp8NanReg32, MAX_EXP_FOR_FP8_IN_FP32);

        // Phase 2: convert 64 cached maxima per VF, then scatter eight rows
        // back to the original 32-byte-aligned scale/reciprocal layout.
        __ubuf__ uint16_t* readMaxAddr = mxScale2ReciprocalAddr;
        uint16_t scaleCount = dataLen / BLOCK_SIZE;
        uint16_t batchCount = ops::CeilDiv(static_cast<uint16_t>(blockCount * scaleCount),
                                           static_cast<uint16_t>(VF_LEN_FP32));
        for (uint16_t j = 0; j < batchCount; j++) {
            Reg::LoadAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_UNPACK_B16>(
                absMaxDim1, readMaxAddr, VF_LEN_FP32);
            Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>((Reg::RegTensor<float>&)max32,
                                                          (Reg::RegTensor<xDtype>&)absMaxDim1, maskAll);
            Reg::Compare<uint32_t, CMPMODE::LT>(cmpResult, max32, expMaskReg, maskAll32);
            Reg::Compare<uint32_t, CMPMODE::NE>(zeroMask, max32, zeroReg32, maskAll32);
            Reg::Mul((Reg::RegTensor<float>&)max32, (Reg::RegTensor<float>&)max32, (Reg::RegTensor<float>&)invMax,
                     maskAll32);
            Reg::ShiftRights(exp32, max32, SHR_NUM_FOR_FP32, maskAll32);
            Reg::And(man32, max32, manMaskReg, maskAll32);
            Reg::Compares<uint32_t, CMPMODE::GT>(p0, exp32, static_cast<uint32_t>(0), maskAll32);
            Reg::Compares<uint32_t, CMPMODE::LT>(p0, exp32, EXP_254, p0);
            Reg::Compares<uint32_t, CMPMODE::GT>(p0, man32, static_cast<uint32_t>(0), p0);
            Reg::Compares<uint32_t, CMPMODE::EQ>(p1, exp32, static_cast<uint32_t>(0), maskAll32);
            Reg::Compares<uint32_t, CMPMODE::GT>(p1, man32, HALF_FOR_MAN, p1);
            Reg::Or(p0, p0, p1, maskAll32);
            Reg::Adds(expAddOne32, exp32, 1, maskAll32);
            Reg::Select(extractExp, expAddOne32, exp32, p0);
            Reg::Select<uint32_t>(extractExp, extractExp, fp8NanReg32, cmpResult);
            Reg::Select<uint32_t>(extractExp, extractExp, zeroReg32, zeroMask);
            Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>(scale16, extractExp);
            Reg::Pack<uint8_t, uint16_t, Reg::HighLowPart::LOWEST>(scale8Reg, scale16);
            Reg::ShiftLefts(extractExp, extractExp, SHR_NUM_FOR_BF16, maskAll32);
            Reg::Sub(halfScale, scaleBiasReg, extractExp, maskAll32);
            Reg::Select<uint32_t>(halfScale, halfScale, nanReg32, cmpResult);
            Reg::Select<uint32_t>(halfScale, halfScale, zeroReg32, zeroMask);
            Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>(recip16Reg, halfScale);

            for (uint16_t k = 0; k < 8; k++) {
                Reg::Arange(extractIdx, static_cast<int8_t>(k * 8));
                Reg::Gather(scale8Row, scale8Reg, (Reg::RegTensor<uint8_t>&)extractIdx);
                Reg::StoreAlign<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE>(mxScale1Addr, scale8Row, UB_BLOCK_SIZE,
                                                                             maskReduceB8);
                Reg::Arange(extractIdx, static_cast<int8_t>(k * 16));
                Reg::Gather((Reg::RegTensor<uint8_t>&)recip16Row, (Reg::RegTensor<uint8_t>&)recip16Reg,
                            (Reg::RegTensor<uint8_t>&)extractIdx);
                Reg::StoreAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(
                    mxScale1ReciprocalAddr, recip16Row, SCALE1_RECIPROCAL_ROW_ELEMS, maskReduceB16);
            }
        }

        // Generate y1 immediately after scale1 while staying in the same vector scope.
        Reg::MaskReg maskAllB8 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
        Reg::RegTensor<uint16_t> scaleForMulFP16;
        Reg::RegTensor<float> scaleForMulFP32;
        Reg::RegTensor<float> x0ZeroFP32;
        Reg::RegTensor<float> x0OneFP32;
        Reg::RegTensor<float> x1ZeroFP32;
        Reg::RegTensor<float> x1OneFP32;
        Reg::RegTensor<y1Dtype> x0ZeroFP8;
        Reg::RegTensor<y1Dtype> x0OneFP8;
        Reg::RegTensor<y1Dtype> x1ZeroFP8;
        Reg::RegTensor<y1Dtype> x1OneFP8;
        __ubuf__ xDtype* xReadAddr = xAddrBase;
        __ubuf__ uint16_t* recipReadAddr = reciprocalBase;

        for (uint16_t i = 0; i < blockCount; i++) {
            Reg::LoadAlign<xDtype, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B16>(
                x0, x1, xReadAddr, dataLen);
            Reg::LoadAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_E2B_B16>(
                scaleForMulFP16, recipReadAddr, SCALE1_RECIPROCAL_ROW_ELEMS);

            if constexpr (IsSameType<xDtype, half>::value) {
                Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(x0ZeroFP32, x0, maskAll);
                Reg::Cast<float, xDtype, CAST_X_TO_FP32_ONE>(x0OneFP32, x0, maskAll);
                Reg::Cast<float, bfloat16_t, CAST_X_TO_FP32_ZERO>(
                    scaleForMulFP32, (Reg::RegTensor<bfloat16_t>&)scaleForMulFP16, maskAll);
                Reg::Mul(x0ZeroFP32, x0ZeroFP32, scaleForMulFP32, maskAll);
                Reg::Mul(x0OneFP32, x0OneFP32, scaleForMulFP32, maskAll);
                Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(x1ZeroFP32, x1, maskAll);
                Reg::Cast<float, xDtype, CAST_X_TO_FP32_ONE>(x1OneFP32, x1, maskAll);
                Reg::Mul(x1ZeroFP32, x1ZeroFP32, scaleForMulFP32, maskAll);
                Reg::Mul(x1OneFP32, x1OneFP32, scaleForMulFP32, maskAll);
            } else {
                Reg::Mul(x0, x0, (Reg::RegTensor<xDtype>&)scaleForMulFP16, maskAll);
                Reg::Mul(x1, x1, (Reg::RegTensor<xDtype>&)scaleForMulFP16, maskAll);
                Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(x0ZeroFP32, x0, maskAll);
                Reg::Cast<float, xDtype, CAST_X_TO_FP32_ONE>(x0OneFP32, x0, maskAll);
                Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(x1ZeroFP32, x1, maskAll);
                Reg::Cast<float, xDtype, CAST_X_TO_FP32_ONE>(x1OneFP32, x1, maskAll);
            }
            Reg::Cast<y1Dtype, float, CAST_32_TO_80>(x0ZeroFP8, x0ZeroFP32, maskAll);
            Reg::Cast<y1Dtype, float, CAST_32_TO_82>(x0OneFP8, x0OneFP32, maskAll);
            Reg::Cast<y1Dtype, float, CAST_32_TO_81>(x1ZeroFP8, x1ZeroFP32, maskAll);
            Reg::Cast<y1Dtype, float, CAST_32_TO_83>(x1OneFP8, x1OneFP32, maskAll);
            Reg::Add((Reg::RegTensor<uint8_t>&)x0ZeroFP8, (Reg::RegTensor<uint8_t>&)x0ZeroFP8,
                     (Reg::RegTensor<uint8_t>&)x0OneFP8, maskAllB8);
            Reg::Add((Reg::RegTensor<uint8_t>&)x0ZeroFP8, (Reg::RegTensor<uint8_t>&)x0ZeroFP8,
                     (Reg::RegTensor<uint8_t>&)x1ZeroFP8, maskAllB8);
            Reg::Add((Reg::RegTensor<uint8_t>&)x0ZeroFP8, (Reg::RegTensor<uint8_t>&)x0ZeroFP8,
                     (Reg::RegTensor<uint8_t>&)x1OneFP8, maskAllB8);
            Reg::StoreAlign<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_NORM_B8>(
                y1Addr, (Reg::RegTensor<uint8_t>&)x0ZeroFP8, dataLen, maskAllB8);
        }

        // Reuse the temporary area for the normal axis=-2 maxima output.
        Reg::StoreAlign<uint16_t, Reg::StoreDist::DIST_INTLV_B16>(mxScale2ReciprocalAddr, scale2Slot0Part0,
                                                                  scale2Slot0Part1, maskAll);
        if (blockCount > BLOCK_SIZE) {
            Reg::StoreAlign<uint16_t, Reg::StoreDist::DIST_INTLV_B16>(mxScale2ReciprocalAddr + dataLen,
                                                                      scale2Slot1Part0, scale2Slot1Part1, maskAll);
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
        Reg::MaskReg maskAll = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg maskAllB8 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
        Reg::RegTensor<uint16_t> scaleForMulFP16;
        Reg::RegTensor<float> scaleForMulFP32;
        Reg::RegTensor<xDtype> x0;
        Reg::RegTensor<xDtype> x1;
        Reg::RegTensor<bfloat16_t> x0BF16;
        Reg::RegTensor<bfloat16_t> x1BF16;
        Reg::RegTensor<float> x0ZeroFP32;
        Reg::RegTensor<float> x0OneFP32;
        Reg::RegTensor<float> x1ZeroFP32;
        Reg::RegTensor<float> x1OneFP32;
        Reg::RegTensor<y1Dtype> x0ZeroFP8;
        Reg::RegTensor<y1Dtype> x0OneFP8;
        Reg::RegTensor<y1Dtype> x1ZeroFP8;
        Reg::RegTensor<y1Dtype> x1OneFP8;

        for (uint16_t i = 0; i < blockCount; i++) {
            Reg::LoadAlign<xDtype, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B16>(x0, x1, xAddr,
                                                                                                       ONCE_ROW_LEN);
            Reg::LoadAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_E2B_B16>(
                scaleForMulFP16, mxScale1ReciprocalAddr, SCALE1_RECIPROCAL_ROW_ELEMS);

            if constexpr (IsSameType<xDtype, half>::value) {
                Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(x0ZeroFP32, x0, maskAll);
                Reg::Cast<float, xDtype, CAST_X_TO_FP32_ONE>(x0OneFP32, x0, maskAll);
                Reg::Cast<float, bfloat16_t, CAST_X_TO_FP32_ZERO>(
                    scaleForMulFP32, (Reg::RegTensor<bfloat16_t>&)scaleForMulFP16, maskAll);

                Reg::Mul(x0ZeroFP32, x0ZeroFP32, scaleForMulFP32, maskAll);
                Reg::Mul(x0OneFP32, x0OneFP32, scaleForMulFP32, maskAll);

                Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(x1ZeroFP32, x1, maskAll);
                Reg::Cast<float, xDtype, CAST_X_TO_FP32_ONE>(x1OneFP32, x1, maskAll);
                Reg::Mul(x1ZeroFP32, x1ZeroFP32, scaleForMulFP32, maskAll);
                Reg::Mul(x1OneFP32, x1OneFP32, scaleForMulFP32, maskAll);
            } else {
                Reg::Mul(x0, x0, (Reg::RegTensor<xDtype>&)scaleForMulFP16, maskAll);
                Reg::Mul(x1, x1, (Reg::RegTensor<xDtype>&)scaleForMulFP16, maskAll);

                AscendC::Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(x0ZeroFP32, x0, maskAll);
                AscendC::Reg::Cast<float, xDtype, CAST_X_TO_FP32_ONE>(x0OneFP32, x0, maskAll);
                AscendC::Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(x1ZeroFP32, x1, maskAll);
                AscendC::Reg::Cast<float, xDtype, CAST_X_TO_FP32_ONE>(x1OneFP32, x1, maskAll);
            }
            AscendC::Reg::Cast<y1Dtype, float, CAST_32_TO_80>(x0ZeroFP8, x0ZeroFP32, maskAll);
            AscendC::Reg::Cast<y1Dtype, float, CAST_32_TO_82>(x0OneFP8, x0OneFP32, maskAll);
            AscendC::Reg::Cast<y1Dtype, float, CAST_32_TO_81>(x1ZeroFP8, x1ZeroFP32, maskAll);
            AscendC::Reg::Cast<y1Dtype, float, CAST_32_TO_83>(x1OneFP8, x1OneFP32, maskAll);

            AscendC::Reg::Add((AscendC::Reg::RegTensor<uint8_t>&)x0ZeroFP8,
                              (AscendC::Reg::RegTensor<uint8_t>&)x0ZeroFP8, (AscendC::Reg::RegTensor<uint8_t>&)x0OneFP8,
                              maskAllB8);
            AscendC::Reg::Add((AscendC::Reg::RegTensor<uint8_t>&)x0ZeroFP8,
                              (AscendC::Reg::RegTensor<uint8_t>&)x0ZeroFP8,
                              (AscendC::Reg::RegTensor<uint8_t>&)x1ZeroFP8, maskAllB8);
            AscendC::Reg::Add((AscendC::Reg::RegTensor<uint8_t>&)x0ZeroFP8,
                              (AscendC::Reg::RegTensor<uint8_t>&)x0ZeroFP8, (AscendC::Reg::RegTensor<uint8_t>&)x1OneFP8,
                              maskAllB8);
            AscendC::Reg::StoreAlign<uint8_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                     AscendC::Reg::StoreDist::DIST_NORM_B8>(
                y1Addr, (AscendC::Reg::RegTensor<uint8_t>&)x0ZeroFP8, ONCE_ROW_LEN, maskAllB8);
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
        Reg::MaskReg dataMaskB8 = Reg::CreateMask<uint8_t>();
        Reg::MaskReg dataMaskB16 = Reg::CreateMask<half>();
        Reg::MaskReg dataMaskB32 = Reg::CreateMask<float>();
        Reg::RegTensor<uint16_t> scaleForMulFP16;
        Reg::RegTensor<xDtype> x0;
        Reg::RegTensor<xDtype> x1;

        Reg::RegTensor<float> x0ZeroFP32;
        Reg::RegTensor<float> x0OneFP32;
        Reg::RegTensor<float> x1ZeroFP32;
        Reg::RegTensor<float> x1OneFP32;
        Reg::RegTensor<float> scaleForMulZeroFP32;
        Reg::RegTensor<float> scaleForMulOneFP32;

        Reg::RegTensor<bfloat16_t> x0ZeroBF16;
        Reg::RegTensor<bfloat16_t> x0OneBF16;
        Reg::RegTensor<bfloat16_t> x1ZeroBF16;
        Reg::RegTensor<bfloat16_t> x1OneBF16;

        Reg::RegTensor<y1Dtype> x0FP4;
        Reg::RegTensor<y1Dtype> x1FP4;

        for (uint16_t i = 0; i < blockCount; i++) {
            Reg::LoadAlign<xDtype, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B16>(x0, x1, xAddr,
                                                                                                       ONCE_ROW_LEN);
            Reg::LoadAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_E2B_B16>(
                scaleForMulFP16, mxScale1ReciprocalAddr, SCALE1_RECIPROCAL_ROW_ELEMS);

            if constexpr (IsSameType<xDtype, half>::value) {
                Reg::Cast<float, bfloat16_t, CAST_X_TO_FP32_ZERO>(
                    scaleForMulZeroFP32, (Reg::RegTensor<bfloat16_t>&)scaleForMulFP16, dataMaskB16);

                // x0 cast to fp32 and multiply by 1/scale
                Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(x0ZeroFP32, x0, dataMaskB16);
                Reg::Cast<float, xDtype, CAST_X_TO_FP32_ONE>(x0OneFP32, x0, dataMaskB16);

                Reg::Mul(x0ZeroFP32, scaleForMulZeroFP32, x0ZeroFP32, dataMaskB32);
                Reg::Mul(x0OneFP32, scaleForMulZeroFP32, x0OneFP32, dataMaskB32);
                ComputeFP4FromHalf(x0ZeroFP32);
                ComputeFP4FromHalf(x0OneFP32);
                Reg::Cast<bfloat16_t, float, castTraitFp32toBF16>(x0ZeroBF16, x0ZeroFP32, dataMaskB32);
                Reg::Cast<bfloat16_t, float, castTraitFp32toBF16>(x0OneBF16, x0OneFP32, dataMaskB32);
                Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>((Reg::RegTensor<uint16_t>&)x0ZeroBF16,
                                                                        (Reg::RegTensor<uint32_t>&)x0ZeroBF16);
                Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>((Reg::RegTensor<uint16_t>&)x0OneBF16,
                                                                        (Reg::RegTensor<uint32_t>&)x0OneBF16);
                Reg::Interleave(x0ZeroBF16, x0OneBF16, x0ZeroBF16, x0OneBF16);

                // x1 cast to fp32 and multiply by 1/scale
                Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(x1ZeroFP32, x1, dataMaskB16);
                Reg::Cast<float, xDtype, CAST_X_TO_FP32_ONE>(x1OneFP32, x1, dataMaskB16);

                Reg::Mul(x1ZeroFP32, scaleForMulZeroFP32, x1ZeroFP32, dataMaskB32);
                Reg::Mul(x1OneFP32, scaleForMulZeroFP32, x1OneFP32, dataMaskB32);
                ComputeFP4FromHalf(x1ZeroFP32);
                ComputeFP4FromHalf(x1OneFP32);
                Reg::Cast<bfloat16_t, float, castTraitFp32toBF16>(x1ZeroBF16, x1ZeroFP32, dataMaskB32);
                Reg::Cast<bfloat16_t, float, castTraitFp32toBF16>(x1OneBF16, x1OneFP32, dataMaskB32);
                Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>((Reg::RegTensor<uint16_t>&)x1ZeroBF16,
                                                                        (Reg::RegTensor<uint32_t>&)x1ZeroBF16);
                Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>((Reg::RegTensor<uint16_t>&)x1OneBF16,
                                                                        (Reg::RegTensor<uint32_t>&)x1OneBF16);
                Reg::Interleave(x1ZeroBF16, x1OneBF16, x1ZeroBF16, x1OneBF16);

                // Interleave x0 and x1, then cast to FP4
                Reg::Interleave(x0ZeroBF16, x1ZeroBF16, x0ZeroBF16, x1ZeroBF16);
                Reg::Cast<y1Dtype, bfloat16_t, castTraitBF16toFp4>(x0FP4, x0ZeroBF16, dataMaskB16);
                Reg::Cast<y1Dtype, bfloat16_t, castTraitBF16toFp4>(x1FP4, x1ZeroBF16, dataMaskB16);
            } else {
                // BF16 input path
                Reg::Mul(x0, x0, (Reg::RegTensor<xDtype>&)scaleForMulFP16, dataMaskB16);
                Reg::Mul(x1, x1, (Reg::RegTensor<xDtype>&)scaleForMulFP16, dataMaskB16);
                Reg::Interleave(x0, x1, x0, x1);
                Reg::Cast<y1Dtype, xDtype, castTraitBF16toFp4>(x0FP4, x0, dataMaskB16);
                Reg::Cast<y1Dtype, xDtype, castTraitBF16toFp4>(x1FP4, x1, dataMaskB16);
            }

            Reg::StoreAlign<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_PACK4_B32>(
                y1Addr, (Reg::RegTensor<uint8_t>&)x0FP4, OUT_ELE_NUM_ONE_BLK, dataMaskB8);
            Reg::StoreAlign<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_PACK4_B32>(
                y1Addr, (Reg::RegTensor<uint8_t>&)x1FP4, OUT_ELE_NUM_ONE_BLK, dataMaskB8);
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
        Reg::RegTensor<xDtype> xEven;
        Reg::RegTensor<xDtype> xOdd;
        Reg::RegTensor<uint16_t> reciprocalEven;
        Reg::RegTensor<uint16_t> reciprocalOdd;
        Reg::RegTensor<float> reciprocalEvenFP32Layout0;
        Reg::RegTensor<float> reciprocalEvenFP32Layout1;
        Reg::RegTensor<float> reciprocalOddFP32Layout0;
        Reg::RegTensor<float> reciprocalOddFP32Layout1;
        Reg::RegTensor<float> xEvenFP32Layout0;
        Reg::RegTensor<float> xEvenFP32Layout1;
        Reg::RegTensor<float> xOddFP32Layout0;
        Reg::RegTensor<float> xOddFP32Layout1;
        Reg::RegTensor<y1Dtype> yEvenFP8Layout0;
        Reg::RegTensor<y1Dtype> yEvenFP8Layout2;
        Reg::RegTensor<y1Dtype> yOddFP8Layout1;
        Reg::RegTensor<y1Dtype> yOddFP8Layout3;

        Reg::MaskReg maskB8 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg maskB16 = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg maskB32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();

        // Load all 256 reciprocal values once and reuse them for all 32 rows in this scale2 slot.
        Reg::LoadAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B16>(
            reciprocalEven, reciprocalOdd, mxScale2ReciprocalAddr, dualLoadLen);
        if constexpr (IsSameType<xDtype, half>::value) {
            Reg::Cast<float, bfloat16_t, CAST_X_TO_FP32_ZERO>(reciprocalEvenFP32Layout0,
                                                              (Reg::RegTensor<bfloat16_t>&)reciprocalEven, maskB16);
            Reg::Cast<float, bfloat16_t, CAST_X_TO_FP32_ONE>(reciprocalEvenFP32Layout1,
                                                             (Reg::RegTensor<bfloat16_t>&)reciprocalEven, maskB16);
            Reg::Cast<float, bfloat16_t, CAST_X_TO_FP32_ZERO>(reciprocalOddFP32Layout0,
                                                              (Reg::RegTensor<bfloat16_t>&)reciprocalOdd, maskB16);
            Reg::Cast<float, bfloat16_t, CAST_X_TO_FP32_ONE>(reciprocalOddFP32Layout1,
                                                             (Reg::RegTensor<bfloat16_t>&)reciprocalOdd, maskB16);
        }

        for (uint16_t row = 0; row < blockCount; row++) {
            __ubuf__ xDtype* xCursor = xAddr + row * localUbRowLen;
            Reg::LoadAlign<xDtype, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B16>(
                xEven, xOdd, xCursor, dualLoadLen);
            if constexpr (IsSameType<xDtype, half>::value) {
                Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(xEvenFP32Layout0, xEven, maskB16);
                Reg::Cast<float, xDtype, CAST_X_TO_FP32_ONE>(xEvenFP32Layout1, xEven, maskB16);
                Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(xOddFP32Layout0, xOdd, maskB16);
                Reg::Cast<float, xDtype, CAST_X_TO_FP32_ONE>(xOddFP32Layout1, xOdd, maskB16);

                Reg::Mul(xEvenFP32Layout0, xEvenFP32Layout0, reciprocalEvenFP32Layout0, maskB32);
                Reg::Mul(xEvenFP32Layout1, xEvenFP32Layout1, reciprocalEvenFP32Layout1, maskB32);
                Reg::Mul(xOddFP32Layout0, xOddFP32Layout0, reciprocalOddFP32Layout0, maskB32);
                Reg::Mul(xOddFP32Layout1, xOddFP32Layout1, reciprocalOddFP32Layout1, maskB32);
            } else {
                Reg::Mul(xEven, xEven, (Reg::RegTensor<xDtype>&)reciprocalEven, maskB16);
                Reg::Mul(xOdd, xOdd, (Reg::RegTensor<xDtype>&)reciprocalOdd, maskB16);
                Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(xEvenFP32Layout0, xEven, maskB16);
                Reg::Cast<float, xDtype, CAST_X_TO_FP32_ONE>(xEvenFP32Layout1, xEven, maskB16);
                Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(xOddFP32Layout0, xOdd, maskB16);
                Reg::Cast<float, xDtype, CAST_X_TO_FP32_ONE>(xOddFP32Layout1, xOdd, maskB16);
            }

            Reg::Cast<y1Dtype, float, CAST_32_TO_80>(yEvenFP8Layout0, xEvenFP32Layout0, maskB32);
            Reg::Cast<y1Dtype, float, CAST_32_TO_82>(yEvenFP8Layout2, xEvenFP32Layout1, maskB32);
            Reg::Cast<y1Dtype, float, CAST_32_TO_81>(yOddFP8Layout1, xOddFP32Layout0, maskB32);
            Reg::Cast<y1Dtype, float, CAST_32_TO_83>(yOddFP8Layout3, xOddFP32Layout1, maskB32);

            Reg::Add((Reg::RegTensor<uint8_t>&)yEvenFP8Layout0, (Reg::RegTensor<uint8_t>&)yEvenFP8Layout0,
                     (Reg::RegTensor<uint8_t>&)yEvenFP8Layout2, maskB8);
            Reg::Add((Reg::RegTensor<uint8_t>&)yOddFP8Layout1, (Reg::RegTensor<uint8_t>&)yOddFP8Layout1,
                     (Reg::RegTensor<uint8_t>&)yOddFP8Layout3, maskB8);
            Reg::Add((Reg::RegTensor<uint8_t>&)yEvenFP8Layout0, (Reg::RegTensor<uint8_t>&)yEvenFP8Layout0,
                     (Reg::RegTensor<uint8_t>&)yOddFP8Layout1, maskB8);

            __ubuf__ uint8_t* yCursor = y2Addr + row * localUbRowLen;
            Reg::StoreAlign<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_NORM_B8>(
                yCursor, (Reg::RegTensor<uint8_t>&)yEvenFP8Layout0, dualLoadLen, maskB8);
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
        Reg::RegTensor<xDtype> x;
        Reg::RegTensor<bfloat16_t> x0BF16;
        Reg::RegTensor<bfloat16_t> x1BF16;
        Reg::RegTensor<bfloat16_t> xBF16;
        Reg::RegTensor<float> x0FP32;
        Reg::RegTensor<float> x1FP32;
        Reg::RegTensor<uint16_t> reversedShareExp;
        Reg::RegTensor<float> reversedShareExp0FP32;
        Reg::RegTensor<float> reversedShareExp1FP32;
        Reg::RegTensor<y1Dtype> yZeroFP4;

        Reg::MaskReg pregAll8 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg pregAll16 = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg pregAll32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();

        // Load per-column 1/scale
        Reg::LoadAlign<uint16_t, Reg::LoadDist::DIST_NORM>(reversedShareExp, mxScale2ReciprocalAddr);

        for (uint16_t j = 0; j < blockCount; j++) {
            Reg::LoadAlign<xDtype, Reg::LoadDist::DIST_NORM>(x, xAddr + j * localUbRowLen);

            if constexpr (IsSameType<xDtype, half>::value) {
                Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(x0FP32, x, pregAll16);
                Reg::Cast<float, xDtype, CAST_X_TO_FP32_ONE>(x1FP32, x, pregAll16);

                Reg::Cast<float, bfloat16_t, CAST_X_TO_FP32_ZERO>(
                    reversedShareExp0FP32, (Reg::RegTensor<bfloat16_t>&)reversedShareExp, pregAll16);

                Reg::Cast<float, bfloat16_t, CAST_X_TO_FP32_ONE>(
                    reversedShareExp1FP32, (Reg::RegTensor<bfloat16_t>&)reversedShareExp, pregAll16);

                Reg::Mul(x0FP32, x0FP32, reversedShareExp0FP32, pregAll32);
                Reg::Mul(x1FP32, x1FP32, reversedShareExp1FP32, pregAll32);

                ComputeFP4FromHalf(x0FP32);
                ComputeFP4FromHalf(x1FP32);

                Reg::Cast<bfloat16_t, float, castTraitFp32toBF16>((Reg::RegTensor<bfloat16_t>&)x0BF16, x0FP32,
                                                                  pregAll32);
                Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>((Reg::RegTensor<uint16_t>&)x0BF16,
                                                                        (Reg::RegTensor<uint32_t>&)x0BF16);

                Reg::Cast<bfloat16_t, float, castTraitFp32toBF16>((Reg::RegTensor<bfloat16_t>&)x1BF16, x1FP32,
                                                                  pregAll32);

                Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>((Reg::RegTensor<uint16_t>&)x1BF16,
                                                                        (Reg::RegTensor<uint32_t>&)x1BF16);

                Reg::Interleave(x0BF16, x1BF16, x0BF16, x1BF16);
                Reg::Cast<y1Dtype, bfloat16_t, castTraitBF16toFp4>(yZeroFP4, (Reg::RegTensor<bfloat16_t>&)x0BF16,
                                                                   pregAll16);
            } else {
                Reg::Mul(xBF16, x, (Reg::RegTensor<bfloat16_t>&)reversedShareExp, pregAll16);
                Reg::Cast<y1Dtype, bfloat16_t, castTraitBF16toFp4>(yZeroFP4, xBF16, pregAll16);
            }

            Reg::StoreAlign<uint8_t, Reg::StoreDist::DIST_PACK4_B32>(y2Addr + (j * 128),
                                                                     (Reg::RegTensor<uint8_t>&)yZeroFP4, pregAll8);
        }
    }
}

// ============================================================================
// ComputeFP4FromHalf — FP4 quantization rounding logic (E2M1 / E1M2)
// ============================================================================
template <typename xDtype, typename y1Dtype, uint64_t mode, AscendC::RoundMode roundMode, uint64_t scaleAlg,
          uint64_t isGroupIdx>
__aicore__ inline void SwigluMxQuantWithDualAxisBase<xDtype, y1Dtype, mode, roundMode, scaleAlg,
                                                     isGroupIdx>::ComputeFP4FromHalf(Reg::RegTensor<float>& Reg)
{
    Reg::MaskReg pregAll32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::MaskReg zeroMask;
    Reg::MaskReg specialMask;
    Reg::MaskReg negInfMask;

    Reg::RegTensor<int32_t> negZero;
    Reg::RegTensor<int32_t> maxExpFP32;
    Reg::RegTensor<int32_t> exp0FP32;
    Reg::RegTensor<int32_t> exp1FP32;

    Reg::Duplicate(negZero, NEG_ZERO);

    Reg::Compare<int32_t, CMPMODE::EQ>(negInfMask, (Reg::RegTensor<int32_t>&)Reg, negZero, pregAll32);
    if constexpr (IsSameType<y1Dtype, fp4x2_e1m2_t>::value) {
        // E1M2: multiply by 4, truncate, divide by 4
        Reg::Muls(Reg, Reg, FOUR, pregAll32);
        Reg::Compares<float, CMPMODE::LT>(specialMask, Reg, 0, pregAll32);
        Reg::Truncate<float, roundMode>(Reg, Reg, pregAll32);
        Reg::Muls(Reg, Reg, ONE_FOURTH, pregAll32);
    } else {
        // E2M1: bit-level exponent manipulation for correct rounding
        Reg::Duplicate(maxExpFP32, MAX_EXP_FOR_FP32);
        Reg::And(exp0FP32, (Reg::RegTensor<int32_t>&)Reg, maxExpFP32, pregAll32);
        Reg::ShiftRights(exp0FP32, exp0FP32, SHR_NUM_FOR_FP32, pregAll32);
        Reg::Adds(exp0FP32, exp0FP32, FP32_BIAS_NEG, pregAll32);
        Reg::Maxs(exp0FP32, exp0FP32, 0, pregAll32);
        Reg::Adds(exp0FP32, exp0FP32, NEG_ONE, pregAll32);
        Reg::Muls(exp1FP32, exp0FP32, NEG_ONE, pregAll32);
        Reg::Adds(exp1FP32, exp1FP32, FP32_BIAS, pregAll32);
        Reg::ShiftLefts(exp1FP32, exp1FP32, SHR_NUM_FOR_FP32, pregAll32);

        Reg::Mul(Reg, Reg, (Reg::RegTensor<float>&)exp1FP32, pregAll32);
        Reg::Adds(exp0FP32, exp0FP32, FP32_BIAS, pregAll32);
        Reg::ShiftLefts(exp0FP32, exp0FP32, SHR_NUM_FOR_FP32, pregAll32);
        Reg::Compares<float, CMPMODE::LT>(specialMask, Reg, 0, pregAll32);
        Reg::Truncate<float, roundMode>(Reg, Reg, pregAll32);
        Reg::Mul(Reg, Reg, (Reg::RegTensor<float>&)exp0FP32, pregAll32);
    }
    // Handle negative zero
    Reg::Compares<float, CMPMODE::EQ>(zeroMask, Reg, 0, pregAll32);
    Reg::And(zeroMask, specialMask, zeroMask, pregAll32);
    Reg::Or(zeroMask, negInfMask, zeroMask, pregAll32);
    Reg::Select<int32_t>((Reg::RegTensor<int32_t>&)Reg, negZero, (Reg::RegTensor<int32_t>&)Reg, zeroMask);
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
