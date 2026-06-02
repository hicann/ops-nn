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
 * \file swiglu_backward_mx_quant_with_dual_axis_regbase.h
 * \brief Fused SwiGLU + grouped dual-axis MX quantization kernel for Ascend950 (regbase)
 */

#ifndef OPS_NN_SWIGLU_BACKWARD_MX_QUANT_WITH_DUAL_AXIS_REGBASE_H
#define OPS_NN_SWIGLU_BACKWARD_MX_QUANT_WITH_DUAL_AXIS_REGBASE_H

#define FLOAT_OVERFLOW_MODE_CTRL 60

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../inc/platform.h"
#include "../inc/kernel_utils.h"
#include "swiglu_backward_mx_quant_with_dual_axis_tiling_key.h"
#include "swiglu_backward_mx_quant_with_dual_axis_tilingdata.h"

namespace SwigluBackwardMxQuantWithDualAxis {
using namespace AscendC;

constexpr int64_t DB_BUFFER = 2;
constexpr int64_t DIGIT_TWO = 2;
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
constexpr int64_t UB_BLOCK_SIZE = platform::GetUbBlockSize();

static constexpr MicroAPI::CastTrait CAST_X_TO_FP32_ZERO = { MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
    MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN };
static constexpr MicroAPI::CastTrait CAST_X_TO_FP32_ONE = { MicroAPI::RegLayout::ONE, MicroAPI::SatMode::UNKNOWN,
    MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN };
static constexpr MicroAPI::CastTrait CAST_HALF_TO_BF16 = { MicroAPI::RegLayout::UNKNOWN, MicroAPI::SatMode::UNKNOWN,
    MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_TRUNC };

static constexpr MicroAPI::CastTrait CAST_FP32_TO_FP16_BF16 = { MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
    MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT };
static constexpr AscendC::MicroAPI::CastTrait CAST_32_TO_80 = { AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT };
static constexpr AscendC::MicroAPI::CastTrait CAST_32_TO_81 = { AscendC::MicroAPI::RegLayout::ONE,
    AscendC::MicroAPI::SatMode::SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT };
static constexpr AscendC::MicroAPI::CastTrait CAST_32_TO_82 = { AscendC::MicroAPI::RegLayout::TWO,
    AscendC::MicroAPI::SatMode::SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT };
static constexpr AscendC::MicroAPI::CastTrait CAST_32_TO_83 = { AscendC::MicroAPI::RegLayout::THREE,
    AscendC::MicroAPI::SatMode::SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT };

template <typename xDtype, typename y1Dtype, uint64_t mode, AscendC::RoundMode roundMode, uint64_t scaleAlg,
    uint64_t isGroupIdx>
class SwigluBackwardMxQuantWithDualAxisBase {
public:
    __aicore__ inline SwigluBackwardMxQuantWithDualAxisBase(const SwigluBackwardMxQuantWithDualAxisTilingData *tilingData, TPipe *pipe)
        : tilingData_(tilingData), pipe_(pipe){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR yGrad, GM_ADDR groupIndex, GM_ADDR y1, GM_ADDR mxScale1, GM_ADDR y2,
        GM_ADDR mxScale2);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitParams();
    __aicore__ inline void CopyIn(int64_t absRowStart, int64_t colOffsetAB, int64_t calcRow, int64_t calcColAB);
    __aicore__ inline void CopyInYGrad(int64_t absRowStart, int64_t colOffsetAB, int64_t calcRow, int64_t calcColAB);
    __aicore__ inline void ComputeSwigluBackward(uint16_t dataLenAB, uint16_t blockCount, __ubuf__ xDtype *actAddr,
        __ubuf__ xDtype *gateAddr, __ubuf__ xDtype *yGradAddr, __ubuf__ xDtype *gradXAddr, uint32_t rowWidth);
    __aicore__ inline void PadZeroM(__ubuf__ xDtype *gradXAddr, uint32_t num);
    __aicore__ inline void ComputeScaleOcp(uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype *xAddr,
        __ubuf__ uint8_t *mxScale1Addr, __ubuf__ uint16_t *mxScale1ReciprocalAddr, __ubuf__ uint8_t *mxScale2Addr,
        __ubuf__ uint16_t *mxScale2ReciprocalAddr);
    __aicore__ inline void ComputeScaleCuBLAS(uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype *xAddr,
        __ubuf__ uint8_t *mxScaleA1Addr, __ubuf__ uint8_t *mxScaleA2Addr, __ubuf__ uint16_t *mxScale1ReciprocalAddr,
        __ubuf__ uint8_t *mxScale2Addr, __ubuf__ uint16_t *mxScale2ReciprocalAddr, int64_t dataLenAB,
        __ubuf__ uint8_t *y1Addr);
    __aicore__ inline void ComputeScaleCuBLASSecondLast(uint16_t dataLen, uint32_t localInvDtypeMax,
        __ubuf__ uint16_t *mxScale2ReciprocalAddr, __ubuf__ uint8_t *mxScale2Addr);
    __aicore__ inline void ComputeY1ToFP8(uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype *xAddr,
        __ubuf__ uint16_t *mxScale1ReciprocalAddr, __ubuf__ uint8_t *y1Addr);
    __aicore__ inline void ComputeY1ToFP4(uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype *xAddr,
        __ubuf__ uint16_t *mxScale1ReciprocalAddr, __ubuf__ uint8_t *y1Addr);
    __aicore__ inline void ComputeY2ToFP8(uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype *xAddr,
        __ubuf__ uint16_t *mxScale2ReciprocalAddr, __ubuf__ uint8_t *y2Addr);
    __aicore__ inline void ComputeY2ToFP4(uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype *xAddr,
        __ubuf__ uint16_t *mxScale2ReciprocalAddr, __ubuf__ uint8_t *y2Addr);
    __aicore__ inline void ComputeFP4FromHalf(MicroAPI::RegTensor<float> &Reg);
    __aicore__ inline void CopyOut(int64_t yOffsetA, int64_t yOffsetB, int64_t scale1OutOffset,
        int64_t scale2OutOffsetA, int64_t scale2OutOffsetB, int64_t blockCount, int64_t blockCountAlign,
        int64_t dataLenAB, int64_t rowWidth);

protected:
    static constexpr MicroAPI::CastTrait castTraitBF16toFp4 = { MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT,
        MicroAPI::MaskMergeMode::ZEROING, roundMode };
    static constexpr MicroAPI::CastTrait castTraitFp32toBF16 = { MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
        MicroAPI::MaskMergeMode::ZEROING, roundMode };
    static constexpr MicroAPI::CastTrait castTraitFp32toYdtype = { MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT,
        MicroAPI::MaskMergeMode::ZEROING, roundMode };

private:
    const SwigluBackwardMxQuantWithDualAxisTilingData *tilingData_;

    TPipe *pipe_;
    TQue<QuePosition::VECIN, DB_BUFFER> inQueue_;
    TQue<QuePosition::VECIN, DB_BUFFER> yGradQueue_;
    TBuf<TPosition::VECCALC> gradXBuf_;
    TQue<QuePosition::VECOUT, DB_BUFFER> outQueue1_;
    TQue<QuePosition::VECOUT, DB_BUFFER> outQueue2_;
    TQue<QuePosition::VECOUT, DB_BUFFER> mxScaleAQueue1_;
    TQue<QuePosition::VECOUT, DB_BUFFER> mxScaleBQueue1_;
    TQue<QuePosition::VECOUT, DB_BUFFER> mxScaleQueue2_;
    TBuf<TPosition::VECCALC> mxScale1ReciprocalBuf_;
    TBuf<TPosition::VECCALC> mxScale2ReciprocalBuf_;

    GlobalTensor<xDtype> xGm_;
    GlobalTensor<xDtype> yGradGm_;
    GlobalTensor<int64_t> groupIndexGm_;
    GlobalTensor<uint8_t> yGm1_;
    GlobalTensor<uint8_t> mxScaleGm1_;
    GlobalTensor<uint8_t> yGm2_;
    GlobalTensor<uint8_t> mxScaleGm2_;

    int64_t blockIdx_ = 0;
    int64_t ubRowLen_ = 0;
    int64_t ubRowLenTail_ = 0;
    int64_t ubRowCount_ = 0; 
    int64_t dimNeg1ScaleNum_ = 0;
    uint32_t invDtypeMax_ = 0;
    uint16_t dtypeYMaxExp_ = 0;
    int64_t activateLeft_ = 1;
    int64_t inHalfSize_ = 0;
    int64_t dimN_ = 0;
    int64_t dimGradX_ = 0;

    int64_t oneBlockCountB16_ = UB_BLOCK_SIZE / sizeof(xDtype);
    int64_t oneBlockCountB8_ = UB_BLOCK_SIZE / sizeof(uint8_t);
};

template <typename xDtype, typename y1Dtype, uint64_t mode, AscendC::RoundMode roundMode, uint64_t scaleAlg,
    uint64_t isGroupIdx>
__aicore__ inline void
SwigluBackwardMxQuantWithDualAxisBase<xDtype, y1Dtype, mode, roundMode, scaleAlg, isGroupIdx>::InitParams()
{
    blockIdx_ = GetBlockIdx();
    ubRowLen_ = tilingData_->blockW;
    ubRowLenTail_ = tilingData_->dimNTail;
    ubRowCount_ = tilingData_->splitBlockH;
    activateLeft_ = tilingData_->activateLeft;
    dimN_ = tilingData_->dimN;
    dimGradX_ = tilingData_->dimGradX;

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

template <typename xDtype, typename y1Dtype, uint64_t mode, AscendC::RoundMode roundMode, uint64_t scaleAlg,
    uint64_t isGroupIdx>
__aicore__ inline void 
SwigluBackwardMxQuantWithDualAxisBase<xDtype, y1Dtype, mode, roundMode, scaleAlg, isGroupIdx>::Init(
    GM_ADDR x, GM_ADDR yGrad, GM_ADDR groupIndex, GM_ADDR y1, GM_ADDR mxScale1, GM_ADDR y2, GM_ADDR mxScale2)
{
#if (__NPU_ARCH__ == 3510)
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
#endif

    InitParams();

    xGm_.SetGlobalBuffer((__gm__ xDtype *)(x));
    yGradGm_.SetGlobalBuffer((__gm__ xDtype *)(yGrad));
    if constexpr (isGroupIdx == static_cast<uint64_t>(1)) {
        groupIndexGm_.SetGlobalBuffer((__gm__ int64_t *)(groupIndex));
        groupIndexGm_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    }
    xGm_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    yGradGm_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    yGm1_.SetGlobalBuffer((__gm__ uint8_t *)(y1));
    mxScaleGm1_.SetGlobalBuffer((__gm__ uint8_t *)(mxScale1));
    yGm2_.SetGlobalBuffer((__gm__ uint8_t *)(y2));
    mxScaleGm2_.SetGlobalBuffer((__gm__ uint8_t *)(mxScale2));

    inHalfSize_ = ubRowLen_ * ubRowCount_;
    int64_t inBufferSize = inHalfSize_ * static_cast<int64_t>(sizeof(xDtype));
    int64_t gradXBufferSize = inHalfSize_ * DIGIT_TWO * static_cast<int64_t>(sizeof(xDtype));
    int64_t mxScale2BufferSize = (ubRowLen_ * DIGIT_TWO) * ((ubRowCount_ / DOUBLE_BLOCK_SIZE) * DIGIT_TWO);
    int64_t mxScale1BufferSize = ubRowCount_ * UB_BLOCK_SIZE;
    int64_t tmpScale2BufferSize = (ubRowLen_ * DIGIT_TWO) * ((ubRowCount_ / DOUBLE_BLOCK_SIZE) * DIGIT_TWO)
        * static_cast<int64_t>(sizeof(xDtype));
    pipe_->InitBuffer(inQueue_, DB_BUFFER, inBufferSize * DIGIT_TWO);
    pipe_->InitBuffer(yGradQueue_, DB_BUFFER, inBufferSize);
    pipe_->InitBuffer(gradXBuf_, gradXBufferSize);
    pipe_->InitBuffer(outQueue1_, DB_BUFFER, inHalfSize_ * DIGIT_TWO);
    pipe_->InitBuffer(outQueue2_, DB_BUFFER, inHalfSize_ * DIGIT_TWO);
    pipe_->InitBuffer(mxScaleAQueue1_, DB_BUFFER, mxScale1BufferSize);
    pipe_->InitBuffer(mxScaleBQueue1_, DB_BUFFER, mxScale1BufferSize);
    pipe_->InitBuffer(mxScaleQueue2_, DB_BUFFER, mxScale2BufferSize);
    pipe_->InitBuffer(mxScale1ReciprocalBuf_, mxScale1BufferSize);
    pipe_->InitBuffer(mxScale2ReciprocalBuf_, tmpScale2BufferSize);
}

template <typename xDtype, typename y1Dtype, uint64_t mode, AscendC::RoundMode roundMode, uint64_t scaleAlg,
    uint64_t isGroupIdx>
__aicore__ inline void 
SwigluBackwardMxQuantWithDualAxisBase<xDtype, y1Dtype, mode, roundMode, scaleAlg, isGroupIdx>::Process()
{
    int64_t numGroups = 1;
    int64_t batchCount = tilingData_->dimBatch;
    if constexpr (isGroupIdx == static_cast<uint64_t>(1)) {
        numGroups = tilingData_->numGroups;
    } else if (batchCount > 1) {
        numGroups = batchCount;
    }
    int64_t dimM = tilingData_->dimM;
    dimNeg1ScaleNum_ = (ops::CeilDiv(dimGradX_, DOUBLE_BLOCK_SIZE)) * DIGIT_TWO;
    int64_t dimNBlockNum = tilingData_->dimNBlockNum;
    int64_t totalCoreNum = tilingData_->usedCoreNum;
    if (blockIdx_ >= totalCoreNum) {
        return;
    }

    [[maybe_unused]] int64_t coreRotateOffset = 0;

    for (int64_t g = 0; g < numGroups; g++) {
        int64_t groupStart = 0;
        int64_t groupEnd = dimM;
        if constexpr (isGroupIdx == static_cast<uint64_t>(1)) {
            groupStart = (g > 0) ? groupIndexGm_.GetValue(g - 1) : 0;
            groupEnd = groupIndexGm_.GetValue(g);
        } else if (batchCount > 1) {
            groupStart = g * dimM;
            groupEnd = groupStart + dimM;
        }
        int64_t groupRows = groupEnd - groupStart;
        if (groupRows <= 0) {
            continue;
        }
        int64_t scale2GmRowOffset = 0;
        if constexpr (isGroupIdx == static_cast<uint64_t>(1)) {
            scale2GmRowOffset = (groupStart / DOUBLE_BLOCK_SIZE + g) * DIGIT_TWO;
        } else if (batchCount > 1) {
            int64_t rowsPerBatch = ops::CeilDiv(dimM, DOUBLE_BLOCK_SIZE) * DIGIT_TWO;
            scale2GmRowOffset = g * rowsPerBatch;
        } else {
            scale2GmRowOffset = (groupStart / DOUBLE_BLOCK_SIZE) * DIGIT_TWO;
        }

        int64_t dimMSplitG = ops::CeilDiv(groupRows, ubRowCount_);
        int64_t blockCountG = dimMSplitG * dimNBlockNum;

        int64_t loopPerCoreG = 0;
        int64_t blockOffsetG = 0;

        if constexpr (mode == TPL_MODE_ROTATE) {
            int64_t usedCoreNumG = (blockCountG < totalCoreNum) ? blockCountG : totalCoreNum;
            int64_t myCoreInGroup = blockIdx_ - coreRotateOffset;
            if (myCoreInGroup < 0) {
                myCoreInGroup += totalCoreNum;
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
            coreRotateOffset = (coreRotateOffset + blockCountG) % totalCoreNum;
            if (loopPerCoreG == 0) {
                continue;
            }
        } else {
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

        int64_t dimMTailG = groupRows % ubRowCount_ == 0 ? ubRowCount_ : groupRows % ubRowCount_;

        for (int64_t i = 0; i < loopPerCoreG; i++) {
            int64_t blockInGroup = blockOffsetG + i;
            int64_t rowBlockIdx = blockInGroup / dimNBlockNum;
            int64_t colBlockIdx = blockInGroup % dimNBlockNum;

            int64_t calcColAB = (colBlockIdx == dimNBlockNum - 1) ? ubRowLenTail_ : ubRowLen_;
            int64_t calcRow = (rowBlockIdx == dimMSplitG - 1) ? dimMTailG : ubRowCount_;
            int64_t absRowStart = groupStart + rowBlockIdx * ubRowCount_;
            int64_t colOffsetAB = colBlockIdx * ubRowLen_;
            int64_t colOffsetGradX = colBlockIdx * ubRowLen_ * DIGIT_TWO;
            CopyIn(absRowStart, colOffsetAB, calcRow, calcColAB);
            CopyInYGrad(absRowStart, colOffsetAB, calcRow, calcColAB);

            LocalTensor<xDtype> xLocal = inQueue_.template DeQue<xDtype>();
            LocalTensor<xDtype> yGradLocal = yGradQueue_.template DeQue<xDtype>();
            LocalTensor<xDtype> gradXLocal = gradXBuf_.template Get<xDtype>();

            auto actAddr = (__ubuf__ xDtype *)xLocal.GetPhyAddr();
            auto gateAddr = (__ubuf__ xDtype *)xLocal[inHalfSize_].GetPhyAddr();
            if (activateLeft_ == 0) {
                actAddr = (__ubuf__ xDtype *)xLocal[inHalfSize_].GetPhyAddr();
                gateAddr = (__ubuf__ xDtype *)xLocal.GetPhyAddr();
            }
            auto yGradAddr = (__ubuf__ xDtype *)yGradLocal.GetPhyAddr();
            auto gradXAddr = (__ubuf__ xDtype *)gradXLocal.GetPhyAddr();
            uint32_t calcPadRowAlgin = ops::CeilDiv(calcRow, DOUBLE_BLOCK_SIZE) * DOUBLE_BLOCK_SIZE;
            uint32_t rowWidth = ops::CeilDiv(calcColAB * DIGIT_TWO, static_cast<int64_t>(VF_LEN_B16)) * VF_LEN_B16;

            ComputeSwigluBackward(static_cast<uint16_t>(calcColAB), static_cast<uint16_t>(calcRow), actAddr, gateAddr,
                yGradAddr, gradXAddr, rowWidth);
            inQueue_.template FreeTensor(xLocal);
            yGradQueue_.template FreeTensor(yGradLocal);
            if (calcRow % DOUBLE_BLOCK_SIZE != 0) {
                uint32_t calcPadRow = calcPadRowAlgin - calcRow;
                uint32_t allNumZero = calcPadRow * rowWidth;
                auto gradXAddrPadZero = (__ubuf__ xDtype *)gradXLocal[calcRow * rowWidth].GetPhyAddr();
                PadZeroM(gradXAddrPadZero, allNumZero);
            }

            LocalTensor<uint8_t> mxScaleA1 = mxScaleAQueue1_.template AllocTensor<uint8_t>();
            LocalTensor<uint8_t> mxScaleB1 = mxScaleBQueue1_.template AllocTensor<uint8_t>();
            LocalTensor<uint8_t> mxScale2 = mxScaleQueue2_.template AllocTensor<uint8_t>();
            LocalTensor<uint8_t> y1 = outQueue1_.template AllocTensor<uint8_t>();
            LocalTensor<uint8_t> y2 = outQueue2_.template AllocTensor<uint8_t>();
            LocalTensor<uint16_t> mxScale1Reciprocal = mxScale1ReciprocalBuf_.template Get<uint16_t>();
            LocalTensor<uint16_t> mxScale2Reciprocal = mxScale2ReciprocalBuf_.template Get<uint16_t>();

            auto y1Addr = (__ubuf__ uint8_t *)y1.GetPhyAddr();
            auto y2Addr = (__ubuf__ uint8_t *)y2.GetPhyAddr();
            auto msA1Addr = (__ubuf__ uint8_t *)mxScaleA1.GetPhyAddr();
            auto msB1Addr = (__ubuf__ uint8_t *)mxScaleB1.GetPhyAddr();
            auto ms2Addr = (__ubuf__ uint8_t *)mxScale2.GetPhyAddr();
            auto ms1RecipAddr = (__ubuf__ uint16_t *)mxScale1Reciprocal.GetPhyAddr();
            auto ms2RecipAddr = (__ubuf__ uint16_t *)mxScale2Reciprocal.GetPhyAddr();

            int64_t calcBlockLoop = calcPadRowAlgin / BLOCK_SIZE;
            uint32_t dataLenGradX = rowWidth;

            for (int64_t blk = 0; blk < calcBlockLoop; blk++) {
                int64_t sOff = blk * BLOCK_SIZE * rowWidth;
                int64_t yOff = sOff;
                if constexpr (IsSameType<y1Dtype, fp4x2_e2m1_t>::value || IsSameType<y1Dtype, fp4x2_e1m2_t>::value) {
                    yOff = sOff / DIGIT_TWO;
                }
                int64_t s1Off = blk * BLOCK_SIZE * ops::CeilAlign(rowWidth / BLOCK_SIZE, oneBlockCountB8_);
                int64_t s2Off = blk * rowWidth;
                int64_t r1Off = blk * BLOCK_SIZE * ops::CeilAlign(rowWidth / BLOCK_SIZE, oneBlockCountB16_);
                int64_t r2Off = blk * rowWidth;

                if constexpr (scaleAlg == TPL_SCALE_ALG_0) {
                    ComputeScaleOcp(static_cast<uint16_t>(rowWidth), static_cast<uint16_t>(BLOCK_SIZE),
                        gradXAddr + sOff, msA1Addr + s1Off, ms1RecipAddr + r1Off, ms2Addr + s2Off,
                        ms2RecipAddr + r2Off);
                } else {
                    ComputeScaleCuBLAS(static_cast<uint16_t>(rowWidth), static_cast<uint16_t>(BLOCK_SIZE),
                        gradXAddr + sOff, msA1Addr + s1Off, msB1Addr + s1Off, ms1RecipAddr + r1Off, ms2Addr + s2Off,
                        ms2RecipAddr + r2Off, calcColAB, y1Addr + yOff);
                }
                if constexpr (IsSameType<y1Dtype, fp4x2_e2m1_t>::value || IsSameType<y1Dtype, fp4x2_e1m2_t>::value) {
                    if constexpr (scaleAlg == TPL_SCALE_ALG_0) {
                        ComputeY1ToFP4(static_cast<uint16_t>(rowWidth), static_cast<uint16_t>(BLOCK_SIZE),
                            gradXAddr + sOff, ms1RecipAddr + r1Off, y1Addr + yOff);
                    }
                    ComputeY2ToFP4(static_cast<uint16_t>(rowWidth), static_cast<uint16_t>(BLOCK_SIZE),
                        gradXAddr + sOff, ms2RecipAddr + r2Off, y2Addr + yOff);
                    if (rowWidth > VF_LEN_B16) {
                        ComputeY2ToFP4(static_cast<uint16_t>(rowWidth), static_cast<uint16_t>(BLOCK_SIZE),
                            gradXAddr + sOff + VF_LEN_B16, ms2RecipAddr + r2Off + VF_LEN_B16,
                            y2Addr + yOff + VF_LEN_B16 / DIGIT_TWO);
                    }
                } else {
                    if constexpr (scaleAlg == TPL_SCALE_ALG_0) {
                        ComputeY1ToFP8(static_cast<uint16_t>(rowWidth), static_cast<uint16_t>(BLOCK_SIZE),
                            gradXAddr + sOff, ms1RecipAddr + r1Off, y1Addr + yOff);
                    }
                    ComputeY2ToFP8(static_cast<uint16_t>(rowWidth), static_cast<uint16_t>(BLOCK_SIZE),
                        gradXAddr + sOff, ms2RecipAddr + r2Off, y2Addr + yOff);
                    if (rowWidth > VF_LEN_B16) {
                        ComputeY2ToFP8(static_cast<uint16_t>(rowWidth), static_cast<uint16_t>(BLOCK_SIZE),
                            gradXAddr + sOff + VF_LEN_B16, ms2RecipAddr + r2Off + VF_LEN_B16,
                            y2Addr + yOff + VF_LEN_B16);
                    }
                }
            }
            for (int64_t blk = 1; blk < calcBlockLoop; blk += 2) {
                Interleave(mxScale2[(blk - 1) * rowWidth], mxScale2[blk * rowWidth],
                    mxScale2[(blk - 1) * rowWidth], mxScale2[blk * rowWidth],
                    rowWidth);
            }

            mxScaleAQueue1_.template EnQue(mxScaleA1);
            mxScaleBQueue1_.template EnQue(mxScaleB1);
            outQueue1_.template EnQue(y1);
            mxScaleQueue2_.template EnQue(mxScale2);
            outQueue2_.template EnQue(y2);

            int64_t yGmOffsetA = absRowStart * dimGradX_ + colOffsetAB;
            int64_t yGmOffsetB = absRowStart * dimGradX_ + dimN_ + colOffsetAB;
            int64_t scale1GmOffset = absRowStart * dimNeg1ScaleNum_ + colOffsetAB / BLOCK_SIZE;
            int64_t scale2RowIdx = scale2GmRowOffset + rowBlockIdx * ubRowCount_ / BLOCK_SIZE;
            
            int64_t scale2GmOffsetA = scale2RowIdx * dimGradX_ + colOffsetGradX;
            int64_t scale2GmOffsetB = scale2RowIdx * dimGradX_ + colOffsetGradX + dimN_ * 2;

            CopyOut(yGmOffsetA, yGmOffsetB, scale1GmOffset, scale2GmOffsetA, scale2GmOffsetB, calcRow, calcPadRowAlgin,
                calcColAB, rowWidth);
        }
    }
}

template <typename xDtype, typename y1Dtype, uint64_t mode, AscendC::RoundMode roundMode, uint64_t scaleAlg,
    uint64_t isGroupIdx>
__aicore__ inline void
SwigluBackwardMxQuantWithDualAxisBase<xDtype, y1Dtype, mode, roundMode, scaleAlg, isGroupIdx>::CopyIn(
    int64_t absRowStart, int64_t colOffsetAB, int64_t calcRow, int64_t calcColAB)
{
    int64_t xRowStride = DIGIT_TWO * dimN_;

    LocalTensor<xDtype> xLocal = inQueue_.template AllocTensor<xDtype>();

    DataCopyExtParams copyParams = { 0, 0, 0, 0, 0 };
    DataCopyPadExtParams<xDtype> padParams = { false, 0, 0, 0 };
    copyParams.blockCount = static_cast<uint16_t>(calcRow);
    copyParams.blockLen = static_cast<uint32_t>(calcColAB * static_cast<int64_t>(sizeof(xDtype)));
    copyParams.srcStride = static_cast<uint32_t>((xRowStride - calcColAB) * static_cast<int64_t>(sizeof(xDtype)));

    int64_t leftGmOffset = absRowStart * xRowStride + colOffsetAB;
    DataCopyPad(xLocal, xGm_[leftGmOffset], copyParams, padParams);

    int64_t rightGmOffset = leftGmOffset + dimN_;
    DataCopyPad(xLocal[inHalfSize_], xGm_[rightGmOffset], copyParams, padParams);

    inQueue_.template EnQue(xLocal);
}

template <typename xDtype, typename y1Dtype, uint64_t mode, AscendC::RoundMode roundMode, uint64_t scaleAlg,
    uint64_t isGroupIdx>
__aicore__ inline void
SwigluBackwardMxQuantWithDualAxisBase<xDtype, y1Dtype, mode, roundMode, scaleAlg, isGroupIdx>::CopyInYGrad(
    int64_t absRowStart, int64_t colOffsetAB, int64_t calcRow, int64_t calcColAB)
{
    int64_t yGradRowStride = dimN_;

    LocalTensor<xDtype> yGradLocal = yGradQueue_.template AllocTensor<xDtype>();

    DataCopyExtParams copyParams = { 0, 0, 0, 0, 0 };
    DataCopyPadExtParams<xDtype> padParams = { false, 0, 0, 0 };
    copyParams.blockCount = static_cast<uint16_t>(calcRow);
    copyParams.blockLen = static_cast<uint32_t>(calcColAB * static_cast<int64_t>(sizeof(xDtype)));
    copyParams.srcStride = static_cast<uint32_t>((yGradRowStride - calcColAB) * static_cast<int64_t>(sizeof(xDtype)));

    int64_t yGradGmOffset = absRowStart * yGradRowStride + colOffsetAB;
    DataCopyPad(yGradLocal, yGradGm_[yGradGmOffset], copyParams, padParams);

    yGradQueue_.template EnQue(yGradLocal);
}

template <typename xDtype, typename y1Dtype, uint64_t mode, AscendC::RoundMode roundMode, uint64_t scaleAlg,
    uint64_t isGroupIdx>
__aicore__ inline void
SwigluBackwardMxQuantWithDualAxisBase<xDtype, y1Dtype, mode, roundMode, scaleAlg, isGroupIdx>::ComputeSwigluBackward(
    uint16_t dataLenAB, uint16_t blockCount, __ubuf__ xDtype *actAddr, __ubuf__ xDtype *gateAddr,
    __ubuf__ xDtype *yGradAddr, __ubuf__ xDtype *gradXAddr, uint32_t rowWidth)
{
    uint32_t localOneBlockNum = oneBlockCountB16_;
    uint32_t outAllNum = rowWidth;

    uint16_t dim0VfTimes = blockCount;
    uint32_t localVfLenFp32 = VF_LEN_FP32;
    uint16_t dim1VfTimes = static_cast<uint16_t>(dataLenAB / VF_LEN_FP32);
    uint32_t dim1Tail = dataLenAB % localVfLenFp32;
    uint16_t dim1TailTimes = (dim1Tail > 0) ? 1 : 0;
    uint32_t mask1Num = dim1Tail;
    uint32_t alignDim1In = ((dataLenAB + localOneBlockNum - 1) / localOneBlockNum) * localOneBlockNum;

    float scalarOne = 1.0f;
    float negScalarOne = -1.0f;

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<xDtype> regHalf;
        MicroAPI::RegTensor<float> regA;
        MicroAPI::RegTensor<float> regB;
        MicroAPI::RegTensor<float> regGrad;
        MicroAPI::RegTensor<float> regSig;
        MicroAPI::RegTensor<float> regOut;

        MicroAPI::MaskReg mask;

        for (uint16_t dim0vfLoopIdx = 0; dim0vfLoopIdx < dim0VfTimes; dim0vfLoopIdx++) {
            uint32_t length = dataLenAB;
            for (uint16_t dim1vfLoopIdx = 0; dim1vfLoopIdx < dim1VfTimes + dim1TailTimes; dim1vfLoopIdx++) {
                mask = MicroAPI::UpdateMask<float>(length);
                MicroAPI::AddrReg srcIdxOffset =
                    MicroAPI::CreateAddrReg<xDtype>(dim0vfLoopIdx, alignDim1In, dim1vfLoopIdx, VF_LEN_FP32);
                MicroAPI::DataCopy<xDtype, MicroAPI::LoadDist::DIST_UNPACK_B16>(regHalf, actAddr, srcIdxOffset);
                MicroAPI::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(regA, regHalf, mask);
                MicroAPI::DataCopy<xDtype, MicroAPI::LoadDist::DIST_UNPACK_B16>(regHalf, gateAddr, srcIdxOffset);
                MicroAPI::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(regB, regHalf, mask);
                MicroAPI::DataCopy<xDtype, MicroAPI::LoadDist::DIST_UNPACK_B16>(regHalf, yGradAddr, srcIdxOffset);
                MicroAPI::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(regGrad, regHalf, mask);

                MicroAPI::Muls(regSig, regA, negScalarOne, mask);
                MicroAPI::Exp(regSig, regSig, mask);
                MicroAPI::Adds(regSig, regSig, scalarOne, mask);
                MicroAPI::Duplicate(regOut, scalarOne, mask);
                MicroAPI::Div(regSig, regOut, regSig, mask);

                MicroAPI::Muls(regOut, regSig, negScalarOne, mask);
                MicroAPI::Adds(regOut, regOut, scalarOne, mask);
                MicroAPI::Mul(regOut, regOut, regA, mask);
                MicroAPI::Adds(regOut, regOut, scalarOne, mask);
                MicroAPI::Mul(regOut, regOut, regB, mask);
                MicroAPI::Mul(regOut, regOut, regSig, mask);
                MicroAPI::Mul(regOut, regOut, regGrad, mask);

                MicroAPI::AddrReg outOffsetA =
                    MicroAPI::CreateAddrReg<xDtype>(dim0vfLoopIdx, outAllNum, dim1vfLoopIdx, VF_LEN_FP32);
                MicroAPI::Cast<xDtype, float, CAST_FP32_TO_FP16_BF16>(regHalf, regOut, mask);
                DataCopy<xDtype, MicroAPI::StoreDist::DIST_PACK_B32>(gradXAddr, regHalf, outOffsetA, mask);

                MicroAPI::Mul(regSig, regSig, regGrad, mask);
                MicroAPI::Mul(regSig, regSig, regA, mask);

                MicroAPI::Cast<xDtype, float, CAST_FP32_TO_FP16_BF16>(regHalf, regSig, mask);
                DataCopy<xDtype, MicroAPI::StoreDist::DIST_PACK_B32>(gradXAddr + dataLenAB, regHalf, outOffsetA, mask);
            }
        }
    }
}

template <typename xDtype, typename y1Dtype, uint64_t mode, AscendC::RoundMode roundMode, uint64_t scaleAlg,
    uint64_t isGroupIdx>
__aicore__ inline void
SwigluBackwardMxQuantWithDualAxisBase<xDtype, y1Dtype, mode, roundMode, scaleAlg, isGroupIdx>::PadZeroM(
    __ubuf__ xDtype *swigluOutAddr, uint32_t num)
{
    uint16_t times = CeilDivision(num, 128);
    uint32_t size = num;
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<xDtype> zeroReg;
        AscendC::MicroAPI::MaskReg mask;
        MicroAPI::Duplicate(zeroReg, 0);
        for (uint16_t i = 0; i < times; i++) {
            mask = AscendC::MicroAPI::UpdateMask<xDtype>(size);
            MicroAPI::AddrReg offset = MicroAPI::CreateAddrReg<xDtype>(i, 128);
            AscendC::MicroAPI::DataCopy(swigluOutAddr, zeroReg, offset, mask);
        }
    }
}

template <typename xDtype, typename y1Dtype, uint64_t mode, AscendC::RoundMode roundMode, uint64_t scaleAlg,
    uint64_t isGroupIdx>
__aicore__ inline void
SwigluBackwardMxQuantWithDualAxisBase<xDtype, y1Dtype, mode, roundMode, scaleAlg, isGroupIdx>::ComputeScaleOcp(
    uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype *xAddr, __ubuf__ uint8_t *mxScale1Addr,
    __ubuf__ uint16_t *mxScale1ReciprocalAddr, __ubuf__ uint8_t *mxScale2Addr, __ubuf__ uint16_t *mxScale2ReciprocalAddr)
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
            MicroAPI::Duplicate(x0, static_cast<xDtype>(0));
            MicroAPI::Duplicate(x1, static_cast<xDtype>(0));
            MicroAPI::DataCopy<xDtype, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_DINTLV_B16>(x0,
                x1, xAddr, dataLen);
            if constexpr (IsSameType<xDtype, half>::value) {
                MicroAPI::And(x0ExpFP16, (MicroAPI::RegTensor<uint16_t> &)x0, expMaskFP16, maskAll);
                MicroAPI::And(x1ExpFP16, (MicroAPI::RegTensor<uint16_t> &)x1, expMaskFP16, maskAll);
                MicroAPI::Compare<uint16_t, CMPMODE::NE>(infNanDataMask0, x0ExpFP16, expMaskFP16, maskAll);
                MicroAPI::Compare<uint16_t, CMPMODE::NE>(infNanDataMask1, x1ExpFP16, expMaskFP16, maskAll);
                MicroAPI::Cast<bfloat16_t, xDtype, CAST_HALF_TO_BF16>(x0BF16, x0, maskAll);
                MicroAPI::Cast<bfloat16_t, xDtype, CAST_HALF_TO_BF16>(x1BF16, x1, maskAll);
                MicroAPI::And(x0ExpBF16, (MicroAPI::RegTensor<uint16_t> &)x0BF16, expMaskBF16, maskAll);
                MicroAPI::And(x1ExpBF16, (MicroAPI::RegTensor<uint16_t> &)x1BF16, expMaskBF16, maskAll);
                MicroAPI::Select<uint16_t>(x0ExpBF16, x0ExpBF16, expMaskBF16, infNanDataMask0);
                MicroAPI::Select<uint16_t>(x1ExpBF16, x1ExpBF16, expMaskBF16, infNanDataMask1);
            } else {
                MicroAPI::And(x0ExpBF16, (MicroAPI::RegTensor<uint16_t> &)x0, expMaskBF16, maskAll);
                MicroAPI::And(x1ExpBF16, (MicroAPI::RegTensor<uint16_t> &)x1, expMaskBF16, maskAll);
            }

            MicroAPI::Max(expMaxDim1, x0ExpBF16, x1ExpBF16, maskAll);
            MicroAPI::ReduceMaxWithDataBlock(expMaxDim1, expMaxDim1, maskAll);

            MicroAPI::Max(expMax1Dim2, expMax1Dim2, x0ExpBF16, maskAll);
            MicroAPI::Max(expMax2Dim2, expMax2Dim2, x1ExpBF16, maskAll);

            MicroAPI::Compare<uint16_t, CMPMODE::NE>(infMask, expMaxDim1, expMaskBF16, maskAll);
            MicroAPI::Compare<uint16_t, CMPMODE::NE>(zeroMask, expMaxDim1, zero, maskAll);
            MicroAPI::Compare<uint16_t, CMPMODE::LE>(invalidDataMask, expMaxDim1, yMaxExp, maskAll);
            MicroAPI::Select<uint16_t>(expMaxDim1, yMaxExp, expMaxDim1, invalidDataMask);

            MicroAPI::Sub(expMaxDim1, expMaxDim1, yMaxExp, maskAll);
            MicroAPI::ShiftRights(mxScale1B16, expMaxDim1, SHR_NUM_FOR_BF16, maskAll);
            MicroAPI::Select<uint16_t>(mxScale1B16, mxScale1B16, nanE8M0, infMask);
            MicroAPI::Select<uint16_t>(mxScale1B16, mxScale1B16, zero, zeroMask);

            MicroAPI::Pack<uint8_t, uint16_t, MicroAPI::HighLowPart::LOWEST>(mxScale1B8, mxScale1B16);
            MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(mxScale1Addr, mxScale1B8, 32,
                maskReduceB8);
            MicroAPI::Compare<uint16_t, CMPMODE::EQ>(invalidDataMask, expMaxDim1, biasE8M0, maskAll);

            MicroAPI::Sub(reversedShareExp1, biasE8M0, expMaxDim1, maskAll);
            MicroAPI::Select<uint16_t>(reversedShareExp1, reversedShareExp1, nanBF16, infMask);
            MicroAPI::Select<uint16_t>(reversedShareExp1, reversedShareExp1, zero, zeroMask);
            MicroAPI::Select<uint16_t>(reversedShareExp1, specialExp, reversedShareExp1, invalidDataMask);
            MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(mxScale1ReciprocalAddr,
                reversedShareExp1, 16, maskReduceB16);
        }

        MicroAPI::Compare<uint16_t, CMPMODE::NE>(infMask, expMax1Dim2, expMaskBF16, maskAll);
        MicroAPI::Compare<uint16_t, CMPMODE::NE>(zeroMask, expMax1Dim2, zero, maskAll);
        MicroAPI::Compare<uint16_t, CMPMODE::LE>(invalidDataMask, expMax1Dim2, yMaxExp, maskAll);
        MicroAPI::Select<uint16_t>(expMax1Dim2, yMaxExp, expMax1Dim2, invalidDataMask);
        MicroAPI::Sub(expMax1Dim2, expMax1Dim2, yMaxExp, maskAll);
        MicroAPI::ShiftRights(mxScale2ZeroB16, expMax1Dim2, SHR_NUM_FOR_BF16, maskAll);
        MicroAPI::Select<uint16_t>(mxScale2ZeroB16, mxScale2ZeroB16, nanE8M0, infMask);
        MicroAPI::Select<uint16_t>(mxScale2ZeroB16, mxScale2ZeroB16, zero, zeroMask);

        MicroAPI::Pack<uint8_t, uint16_t, MicroAPI::HighLowPart::LOWEST>(mxScale2ZeroB8, mxScale2ZeroB16);

        MicroAPI::Compare<uint16_t, CMPMODE::EQ>(invalidDataMask, expMax1Dim2, biasE8M0, maskAll);

        MicroAPI::Sub(reversedShareExp2Zero, biasE8M0, expMax1Dim2, maskAll);
        MicroAPI::Select<uint16_t>(reversedShareExp2Zero, reversedShareExp2Zero, nanBF16, infMask);
        MicroAPI::Select<uint16_t>(reversedShareExp2Zero, reversedShareExp2Zero, zero, zeroMask);
        MicroAPI::Select<uint16_t>(reversedShareExp2Zero, specialExp, reversedShareExp2Zero,
            invalidDataMask);

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

        MicroAPI::DataCopy<uint8_t, MicroAPI::StoreDist::DIST_INTLV_B8>(mxScale2Addr, mxScale2ZeroB8, mxScale2OneB8,
            maskB8);
        MicroAPI::DataCopy<uint16_t, MicroAPI::StoreDist::DIST_INTLV_B16>(mxScale2ReciprocalAddr, reversedShareExp2Zero,
            reversedShareExp2One, maskAll);
    }
}

template <typename xDtype, typename y1Dtype, uint64_t mode, AscendC::RoundMode roundMode, uint64_t scaleAlg,
    uint64_t isGroupIdx>
__aicore__ inline void SwigluBackwardMxQuantWithDualAxisBase<xDtype, y1Dtype, mode, roundMode, scaleAlg,
    isGroupIdx>::ComputeScaleCuBLASSecondLast(uint16_t dataLen, uint32_t localInvDtypeMax,
    __ubuf__ uint16_t *mxScale2ReciprocalAddr, __ubuf__ uint8_t *mxScale2Addr)
{
    uint16_t times = dataLen / VF_LEN_FP32;
    __ubuf__ uint16_t *mxScale2ReciprocalOutAddr = mxScale2ReciprocalAddr;
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<uint16_t> max16Reg;
        MicroAPI::RegTensor<uint32_t> max32Reg;
        MicroAPI::RegTensor<uint16_t> absMask;
        MicroAPI::RegTensor<uint32_t> invMax, manMaskReg, expMaskReg, zero32Reg, scaleBiasReg;
        MicroAPI::RegTensor<uint32_t> nan32Reg, fp8Nan32Reg;

        MicroAPI::MaskReg cmpResult, zeroMask, p0, p1, p2;
        MicroAPI::RegTensor<uint32_t> exp32Reg, man32Reg, expOne32Reg, extractExp;
        MicroAPI::RegTensor<uint32_t> halfScale;
        MicroAPI::RegTensor<uint16_t> scale16;
        MicroAPI::RegTensor<uint8_t> scale8Zero;
        MicroAPI::RegTensor<uint16_t> recip16Zero;

        MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<xDtype, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg maskAll32 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg maskB8 = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::VL64>();
        MicroAPI::MaskReg maskB16 = MicroAPI::CreateMask<uint16_t, MicroAPI::MaskPattern::VL64>();
        MicroAPI::Duplicate(scaleBiasReg, FP32_EXP_BIAS_CUBLAS);
        MicroAPI::Duplicate(expMaskReg, MAX_EXP_FOR_FP32);
        MicroAPI::Duplicate(zero32Reg, static_cast<uint32_t>(0));
        MicroAPI::Duplicate(invMax, localInvDtypeMax);
        MicroAPI::Duplicate(manMaskReg, MAN_MASK_FLOAT);
        MicroAPI::Duplicate(fp8Nan32Reg, MAX_EXP_FOR_FP8_IN_FP32);
        MicroAPI::Duplicate(nan32Reg, static_cast<uint32_t>(NAN_CUSTOMIZATION));
        for (uint16_t i = 0; i < times; i++) {
            MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(max16Reg, mxScale2ReciprocalAddr, VF_LEN_FP32);
            MicroAPI::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>((MicroAPI::RegTensor<float> &)max32Reg,
                (MicroAPI::RegTensor<xDtype> &)max16Reg, maskAll);
            MicroAPI::Compare<uint32_t, CMPMODE::LT>(cmpResult, max32Reg, expMaskReg, maskAll32);
            MicroAPI::Compare<uint32_t, CMPMODE::NE>(zeroMask, max32Reg, zero32Reg, maskAll32);
            MicroAPI::Mul((MicroAPI::RegTensor<float> &)max32Reg, (MicroAPI::RegTensor<float> &)max32Reg,
                (MicroAPI::RegTensor<float> &)invMax, maskAll32);
            MicroAPI::ShiftRights(exp32Reg, max32Reg, SHR_NUM_FOR_FP32, maskAll32);
            MicroAPI::And(man32Reg, max32Reg, manMaskReg, maskAll32);
            MicroAPI::CompareScalar<uint32_t, CMPMODE::GT>(p0, exp32Reg, static_cast<uint32_t>(0), maskAll32);
            MicroAPI::CompareScalar<uint32_t, CMPMODE::LT>(p1, exp32Reg, EXP_254, maskAll32);
            MicroAPI::CompareScalar<uint32_t, CMPMODE::GT>(p2, man32Reg, static_cast<uint32_t>(0), maskAll32);
            MicroAPI::MaskAnd(p0, p0, p1, maskAll32);
            MicroAPI::MaskAnd(p0, p0, p2, maskAll32);
            MicroAPI::CompareScalar<uint32_t, CMPMODE::EQ>(p1, exp32Reg, static_cast<uint32_t>(0), maskAll32);
            MicroAPI::CompareScalar<uint32_t, CMPMODE::GT>(p2, man32Reg, HALF_FOR_MAN, maskAll32);
            MicroAPI::MaskAnd(p1, p1, p2, maskAll32);
            MicroAPI::MaskOr(p0, p0, p1, maskAll32);
            MicroAPI::Adds(expOne32Reg, exp32Reg, 1, maskAll32);
            MicroAPI::Select(extractExp, expOne32Reg, exp32Reg, p0);
            MicroAPI::Select<uint32_t>(extractExp, extractExp, fp8Nan32Reg, cmpResult);
            MicroAPI::Select<uint32_t>(extractExp, extractExp, zero32Reg, zeroMask);
            MicroAPI::Pack<uint16_t, uint32_t, MicroAPI::HighLowPart::LOWEST>(scale16, extractExp);
            MicroAPI::Pack<uint8_t, uint16_t, MicroAPI::HighLowPart::LOWEST>(scale8Zero, scale16);

            MicroAPI::ShiftLefts(extractExp, extractExp, SHR_NUM_FOR_BF16, maskAll32);
            MicroAPI::Sub(halfScale, scaleBiasReg, extractExp, maskAll32);
            MicroAPI::Select<uint32_t>(halfScale, halfScale, nan32Reg, cmpResult);
            MicroAPI::Select<uint32_t>(halfScale, halfScale, zero32Reg, zeroMask);
            MicroAPI::Pack<uint16_t, uint32_t, MicroAPI::HighLowPart::LOWEST>(recip16Zero, halfScale);
            MicroAPI::DataCopy<uint8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(mxScale2Addr, scale8Zero,
                VF_LEN_FP32, maskB8);

            MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(mxScale2ReciprocalOutAddr,
                recip16Zero, VF_LEN_FP32, maskB16);
        }
    }
}

template <typename xDtype, typename y1Dtype, uint64_t mode, AscendC::RoundMode roundMode, uint64_t scaleAlg,
    uint64_t isGroupIdx>
__aicore__ inline void SwigluBackwardMxQuantWithDualAxisBase<xDtype, y1Dtype, mode, roundMode, scaleAlg,
    isGroupIdx>::ComputeScaleCuBLAS(uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype *xAddr,
    __ubuf__ uint8_t *mxScaleA1Addr, __ubuf__ uint8_t *mxScaleA2Addr, __ubuf__ uint16_t *mxScale1ReciprocalAddr,
    __ubuf__ uint8_t *mxScale2Addr, __ubuf__ uint16_t *mxScale2ReciprocalAddr, int64_t dataLenAB,
    __ubuf__ uint8_t *y1Addr)
{
    uint32_t localInvDtypeMax = invDtypeMax_;
    uint32_t scale1BIdx = ops::CeilDiv(dataLenAB, BLOCK_SIZE);
    __ubuf__ uint16_t *tempMaxAddr = mxScale2ReciprocalAddr;
    __ubuf__ xDtype *xAddrBase = xAddr;
    __ubuf__ uint16_t *recipBase = mxScale1ReciprocalAddr;

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<xDtype> regA, regB;
        MicroAPI::RegTensor<uint16_t> regU16_0, regU16_1;
        MicroAPI::RegTensor<uint16_t> regU16_2;
        MicroAPI::RegTensor<uint16_t> absMax1Dim2, absMax2Dim2;

        MicroAPI::RegTensor<uint32_t> expAddOne32, extractExp, halfScale;
        MicroAPI::RegTensor<uint32_t> invMax, manMaskReg, expMaskReg, zeroReg32, scaleBiasReg;
        MicroAPI::RegTensor<uint32_t> nanReg32, fp8NanReg32;

        MicroAPI::RegTensor<uint8_t> scale8Reg, scale8Row, scale8Swapped;
        MicroAPI::RegTensor<uint16_t> recip16Row;
        MicroAPI::RegTensor<int8_t> indexShiftS8, extractIdx;
        MicroAPI::RegTensor<uint16_t> absMask;

        MicroAPI::MaskReg cmpResult, zeroMask, p0, p1, p2;
        MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<xDtype, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg maskAll32 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg maskReduceB8 = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::VL8>();
        MicroAPI::MaskReg maskReduceB16 = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::VL16>();
        MicroAPI::UnalignReg ureg;

        MicroAPI::Duplicate(absMask, ABS_MASK_FOR_16BIT);
        MicroAPI::Duplicate(absMax1Dim2, static_cast<uint16_t>(0));
        MicroAPI::Duplicate(absMax2Dim2, static_cast<uint16_t>(0));

        for (uint16_t i = 0; i < blockCount; i++) {
            MicroAPI::DataCopy<xDtype, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_DINTLV_B16>(
                regA, regB, xAddr, dataLen);

            MicroAPI::And(regU16_0, (MicroAPI::RegTensor<uint16_t> &)regA, absMask, maskAll);
            MicroAPI::And(regU16_1, (MicroAPI::RegTensor<uint16_t> &)regB, absMask, maskAll);

            MicroAPI::Max(regU16_2, regU16_0, regU16_1, maskAll);
            MicroAPI::ReduceMaxWithDataBlock(regU16_2, regU16_2, maskAll);
            // 8: 每次循环scale1偏移8个元素
            MicroAPI::StoreUnAlign<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(tempMaxAddr, regU16_2, ureg, 8);

            MicroAPI::Max(absMax1Dim2, absMax1Dim2, regU16_0, maskAll);
            MicroAPI::Max(absMax2Dim2, absMax2Dim2, regU16_1, maskAll);
        }
        MicroAPI::StoreUnAlignPost(tempMaxAddr, ureg, 0);

        MicroAPI::Duplicate(invMax, localInvDtypeMax);
        MicroAPI::Duplicate(manMaskReg, MAN_MASK_FLOAT);
        MicroAPI::Duplicate(expMaskReg, MAX_EXP_FOR_FP32);
        MicroAPI::Duplicate(zeroReg32, static_cast<uint32_t>(0));
        MicroAPI::Duplicate(scaleBiasReg, FP32_EXP_BIAS_CUBLAS);
        MicroAPI::Duplicate(nanReg32, static_cast<uint32_t>(NAN_CUSTOMIZATION));
        MicroAPI::Duplicate(fp8NanReg32, MAX_EXP_FOR_FP8_IN_FP32);

        MicroAPI::Arange(indexShiftS8, static_cast<int8_t>(scale1BIdx));

        __ubuf__ uint16_t *readMaxAddr = mxScale2ReciprocalAddr;
        // 8: 每次处理的数据元素个数=32 * 8
        uint16_t batchCount = ops::CeilDiv(static_cast<uint16_t>(blockCount * 8),
            static_cast<uint16_t>(VF_LEN_FP32));

        for (uint16_t j = 0; j < batchCount; j++) {
            MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE,
                MicroAPI::LoadDist::DIST_UNPACK_B16>(regU16_0, readMaxAddr, VF_LEN_FP32);
            MicroAPI::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(
                (MicroAPI::RegTensor<float> &)regA, (MicroAPI::RegTensor<xDtype> &)regU16_0, maskAll);

            MicroAPI::Compare<uint32_t, CMPMODE::LT>(cmpResult,
                (MicroAPI::RegTensor<uint32_t> &)regA, expMaskReg, maskAll32);
            MicroAPI::Compare<uint32_t, CMPMODE::NE>(zeroMask,
                (MicroAPI::RegTensor<uint32_t> &)regA, zeroReg32, maskAll32);
            MicroAPI::Mul((MicroAPI::RegTensor<float> &)regA, (MicroAPI::RegTensor<float> &)regA,
                (MicroAPI::RegTensor<float> &)invMax, maskAll32);
            MicroAPI::ShiftRights((MicroAPI::RegTensor<uint32_t> &)regB,
                (MicroAPI::RegTensor<uint32_t> &)regA, SHR_NUM_FOR_FP32, maskAll32);
            MicroAPI::And((MicroAPI::RegTensor<uint32_t> &)regU16_1,
                (MicroAPI::RegTensor<uint32_t> &)regA, manMaskReg, maskAll32);

            MicroAPI::CompareScalar<uint32_t, CMPMODE::GT>(p0,
                (MicroAPI::RegTensor<uint32_t> &)regB, static_cast<uint32_t>(0), maskAll32);
            MicroAPI::CompareScalar<uint32_t, CMPMODE::LT>(p1,
                (MicroAPI::RegTensor<uint32_t> &)regB, EXP_254, maskAll32);
            MicroAPI::CompareScalar<uint32_t, CMPMODE::GT>(p2,
                (MicroAPI::RegTensor<uint32_t> &)regU16_1, static_cast<uint32_t>(0), maskAll32);
            MicroAPI::MaskAnd(p0, p0, p1, maskAll32);
            MicroAPI::MaskAnd(p0, p0, p2, maskAll32);
            MicroAPI::CompareScalar<uint32_t, CMPMODE::EQ>(p1,
                (MicroAPI::RegTensor<uint32_t> &)regB, static_cast<uint32_t>(0), maskAll32);
            MicroAPI::CompareScalar<uint32_t, CMPMODE::GT>(p2,
                (MicroAPI::RegTensor<uint32_t> &)regU16_1, HALF_FOR_MAN, maskAll32);
            MicroAPI::MaskAnd(p1, p1, p2, maskAll32);
            MicroAPI::MaskOr(p0, p0, p1, maskAll32);

            MicroAPI::Adds(expAddOne32, (MicroAPI::RegTensor<uint32_t> &)regB, 1, maskAll32);
            MicroAPI::Select(extractExp, expAddOne32, (MicroAPI::RegTensor<uint32_t> &)regB, p0);
            MicroAPI::Select<uint32_t>(extractExp, extractExp, fp8NanReg32, cmpResult);
            MicroAPI::Select<uint32_t>(extractExp, extractExp, zeroReg32, zeroMask);

            MicroAPI::Pack<uint16_t, uint32_t, MicroAPI::HighLowPart::LOWEST>(regU16_2, extractExp);
            MicroAPI::Pack<uint8_t, uint16_t, MicroAPI::HighLowPart::LOWEST>(scale8Reg, regU16_2);

            MicroAPI::ShiftLefts(extractExp, extractExp, SHR_NUM_FOR_BF16, maskAll32);
            MicroAPI::Sub(halfScale, scaleBiasReg, extractExp, maskAll32);
            MicroAPI::Select<uint32_t>(halfScale, halfScale, nanReg32, cmpResult);
            MicroAPI::Select<uint32_t>(halfScale, halfScale, zeroReg32, zeroMask);
            MicroAPI::Pack<uint16_t, uint32_t, MicroAPI::HighLowPart::LOWEST>(regU16_2, halfScale);

            // 8: 一个RegTensor可容纳64个fp32元素，每次循环处理8个元素
            for (uint16_t k = 0; k < 8; k++) {
                // 8: scale1的搬运, 每次循环索引值的更新步长为8
                MicroAPI::Arange(extractIdx, static_cast<int8_t>(k * 8));
                MicroAPI::Gather(scale8Row, scale8Reg, (MicroAPI::RegTensor<uint8_t> &)extractIdx);
                MicroAPI::Gather(scale8Swapped, scale8Row, (MicroAPI::RegTensor<uint8_t> &)indexShiftS8);

                // 32: 偏移32字节
                MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    mxScaleA1Addr, scale8Row, 32, maskReduceB8);
                MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    mxScaleA2Addr, scale8Swapped, 32, maskReduceB8);

                // 16: 1/scale1的搬运, 每次循环索引值的更新步长为16
                MicroAPI::Arange(extractIdx, static_cast<int8_t>(k * 16));
                MicroAPI::Gather((MicroAPI::RegTensor<uint8_t> &)recip16Row,
                    (MicroAPI::RegTensor<uint8_t> &)regU16_2, (MicroAPI::RegTensor<uint8_t> &)extractIdx);
                MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    mxScale1ReciprocalAddr, recip16Row, 16, maskReduceB16);
            }
        }

        MicroAPI::MaskReg maskAllB8 = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::RegTensor<uint16_t> scaleForMulFP16;
        MicroAPI::RegTensor<float> scaleForMulFP32;
        MicroAPI::RegTensor<float> x0ZeroFP32, x0OneFP32, x1ZeroFP32, x1OneFP32;
        MicroAPI::RegTensor<y1Dtype> fp8Part0, fp8Part1, fp8Part2, fp8Part3;

        __ubuf__ xDtype *xReadAddr = xAddrBase;
        __ubuf__ uint16_t *recipReadAddr = recipBase;

        for (uint16_t i = 0; i < blockCount; i++) {
            MicroAPI::DataCopy<xDtype, MicroAPI::PostLiteral::POST_MODE_UPDATE,
                MicroAPI::LoadDist::DIST_DINTLV_B16>(regA, regB, xReadAddr, dataLen);
            MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE,
                MicroAPI::LoadDist::DIST_E2B_B16>(scaleForMulFP16, recipReadAddr, 16); // 1/scale1每次偏移16个元素

            if constexpr (IsSameType<xDtype, half>::value) {
                MicroAPI::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(x0ZeroFP32, regA, maskAll);
                MicroAPI::Cast<float, xDtype, CAST_X_TO_FP32_ONE>(x0OneFP32, regA, maskAll);
                MicroAPI::Cast<float, bfloat16_t, CAST_X_TO_FP32_ZERO>(scaleForMulFP32,
                    (MicroAPI::RegTensor<bfloat16_t> &)scaleForMulFP16, maskAll);
                MicroAPI::Mul(x0ZeroFP32, x0ZeroFP32, scaleForMulFP32, maskAll);
                MicroAPI::Mul(x0OneFP32, x0OneFP32, scaleForMulFP32, maskAll);
                MicroAPI::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(x1ZeroFP32, regB, maskAll);
                MicroAPI::Cast<float, xDtype, CAST_X_TO_FP32_ONE>(x1OneFP32, regB, maskAll);
                MicroAPI::Mul(x1ZeroFP32, x1ZeroFP32, scaleForMulFP32, maskAll);
                MicroAPI::Mul(x1OneFP32, x1OneFP32, scaleForMulFP32, maskAll);
            } else {
                MicroAPI::Mul(regA, regA, (MicroAPI::RegTensor<xDtype> &)scaleForMulFP16, maskAll);
                MicroAPI::Mul(regB, regB, (MicroAPI::RegTensor<xDtype> &)scaleForMulFP16, maskAll);
                MicroAPI::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(x0ZeroFP32, regA, maskAll);
                MicroAPI::Cast<float, xDtype, CAST_X_TO_FP32_ONE>(x0OneFP32, regA, maskAll);
                MicroAPI::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(x1ZeroFP32, regB, maskAll);
                MicroAPI::Cast<float, xDtype, CAST_X_TO_FP32_ONE>(x1OneFP32, regB, maskAll);
            }
            AscendC::MicroAPI::Cast<y1Dtype, float, CAST_32_TO_80>(fp8Part0, x0ZeroFP32, maskAll);
            AscendC::MicroAPI::Cast<y1Dtype, float, CAST_32_TO_82>(fp8Part1, x0OneFP32, maskAll);
            AscendC::MicroAPI::Cast<y1Dtype, float, CAST_32_TO_81>(fp8Part2, x1ZeroFP32, maskAll);
            AscendC::MicroAPI::Cast<y1Dtype, float, CAST_32_TO_83>(fp8Part3, x1OneFP32, maskAll);
            AscendC::MicroAPI::Add((MicroAPI::RegTensor<uint8_t> &)fp8Part0,
                (MicroAPI::RegTensor<uint8_t> &)fp8Part0, (MicroAPI::RegTensor<uint8_t> &)fp8Part1, maskAllB8);
            AscendC::MicroAPI::Add((MicroAPI::RegTensor<uint8_t> &)fp8Part0,
                (MicroAPI::RegTensor<uint8_t> &)fp8Part0, (MicroAPI::RegTensor<uint8_t> &)fp8Part2, maskAllB8);
            AscendC::MicroAPI::Add((MicroAPI::RegTensor<uint8_t> &)fp8Part0,
                (MicroAPI::RegTensor<uint8_t> &)fp8Part0, (MicroAPI::RegTensor<uint8_t> &)fp8Part3, maskAllB8);
            AscendC::MicroAPI::DataCopy<uint8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                AscendC::MicroAPI::StoreDist::DIST_NORM_B8>(y1Addr,
                (MicroAPI::RegTensor<uint8_t> &)fp8Part0, dataLen, maskAllB8);
        }

        MicroAPI::DataCopy<uint16_t, MicroAPI::StoreDist::DIST_INTLV_B16>(mxScale2ReciprocalAddr, absMax1Dim2,
            absMax2Dim2, maskAll);
    }
    ComputeScaleCuBLASSecondLast(dataLen, localInvDtypeMax, mxScale2ReciprocalAddr, mxScale2Addr);
}

template <typename xDtype, typename y1Dtype, uint64_t mode, AscendC::RoundMode roundMode, uint64_t scaleAlg,
    uint64_t isGroupIdx>
__aicore__ inline void
SwigluBackwardMxQuantWithDualAxisBase<xDtype, y1Dtype, mode, roundMode, scaleAlg, isGroupIdx>::ComputeY1ToFP8(
    uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype *xAddr, __ubuf__ uint16_t *mxScale1ReciprocalAddr,
    __ubuf__ uint8_t *y1Addr)
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
            MicroAPI::DataCopy<xDtype, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_DINTLV_B16>(x0,
                x1, xAddr, dataLen);
            MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_E2B_B16>(
                scaleForMulFP16, mxScale1ReciprocalAddr, 16);

            if constexpr (IsSameType<xDtype, half>::value) {
                MicroAPI::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(x0ZeroFP32, x0, maskAll);
                MicroAPI::Cast<float, xDtype, CAST_X_TO_FP32_ONE>(x0OneFP32, x0, maskAll);
                MicroAPI::Cast<float, bfloat16_t, CAST_X_TO_FP32_ZERO>(scaleForMulFP32,
                    (MicroAPI::RegTensor<bfloat16_t> &)scaleForMulFP16, maskAll);

                MicroAPI::Mul(x0ZeroFP32, x0ZeroFP32, scaleForMulFP32, maskAll);
                MicroAPI::Mul(x0OneFP32, x0OneFP32, scaleForMulFP32, maskAll);

                MicroAPI::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(x1ZeroFP32, x1, maskAll);
                MicroAPI::Cast<float, xDtype, CAST_X_TO_FP32_ONE>(x1OneFP32, x1, maskAll);
                MicroAPI::Mul(x1ZeroFP32, x1ZeroFP32, scaleForMulFP32, maskAll);
                MicroAPI::Mul(x1OneFP32, x1OneFP32, scaleForMulFP32, maskAll);
            } else {
                MicroAPI::Mul(x0, x0, (MicroAPI::RegTensor<xDtype> &)scaleForMulFP16, maskAll);
                MicroAPI::Mul(x1, x1, (MicroAPI::RegTensor<xDtype> &)scaleForMulFP16, maskAll);

                AscendC::MicroAPI::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(x0ZeroFP32, x0, maskAll);
                AscendC::MicroAPI::Cast<float, xDtype, CAST_X_TO_FP32_ONE>(x0OneFP32, x0, maskAll);
                AscendC::MicroAPI::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(x1ZeroFP32, x1, maskAll);
                AscendC::MicroAPI::Cast<float, xDtype, CAST_X_TO_FP32_ONE>(x1OneFP32, x1, maskAll);
            }
            AscendC::MicroAPI::Cast<y1Dtype, float, CAST_32_TO_80>(x0ZeroFP8, x0ZeroFP32, maskAll);
            AscendC::MicroAPI::Cast<y1Dtype, float, CAST_32_TO_82>(x0OneFP8, x0OneFP32, maskAll);
            AscendC::MicroAPI::Cast<y1Dtype, float, CAST_32_TO_81>(x1ZeroFP8, x1ZeroFP32, maskAll);
            AscendC::MicroAPI::Cast<y1Dtype, float, CAST_32_TO_83>(x1OneFP8, x1OneFP32, maskAll);

            AscendC::MicroAPI::Add((AscendC::MicroAPI::RegTensor<uint8_t> &)x0ZeroFP8,
                (AscendC::MicroAPI::RegTensor<uint8_t> &)x0ZeroFP8, (AscendC::MicroAPI::RegTensor<uint8_t> &)x0OneFP8,
                maskAllB8);
            AscendC::MicroAPI::Add((AscendC::MicroAPI::RegTensor<uint8_t> &)x0ZeroFP8,
                (AscendC::MicroAPI::RegTensor<uint8_t> &)x0ZeroFP8, (AscendC::MicroAPI::RegTensor<uint8_t> &)x1ZeroFP8,
                maskAllB8);
            AscendC::MicroAPI::Add((AscendC::MicroAPI::RegTensor<uint8_t> &)x0ZeroFP8,
                (AscendC::MicroAPI::RegTensor<uint8_t> &)x0ZeroFP8, (AscendC::MicroAPI::RegTensor<uint8_t> &)x1OneFP8,
                maskAllB8);
            AscendC::MicroAPI::DataCopy<uint8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                AscendC::MicroAPI::StoreDist::DIST_NORM_B8>(y1Addr, (AscendC::MicroAPI::RegTensor<uint8_t> &)x0ZeroFP8,
                dataLen, maskAllB8);
        }
    }
}

template <typename xDtype, typename y1Dtype, uint64_t mode, AscendC::RoundMode roundMode, uint64_t scaleAlg,
    uint64_t isGroupIdx>
__aicore__ inline void
SwigluBackwardMxQuantWithDualAxisBase<xDtype, y1Dtype, mode, roundMode, scaleAlg, isGroupIdx>::ComputeY1ToFP4(
    uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype *xAddr, __ubuf__ uint16_t *mxScale1ReciprocalAddr,
    __ubuf__ uint8_t *y1Addr)
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
            MicroAPI::Duplicate(x0, static_cast<xDtype>(0));
            MicroAPI::Duplicate(x1, static_cast<xDtype>(0));
            MicroAPI::DataCopy<xDtype, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_DINTLV_B16>(x0,
                x1, xAddr, dataLen);
            MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_E2B_B16>(
                scaleForMulFP16, mxScale1ReciprocalAddr, 16);

            if constexpr (IsSameType<xDtype, half>::value) {
                MicroAPI::Cast<float, bfloat16_t, CAST_X_TO_FP32_ZERO>(scaleForMulZeroFP32,
                    (MicroAPI::RegTensor<bfloat16_t> &)scaleForMulFP16, dataMaskB16);

                MicroAPI::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(x0ZeroFP32, x0, dataMaskB16);
                MicroAPI::Cast<float, xDtype, CAST_X_TO_FP32_ONE>(x0OneFP32, x0, dataMaskB16);

                MicroAPI::Mul(x0ZeroFP32, scaleForMulZeroFP32, x0ZeroFP32, dataMaskB32);
                MicroAPI::Mul(x0OneFP32, scaleForMulZeroFP32, x0OneFP32, dataMaskB32);
                ComputeFP4FromHalf(x0ZeroFP32);
                ComputeFP4FromHalf(x0OneFP32);
                MicroAPI::Cast<bfloat16_t, float, castTraitFp32toBF16>(x0ZeroBF16, x0ZeroFP32, dataMaskB32);
                MicroAPI::Cast<bfloat16_t, float, castTraitFp32toBF16>(x0OneBF16, x0OneFP32, dataMaskB32);
                MicroAPI::Pack<uint16_t, uint32_t, MicroAPI::HighLowPart::LOWEST>(
                    (MicroAPI::RegTensor<uint16_t> &)x0ZeroBF16, (MicroAPI::RegTensor<uint32_t> &)x0ZeroBF16);
                MicroAPI::Pack<uint16_t, uint32_t, MicroAPI::HighLowPart::LOWEST>(
                    (MicroAPI::RegTensor<uint16_t> &)x0OneBF16, (MicroAPI::RegTensor<uint32_t> &)x0OneBF16);
                MicroAPI::Interleave(x0ZeroBF16, x0OneBF16, x0ZeroBF16, x0OneBF16);

                MicroAPI::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(x1ZeroFP32, x1, dataMaskB16);
                MicroAPI::Cast<float, xDtype, CAST_X_TO_FP32_ONE>(x1OneFP32, x1, dataMaskB16);

                MicroAPI::Mul(x1ZeroFP32, scaleForMulZeroFP32, x1ZeroFP32, dataMaskB32);
                MicroAPI::Mul(x1OneFP32, scaleForMulZeroFP32, x1OneFP32, dataMaskB32);
                ComputeFP4FromHalf(x1ZeroFP32);
                ComputeFP4FromHalf(x1OneFP32);
                MicroAPI::Cast<bfloat16_t, float, castTraitFp32toBF16>(x1ZeroBF16, x1ZeroFP32, dataMaskB32);
                MicroAPI::Cast<bfloat16_t, float, castTraitFp32toBF16>(x1OneBF16, x1OneFP32, dataMaskB32);
                MicroAPI::Pack<uint16_t, uint32_t, MicroAPI::HighLowPart::LOWEST>(
                    (MicroAPI::RegTensor<uint16_t> &)x1ZeroBF16, (MicroAPI::RegTensor<uint32_t> &)x1ZeroBF16);
                MicroAPI::Pack<uint16_t, uint32_t, MicroAPI::HighLowPart::LOWEST>(
                    (MicroAPI::RegTensor<uint16_t> &)x1OneBF16, (MicroAPI::RegTensor<uint32_t> &)x1OneBF16);
                MicroAPI::Interleave(x1ZeroBF16, x1OneBF16, x1ZeroBF16, x1OneBF16);

                MicroAPI::Interleave(x0ZeroBF16, x1ZeroBF16, x0ZeroBF16, x1ZeroBF16);
                MicroAPI::Cast<y1Dtype, bfloat16_t, castTraitBF16toFp4>(x0FP4, x0ZeroBF16, dataMaskB16);
                MicroAPI::Cast<y1Dtype, bfloat16_t, castTraitBF16toFp4>(x1FP4, x1ZeroBF16, dataMaskB16);
            } else {
                MicroAPI::Mul(x0, x0, (MicroAPI::RegTensor<xDtype> &)scaleForMulFP16, dataMaskB16);
                MicroAPI::Mul(x1, x1, (MicroAPI::RegTensor<xDtype> &)scaleForMulFP16, dataMaskB16);
                MicroAPI::Interleave(x0, x1, x0, x1);
                MicroAPI::Cast<y1Dtype, xDtype, castTraitBF16toFp4>(x0FP4, x0, dataMaskB16);
                MicroAPI::Cast<y1Dtype, xDtype, castTraitBF16toFp4>(x1FP4, x1, dataMaskB16);
            }

            MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::StoreDist::DIST_PACK4_B32>(
                y1Addr, (MicroAPI::RegTensor<uint8_t> &)x0FP4, OUT_ELE_NUM_ONE_BLK, dataMaskB8);
            MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::StoreDist::DIST_PACK4_B32>(
                y1Addr, (MicroAPI::RegTensor<uint8_t> &)x1FP4, OUT_ELE_NUM_ONE_BLK, dataMaskB8);
        }
    }
    return;
}

template <typename xDtype, typename y1Dtype, uint64_t mode, AscendC::RoundMode roundMode, uint64_t scaleAlg,
    uint64_t isGroupIdx>
__aicore__ inline void
SwigluBackwardMxQuantWithDualAxisBase<xDtype, y1Dtype, mode, roundMode, scaleAlg, isGroupIdx>::ComputeY2ToFP8(
    uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype *xAddr, __ubuf__ uint16_t *mxScale2ReciprocalAddr,
    __ubuf__ uint8_t *y2Addr)
{
    int64_t localUbRowLen = dataLen;

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<xDtype> x;
        MicroAPI::RegTensor<float> x0FP32;
        MicroAPI::RegTensor<float> x1FP32;
        MicroAPI::RegTensor<uint16_t> reversedShareExp;
        MicroAPI::RegTensor<float> reversedShareExp0FP32;
        MicroAPI::RegTensor<float> reversedShareExp1FP32;
        MicroAPI::RegTensor<y1Dtype> yZeroFP8;
        MicroAPI::RegTensor<y1Dtype> yOneFP8;

        MicroAPI::MaskReg pregAll8 = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg pregAll16 = MicroAPI::CreateMask<uint16_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg pregAll32 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();

        MicroAPI::DataCopy<uint16_t, MicroAPI::LoadDist::DIST_NORM>(reversedShareExp, mxScale2ReciprocalAddr);
        MicroAPI::Cast<float, bfloat16_t, CAST_X_TO_FP32_ZERO>(reversedShareExp0FP32,
            (MicroAPI::RegTensor<bfloat16_t> &)reversedShareExp, pregAll16);
        MicroAPI::Cast<float, bfloat16_t, CAST_X_TO_FP32_ONE>(reversedShareExp1FP32,
            (MicroAPI::RegTensor<bfloat16_t> &)reversedShareExp, pregAll16);

        for (uint16_t j = 0; j < blockCount; j++) {
            MicroAPI::DataCopy<xDtype, MicroAPI::LoadDist::DIST_NORM>(x, xAddr + j * localUbRowLen);
            MicroAPI::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(x0FP32, x, pregAll16);
            MicroAPI::Cast<float, xDtype, CAST_X_TO_FP32_ONE>(x1FP32, x, pregAll16);

            MicroAPI::Mul(x0FP32, x0FP32, reversedShareExp0FP32, pregAll32);
            MicroAPI::Mul(x1FP32, x1FP32, reversedShareExp1FP32, pregAll32);

            MicroAPI::Cast<y1Dtype, float, castTraitFp32toYdtype>(yZeroFP8, (MicroAPI::RegTensor<float> &)x0FP32,
                pregAll32);
            MicroAPI::Pack<uint16_t, uint32_t, MicroAPI::HighLowPart::LOWEST>((MicroAPI::RegTensor<uint16_t> &)yZeroFP8,
                (MicroAPI::RegTensor<uint32_t> &)yZeroFP8);

            MicroAPI::Cast<y1Dtype, float, castTraitFp32toYdtype>(yOneFP8, (MicroAPI::RegTensor<float> &)x1FP32,
                pregAll32);
            MicroAPI::Pack<uint16_t, uint32_t, MicroAPI::HighLowPart::LOWEST>((MicroAPI::RegTensor<uint16_t> &)yOneFP8,
                (MicroAPI::RegTensor<uint32_t> &)yOneFP8);

            MicroAPI::Interleave((MicroAPI::RegTensor<uint16_t> &)yZeroFP8, (MicroAPI::RegTensor<uint16_t> &)yOneFP8,
                (MicroAPI::RegTensor<uint16_t> &)yZeroFP8, (MicroAPI::RegTensor<uint16_t> &)yOneFP8);

            MicroAPI::Pack<uint8_t, uint16_t, MicroAPI::HighLowPart::LOWEST>((MicroAPI::RegTensor<uint8_t> &)yZeroFP8,
                (MicroAPI::RegTensor<uint16_t> &)yZeroFP8);

            DataCopy(y2Addr + (j * localUbRowLen), (MicroAPI::RegTensor<uint8_t> &)yZeroFP8, pregAll8);
        }
    }
}

template <typename xDtype, typename y1Dtype, uint64_t mode, AscendC::RoundMode roundMode, uint64_t scaleAlg,
    uint64_t isGroupIdx>
__aicore__ inline void
SwigluBackwardMxQuantWithDualAxisBase<xDtype, y1Dtype, mode, roundMode, scaleAlg, isGroupIdx>::ComputeY2ToFP4(
    uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype *xAddr, __ubuf__ uint16_t *mxScale2ReciprocalAddr,
    __ubuf__ uint8_t *y2Addr)
{
    int64_t localUbRowLen = dataLen;

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

        MicroAPI::DataCopy<uint16_t, MicroAPI::LoadDist::DIST_NORM>(reversedShareExp, mxScale2ReciprocalAddr);

        for (uint16_t j = 0; j < blockCount; j++) {
            MicroAPI::DataCopy<xDtype, MicroAPI::LoadDist::DIST_NORM>(x, xAddr + j * localUbRowLen);

            if constexpr (IsSameType<xDtype, half>::value) {
                MicroAPI::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(x0FP32, x, pregAll16);
                MicroAPI::Cast<float, xDtype, CAST_X_TO_FP32_ONE>(x1FP32, x, pregAll16);

                MicroAPI::Cast<float, bfloat16_t, CAST_X_TO_FP32_ZERO>(reversedShareExp0FP32,
                    (MicroAPI::RegTensor<bfloat16_t> &)reversedShareExp, pregAll16);

                MicroAPI::Cast<float, bfloat16_t, CAST_X_TO_FP32_ONE>(reversedShareExp1FP32,
                    (MicroAPI::RegTensor<bfloat16_t> &)reversedShareExp, pregAll16);

                MicroAPI::Mul(x0FP32, x0FP32, reversedShareExp0FP32, pregAll32);
                MicroAPI::Mul(x1FP32, x1FP32, reversedShareExp1FP32, pregAll32);

                ComputeFP4FromHalf(x0FP32);
                ComputeFP4FromHalf(x1FP32);

                MicroAPI::Cast<bfloat16_t, float, castTraitFp32toBF16>((MicroAPI::RegTensor<bfloat16_t> &)x0BF16,
                    x0FP32, pregAll32);
                MicroAPI::Pack<uint16_t, uint32_t, MicroAPI::HighLowPart::LOWEST>(
                    (MicroAPI::RegTensor<uint16_t> &)x0BF16, (MicroAPI::RegTensor<uint32_t> &)x0BF16);

                MicroAPI::Cast<bfloat16_t, float, castTraitFp32toBF16>((MicroAPI::RegTensor<bfloat16_t> &)x1BF16,
                    x1FP32, pregAll32);

                MicroAPI::Pack<uint16_t, uint32_t, MicroAPI::HighLowPart::LOWEST>(
                    (MicroAPI::RegTensor<uint16_t> &)x1BF16, (MicroAPI::RegTensor<uint32_t> &)x1BF16);

                MicroAPI::Interleave(x0BF16, x1BF16, x0BF16, x1BF16);
                MicroAPI::Cast<y1Dtype, bfloat16_t, castTraitBF16toFp4>(yZeroFP4,
                    (MicroAPI::RegTensor<bfloat16_t> &)x0BF16, pregAll16);
            } else {
                MicroAPI::Mul(xBF16, x, (MicroAPI::RegTensor<bfloat16_t> &)reversedShareExp, pregAll16);
                MicroAPI::Cast<y1Dtype, bfloat16_t, castTraitBF16toFp4>(yZeroFP4, xBF16, pregAll16);
            }

            DataCopy<uint8_t, MicroAPI::StoreDist::DIST_PACK4_B32>(y2Addr + (j * dataLen / DIGIT_TWO),
                (MicroAPI::RegTensor<uint8_t> &)yZeroFP4, pregAll8);
        }
    }
}

template <typename xDtype, typename y1Dtype, uint64_t mode, AscendC::RoundMode roundMode, uint64_t scaleAlg,
    uint64_t isGroupIdx>
__aicore__ inline void SwigluBackwardMxQuantWithDualAxisBase<xDtype, y1Dtype, mode, roundMode, scaleAlg,
    isGroupIdx>::ComputeFP4FromHalf(MicroAPI::RegTensor<float> &Reg)
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

    MicroAPI::Compare<int32_t, CMPMODE::EQ>(negInfMask, (MicroAPI::RegTensor<int32_t> &)Reg, negZero, pregAll32);
    if constexpr (IsSameType<y1Dtype, fp4x2_e1m2_t>::value) {
        MicroAPI::Muls(Reg, Reg, FOUR, pregAll32);
        MicroAPI::CompareScalar<float, CMPMODE::LT>(specialMask, Reg, 0, pregAll32);
        MicroAPI::Truncate<float, roundMode>(Reg, Reg, pregAll32);
        MicroAPI::Muls(Reg, Reg, ONE_FOURTH, pregAll32);
    } else {
        MicroAPI::Duplicate(maxExpFP32, MAX_EXP_FOR_FP32);
        MicroAPI::And(exp0FP32, (MicroAPI::RegTensor<int32_t> &)Reg, maxExpFP32, pregAll32);
        MicroAPI::ShiftRights(exp0FP32, exp0FP32, SHR_NUM_FOR_FP32, pregAll32);
        MicroAPI::Adds(exp0FP32, exp0FP32, FP32_BIAS_NEG, pregAll32);
        MicroAPI::Maxs(exp0FP32, exp0FP32, 0, pregAll32);
        MicroAPI::Adds(exp0FP32, exp0FP32, NEG_ONE, pregAll32);
        MicroAPI::Muls(exp1FP32, exp0FP32, NEG_ONE, pregAll32);
        MicroAPI::Adds(exp1FP32, exp1FP32, FP32_BIAS, pregAll32);
        MicroAPI::ShiftLefts(exp1FP32, exp1FP32, SHR_NUM_FOR_FP32, pregAll32);

        MicroAPI::Mul(Reg, Reg, (MicroAPI::RegTensor<float> &)exp1FP32, pregAll32);
        MicroAPI::Adds(exp0FP32, exp0FP32, FP32_BIAS, pregAll32);
        MicroAPI::ShiftLefts(exp0FP32, exp0FP32, SHR_NUM_FOR_FP32, pregAll32);
        MicroAPI::CompareScalar<float, CMPMODE::LT>(specialMask, Reg, 0, pregAll32);
        MicroAPI::Truncate<float, roundMode>(Reg, Reg, pregAll32);
        MicroAPI::Mul(Reg, Reg, (MicroAPI::RegTensor<float> &)exp0FP32, pregAll32);
    }
    MicroAPI::CompareScalar<float, CMPMODE::EQ>(zeroMask, Reg, 0, pregAll32);
    MicroAPI::MaskAnd(zeroMask, specialMask, zeroMask, pregAll32);
    MicroAPI::MaskOr(zeroMask, negInfMask, zeroMask, pregAll32);
    MicroAPI::Select<int32_t>((MicroAPI::RegTensor<int32_t> &)Reg, negZero, (MicroAPI::RegTensor<int32_t> &)Reg,
        zeroMask);
}

template <typename xDtype, typename y1Dtype, uint64_t mode, AscendC::RoundMode roundMode, uint64_t scaleAlg,
    uint64_t isGroupIdx>
__aicore__ inline void
SwigluBackwardMxQuantWithDualAxisBase<xDtype, y1Dtype, mode, roundMode, scaleAlg, isGroupIdx>::CopyOut(
    int64_t yOffsetA, int64_t yOffsetB, int64_t scale1OutOffset, int64_t scale2OutOffsetA, int64_t scale2OutOffsetB,
    int64_t blockCount, int64_t blockCountAlign, int64_t dataLenAB, int64_t rowWidth)
{
    uint16_t outBurst = static_cast<uint16_t>(blockCount);
    uint32_t outBlockLen = 0;
    uint32_t srcStride = 0;
    uint32_t dstStride = 0;
    uint32_t gradBOffset = 0;

    bool yRowByRow = false; // 当每行处理的数据量不是32B对齐时，采用RowByRow搬运方式
    uint32_t yRowPitch = 0;
    if constexpr (IsSameType<y1Dtype, fp4x2_e2m1_t>::value || IsSameType<y1Dtype, fp4x2_e1m2_t>::value) {
        outBlockLen = static_cast<uint32_t>(dataLenAB / DIGIT_TWO * sizeof(uint8_t));
        yRowPitch = static_cast<uint32_t>(rowWidth / DIGIT_TWO * sizeof(uint8_t));
        srcStride = static_cast<uint32_t>((yRowPitch - outBlockLen) / UB_BLOCK_SIZE);
        dstStride = static_cast<uint32_t>((dimGradX_ - dataLenAB) / DIGIT_TWO * sizeof(uint8_t));
        yOffsetA = yOffsetA / DIGIT_TWO;
        yOffsetB = yOffsetB / DIGIT_TWO;
        gradBOffset = dataLenAB / DIGIT_TWO;
        if ((yRowPitch - outBlockLen) % UB_BLOCK_SIZE != 0) {
            yRowByRow = true;
        }
    } else {
        outBlockLen = static_cast<uint32_t>(dataLenAB * sizeof(uint8_t));
        yRowPitch = static_cast<uint32_t>(rowWidth * sizeof(uint8_t));
        srcStride = static_cast<uint32_t>((yRowPitch - outBlockLen) / UB_BLOCK_SIZE);
        dstStride = static_cast<uint32_t>((dimGradX_ - dataLenAB) * sizeof(uint8_t));
        gradBOffset = dataLenAB;
        if ((yRowPitch - outBlockLen) % UB_BLOCK_SIZE != 0) {
            yRowByRow = true;
        }
    }

    uint32_t scale1OutLen = ops::CeilDiv(dataLenAB, BLOCK_SIZE);
    DataCopyExtParams scale1CopyOutParams = { outBurst, static_cast<uint32_t>(scale1OutLen * sizeof(uint8_t)),
        static_cast<uint32_t>(0),
        static_cast<uint32_t>(ops::CeilAlign(dimGradX_, DOUBLE_BLOCK_SIZE) / BLOCK_SIZE - scale1OutLen),
        static_cast<uint32_t>(0) };

    uint32_t scale2BlockLen = dataLenAB * DIGIT_TWO * sizeof(uint8_t);
    uint32_t scale2GroupSize = scale2BlockLen * DIGIT_TWO;
    uint32_t scale2SrcStride = (scale2GroupSize - scale2BlockLen) / UB_BLOCK_SIZE;
    uint32_t scale2DstStride = (dimGradX_ - dataLenAB * DIGIT_TWO) * sizeof(uint8_t);

    DataCopyExtParams scale2CopyOutParamsA = { static_cast<uint16_t>(blockCountAlign / DOUBLE_BLOCK_SIZE),
        scale2BlockLen, scale2SrcStride, scale2DstStride, static_cast<uint32_t>(0) };

    LocalTensor<uint8_t> y1Local = outQueue1_.template DeQue<uint8_t>();
    if (yRowByRow) {
        int64_t gmRowStride = dimGradX_;
        if constexpr (IsSameType<y1Dtype, fp4x2_e2m1_t>::value || IsSameType<y1Dtype, fp4x2_e1m2_t>::value) {
            gmRowStride = dimGradX_ / DIGIT_TWO;
        }
        DataCopyExtParams rowParams = { 1, outBlockLen, 0, 0, 0 };
        for (int64_t row = 0; row < blockCount; row++) {
            DataCopyPad(yGm1_[yOffsetA + row * gmRowStride], y1Local[row * yRowPitch], rowParams);
            DataCopyPad(yGm1_[yOffsetB + row * gmRowStride], y1Local[row * yRowPitch + gradBOffset], rowParams);
        }
    } else {
        DataCopyExtParams yCopyOutParams = { outBurst, outBlockLen, srcStride, dstStride, 0 };
        DataCopyPad(yGm1_[yOffsetA], y1Local, yCopyOutParams);
        DataCopyPad(yGm1_[yOffsetB], y1Local[gradBOffset], yCopyOutParams);
    }
    outQueue1_.FreeTensor(y1Local);

    LocalTensor<uint8_t> y2Local = outQueue2_.template DeQue<uint8_t>();
    if (yRowByRow) {
        int64_t gmRowStride = dimGradX_;
        if constexpr (IsSameType<y1Dtype, fp4x2_e2m1_t>::value || IsSameType<y1Dtype, fp4x2_e1m2_t>::value) {
            gmRowStride = dimGradX_ / DIGIT_TWO;
        }
        DataCopyExtParams rowParams = { 1, outBlockLen, 0, 0, 0 };
        for (int64_t row = 0; row < blockCount; row++) {
            DataCopyPad(yGm2_[yOffsetA + row * gmRowStride], y2Local[row * yRowPitch], rowParams);
            DataCopyPad(yGm2_[yOffsetB + row * gmRowStride], y2Local[row * yRowPitch + gradBOffset], rowParams);
        }
    } else {
        DataCopyExtParams yCopyOutParams = { outBurst, outBlockLen, srcStride, dstStride, 0 };
        DataCopyPad(yGm2_[yOffsetA], y2Local, yCopyOutParams);
        DataCopyPad(yGm2_[yOffsetB], y2Local[gradBOffset], yCopyOutParams);
    }
    outQueue2_.FreeTensor(y2Local);

    LocalTensor<uint8_t> mxScale1Local = mxScaleAQueue1_.template DeQue<uint8_t>();
    LocalTensor<uint8_t> mxScaleB1Local = mxScaleBQueue1_.template DeQue<uint8_t>();
    DataCopyPad(mxScaleGm1_[scale1OutOffset], mxScale1Local, scale1CopyOutParams);
    DataCopyPad(mxScaleGm1_[scale1OutOffset + dimN_ / BLOCK_SIZE], mxScaleB1Local, scale1CopyOutParams);
    mxScaleAQueue1_.FreeTensor(mxScale1Local);
    mxScaleBQueue1_.FreeTensor(mxScaleB1Local);

    LocalTensor<uint8_t> mxScale2Local = mxScaleQueue2_.template DeQue<uint8_t>();
    DataCopyPad(mxScaleGm2_[scale2OutOffsetA], mxScale2Local, scale2CopyOutParamsA);
    DataCopyPad(mxScaleGm2_[scale2OutOffsetB], mxScale2Local[scale2BlockLen], scale2CopyOutParamsA);
    
    mxScaleQueue2_.FreeTensor(mxScale2Local);
}
} // namespace SwigluBackwardMxQuantWithDualAxis

#endif // OPS_NN_SWIGLU_BACKWARD_MX_QUANT_WITH_DUAL_AXIS_REGBASE_H
