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
 * \file swiglu_mx_quant_last_not_last.h
 * \brief Regbase implementation for SwiGLU + MX quantization (activate_dim=-1, axis=-2)
 * No group_index path. 3D view [inputDim0, inputDim1, inputDim2].
 * SwiGLU on axis=-1, MX quant on axis=-2.
 * Reference: swiglu_mx_quant_with_dual_axis / dynamic_mx_quant_with_dual_axis
 */

#ifndef SWIGLU_MX_QUANT_AXIS_NOT_LAST_H
#define SWIGLU_MX_QUANT_AXIS_NOT_LAST_H

#include "swiglu_mx_quant_common.h"
#include "kernel_operator.h"
#include "op_kernel/math_util.h"
#include "op_kernel/platform_util.h"
#include "kernel_tiling/kernel_tiling.h"

namespace SwigluMxQuant {
using namespace AscendC;

template <typename T, typename U, typename T_IDX, bool isGroupIndex, RoundMode roundMode, bool isLast>
class SwigluMxQuantAxisNotLast {
public:
    __aicore__ inline SwigluMxQuantAxisNotLast(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR group_index, GM_ADDR y, GM_ADDR mxscale, GM_ADDR workspace,
                                const SwigluMxQuantTilingData* __restrict tilingData, AscendC::TPipe* pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void Compute(int64_t loopPerCoreG, int64_t blockOffsetG, int64_t dimMTailG, int64_t dimMBlockNum,
                                   int64_t groupStart, int64_t scaleGmOffsetGroup);
    __aicore__ inline void CopyIn(int64_t absRowStart, int64_t colOffset, int64_t calcRow, int64_t calcCol);
    __aicore__ inline void ComputeSwigluV2(int64_t calcCol, int64_t calcRow, __ubuf__ T* actAddr, __ubuf__ T* gateAddr,
                                           __ubuf__ T* swigluAddr);
    __aicore__ inline void CopyOut(int64_t yOffset, int64_t scaleOutOffset, int64_t blockCount, int64_t blockCountAlign,
                                   int64_t dataLen, int64_t dataLenAlign);

private:
    GlobalTensor<T> xGm_;
    GlobalTensor<uint8_t> yGm_;
    GlobalTensor<uint8_t> scaleGm_;
    GlobalTensor<T_IDX> groupIndexGm_;
    const SwigluMxQuantTilingData* tiling_;
    AscendC::TPipe* pipe_;
    int32_t blockIdx_ = 0;

    TQue<QuePosition::VECIN, 1> inQueue_;
    TQue<QuePosition::VECOUT, 1> outQueue_;
    TQue<QuePosition::VECOUT, 1> scaleQueue_;

    TBuf<QuePosition::VECCALC> swigluBuf_;
    TBuf<QuePosition::VECCALC> scaleRecipBuf_;

    // Tiling params
    int64_t realCoreNum_ = 0;
    int64_t dimN_ = 0;               // SwiGLU output columns = N
    int64_t dimM_ = 0;               // Quant axis dim = M
    int64_t dimNBlockNum_ = 0;       // CeilDiv(N, 256)
    int64_t dimMBlockNum_ = 0;       // CeilDiv(M, 64)
    int64_t blockCountPerBatch_ = 0; // M_blocks * N_blocks
    int64_t loopPerCore_ = 0;
    int64_t blockOffset_ = 0;
    // Group index params
    int64_t numGroups_ = 1;

    int64_t ubRowLen_ = QUANT_ONCE_NUM;
    int64_t ubRowLenTail_ = 0;
    int64_t ubRowCount_ = CONST_64;
    int64_t ubRowCountTail_ = 0;
    int64_t inHalfSize_ = 0;
    int64_t activateLeft_ = 0;

    float clampLimit_ = 0.0f;
    float gluAlpha_ = 0.0f;
    float gluBias_ = 0.0f;
    int64_t swigluMode_ = 0;

    int64_t scaleAlg_ = 0;
    uint16_t dtypeYMaxExp_ = 0;
    uint32_t invDtypeMax_ = 0;
    DataCopyExtParams copyParams_ = {0, 0, 0, 0, 0};
    DataCopyPadExtParams<T> padParams_ = {false, 0, 0, 0};
};

// ==================== Init ====================
template <typename T, typename U, typename T_IDX, bool isGroupIndex, RoundMode roundMode, bool isLast>
__aicore__ inline void SwigluMxQuantAxisNotLast<T, U, T_IDX, isGroupIndex, roundMode, isLast>::Init(
    GM_ADDR x, GM_ADDR group_index, GM_ADDR y, GM_ADDR mxscale, GM_ADDR workspace,
    const SwigluMxQuantTilingData* __restrict tilingData, AscendC::TPipe* pipe)
{
#if (__NPU_ARCH__ == 3510)
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
#endif
    tiling_ = tilingData;
    pipe_ = pipe;
    blockIdx_ = GetBlockIdx();

    int64_t inputDim0 = tiling_->inputDim0; // -3轴的值 在groupIndex存在时，它必然为1
    dimN_ = tiling_->inputDim2;             // SwiGLU output = N
    dimM_ = tiling_->inputDim1;
    dimNBlockNum_ = tiling_->dimNBlockNum; // N方向多少个256
    dimMBlockNum_ = tiling_->dimMBlockNum; // M方向多少个64
    blockCountPerBatch_ = 0;               // dimMBlockNum * dimNBlockNum M方向块数 * N方向块数
    realCoreNum_ = tiling_->usedCoreNum; // 实际运行总核数 groupIndex时kernel侧动态分核，不存在时tiling侧计算
    activateLeft_ = tiling_->activateLeft;
    if constexpr (isLast) {
        clampLimit_ = tiling_->clampLimit;
        gluAlpha_ = tiling_->gluAlpha;
        gluBias_ = tiling_->gluBias;
        swigluMode_ = tiling_->swigluMode;
    }
    scaleAlg_ = tiling_->scaleAlg;
    ubRowLenTail_ = tiling_->dimNTail;   // N方向的尾块
    ubRowCountTail_ = tiling_->dimMTail; // M方向尾块
    if constexpr (isGroupIndex) {
        numGroups_ = tiling_->groupIndexNum;
        groupIndexGm_.SetGlobalBuffer((__gm__ T_IDX*)group_index);
    } else {
        blockCountPerBatch_ = tiling_->blockCountPerBatch;
        //  int64_t totalBlocks = inputDim0 * blockCountPerBatch_; // 所有的块数
        int64_t loopPerCoreNow = tiling_->tailCoreBasicNumDim1; // totalBlocks / usedCoreNum_;
        int64_t headCoreNum = tiling_->frontCoreNum;            // totalBlocks % usedCoreNum_;
        if (blockIdx_ < headCoreNum) {
            loopPerCore_ = loopPerCoreNow + 1;
        } else {
            loopPerCore_ = loopPerCoreNow;
        }
        if (blockIdx_ < headCoreNum) {
            blockOffset_ = blockIdx_ * loopPerCore_;
        } else {
            blockOffset_ = headCoreNum * (loopPerCoreNow + 1) + (blockIdx_ - headCoreNum) * loopPerCoreNow;
        }
    }
    xGm_.SetGlobalBuffer((__gm__ T*)x);
    yGm_.SetGlobalBuffer((__gm__ uint8_t*)y);
    scaleGm_.SetGlobalBuffer((__gm__ uint8_t*)mxscale);

    inHalfSize_ = ubRowLen_ * ubRowCount_; // 256 * 64
    int64_t inBufferSize = inHalfSize_ * sizeof(T);

    // Allocate UB: SwiGLU needs double x input (gate+hidden) with double buffer
    pipe_->InitBuffer(inQueue_, 2, inBufferSize * CONST_2);
    pipe_->InitBuffer(swigluBuf_, inBufferSize);

    pipe_->InitBuffer(outQueue_, 2, inHalfSize_);

    // Scale buffer: axis=-2 produces 2 rows per 32-row group (even/odd interleaved)
    int64_t scalePerBlock = ubRowLen_ * 3 * sizeof(int8_t); // 256 * 2 bytes per block
    int64_t scaleBufSize = ((scalePerBlock + ONE_BLOCK_UB - 1) / ONE_BLOCK_UB) * ONE_BLOCK_UB;
    pipe_->InitBuffer(scaleQueue_, 2, scaleBufSize);

    // Reciprocal buffer: FP16 reciprocal for both even and odd halves
    int64_t recipPerBlock = ubRowLen_ * 2 * sizeof(uint16_t);
    int64_t recipBufSize = ((recipPerBlock + ONE_BLOCK_UB - 1) / ONE_BLOCK_UB) * ONE_BLOCK_UB;
    pipe_->InitBuffer(scaleRecipBuf_, recipBufSize);

    if constexpr (IsSame<U, fp4x2_e2m1_t>::value) {
        dtypeYMaxExp_ = FP4_E2M1_BF16_MAX_EXP;
    }
    if constexpr (IsSame<U, fp8_e4m3fn_t>::value) {
        dtypeYMaxExp_ = FP8_E4M3_MAX_EXP;
        invDtypeMax_ = FP8_E4M3_MAX;
    }
    if constexpr (IsSame<U, fp8_e5m2_t>::value) {
        dtypeYMaxExp_ = FP8_E5M2_MAX_EXP;
        invDtypeMax_ = FP8_E5M2_MAX;
    }
}

// ==================== Process ====================
template <typename T, typename U, typename T_IDX, bool isGroupIndex, RoundMode roundMode, bool isLast>
__aicore__ inline void SwigluMxQuantAxisNotLast<T, U, T_IDX, isGroupIndex, roundMode, isLast>::Process()
{
    if (blockIdx_ >= realCoreNum_)
        return;
    [[maybe_unused]] int64_t coreRotateOffset = 0; // 每个group从哪个物理核开始用

    // group_index exists: count mode, kernel-side cumsum + ROTATE core distribution group_index not exists: numGroups_
    // = 1

    [[maybe_unused]] int64_t cumuRowStart = 0;    // cumulative row start (count mode cumsum)
    [[maybe_unused]] int64_t cumuScaleOffset = 0; // cumulative CeilDiv(groupIndex[k], 64) for scale offset

    for (int64_t g = 0; g < numGroups_; g++) {
        // count mode: read groupIndexGm_[g] = rows in this group
        int64_t groupRows = 0;
        int64_t loopPerCoreG = 0;
        int64_t blockOffsetG = 0;
        int64_t dimMTailG = 0;
        int64_t dimMBlockNum = 0;
        [[maybe_unused]] int64_t groupStart = 0;
        [[maybe_unused]] int64_t scaleGmOffsetGroup = 0;
        if constexpr (isGroupIndex) {
            groupRows = groupIndexGm_.GetValue(g);
            if (groupRows <= 0)
                continue;
            groupStart = cumuRowStart;
            cumuRowStart += groupRows;

            // Scale offset: cumuCeilDiv(groupRows, 64) from previous groups
            scaleGmOffsetGroup = (cumuScaleOffset * dimN_ * 2);
            dimMBlockNum = Ops::Base::CeilDiv(groupRows, CONST_64);
            cumuScaleOffset += dimMBlockNum; // (groupStart / DOUBLE_BLOCK_SIZE + g) * DIGIT_TWO;

            // Per-group block count
            int64_t blockCountG = dimMBlockNum * dimNBlockNum_; // 每个group总共有多少块

            // ROTATE core distribution (same as dual_axis)
            int64_t usedCoreNumG = (blockCountG < realCoreNum_) ? blockCountG :
                                                                  realCoreNum_; // 当前group的行数要使用多少个核去计算
            int64_t myCoreInGroup = blockIdx_ - coreRotateOffset;

            if (myCoreInGroup < 0) {
                myCoreInGroup += realCoreNum_; // wrap around
            }
            if (myCoreInGroup < usedCoreNumG) {
                int64_t headCoreNumG = blockCountG % usedCoreNumG;
                int64_t blockPerHeadCoreG = Ops::Base::CeilDiv(blockCountG, usedCoreNumG);
                int64_t blockPerTailCoreG = blockCountG / usedCoreNumG;
                if (myCoreInGroup < headCoreNumG) {
                    loopPerCoreG = blockPerHeadCoreG;
                    blockOffsetG = myCoreInGroup * loopPerCoreG;
                } else {
                    loopPerCoreG = blockPerTailCoreG;
                    blockOffsetG = headCoreNumG * blockPerHeadCoreG + (myCoreInGroup - headCoreNumG) * loopPerCoreG;
                }
            }
            coreRotateOffset = (coreRotateOffset + blockCountG) % realCoreNum_;
            if (loopPerCoreG == 0) {
                continue;
            }
            dimMTailG = groupRows % CONST_64;
            if (dimMTailG == 0)
                dimMTailG = CONST_64;
        } else {
            loopPerCoreG = loopPerCore_;
            blockOffsetG = blockOffset_; // 处理第几块
            dimMTailG = ubRowCountTail_;
            dimMBlockNum = dimMBlockNum_;
        }
        Compute(loopPerCoreG, blockOffsetG, dimMTailG, dimMBlockNum, groupStart, scaleGmOffsetGroup);
    }
}

__aicore__ inline void ComputeInterleave(__ubuf__ uint8_t* dstAddr, __ubuf__ uint8_t* src0Addr,
                                         __ubuf__ uint8_t* src1Addr)
{
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<uint8_t> src0Reg;
        MicroAPI::RegTensor<uint8_t> src1Reg;
        MicroAPI::MaskReg maskB8 = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::DataCopy(src0Reg, src0Addr);
        MicroAPI::DataCopy(src1Reg, src1Addr);
        MicroAPI::DataCopy<uint8_t, MicroAPI::StoreDist::DIST_INTLV_B8>(dstAddr, src0Reg, src1Reg, maskB8);
    }
}

template <typename T, typename U, typename T_IDX, bool isGroupIndex, RoundMode roundMode, bool isLast>
__aicore__ inline void SwigluMxQuantAxisNotLast<T, U, T_IDX, isGroupIndex, roundMode, isLast>::Compute(
    int64_t loopPerCoreG, int64_t blockOffsetG, int64_t dimMTailG, int64_t dimMBlockNum, int64_t groupStart,
    int64_t scaleGmOffsetGroup)
{
    for (int64_t i = 0; i < loopPerCoreG; i++) {
        int64_t blockInGroup = blockOffsetG + i;
        [[maybe_unused]] int64_t batchIdx = 0;
        int64_t rowBlockIdx = 0;
        int64_t colBlockIdx = 0;
        int64_t absRowStart = 0;
        int64_t scaleGmOffset = 0;
        int64_t colOffset = 0;
        if constexpr (isGroupIndex) {
            rowBlockIdx = blockInGroup / dimNBlockNum_;
            colBlockIdx = blockInGroup % dimNBlockNum_;
            absRowStart = groupStart + rowBlockIdx * 64;
            colOffset = colBlockIdx * 256;
            scaleGmOffset = scaleGmOffsetGroup + rowBlockIdx * 2 * dimN_ + colOffset * 2;
        } else {
            batchIdx = blockInGroup / blockCountPerBatch_;
            int64_t innerIdx = blockInGroup % blockCountPerBatch_;
            rowBlockIdx = innerIdx / dimNBlockNum_; // M方向位置
            colBlockIdx = innerIdx % dimNBlockNum_; // N方向位置
            colOffset = colBlockIdx * 256;
            if constexpr (isLast) {
                absRowStart = batchIdx * dimM_ + rowBlockIdx * 64;
            } else {
                absRowStart = batchIdx * dimM_ * 2 + rowBlockIdx * 64;
            }
            scaleGmOffset = batchIdx * (dimMBlockNum * dimN_ * 2) + rowBlockIdx * 2 * dimN_ + colOffset * 2;
        }

        int64_t calcCol = (colBlockIdx == dimNBlockNum_ - 1) ? ubRowLenTail_ : ubRowLen_; // N方向处理数据量
        int64_t calcRow = (rowBlockIdx == dimMBlockNum - 1) ? dimMTailG : ubRowCount_;    // M方向处理数据量
        // 1. CopyIn + SwiGLU
        CopyIn(absRowStart, colOffset, calcRow, calcCol); // 把输入copy进ub

        LocalTensor<T> xLocal = inQueue_.DeQue<T>();

        auto actAddr = (__ubuf__ T*)xLocal.GetPhyAddr();
        auto gateAddr = (__ubuf__ T*)xLocal[inHalfSize_].GetPhyAddr();
        LocalTensor<T> swigluUb = swigluBuf_.Get<T>();
        auto swigluAddr = (__ubuf__ T*)swigluUb.GetPhyAddr();
        if constexpr (isLast) {
            if (swigluMode_ == 0 && activateLeft_ == 0) {
                actAddr = (__ubuf__ T*)xLocal[inHalfSize_].GetPhyAddr();
                gateAddr = (__ubuf__ T*)xLocal.GetPhyAddr();
            }
            if (swigluMode_ == 1) {
                actAddr = (__ubuf__ T*)xLocal.GetPhyAddr();
                gateAddr = (__ubuf__ T*)xLocal[64].GetPhyAddr();
            }
            if (swigluMode_ == 0) {
                ComputeSwigluV1Second<T>(calcCol, calcRow, actAddr, gateAddr, swigluAddr);
            } else {
                ComputeSwigluV2(calcCol, calcRow, actAddr, gateAddr, swigluAddr);
            }
        } else {
            if (activateLeft_ == 0) {
                actAddr = (__ubuf__ T*)xLocal[inHalfSize_].GetPhyAddr();
                gateAddr = (__ubuf__ T*)xLocal.GetPhyAddr();
            }
            ComputeSwigluV1Second<T>(calcCol, calcRow, actAddr, gateAddr, swigluAddr);
        }

        inQueue_.FreeTensor(xLocal);
        int64_t calcPadRowAlign = ((calcRow + 64 - 1) / 64) * 64;
        if (calcRow % 64 != 0) { // M方向如果不是64对齐，全补0，方便后续操作
            uint32_t calcPadRow = calcPadRowAlign - calcRow;
            uint32_t allNumZero = calcPadRow * ubRowLen_;
            auto swigluAddrPadZero = (__ubuf__ T*)swigluUb[calcRow * ubRowLen_].GetPhyAddr();
            SwigluMxQuant::PadZeroM<T>(swigluAddrPadZero, allNumZero);
        }
        // 2. Compute scale + cast in 32-row strips
        LocalTensor<uint8_t> mxScaleLocal = scaleQueue_.AllocTensor<uint8_t>();
        auto mxScaleAddr = (__ubuf__ uint8_t*)mxScaleLocal.GetPhyAddr();
        auto mxScaleRecipAddr = (__ubuf__ uint16_t*)scaleRecipBuf_.Get<uint16_t>().GetPhyAddr();
        LocalTensor<uint8_t> yLocal = outQueue_.AllocTensor<uint8_t>();
        auto yAddr = (__ubuf__ uint8_t*)yLocal.GetPhyAddr();
        int64_t calcBlockLoop = calcPadRowAlign / 32;
        for (int64_t blk = 0; blk < calcBlockLoop; blk++) { // 一次处理64行，所以这里固定是2，vf内部处理32行
            auto sAddr = swigluAddr + blk * 32 * ubRowLen_;
            auto mAddr = mxScaleAddr + blk * ubRowLen_;
            auto rAddr = mxScaleRecipAddr + blk * ubRowLen_;
            if (scaleAlg_ == 0) {
                ComputeScaleOcpNotLastAxis<T, U, roundMode>(32, sAddr, mAddr, rAddr, dtypeYMaxExp_);
            } else {
                ComputeScaleCuBLASNotLastAxis<T, U, roundMode>(32, sAddr, mAddr, rAddr, invDtypeMax_);
            }

            if constexpr (IsSame<U, fp8_e4m3fn_t>::value || IsSame<U, fp8_e5m2_t>::value) {
                ComputeCastToFP8NotLastAxis<T, U, roundMode>(ubRowLen_, 32, sAddr, rAddr, yAddr + blk * 32 * ubRowLen_);
                ComputeCastToFP8NotLastAxis<T, U, roundMode>(ubRowLen_, 32, sAddr + 128, rAddr + 128,
                                                             yAddr + blk * 32 * ubRowLen_ + 128);
            } else {
                if constexpr (IsSameType<T, half>::value) {
                    ComputeYFP16ToFP4NotLastAxis<T, U, roundMode>(ubRowLen_, 32, sAddr, rAddr,
                                                                  yAddr + (blk * 32 * ubRowLen_ / 2));
                } else {
                    ComputeYBF16ToFP4NotLastAxis<T, U, roundMode>(ubRowLen_, 32, sAddr, rAddr,
                                                                  yAddr + (blk * 32 * ubRowLen_ / 2));
                }
            }
        }
        // Scale2 interleave
        for (int64_t blk = 1; blk < calcBlockLoop; blk += 2) {
            auto src0Addr = (__ubuf__ uint8_t*)mxScaleLocal[(blk - 1) * ubRowLen_].GetPhyAddr();
            auto src1Addr = (__ubuf__ uint8_t*)mxScaleLocal[blk * ubRowLen_].GetPhyAddr();
            auto dstAddr = (__ubuf__ uint8_t*)mxScaleLocal[(blk - 1) * ubRowLen_].GetPhyAddr();
            ComputeInterleave(dstAddr, src0Addr, src1Addr);
        }
        scaleQueue_.EnQue(mxScaleLocal);
        outQueue_.EnQue(yLocal);

        // 3. CopyOut
        int64_t yGmOffset = 0;
        if constexpr (isLast) {
            yGmOffset = absRowStart * dimN_ + colOffset;
        } else {
            if constexpr (isGroupIndex) {
                yGmOffset = absRowStart * dimN_ + colOffset;
            } else {
                yGmOffset = (batchIdx * dimM_ + rowBlockIdx * 64) * dimN_ + colOffset;
            }
        }
        CopyOut(yGmOffset, scaleGmOffset, calcRow, calcPadRowAlign, calcCol, ubRowLen_);
    }
}
// ==================== CopyIn (from dual_axis) ====================
template <typename T, typename U, typename T_IDX, bool isGroupIndex, RoundMode roundMode, bool isLast>
__aicore__ inline void SwigluMxQuantAxisNotLast<T, U, T_IDX, isGroupIndex, roundMode, isLast>::CopyIn(
    int64_t absRowStart, int64_t colOffset, int64_t calcRow, int64_t calcCol)
{
    LocalTensor<T> xLocal = inQueue_.AllocTensor<T>();
    if constexpr (isLast) {
        if (swigluMode_ == 0) {
            copyParams_.blockCount = calcRow;
            copyParams_.blockLen = calcCol * sizeof(T);
            copyParams_.srcStride = (dimN_ * 2 - calcCol) * sizeof(T);
            copyParams_.dstStride = 0;
            int64_t leftGmOffset = absRowStart * dimN_ * 2 + colOffset;
            // Left half (gate/act): x[absRowStart..absRowStart+calcRow][colOffset..colOffset+calcCol]

            DataCopyPad(xLocal, xGm_[leftGmOffset], copyParams_, padParams_);

            // Right half (hidden): x[absRowStart..absRowStart+calcRow][dimN_+colOffset..dimN_+colOffset+calcCol]
            int64_t rightGmOffset = leftGmOffset + dimN_;
            DataCopyPad(xLocal[inHalfSize_], xGm_[rightGmOffset], copyParams_, padParams_);
        } else {
            int64_t offset = absRowStart * dimN_ * 2 + colOffset * 2;
            copyParams_.blockCount = calcRow;
            copyParams_.blockLen = calcCol * 2 * sizeof(T);
            copyParams_.srcStride = (dimN_ * 2 - calcCol * 2) * sizeof(T);
            copyParams_.dstStride = 0;
            DataCopyPad(xLocal, xGm_[offset], copyParams_, padParams_);
        }
    } else {
        copyParams_.blockCount = calcRow;
        copyParams_.blockLen = calcCol * sizeof(T);
        copyParams_.srcStride = (dimN_ - calcCol) * sizeof(T);
        copyParams_.dstStride = 0;
        int64_t actOff = absRowStart * dimN_ + colOffset;
        DataCopyPad(xLocal, xGm_[actOff], copyParams_, padParams_);
        int64_t gateOff = actOff + dimM_ * dimN_;
        DataCopyPad(xLocal[inHalfSize_], xGm_[gateOff], copyParams_, padParams_);
    }
    inQueue_.EnQue(xLocal);
}

template <typename T, typename U, typename T_IDX, bool isGroupIndex, RoundMode roundMode, bool isLast>
__aicore__ inline void SwigluMxQuantAxisNotLast<T, U, T_IDX, isGroupIndex, roundMode, isLast>::ComputeSwigluV2(
    int64_t calcCol, int64_t calcRow, __ubuf__ T* actAddr, __ubuf__ T* gateAddr, __ubuf__ T* swigluAddr)
{
    int32_t once2nNum = calcCol * 2;
    int32_t onceIn2n = ((once2nNum + ONE_BLOCK_NUM - 1) / ONE_BLOCK_NUM) * ONE_BLOCK_NUM;
    uint16_t dim0VfTimes = static_cast<uint16_t>(calcRow);
    uint16_t dim1VfTimes = CeilDivision(once2nNum, 128);
    int32_t ubOutRow = QUANT_ONCE_NUM;

    float clampLimit = clampLimit_;
    float negClampLimit = -clampLimit_;
    float negAplha = -gluAlpha_;
    float scalarOne = 1.0f;
    float gluBias = gluBias_;

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<T> vregX1;
        AscendC::MicroAPI::RegTensor<T> vregX2;
        AscendC::MicroAPI::RegTensor<float> vregX1F;
        AscendC::MicroAPI::RegTensor<float> vregX2F;
        AscendC::MicroAPI::RegTensor<float> vregX1DeF;
        AscendC::MicroAPI::RegTensor<float> vregX2DeF;
        AscendC::MicroAPI::RegTensor<float> minsReg;
        AscendC::MicroAPI::RegTensor<float> mulsReg;
        AscendC::MicroAPI::RegTensor<float> expReg;
        AscendC::MicroAPI::RegTensor<float> addsReg;
        AscendC::MicroAPI::RegTensor<float> sigmoidReg;
        AscendC::MicroAPI::RegTensor<float> outFReg;
        AscendC::MicroAPI::RegTensor<T> outTReg;
        AscendC::MicroAPI::MaskReg mask = AscendC::MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
        for (uint16_t dim0vfLoopIdx = 0; dim0vfLoopIdx < dim0VfTimes; dim0vfLoopIdx++) {
            for (uint16_t dim1vfLoopIdx = 0; dim1vfLoopIdx < dim1VfTimes; dim1vfLoopIdx++) {
                AscendC::MicroAPI::AddrReg srcIdxOffset = AscendC::MicroAPI::CreateAddrReg<T>(dim0vfLoopIdx, onceIn2n,
                                                                                              dim1vfLoopIdx, 128);
                AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(vregX1, actAddr,
                                                                                             srcIdxOffset);
                AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(vregX2, gateAddr,
                                                                                             srcIdxOffset);
                AscendC::MicroAPI::Cast<float, T, CAST_ZERO>(vregX1F, vregX1, mask);
                AscendC::MicroAPI::Cast<float, T, CAST_ZERO>(vregX2F, vregX2, mask);

                AscendC::MicroAPI::DeInterleave(vregX1DeF, vregX2DeF, vregX1F, vregX2F);
                AscendC::MicroAPI::Mins(minsReg, vregX1DeF, clampLimit, mask);
                AscendC::MicroAPI::Muls(mulsReg, minsReg, negAplha, mask);
                AscendC::MicroAPI::Exp(expReg, mulsReg, mask);
                AscendC::MicroAPI::Adds(addsReg, expReg, scalarOne, mask);
                AscendC::MicroAPI::Div(sigmoidReg, minsReg, addsReg, mask);

                AscendC::MicroAPI::Mins(vregX2DeF, vregX2DeF, clampLimit, mask);
                AscendC::MicroAPI::Maxs(vregX2DeF, vregX2DeF, negClampLimit, mask);
                AscendC::MicroAPI::Adds(vregX2DeF, vregX2DeF, gluBias, mask);

                AscendC::MicroAPI::Mul(outFReg, sigmoidReg, vregX2DeF, mask);

                AscendC::MicroAPI::Cast<T, float, CAST_FP32_TO_FP16_BF16>(outTReg, outFReg, mask);
                AscendC::MicroAPI::AddrReg outOffset = AscendC::MicroAPI::CreateAddrReg<T>(dim0vfLoopIdx, ubOutRow,
                                                                                           dim1vfLoopIdx, 64);
                DataCopy<T, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(swigluAddr, outTReg, outOffset, mask);
            }
        }
    }
}

// ==================== CopyOut (from dual_axis) ====================
template <typename T, typename U, typename T_IDX, bool isGroupIndex, RoundMode roundMode, bool isLast>
__aicore__ inline void SwigluMxQuantAxisNotLast<T, U, T_IDX, isGroupIndex, roundMode, isLast>::CopyOut(
    int64_t yOffset, int64_t scaleOutOffset, int64_t blockCount, int64_t blockCountAlign, int64_t dataLen,
    int64_t dataLenAlign)
{
    LocalTensor<uint8_t> mxScaleLocal = scaleQueue_.DeQue<uint8_t>();
    LocalTensor<uint8_t> outLocal = outQueue_.DeQue<uint8_t>();

    if constexpr (IsSame<U, fp4x2_e2m1_t>::value || IsSame<U, fp4x2_e1m2_t>::value) {
        copyParams_.blockCount = blockCount;
        copyParams_.blockLen = dataLen / 2;
        copyParams_.srcStride = ((dataLenAlign - dataLen) / 2) / 32;
        copyParams_.dstStride = (dimN_ - dataLen) / 2;
        yOffset = yOffset / 2;
    } else {
        copyParams_.blockCount = blockCount;
        copyParams_.blockLen = dataLen;
        copyParams_.srcStride = (dataLenAlign - dataLen) / 32;
        copyParams_.dstStride = dimN_ - dataLen;
    }
    DataCopyPad<uint8_t>(yGm_[yOffset], outLocal, copyParams_);
    uint32_t scaleSrcStride = 2 * ops::CeilDiv(dataLen, int64_t(32)) - ops::CeilDiv(2 * dataLen, int64_t(32));
    DataCopyExtParams scaleCopyOutParams = {0, 0, 0, 0, 0};
    scaleCopyOutParams.blockCount = blockCountAlign / 64;
    scaleCopyOutParams.blockLen = dataLen * 2;
    scaleCopyOutParams.srcStride = scaleSrcStride;
    scaleCopyOutParams.dstStride = (dimN_ - dataLen) * 2;
    DataCopyPad(scaleGm_[scaleOutOffset], mxScaleLocal, scaleCopyOutParams);
    outQueue_.FreeTensor(outLocal);
    scaleQueue_.FreeTensor(mxScaleLocal);
}
} // namespace SwigluMxQuant

#endif // SWIGLU_MX_QUANT_AXIS_NOT_LAST_H
