/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dynamic_mx_quant_tail_axis.h
 * \brief
 */

#ifndef DYNAMIC_MX_QUANT_TAIL_AXIS_H
#define DYNAMIC_MX_QUANT_TAIL_AXIS_H
#define FLOAT_OVERFLOW_MODE_CTRL 60
#include "kernel_operator.h"
#include "op_kernel/platform_util.h"

#include "dynamic_mx_quant_common.h"
namespace DynamicMxQuant {
using namespace AscendC;

struct LoopParams {
    int64_t runningBlockFactorDim0{0};
    int64_t runningBlockFactorDim1{0};
    int64_t runningUbFactorDim0{0};
    int64_t runningUbFactorDim1{0};
    int64_t runningTailUbFactorDim0{0};
    int64_t runningTailUbFactorDim1{0};
    int64_t isPad{0};
};

template <typename T, typename U>
class DynamicMxQuantTailAxis {
public:
    __aicore__ inline DynamicMxQuantTailAxis()
    {}
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR y, GM_ADDR mxScale, GM_ADDR workspace, const DynamicMxQuantTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(
        int64_t dim0LoopIdx, int64_t dim1LoopIdx, int64_t usedFactorDim0, int64_t usedFactorDim1, int64_t isContainPad);
    __aicore__ inline void CopyOut(
        int64_t dim0LoopIdx, int64_t dim1LoopIdx, int64_t usedFactorDim0, int64_t usedFactorDim1, int64_t isContainPad);
    template <AscendC::RoundMode toBf16RoundMode, AscendC::RoundMode roundMode>
    __aicore__ inline void Compute(int64_t realUbFactorDim0, int64_t realUbFactorDim1, int64_t isContainPad);
    __aicore__ inline void ProcessDim1NotSplit();
    __aicore__ inline void ProcessDim1SplitNormalCore();
    __aicore__ inline void ProcessDim1SplitTailCore();
    __aicore__ inline void ProcessDim1Split();
    template <AscendC::RoundMode toBf16RoundMode, AscendC::RoundMode roundMode>
    __aicore__ inline void CommonFlowProcess(const LoopParams& runningParams);
    __aicore__ inline void ComputeMaxExp(
        __ubuf__ T* srcAddr, __ubuf__ uint16_t* maxExpAddr, uint32_t totalCountInUB, uint16_t loopNum);

    __aicore__ inline void ComputeScale(
        __ubuf__ uint16_t* maxExpAddr, __ubuf__ uint16_t* mxScaleLocalAddr, __ubuf__ uint16_t* halfScaleLocalAddr,
        uint32_t totalScaleInUB, uint16_t loopNumScale);
    template <AscendC::RoundMode toBf16RoundMode, AscendC::RoundMode roundMode>
    __aicore__ inline void ComputeData(
        __ubuf__ T* srcAddr, __ubuf__ uint16_t* halfScaleLocalAddr, __ubuf__ int8_t* outLocalAddr,
        uint32_t totalCountInUB, uint16_t loopNum);
    __aicore__ inline void DeletePadData(
        __ubuf__ int8_t* outLocalAddr, __ubuf__ int8_t* outBufferLocalAddr, uint16_t loopNum,
        uint32_t inputUpdateStride, uint32_t outputUpdateStride);
    __aicore__ inline void FP16Convert(
        AscendC::MicroAPI::RegTensor<half>& output, AscendC::MicroAPI::RegTensor<half>& input,
        AscendC::MicroAPI::MaskReg& mask);
    __aicore__ inline void ParseTilingData(const DynamicMxQuantTilingData* tilingData);

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, DB_BUFFER> inQueue_;
    GlobalTensor<T> xGm_;
    TQue<QuePosition::VECOUT, DB_BUFFER> outQueue_;
    GlobalTensor<uint8_t> yGm_;
    TQue<QuePosition::VECOUT, DB_BUFFER> mxScaleQueue_;
    GlobalTensor<uint8_t> mxScaleGm_;
    GlobalTensor<uint8_t> workspaceGm_;

    TBuf<QuePosition::VECCALC> maxExpBuffer_;
    TBuf<QuePosition::VECCALC> maxhalfScaleBuffer_;
    TBuf<QuePosition::VECCALC> outBuffer_;

    int64_t roundMode_ = 0;
    int64_t blockSize_ = 0;
    int64_t dstType_ = 0;

    int64_t ubFactorDim0_ = 0; // ub最大处理的行数
    int64_t ubFactorDim1_ = 0; // ub最大处理的列方向BLOCK数

    int64_t blockFactorDim0_ = 0; // 每个核的dim0方向循环次数
    int64_t blockFactorDim1_ = 0; // 每个核的dim1列方向循环次数

    int64_t blockFactorDim0Tail_ = 0; // 行方向尾部核数循环次数
    int64_t ubFactorDim0Tail_ = 0;    // 行方向尾核尾循环行数

    int64_t blockIdx_ = 0;
    int64_t totalCoreNum_ = 0;
    int64_t usedCoreNum_ = 0;

    int64_t coreDim0RowCount_ = 0;         // 每个核处理总行数
    int64_t tailCoreDim1NotSplitDim1_ = 0; // 列方向尾块block数

    int64_t currentBlockFactorDim0_ = 0;

    int64_t tailDim0StartCoreIdx_ = 0;
    int64_t inputRowBytes_ = 0; // 每行的字节数，用于计算srcStride
    int64_t outputDataRowBytes_ = 0;
    int64_t outputScaleRowBytes_ = 0;

    int64_t isSplitDim1_ = 0; // 分核是否切分1轴

    int64_t blockFactorDim1ForSplitDim1_ = 0;           // 切分dim1后非尾核循环次数
    int64_t tailBlockFactorDim1ForSplitDim1_ = 0;       // 切分dim1后尾核循环次数
    int64_t blockCountDim1EachCoreForSplitDim1_ = 0;    // 切分dim1后非尾核处理的总block数
    int64_t tailLoopBlockDim1ForSplitDim1_ = 0;         // 切分dim1后非尾核尾循环block数
    int64_t tailLoopBlockTailCoreDim1ForSplitDim1_ = 0; // 切分dim1后尾核尾循环block数

    int64_t cutNumberForDim1_ = 0; // Dim1切成的份数
    int64_t inputShapeDim1_ = 0;   // 尾轴shape大小
    uint16_t f4Emax_ = 0;
    uint32_t vlForHalfNumber_ = 0;
    uint16_t elementAfterReduce_ = 0;
    int64_t isPad_ = 0;
    int64_t tailDataCountNotAlign_ = 0;
    int64_t oneBlockCount_ = 0;
    int64_t UBBlockSize_ = 0;
};

template <typename T, typename U>
__aicore__ inline void DynamicMxQuantTailAxis<T, U>::Init(
    GM_ADDR x, GM_ADDR y, GM_ADDR mxScale, GM_ADDR workspace, const DynamicMxQuantTilingData* tilingData)
{
#if (__NPU_ARCH__ == 3101)
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
#endif
    ParseTilingData(tilingData);
    inputRowBytes_ = inputShapeDim1_ * sizeof(T);
    outputDataRowBytes_ = inputShapeDim1_ * sizeof(uint8_t) / DIGIT_TWO;
    outputScaleRowBytes_ = (inputShapeDim1_ + blockSize_ - 1) / blockSize_;
    coreDim0RowCount_ = ubFactorDim0_ * blockFactorDim0_;

    vlForHalfNumber_ = Ops::Base::GetVRegSize() / sizeof(T);
    UBBlockSize_ = Ops::Base::GetUbBlockSize();
    elementAfterReduce_ = Ops::Base::GetVRegSize() / UBBlockSize_;
    oneBlockCount_ = UBBlockSize_ / sizeof(T);

    pipe_.InitBuffer(
        inQueue_, DB_BUFFER,
        (ubFactorDim0_ * ubFactorDim1_ * blockSize_ * sizeof(T) + UBBlockSize_ - 1) / UBBlockSize_ * UBBlockSize_);
    pipe_.InitBuffer(
        outQueue_, DB_BUFFER,
        (ubFactorDim0_ * ubFactorDim1_ * blockSize_ / DIGIT_TWO + vlForHalfNumber_ - 1) / vlForHalfNumber_ * vlForHalfNumber_);
    pipe_.InitBuffer(
        mxScaleQueue_, DB_BUFFER, (ubFactorDim0_ * ubFactorDim1_ + UBBlockSize_ - 1) / UBBlockSize_ * UBBlockSize_);

    pipe_.InitBuffer(
        maxExpBuffer_, (ubFactorDim0_ * ubFactorDim1_ * sizeof(T) + UBBlockSize_ - 1) / UBBlockSize_ * UBBlockSize_);
    pipe_.InitBuffer(
        maxhalfScaleBuffer_,
        (ubFactorDim0_ * ubFactorDim1_ * sizeof(T) + UBBlockSize_ - 1) / UBBlockSize_ * UBBlockSize_);
    pipe_.InitBuffer(
        outBuffer_,
        (ubFactorDim0_ * ubFactorDim1_ * blockSize_ / DIGIT_TWO + vlForHalfNumber_ - 1) / vlForHalfNumber_ * vlForHalfNumber_);

    blockIdx_ = GetBlockIdx();

    xGm_.SetGlobalBuffer((__gm__ T*)x + blockIdx_ * coreDim0RowCount_ * (inputRowBytes_ / sizeof(T)));
    yGm_.SetGlobalBuffer((__gm__ uint8_t*)y + blockIdx_ * coreDim0RowCount_ * outputDataRowBytes_);
    mxScaleGm_.SetGlobalBuffer((__gm__ uint8_t*)mxScale + blockIdx_ * coreDim0RowCount_ * outputScaleRowBytes_);
    workspaceGm_.SetGlobalBuffer((__gm__ uint8_t*)workspace + blockIdx_ * coreDim0RowCount_ * outputScaleRowBytes_);

    if (isSplitDim1_) {
        xGm_.SetGlobalBuffer(
            (__gm__ T*)x + (blockIdx_ / cutNumberForDim1_) * coreDim0RowCount_ * (inputRowBytes_ / sizeof(T)) +
            (blockIdx_ % cutNumberForDim1_) * blockCountDim1EachCoreForSplitDim1_ * blockSize_);

        yGm_.SetGlobalBuffer(
            (__gm__ uint8_t*)y + (blockIdx_ / cutNumberForDim1_) * coreDim0RowCount_ * outputDataRowBytes_ +
            (blockIdx_ % cutNumberForDim1_) * blockCountDim1EachCoreForSplitDim1_ * blockSize_ / DIGIT_TWO);

        mxScaleGm_.SetGlobalBuffer(
            (__gm__ uint8_t*)mxScale + (blockIdx_ / cutNumberForDim1_) * coreDim0RowCount_ * outputScaleRowBytes_ +
            (blockIdx_ % cutNumberForDim1_) * blockCountDim1EachCoreForSplitDim1_);

        workspaceGm_.SetGlobalBuffer(
            (__gm__ uint8_t*)workspace + (blockIdx_ / cutNumberForDim1_) * coreDim0RowCount_ * outputScaleRowBytes_ +
            (blockIdx_ % cutNumberForDim1_) * blockCountDim1EachCoreForSplitDim1_);
    }
    currentBlockFactorDim0_ = blockFactorDim0_;

    if (blockIdx_ >= tailDim0StartCoreIdx_) {
        currentBlockFactorDim0_ = blockFactorDim0Tail_ - 1; // 尾核的尾loop单独处理
    }

    if constexpr (IsSame<U, fp4x2_e2m1_t>::value) {
        f4Emax_ = FP4_E2M1_BF16_MAX_EXP;
    } else {
        f4Emax_ = FP4_E1M2_MAX_EXP;
    }
}
template <typename T, typename U>
__aicore__ inline void DynamicMxQuantTailAxis<T, U>::FP16Convert(
    AscendC::MicroAPI::RegTensor<half>& output, AscendC::MicroAPI::RegTensor<half>& input,
    AscendC::MicroAPI::MaskReg& mask)
{
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<uint16_t> specialValueTensor;
        AscendC::MicroAPI::RegTensor<uint16_t> newMantissa;
        AscendC::MicroAPI::RegTensor<uint16_t> andResult;
        AscendC::MicroAPI::RegTensor<uint16_t> newValue;
        AscendC::MicroAPI::MaskReg specialMask;
        AscendC::MicroAPI::MaskReg nonzeroMask;
        uint16_t specialValue = SPECIAL_VALUE_E1M2;
        if constexpr (IsSame<U, fp4x2_e2m1_t>::value) {
            specialValue = SPECIAL_VALUE_E2M1;
        }
        AscendC::MicroAPI::Duplicate(specialValueTensor, specialValue);
        AscendC::MicroAPI::Duplicate(newMantissa, NEW_MANTISSA);
        AscendC::MicroAPI::And(andResult, (AscendC::MicroAPI::RegTensor<uint16_t>&)input, specialValueTensor, mask);
        AscendC::MicroAPI::CompareScalar<uint16_t, CMPMODE::GT>(nonzeroMask, andResult, 0, mask);
        AscendC::MicroAPI::CompareScalar<uint16_t, CMPMODE::LT>(specialMask, andResult, NEW_MANTISSA, mask);
        AscendC::MicroAPI::MaskAnd(specialMask, specialMask, nonzeroMask, mask);
        AscendC::MicroAPI::Or(newValue, (AscendC::MicroAPI::RegTensor<uint16_t>&)input, newMantissa, mask);
        AscendC::MicroAPI::Select<uint16_t>(
            (AscendC::MicroAPI::RegTensor<uint16_t>&)output, newValue, (AscendC::MicroAPI::RegTensor<uint16_t>&)input,
            specialMask);
    }
    return;
}
template <typename T, typename U>
__aicore__ inline void DynamicMxQuantTailAxis<T, U>::ParseTilingData(const DynamicMxQuantTilingData* tilingData)
{
    totalCoreNum_ = tilingData->totalCoreNum;
    usedCoreNum_ = tilingData->usedCoreNum;
    roundMode_ = tilingData->roundMode;
    blockSize_ = tilingData->blockSize;
    dstType_ = tilingData->dstType;

    ubFactorDim0_ = tilingData->ubFactorDim0TailAxis;
    ubFactorDim1_ = tilingData->ubFactorDim1TailAxis;
    blockFactorDim0_ = tilingData->blockFactorDim0TailAxis;
    blockFactorDim1_ = tilingData->blockFactorDim1TailAxis;
    blockFactorDim0Tail_ = tilingData->tailBlockFactorDim0TailAxis;
    ubFactorDim0Tail_ = tilingData->tailUbFactorDim0TailAxis;

    tailCoreDim1NotSplitDim1_ = tilingData->tailBlockCountDim1TailAxis;
    tailDim0StartCoreIdx_ = tilingData->tailCoreStartIdxDim0TailAxis;
    isSplitDim1_ = tilingData->isSplitDim1TailAxis;

    blockFactorDim1ForSplitDim1_ = tilingData->blockFactorDim1TailAxis;
    tailBlockFactorDim1ForSplitDim1_ = tilingData->tailBlockFactorDim1ForSplitDim1;

    blockCountDim1EachCoreForSplitDim1_ = tilingData->dim1EachCoreForSplitDim1;

    tailLoopBlockDim1ForSplitDim1_ = tilingData->tailBlockCountDim1TailAxis;
    tailLoopBlockTailCoreDim1ForSplitDim1_ = tilingData->tailLoopBlockTailCoreDim1ForSplitDim1;

    cutNumberForDim1_ = tilingData->cutNumberForDim1TailAxis;
    inputShapeDim1_ = tilingData->inputShapeDim1TailAxis;
    isPad_ = tilingData->isPad;
    tailDataCountNotAlign_ = tilingData->tailBlockSize;
    return;
}

template <typename T, typename U>
__aicore__ inline void DynamicMxQuantTailAxis<T, U>::Process()
{
    if (blockIdx_ >= usedCoreNum_) {
        return;
    }
    if (isSplitDim1_ == 1) {
        ProcessDim1Split();
    } else {
        ProcessDim1NotSplit();
    }
    return;
}
template <typename T, typename U>
template <AscendC::RoundMode toBf16RoundMode, AscendC::RoundMode roundMode>
__aicore__ inline void DynamicMxQuantTailAxis<T, U>::CommonFlowProcess(const LoopParams& runningParams)
{
    for (int64_t dim0LoopIdx = 0; dim0LoopIdx < runningParams.runningBlockFactorDim0; dim0LoopIdx++) {
        for (int64_t dim1LoopIdx = 0; dim1LoopIdx < runningParams.runningBlockFactorDim1 - 1; dim1LoopIdx++) {
            CopyIn(dim0LoopIdx, dim1LoopIdx, runningParams.runningUbFactorDim0, runningParams.runningUbFactorDim1, 0);
            Compute<toBf16RoundMode, roundMode>(
                runningParams.runningUbFactorDim0, runningParams.runningUbFactorDim1, 0);
            CopyOut(dim0LoopIdx, dim1LoopIdx, runningParams.runningUbFactorDim0, runningParams.runningUbFactorDim1, 0);
        }

        CopyIn(
            dim0LoopIdx, runningParams.runningBlockFactorDim1 - 1, runningParams.runningUbFactorDim0,
            runningParams.runningTailUbFactorDim1, runningParams.isPad);

        Compute<toBf16RoundMode, roundMode>(
            runningParams.runningUbFactorDim0, runningParams.runningTailUbFactorDim1, runningParams.isPad);

        CopyOut(
            dim0LoopIdx, runningParams.runningBlockFactorDim1 - 1, runningParams.runningUbFactorDim0,
            runningParams.runningTailUbFactorDim1, runningParams.isPad);
    }
    if (blockIdx_ >= tailDim0StartCoreIdx_) {
        for (int64_t dim1LoopIdx = 0; dim1LoopIdx < runningParams.runningBlockFactorDim1 - 1; dim1LoopIdx++) {
            CopyIn(
                runningParams.runningBlockFactorDim0, dim1LoopIdx, runningParams.runningTailUbFactorDim0,
                runningParams.runningUbFactorDim1, 0);
            Compute<toBf16RoundMode, roundMode>(
                runningParams.runningTailUbFactorDim0, runningParams.runningUbFactorDim1, 0);
            CopyOut(
                runningParams.runningBlockFactorDim0, dim1LoopIdx, runningParams.runningTailUbFactorDim0,
                runningParams.runningUbFactorDim1, 0);
        }

        CopyIn(
            runningParams.runningBlockFactorDim0, runningParams.runningBlockFactorDim1 - 1,
            runningParams.runningTailUbFactorDim0, runningParams.runningTailUbFactorDim1, runningParams.isPad);
        Compute<toBf16RoundMode, roundMode>(
            runningParams.runningTailUbFactorDim0, runningParams.runningTailUbFactorDim1, runningParams.isPad);
        CopyOut(
            runningParams.runningBlockFactorDim0, runningParams.runningBlockFactorDim1 - 1,
            runningParams.runningTailUbFactorDim0, runningParams.runningTailUbFactorDim1, runningParams.isPad);
    }
    return;
}

template <typename T, typename U>
__aicore__ inline void DynamicMxQuantTailAxis<T, U>::ProcessDim1NotSplit()
{
    LoopParams runningParams;
    runningParams.runningBlockFactorDim0 = currentBlockFactorDim0_;
    runningParams.runningBlockFactorDim1 = blockFactorDim1_;
    runningParams.runningUbFactorDim0 = ubFactorDim0_;
    runningParams.runningUbFactorDim1 = ubFactorDim1_;
    runningParams.runningTailUbFactorDim0 = ubFactorDim0Tail_;
    runningParams.runningTailUbFactorDim1 = tailCoreDim1NotSplitDim1_;
    runningParams.isPad = isPad_;

    if (this->roundMode_ == MODE_RINT) {
        CommonFlowProcess<RoundMode::CAST_TRUNC, RoundMode::CAST_RINT>(runningParams);
    } else if (this->roundMode_ == MODE_ROUND) {
        CommonFlowProcess<RoundMode::CAST_TRUNC, RoundMode::CAST_ROUND>(runningParams);
    } else if (this->roundMode_ == MODE_FLOOR) {
        CommonFlowProcess<RoundMode::CAST_FLOOR, RoundMode::CAST_FLOOR>(runningParams);
    }
    return;
}

template <typename T, typename U>
__aicore__ inline void DynamicMxQuantTailAxis<T, U>::ProcessDim1SplitNormalCore()
{
    LoopParams runningParams;
    runningParams.runningBlockFactorDim0 = currentBlockFactorDim0_;
    runningParams.runningBlockFactorDim1 = blockFactorDim1ForSplitDim1_;
    runningParams.runningUbFactorDim1 = ubFactorDim1_;
    runningParams.runningUbFactorDim0 = ubFactorDim0_;
    runningParams.runningTailUbFactorDim1 = tailLoopBlockDim1ForSplitDim1_;
    runningParams.runningTailUbFactorDim0 = ubFactorDim0Tail_;
    runningParams.isPad = 0;

    if (this->roundMode_ == MODE_RINT) {
        CommonFlowProcess<RoundMode::CAST_TRUNC, RoundMode::CAST_RINT>(runningParams);
    } else if (this->roundMode_ == MODE_ROUND) {
        CommonFlowProcess<RoundMode::CAST_TRUNC, RoundMode::CAST_ROUND>(runningParams);
    } else if (this->roundMode_ == MODE_FLOOR) {
        CommonFlowProcess<RoundMode::CAST_FLOOR, RoundMode::CAST_FLOOR>(runningParams);
    }
    return;
}
template <typename T, typename U>
__aicore__ inline void DynamicMxQuantTailAxis<T, U>::ProcessDim1SplitTailCore()
{
    LoopParams runningParams;
    runningParams.runningBlockFactorDim0 = currentBlockFactorDim0_;
    runningParams.runningBlockFactorDim1 = tailBlockFactorDim1ForSplitDim1_;
    runningParams.runningUbFactorDim0 = ubFactorDim0_;
    runningParams.runningUbFactorDim1 = ubFactorDim1_;
    runningParams.runningTailUbFactorDim0 = ubFactorDim0Tail_;
    runningParams.runningTailUbFactorDim1 = tailLoopBlockTailCoreDim1ForSplitDim1_;
    runningParams.isPad = isPad_;

    if (this->roundMode_ == MODE_RINT) {
        CommonFlowProcess<RoundMode::CAST_TRUNC, RoundMode::CAST_RINT>(runningParams);
    } else if (this->roundMode_ == MODE_ROUND) {
        CommonFlowProcess<RoundMode::CAST_TRUNC, RoundMode::CAST_ROUND>(runningParams);
    } else if (this->roundMode_ == MODE_FLOOR) {
        CommonFlowProcess<RoundMode::CAST_FLOOR, RoundMode::CAST_FLOOR>(runningParams);
    }
    return;
}
template <typename T, typename U>
__aicore__ inline void DynamicMxQuantTailAxis<T, U>::ProcessDim1Split()
{
    // 处理列方向切分尾块部分核
    if (((blockIdx_ + 1) % cutNumberForDim1_) == 0) {
        ProcessDim1SplitTailCore();
        return;
    }
    ProcessDim1SplitNormalCore();
    return;
}

template <typename T, typename U>
__aicore__ inline void DynamicMxQuantTailAxis<T, U>::CopyIn(
    int64_t dim0LoopIdx, int64_t dim1LoopIdx, int64_t usedFactorDim0, int64_t usedFactorDim1, int64_t isContainPad)
{
    LocalTensor<T> inLocal = inQueue_.AllocTensor<T>();
    DataCopyExtParams copyInParam = {0, 0, 0, 0, 0};
    DataCopyPadExtParams<T> padParams = {false, 0, 0, 0};
    copyInParam.blockCount = usedFactorDim0;
    copyInParam.blockLen = usedFactorDim1 * blockSize_ * sizeof(T);
    if (isContainPad == 1) {
        copyInParam.blockLen = (usedFactorDim1 - 1) * blockSize_ * sizeof(T) + tailDataCountNotAlign_ * sizeof(T);
        padParams.isPad = true;
        padParams.leftPadding = 0;
        if (tailDataCountNotAlign_ > oneBlockCount_) {
            padParams.rightPadding = oneBlockCount_ - (tailDataCountNotAlign_ - oneBlockCount_);
        } else {
            padParams.rightPadding = oneBlockCount_;
        }
        padParams.paddingValue = 0;
    }
    copyInParam.srcStride = inputRowBytes_ - copyInParam.blockLen;
    copyInParam.dstStride = 0;

    DataCopyPad(
        inLocal,
        xGm_[dim0LoopIdx * ubFactorDim0_ * (inputRowBytes_ / sizeof(T)) + dim1LoopIdx * ubFactorDim1_ * blockSize_],
        copyInParam, padParams);

    inQueue_.EnQue(inLocal);
    return;
}
template <typename T, typename U>
template <AscendC::RoundMode toBf16RoundMode, AscendC::RoundMode roundMode>
__aicore__ inline void DynamicMxQuantTailAxis<T, U>::ComputeData(
    __ubuf__ T* srcAddr, __ubuf__ uint16_t* halfScaleLocalAddr, __ubuf__ int8_t* outLocalAddr, uint32_t totalCountInUB,
    uint16_t loopNum)
{
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::MaskReg dataMask1;
        AscendC::MicroAPI::RegTensor<uint16_t> halfScaleForMul;
        AscendC::MicroAPI::RegTensor<T> vdExp0;
        AscendC::MicroAPI::RegTensor<T> vdExp1;
        AscendC::MicroAPI::RegTensor<T> vdExp0Convert;
        AscendC::MicroAPI::RegTensor<T> vdExp1Convert;

        AscendC::MicroAPI::RegTensor<bfloat16_t> vdExp0BF16;
        AscendC::MicroAPI::RegTensor<bfloat16_t> vdExp1BF16;

        AscendC::MicroAPI::RegTensor<U> vdExp0FP4;
        AscendC::MicroAPI::RegTensor<U> vdExp1FP4;

        AscendC::MicroAPI::RegTensor<bfloat16_t> vdBF16Exp0FP4;
        AscendC::MicroAPI::RegTensor<bfloat16_t> vdBF16Exp1FP4;
        static constexpr AscendC::MicroAPI::CastTrait castTrait = {
            AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, roundMode};
        static constexpr AscendC::MicroAPI::CastTrait castTraitHalf2Bf16 = {
            AscendC::MicroAPI::RegLayout::UNKNOWN, AscendC::MicroAPI::SatMode::UNKNOWN,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, toBf16RoundMode};
        for (uint16_t i = 0; i < loopNum; i++) {
            dataMask1 = AscendC::MicroAPI::UpdateMask<T>(totalCountInUB);
            AscendC::MicroAPI::DataCopy<
                T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE, AscendC::MicroAPI::LoadDist::DIST_DINTLV_B16>(
                vdExp0, vdExp1, srcAddr, vlForHalfNumber_ * DIGIT_TWO);
            AscendC::MicroAPI::DataCopy<
                uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE, AscendC::MicroAPI::LoadDist::DIST_E2B_B16>(
                halfScaleForMul, halfScaleLocalAddr, elementAfterReduce_);

            if constexpr (IsSame<T, half>::value) {
                if constexpr (roundMode == RoundMode::CAST_RINT) {
                    FP16Convert(vdExp0, vdExp0, dataMask1);
                    FP16Convert(vdExp1, vdExp1, dataMask1);
                }
                AscendC::MicroAPI::Cast<bfloat16_t, T, castTraitHalf2Bf16>(vdExp0BF16, vdExp0, dataMask1);
                AscendC::MicroAPI::Cast<bfloat16_t, T, castTraitHalf2Bf16>(vdExp1BF16, vdExp1, dataMask1);
                AscendC::MicroAPI::Mul(
                    vdExp0BF16, vdExp0BF16, (AscendC::MicroAPI::RegTensor<bfloat16_t>&)halfScaleForMul, dataMask1);
                AscendC::MicroAPI::Mul(
                    vdExp1BF16, vdExp1BF16, (AscendC::MicroAPI::RegTensor<bfloat16_t>&)halfScaleForMul, dataMask1);
                AscendC::MicroAPI::Interleave(vdExp0BF16, vdExp1BF16, vdExp0BF16, vdExp1BF16);
                AscendC::MicroAPI::Cast<U, bfloat16_t, castTrait>(vdExp0FP4, vdExp0BF16, dataMask1);
                AscendC::MicroAPI::Cast<U, bfloat16_t, castTrait>(vdExp1FP4, vdExp1BF16, dataMask1);

            } else {
                AscendC::MicroAPI::Mul(vdExp0, vdExp0, (AscendC::MicroAPI::RegTensor<T>&)halfScaleForMul, dataMask1);
                AscendC::MicroAPI::Mul(vdExp1, vdExp1, (AscendC::MicroAPI::RegTensor<T>&)halfScaleForMul, dataMask1);
                AscendC::MicroAPI::Interleave(vdExp0, vdExp1, vdExp0, vdExp1);
                AscendC::MicroAPI::Cast<U, T, castTrait>(vdExp0FP4, vdExp0, dataMask1);
                AscendC::MicroAPI::Cast<U, T, castTrait>(vdExp1FP4, vdExp1, dataMask1);
            }

            AscendC::MicroAPI::DataCopy<
                int8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                outLocalAddr, (AscendC::MicroAPI::RegTensor<int8_t>&)vdExp0FP4, OUT_ELE_NUM_ONE_BLK, dataMask1);
            AscendC::MicroAPI::DataCopy<
                int8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                outLocalAddr, (AscendC::MicroAPI::RegTensor<int8_t>&)vdExp1FP4, OUT_ELE_NUM_ONE_BLK, dataMask1);
        }
    }
    return;
}

template <typename T, typename U>
__aicore__ inline void DynamicMxQuantTailAxis<T, U>::ComputeMaxExp(
    __ubuf__ T* srcAddr, __ubuf__ uint16_t* maxExpAddr, uint32_t totalCountInUB, uint16_t loopNum)
{
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<T> vdExp0;
        AscendC::MicroAPI::RegTensor<T> vdExp1;
        AscendC::MicroAPI::RegTensor<bfloat16_t> vdExp0BF16;
        AscendC::MicroAPI::RegTensor<bfloat16_t> vdExp1BF16;
        AscendC::MicroAPI::RegTensor<uint16_t> vdExpExtract0;
        AscendC::MicroAPI::RegTensor<uint16_t> vdExpExtract1;
        AscendC::MicroAPI::RegTensor<uint16_t> vdExpSelect0;
        AscendC::MicroAPI::RegTensor<uint16_t> vdExpSelect1;
        AscendC::MicroAPI::RegTensor<uint16_t> expMaskBF16;
        AscendC::MicroAPI::Duplicate(expMaskBF16, MAX_EXP_FOR_BF16);
        AscendC::MicroAPI::RegTensor<uint16_t> invalidmaskfp16;
        AscendC::MicroAPI::Duplicate(invalidmaskfp16, INVALID_FLOAT16);
        AscendC::MicroAPI::RegTensor<uint16_t> vdMaxExp;
        AscendC::MicroAPI::MaskReg scaleMask1;
        AscendC::MicroAPI::MaskReg scaleMask2;
        AscendC::MicroAPI::MaskReg invalidDataMask0;
        AscendC::MicroAPI::MaskReg invalidDataMask1;
        AscendC::MicroAPI::UnalignReg u1;
        static constexpr AscendC::MicroAPI::CastTrait castTraitHalf2Bf16 = {
            AscendC::MicroAPI::RegLayout::UNKNOWN, AscendC::MicroAPI::SatMode::UNKNOWN,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_TRUNC};
        for (uint16_t i = 0; i < loopNum; i++) {
            scaleMask1 = AscendC::MicroAPI::UpdateMask<T>(totalCountInUB);
            scaleMask2 = AscendC::MicroAPI::UpdateMask<T>(totalCountInUB);
            AscendC::MicroAPI::DataCopy<
                T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE, AscendC::MicroAPI::LoadDist::DIST_DINTLV_B16>(
                vdExp0, vdExp1, srcAddr, vlForHalfNumber_ * DIGIT_TWO);

            if constexpr (IsSame<T, half>::value) {
                AscendC::MicroAPI::And(
                    vdExpSelect0, (AscendC::MicroAPI::RegTensor<uint16_t>&)vdExp0, invalidmaskfp16, scaleMask1);
                AscendC::MicroAPI::And(
                    vdExpSelect1, (AscendC::MicroAPI::RegTensor<uint16_t>&)vdExp1, invalidmaskfp16, scaleMask1);
                AscendC::MicroAPI::Compare<uint16_t, CMPMODE::NE>(
                    invalidDataMask0, vdExpSelect0, invalidmaskfp16, scaleMask1);
                AscendC::MicroAPI::Compare<uint16_t, CMPMODE::NE>(
                    invalidDataMask1, vdExpSelect1, invalidmaskfp16, scaleMask1);
                AscendC::MicroAPI::Cast<bfloat16_t, T, castTraitHalf2Bf16>(vdExp0BF16, vdExp0, scaleMask1);
                AscendC::MicroAPI::Cast<bfloat16_t, T, castTraitHalf2Bf16>(vdExp1BF16, vdExp1, scaleMask1);
                AscendC::MicroAPI::And(
                    vdExpExtract0, (AscendC::MicroAPI::RegTensor<uint16_t>&)vdExp0BF16, expMaskBF16, scaleMask1);
                AscendC::MicroAPI::And(
                    vdExpExtract1, (AscendC::MicroAPI::RegTensor<uint16_t>&)vdExp1BF16, expMaskBF16, scaleMask1);
                AscendC::MicroAPI::Select<uint16_t>(vdExpExtract0, vdExpExtract0, expMaskBF16, invalidDataMask0);
                AscendC::MicroAPI::Select<uint16_t>(vdExpExtract1, vdExpExtract1, expMaskBF16, invalidDataMask1);
            } else {
                AscendC::MicroAPI::And(
                    vdExpExtract0, (AscendC::MicroAPI::RegTensor<uint16_t>&)vdExp0, expMaskBF16, scaleMask1);
                AscendC::MicroAPI::And(
                    vdExpExtract1, (AscendC::MicroAPI::RegTensor<uint16_t>&)vdExp1, expMaskBF16, scaleMask1);
            }

            AscendC::MicroAPI::Max(vdMaxExp, vdExpExtract0, vdExpExtract1, scaleMask1);
            AscendC::MicroAPI::ReduceMaxWithDataBlock(vdMaxExp, vdMaxExp, scaleMask1);

            AscendC::MicroAPI::DataCopyUnAlign<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                maxExpAddr, vdMaxExp, u1, elementAfterReduce_);
        }
        AscendC::MicroAPI::DataCopyUnAlignPost(maxExpAddr, u1, 0);
    }
    return;
}

template <typename T, typename U>
__aicore__ inline void DynamicMxQuantTailAxis<T, U>::ComputeScale(
    __ubuf__ uint16_t* maxExpAddr, __ubuf__ uint16_t* mxScaleLocalAddr, __ubuf__ uint16_t* halfScaleLocalAddr,
    uint32_t totalScaleInUB, uint16_t loopNumScale)
{
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<uint16_t> expMask;
        AscendC::MicroAPI::Duplicate(expMask, MAX_EXP_FOR_BF16);
        AscendC::MicroAPI::RegTensor<uint16_t> vdMaxExp;

        AscendC::MicroAPI::RegTensor<T> vdExp0;
        AscendC::MicroAPI::RegTensor<T> vdExp1;

        AscendC::MicroAPI::MaskReg cmpResult;
        AscendC::MicroAPI::MaskReg zeroMask;
        AscendC::MicroAPI::MaskReg cmpResultSub;
        AscendC::MicroAPI::MaskReg preMaskScale;
        AscendC::MicroAPI::RegTensor<uint16_t> maxExpValue;
        AscendC::MicroAPI::Duplicate(maxExpValue, f4Emax_);
        AscendC::MicroAPI::RegTensor<uint16_t> sharedExp;
        AscendC::MicroAPI::RegTensor<uint16_t> scaleValue;
        AscendC::MicroAPI::RegTensor<uint16_t> scaleBias;
        AscendC::MicroAPI::Duplicate(scaleBias, BF16_EXP_BIAS);
        AscendC::MicroAPI::RegTensor<uint16_t> halfScale;
        AscendC::MicroAPI::RegTensor<uint16_t> fp8NanRegTensor;
        AscendC::MicroAPI::Duplicate(fp8NanRegTensor, MAX_EXP_FOR_FP8);
        AscendC::MicroAPI::RegTensor<uint16_t> zeroRegTensor;
        AscendC::MicroAPI::Duplicate(zeroRegTensor, 0);
        AscendC::MicroAPI::RegTensor<uint16_t> nanRegTensor;
        AscendC::MicroAPI::Duplicate(nanRegTensor, NAN_CUSTOMIZATION);
        AscendC::MicroAPI::MaskReg invalidDataMask;
        AscendC::MicroAPI::MaskReg specialDataMask;
        AscendC::MicroAPI::RegTensor<uint16_t> specialExpRegTensor;
        AscendC::MicroAPI::Duplicate(specialExpRegTensor, SPECIAL_EXP_THRESHOLD);
        for (uint16_t i = 0; i < loopNumScale; i++) {
            preMaskScale = AscendC::MicroAPI::UpdateMask<uint16_t>(totalScaleInUB);
            AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                vdMaxExp, maxExpAddr, vlForHalfNumber_);
            AscendC::MicroAPI::Compare<uint16_t, CMPMODE::NE>(cmpResult, vdMaxExp, expMask, preMaskScale); // INF/NAN
            AscendC::MicroAPI::Compare<uint16_t, CMPMODE::NE>(zeroMask, vdMaxExp, zeroRegTensor, preMaskScale);
            AscendC::MicroAPI::Compare<uint16_t, CMPMODE::LE>(invalidDataMask, vdMaxExp, maxExpValue, preMaskScale);

            AscendC::MicroAPI::Select<uint16_t>(vdMaxExp, maxExpValue, vdMaxExp, invalidDataMask);

            AscendC::MicroAPI::Sub(sharedExp, vdMaxExp, maxExpValue, preMaskScale);
            AscendC::MicroAPI::ShiftRights(scaleValue, sharedExp, SHR_NUM_FOR_BF16, preMaskScale);

            AscendC::MicroAPI::Select<uint16_t>(scaleValue, scaleValue, fp8NanRegTensor, cmpResult);
            AscendC::MicroAPI::Select<uint16_t>(scaleValue, scaleValue, zeroRegTensor, zeroMask);

            AscendC::MicroAPI::DataCopy<
                uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                AscendC::MicroAPI::StoreDist::DIST_PACK_B16>(
                mxScaleLocalAddr, scaleValue, vlForHalfNumber_ / DIGIT_TWO, preMaskScale);

            AscendC::MicroAPI::Compare<uint16_t, CMPMODE::EQ>(specialDataMask, sharedExp, scaleBias, preMaskScale);
            AscendC::MicroAPI::Sub(halfScale, scaleBias, sharedExp, preMaskScale);
            AscendC::MicroAPI::Select<uint16_t>(halfScale, halfScale, nanRegTensor, cmpResult);
            AscendC::MicroAPI::Select<uint16_t>(halfScale, halfScale, zeroRegTensor, zeroMask);
            AscendC::MicroAPI::Select<uint16_t>(halfScale, specialExpRegTensor, halfScale, specialDataMask);

            AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                halfScaleLocalAddr, halfScale, vlForHalfNumber_, preMaskScale);
        }
    }
    return;
}

template <typename T, typename U>
__aicore__ inline void DynamicMxQuantTailAxis<T, U>::DeletePadData(
    __ubuf__ int8_t* outLocalAddr, __ubuf__ int8_t* outBufferLocalAddr, uint16_t loopNum, uint32_t inputUpdateStride,
    uint32_t outputUpdateStride)
{
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::UnalignReg uIn;
        AscendC::MicroAPI::UnalignReg uOut;
        AscendC::MicroAPI::RegTensor<int8_t> inputRegTensor;
        for (uint16_t i = 0; i < loopNum; i++) {
            AscendC::MicroAPI::DataCopyUnAlignPre(uIn, outBufferLocalAddr);
            AscendC::MicroAPI::DataCopyUnAlign<int8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                inputRegTensor, uIn, outBufferLocalAddr, inputUpdateStride);
            AscendC::MicroAPI::DataCopyUnAlign<int8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                outLocalAddr, inputRegTensor, uOut, outputUpdateStride);
            AscendC::MicroAPI::DataCopyUnAlignPost(outLocalAddr, uOut, 0);
        }
    }
    return;
}

template <typename T, typename U>
template <AscendC::RoundMode toBf16RoundMode, AscendC::RoundMode roundMode>
__aicore__ inline void DynamicMxQuantTailAxis<T, U>::Compute(
    int64_t realUbFactorDim0, int64_t realUbFactorDim1, int64_t isContainPad)
{
    LocalTensor<T> inLocal = inQueue_.DeQue<T>();
    LocalTensor<uint16_t> maxExpLocal = maxExpBuffer_.Get<uint16_t>();

    uint32_t totalScaleInUB = realUbFactorDim0 * realUbFactorDim1;
    uint32_t totalCountInUB = realUbFactorDim0 * realUbFactorDim1 * blockSize_;

    uint16_t loopNum = (totalCountInUB + vlForHalfNumber_ * DIGIT_TWO - 1) / (vlForHalfNumber_ * DIGIT_TWO);
    uint16_t loopNumScale = (totalScaleInUB + vlForHalfNumber_ - 1) / vlForHalfNumber_;

    auto srcAddr = reinterpret_cast<__ubuf__ T*>(inLocal.GetPhyAddr());
    auto maxExpAddr = reinterpret_cast<__ubuf__ uint16_t*>(maxExpLocal.GetPhyAddr());

    LocalTensor<uint16_t> mxScaleLocal = mxScaleQueue_.AllocTensor<uint16_t>();
    auto mxScaleLocalAddr = reinterpret_cast<__ubuf__ uint16_t*>(mxScaleLocal.GetPhyAddr());

    LocalTensor<uint16_t> halfScaleLocal = maxhalfScaleBuffer_.Get<uint16_t>();
    auto halfScaleLocalAddr = reinterpret_cast<__ubuf__ uint16_t*>(halfScaleLocal.GetPhyAddr());

    LocalTensor<int8_t> outLocal = outQueue_.AllocTensor<int8_t>();
    auto outLocalAddr = reinterpret_cast<__ubuf__ int8_t*>(outLocal.GetPhyAddr());

    ComputeMaxExp(srcAddr, maxExpAddr, totalCountInUB, loopNum);

    maxExpAddr = reinterpret_cast<__ubuf__ uint16_t*>(maxExpLocal.GetPhyAddr());
    ComputeScale(maxExpAddr, mxScaleLocalAddr, halfScaleLocalAddr, totalScaleInUB, loopNumScale);

    srcAddr = reinterpret_cast<__ubuf__ T*>(inLocal.GetPhyAddr());
    halfScaleLocalAddr = reinterpret_cast<__ubuf__ uint16_t*>(halfScaleLocal.GetPhyAddr());
    LocalTensor<int8_t> outBufferLocal = outBuffer_.Get<int8_t>();
    auto outBufferLocalAddr = reinterpret_cast<__ubuf__ int8_t*>(outBufferLocal.GetPhyAddr());

    if (isContainPad == 1) {
        ComputeData<toBf16RoundMode, roundMode>(
            srcAddr, halfScaleLocalAddr, outBufferLocalAddr, totalCountInUB, loopNum);
        uint32_t inputUpdateStride = realUbFactorDim1 * blockSize_ / DIGIT_TWO;
        uint32_t outputUpdateStride = ((realUbFactorDim1 - 1) * blockSize_ + tailDataCountNotAlign_) / DIGIT_TWO;
        outBufferLocalAddr = reinterpret_cast<__ubuf__ int8_t*>(outBufferLocal.GetPhyAddr());
        DeletePadData(outLocalAddr, outBufferLocalAddr, realUbFactorDim0, inputUpdateStride, outputUpdateStride);
    } else {
        ComputeData<toBf16RoundMode, roundMode>(srcAddr, halfScaleLocalAddr, outLocalAddr, totalCountInUB, loopNum);
    }

    inQueue_.FreeTensor(inLocal);
    outQueue_.EnQue(outLocal);
    mxScaleQueue_.EnQue(mxScaleLocal);
    return;
}

template <typename T, typename U>
__aicore__ inline void DynamicMxQuantTailAxis<T, U>::CopyOut(
    int64_t dim0LoopIdx, int64_t dim1LoopIdx, int64_t usedFactorDim0, int64_t usedFactorDim1, int64_t isContainPad)
{
    LocalTensor<uint8_t> mxScaleLocal = mxScaleQueue_.DeQue<uint8_t>();
    LocalTensor<uint8_t> outLocal = outQueue_.DeQue<uint8_t>();
    DataCopyExtParams copyOutParamData = {0, 0, 0, 0, 0};
    copyOutParamData.blockCount = usedFactorDim0;
    copyOutParamData.blockLen = usedFactorDim1 * blockSize_ / DIGIT_TWO;
    if (isContainPad == 1) {
        copyOutParamData.blockLen = ((usedFactorDim1 - 1) * blockSize_ + tailDataCountNotAlign_) / DIGIT_TWO;
    }
    copyOutParamData.srcStride = 0;
    copyOutParamData.dstStride = outputDataRowBytes_ - copyOutParamData.blockLen;

    DataCopyPad<uint8_t, PaddingMode::Compact>(
        yGm_[dim0LoopIdx * ubFactorDim0_ * outputDataRowBytes_ + dim1LoopIdx * ubFactorDim1_ * blockSize_ / DIGIT_TWO],
        outLocal, copyOutParamData);

    DataCopyExtParams copyOutParamScale = {0, 0, 0, 0, 0};
    copyOutParamScale.blockCount = usedFactorDim0;
    copyOutParamScale.blockLen = usedFactorDim1;
    copyOutParamScale.srcStride = 0;
    copyOutParamScale.dstStride = outputScaleRowBytes_ - copyOutParamScale.blockLen;

    DataCopyPad<uint8_t, PaddingMode::Compact>(
        workspaceGm_[dim0LoopIdx * ubFactorDim0_ * outputScaleRowBytes_ + dim1LoopIdx * ubFactorDim1_], mxScaleLocal,
        copyOutParamScale);

    outQueue_.FreeTensor(outLocal);
    mxScaleQueue_.FreeTensor(mxScaleLocal);
    return;
}

} // namespace DynamicMxQuant
#endif // DYNAMIC_MX_QUANT_TAIL_AXIS_H
