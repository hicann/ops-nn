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
 * \file dynamic_mx_quant_with_dual_axis_base.h
 * \brief
 */

#ifndef DYNAMIC_DUAL_LEVEL_MX_QUANT_BASE_H
#define DYNAMIC_DUAL_LEVEL_MX_QUANT_BASE_H

#define FLOAT_OVERFLOW_MODE_CTRL 60

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../inc/platform.h"
#include "../inc/kernel_utils.h"
#include "dynamic_dual_level_mx_quant_struct.h"

namespace DynamicDualLevelMxQuant {
using namespace AscendC;

constexpr int64_t DB_BUFFER = 2;
constexpr int64_t DIGIT_TWO = 2;
constexpr int64_t DIGIT_EIGHT = 8;
constexpr int64_t DIGIT_32 = 32;
constexpr int64_t DIGIT_64 = 64;
constexpr int64_t DIGIT_128 = 128;
constexpr int64_t DIGIT_256 = 256;
constexpr int64_t DIGIT_384 = 384;
constexpr uint16_t ABS_FOR_UINT16 = 0x7fff;
constexpr uint16_t NAN_CUSTOMIZATION = 0x7f81;     // 0111 1111 1000 0001
constexpr uint16_t NAN_FOR_FP8_E8M0 = 0x00ff;      // 0000 0000 1111 1111
constexpr uint16_t SPECIAL_EXP_THRESHOLD = 0x0040; // 0000 0000 0100 0000
constexpr int16_t SHR_NUM_FOR_BF16 = 7;
constexpr uint16_t FP4_E2M1_BF16_MAX_EXP = 0x0100;
constexpr uint16_t BF16_EXP_BIAS = 0x7f00;        // 0111 1111 0000 0000
constexpr uint32_t FP4_E2M1_MAX = 0x3e2aaaab;     // 1/6的float32表示 6是E2M1所能表示的最大值
constexpr uint16_t INF_FOR_BF16 = 0x7f80;         // 0111 1111 1000 0000
constexpr uint16_t INF_FOR_FP16 = 0x7c00;         // 0111 1100 0000 0000
constexpr uint32_t INVAILD_FOR_FP32 = 0x00800000; // 0000 0000 1000 0000 2^(-126)
constexpr uint32_t MAX_EXP_FOR_FP32 = 0x7f800000;
constexpr int32_t FP32_BIAS = 127;
constexpr int32_t FP32_BIAS_NEG = -127;
constexpr int32_t NEG_ONE = -1;
constexpr int32_t NEG_ZERO = 0x80000000;
constexpr int16_t SHR_NUM_FOR_FP32 = 23;

template <typename xDtype, AscendC::RoundMode roundMode>
class DynamicDualLevelMxQuantBase {
public:
    __aicore__ inline DynamicDualLevelMxQuantBase(const DynamicDualLevelMxQuantTilingData* tilingData, TPipe* pipe)
        : tilingData_(tilingData), pipe_(pipe){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR level0_scale, GM_ADDR level1_scale);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitParams();
    __aicore__ inline void ProcessOneLoop(
        int64_t xGmAddr, int64_t level0ScaleGmOffset, int64_t level1ScaleGmOffset, int64_t inBlockLen, int64_t inBurst);
    __aicore__ inline void CopyOut(
        int64_t yGmAddr, int64_t level0ScaleGmAddr, int64_t level1ScaleGmAddr, int64_t outBlockLen, int64_t outBurst);
    __aicore__ inline void CopyIn(int64_t xGmAddr, int64_t inBlockLen, int64_t inBurst);
    __aicore__ inline void ComputeAll(int64_t dataLen, int64_t inBurst);
    __aicore__ inline void ComputeLevel0Quant(
        int64_t LoopNum, __ubuf__ xDtype* xUbAddr, __ubuf__ xDtype* xTmpUbAddr, __ubuf__ float* level0ScaleUbAddr);
    __aicore__ inline void CalcXTmp(
        __ubuf__ xDtype* xTmpUbAddr, MicroAPI::RegTensor<float> level0ScaleReg, MicroAPI::RegTensor<xDtype> xReg,
        MicroAPI::RegTensor<float>& xZeroFP32, MicroAPI::RegTensor<float>& xOneFP32);
    __aicore__ inline void ComputeLevel1Scale(
        int64_t loopNum, __ubuf__ xDtype* xTmpUbAddr, __ubuf__ uint8_t* level1ScaleUbAddr,
        __ubuf__ uint16_t* level1ScaleReciprocalUbAddr);
    __aicore__ inline void ComputeY(
        int64_t loopNum, __ubuf__ xDtype* xTmpUbAddr, __ubuf__ uint8_t* yAddr,
        __ubuf__ uint16_t* level1ScaleReciprocalUbAddr);
    __aicore__ inline void ComputeFP4FromHalf(MicroAPI::RegTensor<float>& Reg);

protected:
    static constexpr MicroAPI::CastTrait castTraitXdtypetoFp32Zero = {
        MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING,
        AscendC::RoundMode::UNKNOWN};
    static constexpr MicroAPI::CastTrait castTraitXdtypetoFp32One = {
        MicroAPI::RegLayout::ONE, MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING,
        AscendC::RoundMode::UNKNOWN};
    static constexpr MicroAPI::CastTrait castTraitHalf2BF16 = {
        MicroAPI::RegLayout::UNKNOWN, MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING,
        AscendC::RoundMode::CAST_TRUNC};
    static constexpr MicroAPI::CastTrait castTraitBF16toFp4 = {
        MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT, MicroAPI::MaskMergeMode::ZEROING, roundMode};
    static constexpr MicroAPI::CastTrait castTraitFp32toBF16 = {
        MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT, MicroAPI::MaskMergeMode::ZEROING,
        AscendC::RoundMode::CAST_RINT};

private:
    // tiling data
    const DynamicDualLevelMxQuantTilingData* tilingData_;

    // pipe & queue & buf
    TPipe* pipe_;
    TQue<QuePosition::VECIN, DB_BUFFER> inQueue;
    TQue<QuePosition::VECOUT, DB_BUFFER> outQueue;
    TQue<QuePosition::VECOUT, DB_BUFFER> level0ScaleQueue;
    TQue<QuePosition::VECOUT, DB_BUFFER> level1ScaleQueue;
    TBuf<TPosition::VECCALC> xTmpBuf;
    TBuf<TPosition::VECCALC> level1ScaleReciprocalBuf;

    // gm
    GlobalTensor<xDtype> xGm_;
    GlobalTensor<uint8_t> yGm_;
    GlobalTensor<float> level0ScaleGm_;
    GlobalTensor<uint8_t> level1ScaleGm_;

    int64_t blockIdx_ = 0;
    int64_t ubFactor_ = 0;
    int64_t blockSizeCol_ = 0;
    int64_t blockSizeRow_ = 0;
    int64_t rowBlockNum_ = 0;
    int64_t colBlockNum_ = 0;
    int64_t rowTileNum_ = 0;
    int64_t colTileNum_ = 0;
    int64_t curRowBlockNum_ = 0;
    int64_t curColBlockNum_ = 0;
    int64_t curRowLoopNum_ = 0;
    int64_t curColLoopNum_ = 0;
    int64_t curRowTileSize_ = 0;
    int64_t level1BlockRowNum_ = 0;
    int64_t isRowTail_ = 0;
    int64_t isColTail_ = 0;

    int64_t xBufferSize_ = 0;
    int64_t yBufferSize_ = 0;
    int64_t level0ScaleBufferSize_ = 0;
    int64_t level1ScaleBufferSize_ = 0;
    int64_t xTmpBufferSize_ = 0;
    int64_t level1ScaleReciprocalBufferSize_ = 0;

    int64_t vlForHalfNumber_ = platform::GetVRegSize() / sizeof(uint16_t);
    int64_t vlForFloatNumber_ = platform::GetVRegSize() / sizeof(uint32_t);
    int64_t UBBlockSize_ = platform::GetUbBlockSize();
};

template <typename xDtype, AscendC::RoundMode roundMode>
__aicore__ inline void DynamicDualLevelMxQuantBase<xDtype, roundMode>::Init(
    GM_ADDR x, GM_ADDR y, GM_ADDR level0Scale, GM_ADDR level1Scale)
{
#if (__NPU_ARCH__ == 3101)
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
#endif
    // init block params
    InitParams();

    xGm_.SetGlobalBuffer((__gm__ xDtype*)(x));
    yGm_.SetGlobalBuffer((__gm__ uint8_t*)(y));
    level0ScaleGm_.SetGlobalBuffer((__gm__ float*)(level0Scale));
    level1ScaleGm_.SetGlobalBuffer((__gm__ uint8_t*)(level1Scale));

    xBufferSize_ = ubFactor_ * blockSizeCol_ * blockSizeRow_ * sizeof(xDtype);
    // y 为fp4_e2m1, 两个元素看成一个uint8搬出
    yBufferSize_ = ubFactor_ * blockSizeCol_ * blockSizeRow_ / DIGIT_TWO;

    level0ScaleBufferSize_ = ubFactor_ * blockSizeCol_ * UBBlockSize_;
    level1ScaleBufferSize_ = ubFactor_ * blockSizeCol_ * UBBlockSize_;

    xTmpBufferSize_ = ubFactor_ * blockSizeCol_ * blockSizeRow_ * sizeof(xDtype);
    level1ScaleReciprocalBufferSize_ = ubFactor_ * blockSizeCol_ * UBBlockSize_;

    pipe_->InitBuffer(inQueue, DB_BUFFER, xBufferSize_);
    pipe_->InitBuffer(outQueue, DB_BUFFER, yBufferSize_);
    pipe_->InitBuffer(level0ScaleQueue, DB_BUFFER, level0ScaleBufferSize_);
    pipe_->InitBuffer(level1ScaleQueue, DB_BUFFER, level1ScaleBufferSize_);
    pipe_->InitBuffer(xTmpBuf, xTmpBufferSize_);
    pipe_->InitBuffer(level1ScaleReciprocalBuf, level1ScaleReciprocalBufferSize_);
}

template <typename xDtype, AscendC::RoundMode roundMode>
__aicore__ inline void DynamicDualLevelMxQuantBase<xDtype, roundMode>::InitParams()
{
    blockIdx_ = GetBlockIdx();

    // ub 计算长度
    ubFactor_ = tilingData_->ubFactor;
    // 量化基本块大小 1*512
    blockSizeCol_ = tilingData_->blockSizeCol;
    blockSizeRow_ = tilingData_->blockSizeRow;
    // 行、列方向基本块数
    rowBlockNum_ = tilingData_->rowBlockNum;
    colBlockNum_ = tilingData_->colBlockNum;
    // 行、列方向切分块数
    rowTileNum_ = tilingData_->rowTileNum;
    colTileNum_ = tilingData_->colTileNum;
    // 单行level1ScaleBlock个数，用于偏移计算
    level1BlockRowNum_ = ops::CeilDiv(tilingData_->rowSize, DIGIT_64) * DIGIT_TWO;

    isRowTail_ = blockIdx_ % rowTileNum_ == (rowTileNum_ - 1);
    isColTail_ = blockIdx_ / rowTileNum_ == (colTileNum_ - 1);
    // 当前核计算数据块行、列大小
    if (isRowTail_) {
        curRowBlockNum_ = tilingData_->tailTileRowBlockNum;
        curRowLoopNum_ = tilingData_->tailTileRowLoopNum;
        curRowTileSize_ = tilingData_->tailTileRowSize;
    } else {
        curRowBlockNum_ = tilingData_->normalTileRowBlockNum;
        curRowLoopNum_ = tilingData_->normalTileRowLoopNum;
        curRowTileSize_ = tilingData_->normalTileRowSize;
    }

    if (isColTail_) {
        curColBlockNum_ = tilingData_->tailTileColBlockNum;
        curColLoopNum_ = tilingData_->tailTileColLoopNum;
    } else {
        curColBlockNum_ = tilingData_->normalTileColBlockNum;
        curColLoopNum_ = tilingData_->normalTileColLoopNum;
    }
}

template <typename xDtype, AscendC::RoundMode roundMode>
__aicore__ inline void DynamicDualLevelMxQuantBase<xDtype, roundMode>::Process()
{
    if (blockIdx_ >= tilingData_->usedCoreNum) {
        return;
    }
    // 行列索引
    int64_t colIndex = blockIdx_ / rowTileNum_;
    int64_t rowIndex = blockIdx_ % rowTileNum_;
    // 单次UB处理 数据块个数
    int64_t normalBurst = 0;
    int64_t tailBurst = 0;
    // 单次UB处理 数据块长度
    int64_t normalBlockLen = 0;
    int64_t tailBlockLen = 0;
    // 输入输出gm偏移，y偏移同x
    int64_t xGmOffset = 0;
    int64_t level0ScaleGmOffset = 0;
    int64_t level1ScaleGmOffset = 0;
    // 本核计算起始地址
    int64_t xGmAddr = colIndex * tilingData_->normalTileColBlockNum * tilingData_->rowSize +
                      rowIndex * tilingData_->normalTileRowBlockNum * blockSizeRow_;

    int64_t level0ScaleGmAddr = colIndex * tilingData_->normalTileColBlockNum * tilingData_->rowBlockNum +
                                rowIndex * tilingData_->normalTileRowBlockNum;

    int64_t level1ScaleGmAddr = colIndex * tilingData_->normalTileColBlockNum * level1BlockRowNum_ +
                                rowIndex * ops::CeilDiv(tilingData_->normalTileRowSize, tilingData_->level1BlockSize);

    if (tilingData_->copyMethod) {
        // 单行长度较大，分多次搬入ub计算
        for (int64_t i = 0; i < curColLoopNum_; i++) {
            // 均衡搬入，存在多尾块场景，计算正常块与尾块个数
            int64_t headRowLoopNum =
                curRowBlockNum_ % curRowLoopNum_ == 0 ? curRowLoopNum_ - 1 : curRowBlockNum_ % curRowLoopNum_;
            int64_t tailRowLoopNum =
                curRowBlockNum_ % curRowLoopNum_ == 0 ? 0 : (curRowLoopNum_ - curRowBlockNum_ % curRowLoopNum_ - 1);
            normalBlockLen = ops::CeilDiv(curRowBlockNum_, curRowLoopNum_) * blockSizeRow_;
            tailBlockLen = normalBlockLen - blockSizeRow_;
            // 该场景仅搬一块
            normalBurst = 1;
            for (int64_t j = 0; j < headRowLoopNum; j++) {
                // 正常块场景
                xGmOffset = i * tilingData_->rowSize + j * normalBlockLen;
                level0ScaleGmOffset = i * tilingData_->rowBlockNum + j * (normalBlockLen / blockSizeRow_);
                level1ScaleGmOffset = i * level1BlockRowNum_ + j * (normalBlockLen / DIGIT_32);
                ProcessOneLoop(
                    xGmAddr + xGmOffset, level0ScaleGmAddr + level0ScaleGmOffset,
                    level1ScaleGmAddr + level1ScaleGmOffset, normalBlockLen, normalBurst);
            }
            for (int64_t j = 0; j < tailRowLoopNum; j++) {
                // 对齐尾块场景
                xGmOffset = i * tilingData_->rowSize + headRowLoopNum * normalBlockLen + j * tailBlockLen;
                level0ScaleGmOffset = i * tilingData_->rowBlockNum + headRowLoopNum * (normalBlockLen / blockSizeRow_) +
                                      j * (tailBlockLen / blockSizeRow_);
                level1ScaleGmOffset = i * level1BlockRowNum_ + headRowLoopNum * (normalBlockLen / DIGIT_32) +
                                      j * (tailBlockLen / DIGIT_32);
                ProcessOneLoop(
                    xGmAddr + xGmOffset, level0ScaleGmAddr + level0ScaleGmOffset,
                    level1ScaleGmAddr + level1ScaleGmOffset, tailBlockLen, normalBurst);
            }
            // 非对齐尾块
            xGmOffset = i * tilingData_->rowSize + headRowLoopNum * normalBlockLen + tailRowLoopNum * tailBlockLen;
            level0ScaleGmOffset = i * tilingData_->rowBlockNum + headRowLoopNum * (normalBlockLen / blockSizeRow_) +
                                  tailRowLoopNum * (tailBlockLen / blockSizeRow_);
            level1ScaleGmOffset = i * level1BlockRowNum_ + headRowLoopNum * (normalBlockLen / DIGIT_32) +
                                  tailRowLoopNum * (tailBlockLen / DIGIT_32);
            ProcessOneLoop(
                xGmAddr + xGmOffset, level0ScaleGmAddr + level0ScaleGmOffset, level1ScaleGmAddr + level1ScaleGmOffset,
                curRowTileSize_ - headRowLoopNum * normalBlockLen - tailRowLoopNum * tailBlockLen, normalBurst);
        }
    } else {
        // 单行长度较小，单次多行搬入UB计算
        // 均衡搬入，同上
        int64_t headColLoopNum =
            curColBlockNum_ % curColLoopNum_ == 0 ? curColLoopNum_ : curColBlockNum_ % curColLoopNum_;
        normalBurst = ops::CeilDiv(curColBlockNum_, curColLoopNum_);
        tailBurst = normalBurst - 1;
        // 该场景搬入数据块长度固定
        normalBlockLen = curRowTileSize_;
        for (int64_t i = 0; i < headColLoopNum; i++) {
            // 正常块
            xGmOffset = i * normalBurst * tilingData_->rowSize;
            level0ScaleGmOffset = i * normalBurst * tilingData_->rowBlockNum;
            level1ScaleGmOffset = i * normalBurst * level1BlockRowNum_;
            ProcessOneLoop(
                xGmAddr + xGmOffset, level0ScaleGmAddr + level0ScaleGmOffset, level1ScaleGmAddr + level1ScaleGmOffset,
                normalBlockLen, normalBurst);
        }
        for (int64_t i = headColLoopNum; i < curColLoopNum_; i++) {
            // 尾块
            xGmOffset = headColLoopNum * normalBurst * tilingData_->rowSize +
                        (i - headColLoopNum) * tailBurst * tilingData_->rowSize;
            level0ScaleGmOffset = headColLoopNum * normalBurst * tilingData_->rowBlockNum +
                                  (i - headColLoopNum) * tailBurst * tilingData_->rowBlockNum;
            level1ScaleGmOffset = headColLoopNum * normalBurst * level1BlockRowNum_ +
                                  (i - headColLoopNum) * tailBurst * level1BlockRowNum_;
            ProcessOneLoop(
                xGmAddr + xGmOffset, level0ScaleGmAddr + level0ScaleGmOffset, level1ScaleGmAddr + level1ScaleGmOffset,
                normalBlockLen, tailBurst);
        }
    }
}

template <typename xDtype, AscendC::RoundMode roundMode>
__aicore__ inline void DynamicDualLevelMxQuantBase<xDtype, roundMode>::ProcessOneLoop(
    int64_t xGmAddr, int64_t level0ScaleGmAddr, int64_t level1ScaleGmAddr, int64_t dataLen, int64_t inBurst)
{
    CopyIn(xGmAddr, dataLen, inBurst);
    ComputeAll(dataLen, inBurst);
    CopyOut(xGmAddr, level0ScaleGmAddr, level1ScaleGmAddr, dataLen, inBurst);
}

template <typename xDtype, AscendC::RoundMode roundMode>
__aicore__ inline void DynamicDualLevelMxQuantBase<xDtype, roundMode>::CopyIn(
    int64_t xGmAddr, int64_t dataLen, int64_t inBurst)
{
    int64_t rightPadding =
        ops::CeilAlign(static_cast<int64_t>(dataLen * sizeof(xDtype)), UBBlockSize_) / sizeof(xDtype) - dataLen;
    int64_t srcStride = tilingData_->copyMethod ? 0 : (tilingData_->rowSize - curRowTileSize_);
    int64_t dstStride = (ops::CeilAlign(dataLen, blockSizeRow_) - dataLen) * sizeof(xDtype) / UBBlockSize_;

    DataCopyExtParams copyInParams = {
        static_cast<uint16_t>(inBurst), static_cast<uint32_t>(dataLen * sizeof(xDtype)),
        static_cast<uint32_t>(srcStride * sizeof(xDtype)), static_cast<uint32_t>(dstStride), static_cast<uint32_t>(0)};
    DataCopyPadExtParams<xDtype> padParams{true, 0, static_cast<uint8_t>(rightPadding), 0};

    LocalTensor<xDtype> xLocal = inQueue.template AllocTensor<xDtype>();
    if (isRowTail_ && dataLen % blockSizeRow_ != 0) {
        // 非对齐尾块计算时，提前填充0
        Duplicate<xDtype>(xLocal, static_cast<xDtype>(0), xBufferSize_ / sizeof(xDtype));
        event_t eventIDVToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(eventIDVToMTE2);
        WaitFlag<HardEvent::V_MTE2>(eventIDVToMTE2);
    }
    DataCopyPad(xLocal, xGm_[xGmAddr], copyInParams, padParams);
    inQueue.template EnQue(xLocal);
}

template <typename xDtype, AscendC::RoundMode roundMode>
__aicore__ inline void DynamicDualLevelMxQuantBase<xDtype, roundMode>::ComputeAll(int64_t dataLen, int64_t inBurst)
{
    LocalTensor<xDtype> x = inQueue.template DeQue<xDtype>();
    LocalTensor<float> level0Scale = level0ScaleQueue.template AllocTensor<float>();
    LocalTensor<uint8_t> level1Scale = level1ScaleQueue.template AllocTensor<uint8_t>();
    LocalTensor<uint8_t> y = outQueue.template AllocTensor<uint8_t>();
    LocalTensor<xDtype> xTmp = xTmpBuf.Get<xDtype>();
    LocalTensor<uint16_t> level1ScaleReciprocal = level1ScaleReciprocalBuf.Get<uint16_t>();

    auto xUbAddr = (__ubuf__ xDtype*)x.GetPhyAddr();
    auto yUbAddr = (__ubuf__ uint8_t*)y.GetPhyAddr();
    auto level0ScaleUbAddr = (__ubuf__ float*)level0Scale.GetPhyAddr();
    auto level1ScaleUbAddr = (__ubuf__ uint8_t*)level1Scale.GetPhyAddr();
    auto xTmpUbAddr = (__ubuf__ xDtype*)xTmp.GetPhyAddr();
    auto level1ScaleReciprocalUbAddr = (__ubuf__ uint16_t*)level1ScaleReciprocal.GetPhyAddr();

    int64_t xUbOffset = 0;
    int64_t yUbOffset = 0;
    int64_t level0ScaleUbOffset = 0;
    int64_t level1ScaleUbOffset = 0;
    int64_t level1ScaleReciprocalUbOffset = 0;
    int64_t xTmpUbOffset = 0;

    int64_t rowLoopNum = ops::CeilDiv(dataLen, blockSizeRow_);

    for (int64_t i = 0; i < inBurst; i++) {
        // 以行为单位计算
        xUbOffset = i * ops::CeilAlign(dataLen, blockSizeRow_);
        yUbOffset = xUbOffset / DIGIT_TWO;
        level0ScaleUbOffset = i * ops::CeilAlign(rowLoopNum, DIGIT_EIGHT);
        level1ScaleUbOffset = i * ops::CeilAlign(ops::CeilDiv(dataLen, DIGIT_32), DIGIT_32);
        level1ScaleReciprocalUbOffset = i * rowLoopNum * UBBlockSize_ / sizeof(bfloat16_t) * DIGIT_TWO;
        xTmpUbOffset = xUbOffset;
        ComputeLevel0Quant(
            rowLoopNum, xUbAddr + xUbOffset, xTmpUbAddr + xTmpUbOffset,
            level0ScaleUbAddr + level0ScaleUbOffset); // 512
        ComputeLevel1Scale(
            rowLoopNum * DIGIT_TWO, xTmpUbAddr + xTmpUbOffset, level1ScaleUbAddr + level1ScaleUbOffset,
            level1ScaleReciprocalUbAddr + level1ScaleReciprocalUbOffset); // 256
        ComputeY(
            rowLoopNum * DIGIT_TWO, xTmpUbAddr + xTmpUbOffset, yUbAddr + yUbOffset,
            level1ScaleReciprocalUbAddr + level1ScaleReciprocalUbOffset); // 256
    }

    level0ScaleQueue.template EnQue(level0Scale);
    level1ScaleQueue.template EnQue(level1Scale);
    outQueue.template EnQue(y);
    inQueue.template FreeTensor(x);
}

template <typename xDtype, AscendC::RoundMode roundMode>
__aicore__ inline void DynamicDualLevelMxQuantBase<xDtype, roundMode>::ComputeLevel0Quant(
    int64_t loopNum, __ubuf__ xDtype* xUbAddr, __ubuf__ xDtype* xTmpUbAddr, __ubuf__ float* level0ScaleUbAddr)
{
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<xDtype> x0;
        MicroAPI::RegTensor<xDtype> x1;
        MicroAPI::RegTensor<xDtype> x2;
        MicroAPI::RegTensor<xDtype> x3;
        MicroAPI::RegTensor<uint16_t> absX0;
        MicroAPI::RegTensor<uint16_t> absX1;
        MicroAPI::RegTensor<uint16_t> absX2;
        MicroAPI::RegTensor<uint16_t> absX3;
        MicroAPI::RegTensor<float> level0Scale;
        MicroAPI::RegTensor<float> x0ZeroFP32;
        MicroAPI::RegTensor<float> x0OneFP32;
        MicroAPI::RegTensor<float> x1ZeroFP32;
        MicroAPI::RegTensor<float> x1OneFP32;
        MicroAPI::RegTensor<float> x2ZeroFP32;
        MicroAPI::RegTensor<float> x2OneFP32;
        MicroAPI::RegTensor<float> x3ZeroFP32;
        MicroAPI::RegTensor<float> x3OneFP32;

        MicroAPI::RegTensor<uint32_t> yMaxExp;
        MicroAPI::RegTensor<uint32_t> invaildData;
        MicroAPI::RegTensor<uint16_t> absForX;
        MicroAPI::RegTensor<uint16_t> infForX;
        MicroAPI::RegTensor<uint16_t> zero;
        MicroAPI::MaskReg infMask;
        MicroAPI::MaskReg invalidDataMask;
        MicroAPI::MaskReg maskAll16 = MicroAPI::CreateMask<uint16_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg maskAll32 = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
        MicroAPI::UnalignRegForStore ureg;

        MicroAPI::Duplicate(yMaxExp, FP4_E2M1_MAX);
        MicroAPI::Duplicate(absForX, ABS_FOR_UINT16);
        MicroAPI::Duplicate(infForX, INF_FOR_BF16);
        MicroAPI::Duplicate(invaildData, INVAILD_FOR_FP32);
        MicroAPI::Duplicate(zero, 0);
        if constexpr (IsSameType<xDtype, half>::value) {
            MicroAPI::Duplicate(infForX, INF_FOR_FP16);
        } else {
            MicroAPI::Duplicate(infForX, INF_FOR_BF16);
        }

        for (uint16_t j = 0; j < static_cast<uint16_t>(loopNum); j++) {
            MicroAPI::DataCopy<xDtype, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_NORM>(
                x0, xUbAddr, vlForHalfNumber_);
            MicroAPI::DataCopy<xDtype, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_NORM>(
                x1, xUbAddr, vlForHalfNumber_);
            MicroAPI::DataCopy<xDtype, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_NORM>(
                x2, xUbAddr, vlForHalfNumber_);
            MicroAPI::DataCopy<xDtype, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_NORM>(
                x3, xUbAddr, vlForHalfNumber_);

            MicroAPI::And(absX0, (AscendC::MicroAPI::RegTensor<uint16_t>&)x0, absForX, maskAll16);
            MicroAPI::And(absX1, (AscendC::MicroAPI::RegTensor<uint16_t>&)x1, absForX, maskAll16);
            MicroAPI::And(absX2, (AscendC::MicroAPI::RegTensor<uint16_t>&)x2, absForX, maskAll16);
            MicroAPI::And(absX3, (AscendC::MicroAPI::RegTensor<uint16_t>&)x3, absForX, maskAll16);
            // inf/nan不参与计算，将对应位置填充0
            MicroAPI::Compare<uint16_t, CMPMODE::GE>(infMask, absX0, infForX, maskAll16);
            MicroAPI::Select<uint16_t>(absX0, zero, absX0, infMask);
            MicroAPI::Compare<uint16_t, CMPMODE::GE>(infMask, absX1, infForX, maskAll16);
            MicroAPI::Select<uint16_t>(absX1, zero, absX1, infMask);
            MicroAPI::Compare<uint16_t, CMPMODE::GE>(infMask, absX2, infForX, maskAll16);
            MicroAPI::Select<uint16_t>(absX2, zero, absX2, infMask);
            MicroAPI::Compare<uint16_t, CMPMODE::GE>(infMask, absX3, infForX, maskAll16);
            MicroAPI::Select<uint16_t>(absX3, zero, absX3, infMask);

            MicroAPI::Max(absX0, absX0, absX1, maskAll16);
            MicroAPI::Max(absX2, absX2, absX3, maskAll16);
            MicroAPI::Max(absX0, absX0, absX2, maskAll16);
            MicroAPI::ReduceMax(absX0, absX0, maskAll16);

            MicroAPI::Cast<float, xDtype, castTraitXdtypetoFp32Zero>(
                level0Scale, (AscendC::MicroAPI::RegTensor<xDtype>&)absX0, maskAll16);
            MicroAPI::Mul(level0Scale, level0Scale, (AscendC::MicroAPI::RegTensor<float>&)yMaxExp, maskAll32);
            // 当level0_scale是subnormal时，直接赋0
            MicroAPI::Compare<uint32_t, CMPMODE::LT>(
                invalidDataMask, (AscendC::MicroAPI::RegTensor<uint32_t>&)level0Scale, invaildData, maskAll32);
            MicroAPI::Select<float>(
                level0Scale, (AscendC::MicroAPI::RegTensor<float>&)(zero), level0Scale, invalidDataMask);

            // 输出1个值,非对齐搬出
            MicroAPI::StoreUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                level0ScaleUbAddr, level0Scale, ureg, static_cast<uint32_t>(1));

            MicroAPI::Duplicate(level0Scale, level0Scale, maskAll32);

            // fp32类型，一次计算128个元素，共四次
            CalcXTmp(xTmpUbAddr + j * blockSizeRow_, level0Scale, x0, x0ZeroFP32, x0OneFP32);
            CalcXTmp(xTmpUbAddr + j * blockSizeRow_ + DIGIT_128, level0Scale, x1, x1ZeroFP32, x1OneFP32);
            CalcXTmp(xTmpUbAddr + j * blockSizeRow_ + DIGIT_256, level0Scale, x2, x2ZeroFP32, x2OneFP32);
            CalcXTmp(xTmpUbAddr + j * blockSizeRow_ + DIGIT_384, level0Scale, x3, x3ZeroFP32, x3OneFP32);
        }
        MicroAPI::StoreUnAlignPost<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            level0ScaleUbAddr, ureg, static_cast<int32_t>(0));
    }
}

template <typename xDtype, AscendC::RoundMode roundMode>
__aicore__ inline void DynamicDualLevelMxQuantBase<xDtype, roundMode>::CalcXTmp(
    __ubuf__ xDtype* xTmpUbAddr, MicroAPI::RegTensor<float> level0ScaleReg, MicroAPI::RegTensor<xDtype> xReg,
    MicroAPI::RegTensor<float>& xZeroFP32, MicroAPI::RegTensor<float>& xOneFP32)
{
    MicroAPI::RegTensor<xDtype> xZero;
    MicroAPI::RegTensor<xDtype> xOne;
    MicroAPI::RegTensor<xDtype> zero;

    MicroAPI::MaskReg zeroMask;
    MicroAPI::MaskReg maskAll16 = MicroAPI::CreateMask<xDtype, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg maskAll32 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::Duplicate(zero, 0);
    // 输入是0，输出直接设置0，否则出现 0/0=nan
    MicroAPI::Compare<xDtype, CMPMODE::EQ>(zeroMask, xReg, zero, maskAll16);
    // cast to float for blockQuant
    MicroAPI::Cast<float, xDtype, castTraitXdtypetoFp32Zero>(xZeroFP32, xReg, maskAll16);
    MicroAPI::Cast<float, xDtype, castTraitXdtypetoFp32One>(xOneFP32, xReg, maskAll16);

    MicroAPI::Div(xZeroFP32, xZeroFP32, level0ScaleReg, maskAll32);
    MicroAPI::Div(xOneFP32, xOneFP32, level0ScaleReg, maskAll32);
    // cast to xDtype for mxQuant
    MicroAPI::Cast<xDtype, float, castTraitFp32toBF16>(xZero, xZeroFP32, maskAll32);
    MicroAPI::Cast<xDtype, float, castTraitFp32toBF16>(xOne, xOneFP32, maskAll32);

    MicroAPI::Pack<uint16_t, uint32_t, MicroAPI::HighLowPart::LOWEST>(
        (MicroAPI::RegTensor<uint16_t>&)xZero, (MicroAPI::RegTensor<uint32_t>&)xZero);
    MicroAPI::Pack<uint16_t, uint32_t, MicroAPI::HighLowPart::LOWEST>(
        (MicroAPI::RegTensor<uint16_t>&)xOne, (MicroAPI::RegTensor<uint32_t>&)xOne);

    MicroAPI::Interleave(xZero, xOne, xZero, xOne);
    MicroAPI::Select<xDtype>(xZero, xReg, xZero, zeroMask);

    // 连续搬出128个float32的xTmp
    MicroAPI::DataCopy(xTmpUbAddr, xZero, maskAll16);
}

template <typename xDtype, AscendC::RoundMode roundMode>
__aicore__ inline void DynamicDualLevelMxQuantBase<xDtype, roundMode>::ComputeLevel1Scale(
    int64_t loopNum, __ubuf__ xDtype* xTmpUbAddr, __ubuf__ uint8_t* level1ScaleUbAddr,
    __ubuf__ uint16_t* level1ScaleReciprocalUbAddr)
{
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<xDtype> xTmp0;
        MicroAPI::RegTensor<xDtype> xTmp1;
        MicroAPI::RegTensor<bfloat16_t> xTmp0BF16;
        MicroAPI::RegTensor<bfloat16_t> xTmp1BF16;
        MicroAPI::RegTensor<uint16_t> xTmp0ExpBF16;
        MicroAPI::RegTensor<uint16_t> xTmp1ExpBF16;
        MicroAPI::RegTensor<uint16_t> xTmp0ExpFP16;
        MicroAPI::RegTensor<uint16_t> xTmp1ExpFP16;

        MicroAPI::RegTensor<uint16_t> expMaskBF16;
        MicroAPI::RegTensor<uint16_t> expMaskFP16;
        MicroAPI::RegTensor<uint16_t> yMaxExp;
        MicroAPI::RegTensor<uint16_t> nanE8M0;
        MicroAPI::RegTensor<uint16_t> biasE8M0;
        MicroAPI::RegTensor<uint16_t> zero;
        MicroAPI::RegTensor<uint16_t> nanBF16;
        MicroAPI::RegTensor<uint16_t> specialExp;
        MicroAPI::RegTensor<uint16_t> mxScale1B16;
        MicroAPI::RegTensor<uint8_t> mxScale1B8;
        MicroAPI::RegTensor<uint16_t> reversedShareExp1;

        MicroAPI::MaskReg infMask;
        MicroAPI::MaskReg zeroMask;
        MicroAPI::MaskReg invalidDataMask;
        MicroAPI::MaskReg infNanDataMask0;
        MicroAPI::MaskReg infNanDataMask1;
        MicroAPI::MaskReg maskAll16 = MicroAPI::CreateMask<uint16_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg maskAll32 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg maskReduceB8 = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::VL8>();
        MicroAPI::MaskReg maskReduceB16 = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::VL16>();

        MicroAPI::Duplicate(expMaskBF16, INF_FOR_BF16);
        MicroAPI::Duplicate(expMaskFP16, INF_FOR_FP16);
        MicroAPI::Duplicate(yMaxExp, FP4_E2M1_BF16_MAX_EXP);
        MicroAPI::Duplicate(nanE8M0, NAN_FOR_FP8_E8M0);
        MicroAPI::Duplicate(biasE8M0, BF16_EXP_BIAS);
        MicroAPI::Duplicate(zero, 0);
        MicroAPI::Duplicate(nanBF16, NAN_CUSTOMIZATION);
        MicroAPI::Duplicate(specialExp, SPECIAL_EXP_THRESHOLD);

        for (uint16_t i = 0; i < static_cast<uint16_t>(loopNum); i++) {
            // 交织搬运，一次搬256个B16
            MicroAPI::DataCopy<xDtype, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_DINTLV_B16>(
                xTmp0, xTmp1, xTmpUbAddr, vlForHalfNumber_ * DIGIT_TWO);

            if constexpr (IsSameType<xDtype, half>::value) {
                // 提取指数位
                MicroAPI::And(xTmp0ExpFP16, (MicroAPI::RegTensor<uint16_t>&)xTmp0, expMaskFP16, maskAll16);
                MicroAPI::And(xTmp1ExpFP16, (MicroAPI::RegTensor<uint16_t>&)xTmp1, expMaskFP16, maskAll16);
                // 比较INF/NAN数据
                MicroAPI::Compare<uint16_t, CMPMODE::NE>(infNanDataMask0, xTmp0ExpFP16, expMaskFP16, maskAll16);
                MicroAPI::Compare<uint16_t, CMPMODE::NE>(infNanDataMask1, xTmp1ExpFP16, expMaskFP16, maskAll16);
                // 原始数据转成bf16
                MicroAPI::Cast<bfloat16_t, xDtype, castTraitHalf2BF16>(xTmp0BF16, xTmp0, maskAll16);
                MicroAPI::Cast<bfloat16_t, xDtype, castTraitHalf2BF16>(xTmp1BF16, xTmp1, maskAll16);
                // 提取指数位
                MicroAPI::And(xTmp0ExpBF16, (MicroAPI::RegTensor<uint16_t>&)xTmp0BF16, expMaskBF16, maskAll16);
                MicroAPI::And(xTmp1ExpBF16, (MicroAPI::RegTensor<uint16_t>&)xTmp1BF16, expMaskBF16, maskAll16);
                // 选择数据，INF/NAN数据时设成BF的INF/NAN
                MicroAPI::Select<uint16_t>(xTmp0ExpBF16, xTmp0ExpBF16, expMaskBF16, infNanDataMask0);
                MicroAPI::Select<uint16_t>(xTmp1ExpBF16, xTmp1ExpBF16, expMaskBF16, infNanDataMask1);
            } else {
                // 提取指数位
                MicroAPI::And(xTmp0ExpBF16, (MicroAPI::RegTensor<uint16_t>&)xTmp0, expMaskBF16, maskAll16);
                MicroAPI::And(xTmp1ExpBF16, (MicroAPI::RegTensor<uint16_t>&)xTmp1, expMaskBF16, maskAll16);
            }
            // 计算奇偶位置最大值，相当于计算原始相邻两个数据的最大值
            MicroAPI::Max(xTmp0ExpBF16, xTmp1ExpBF16, xTmp0ExpBF16, maskAll16);
            // ReduceMax一个block，即16个数，配合上一步，可以计算出每32个数的最大值，一共256/32个
            MicroAPI::ReduceMaxWithDataBlock(xTmp0ExpBF16, xTmp0ExpBF16, maskAll16);

            // 计算-1轴的scale和1/scale
            // inf/nan值单独处理，结果为E8M0的nan
            MicroAPI::Compare<uint16_t, CMPMODE::NE>(infMask, xTmp0ExpBF16, expMaskBF16, maskAll16);
            // 0值单独处理，结果为0
            MicroAPI::Compare<uint16_t, CMPMODE::NE>(zeroMask, xTmp0ExpBF16, zero, maskAll16);
            // 指数位不足被量化类型的ele_max时，为subnormal场景，结果为0
            MicroAPI::Compare<uint16_t, CMPMODE::LE>(invalidDataMask, xTmp0ExpBF16, yMaxExp, maskAll16);
            MicroAPI::Select<uint16_t>(xTmp0ExpBF16, yMaxExp, xTmp0ExpBF16, invalidDataMask);
            // 指数位减去expMax，按照BF16的格式处理，例：E2M1的expMax为2，即需要减去0 000000010 0000000
            MicroAPI::Sub(xTmp0ExpBF16, xTmp0ExpBF16, yMaxExp, maskAll16);
            // 右移7位，BF16的指数位移到了末8位
            MicroAPI::ShiftRights(mxScale1B16, xTmp0ExpBF16, SHR_NUM_FOR_BF16, maskAll16);
            MicroAPI::Select<uint16_t>(mxScale1B16, mxScale1B16, nanE8M0, infMask);
            MicroAPI::Select<uint16_t>(mxScale1B16, mxScale1B16, zero, zeroMask);
            // 搬出 256/32 个scale(uint8_t)
            MicroAPI::Pack<uint8_t, uint16_t, MicroAPI::HighLowPart::LOWEST>(mxScale1B8, mxScale1B16);

            // 采用非对齐搬出
            MicroAPI::UnalignRegForStore ureg;
            MicroAPI::StoreUnAlign<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                level1ScaleUbAddr, mxScale1B8, ureg, static_cast<uint32_t>(DIGIT_EIGHT));
            MicroAPI::StoreUnAlignPost<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                level1ScaleUbAddr, ureg, static_cast<int32_t>(0));

            // 公式中的1/X
            // 只有在E1M2时，yMaxExp=0，xTmp0ExpBF16可能会等于biasE8M0
            MicroAPI::Compare<uint16_t, CMPMODE::EQ>(invalidDataMask, xTmp0ExpBF16, biasE8M0, maskAll16);
            MicroAPI::Sub(reversedShareExp1, biasE8M0, xTmp0ExpBF16, maskAll16);
            MicroAPI::Select<uint16_t>(reversedShareExp1, reversedShareExp1, nanBF16, infMask);
            MicroAPI::Select<uint16_t>(reversedShareExp1, reversedShareExp1, zero, zeroMask);
            MicroAPI::Select<uint16_t>(reversedShareExp1, specialExp, reversedShareExp1, invalidDataMask);
            // 搬出8位1/scale，占一个UBBlock方便后续取值计算
            MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                level1ScaleReciprocalUbAddr, reversedShareExp1, UBBlockSize_ / sizeof(uint16_t), maskReduceB16);
        }
    }
}

template <typename xDtype, AscendC::RoundMode roundMode>
__aicore__ inline void DynamicDualLevelMxQuantBase<xDtype, roundMode>::ComputeY(
    int64_t loopNum, __ubuf__ xDtype* xTmpUbAddr, __ubuf__ uint8_t* yAddr,
    __ubuf__ uint16_t* level1ScaleReciprocalUbAddr)
{
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<xDtype> xTmp0;
        MicroAPI::RegTensor<xDtype> xTmp1;
        MicroAPI::RegTensor<uint16_t> scaleForMulFP16;
        MicroAPI::RegTensor<float> scaleForMulZeroFP32;
        MicroAPI::RegTensor<fp4x2_e2m1_t> y0FP4;
        MicroAPI::RegTensor<fp4x2_e2m1_t> y1FP4;

        MicroAPI::RegTensor<float> xTmp0ZeroFP32;
        MicroAPI::RegTensor<float> xTmp0OneFP32;
        MicroAPI::RegTensor<float> xTmp1ZeroFP32;
        MicroAPI::RegTensor<float> xTmp1OneFP32;

        MicroAPI::RegTensor<bfloat16_t> xTmp0ZeroBF16;
        MicroAPI::RegTensor<bfloat16_t> xTmp0OneBF16;
        MicroAPI::RegTensor<bfloat16_t> xTmp1ZeroBF16;
        MicroAPI::RegTensor<bfloat16_t> xTmp1OneBF16;

        MicroAPI::MaskReg dataMaskB8 = MicroAPI::CreateMask<uint8_t>();
        MicroAPI::MaskReg dataMaskB16 = MicroAPI::CreateMask<uint16_t>();
        MicroAPI::MaskReg dataMaskB32 = MicroAPI::CreateMask<uint32_t>();

        for (uint16_t i = 0; i < static_cast<uint16_t>(loopNum); i++) {
            // 搬入8个uint16_t元素,单个元素广播到一个UBBlock中
            MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_E2B_B16>(
                scaleForMulFP16, level1ScaleReciprocalUbAddr, UBBlockSize_ / sizeof(uint16_t));

            // 交织搬入256个xTmp(bfloat16_t)
            MicroAPI::DataCopy<xDtype, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_DINTLV_B16>(
                xTmp0, xTmp1, xTmpUbAddr, vlForHalfNumber_ * DIGIT_TWO);

            if constexpr (IsSameType<xDtype, half>::value) {
                MicroAPI::Cast<float, bfloat16_t, castTraitXdtypetoFp32Zero>(
                    scaleForMulZeroFP32, (MicroAPI::RegTensor<bfloat16_t>&)scaleForMulFP16, dataMaskB16);

                // x0 cast to bf16
                MicroAPI::Cast<float, xDtype, castTraitXdtypetoFp32Zero>(xTmp0ZeroFP32, xTmp0, dataMaskB16);
                MicroAPI::Cast<float, xDtype, castTraitXdtypetoFp32One>(xTmp0OneFP32, xTmp0, dataMaskB16);

                MicroAPI::Mul(xTmp0ZeroFP32, scaleForMulZeroFP32, xTmp0ZeroFP32, dataMaskB32);
                MicroAPI::Mul(xTmp0OneFP32, scaleForMulZeroFP32, xTmp0OneFP32, dataMaskB32);
                ComputeFP4FromHalf(xTmp0ZeroFP32);
                ComputeFP4FromHalf(xTmp0OneFP32);
                MicroAPI::Cast<bfloat16_t, float, castTraitFp32toBF16>(xTmp0ZeroBF16, xTmp0ZeroFP32, dataMaskB32);
                MicroAPI::Cast<bfloat16_t, float, castTraitFp32toBF16>(xTmp0OneBF16, xTmp0OneFP32, dataMaskB32);
                MicroAPI::Pack<uint16_t, uint32_t, MicroAPI::HighLowPart::LOWEST>(
                    (MicroAPI::RegTensor<uint16_t>&)xTmp0ZeroBF16, (MicroAPI::RegTensor<uint32_t>&)xTmp0ZeroBF16);
                MicroAPI::Pack<uint16_t, uint32_t, MicroAPI::HighLowPart::LOWEST>(
                    (MicroAPI::RegTensor<uint16_t>&)xTmp0OneBF16, (MicroAPI::RegTensor<uint32_t>&)xTmp0OneBF16);
                MicroAPI::Interleave(xTmp0ZeroBF16, xTmp0OneBF16, xTmp0ZeroBF16, xTmp0OneBF16);

                // x1 cast to bf16
                MicroAPI::Cast<float, xDtype, castTraitXdtypetoFp32Zero>(xTmp1ZeroFP32, xTmp1, dataMaskB16);
                MicroAPI::Cast<float, xDtype, castTraitXdtypetoFp32One>(xTmp1OneFP32, xTmp1, dataMaskB16);

                MicroAPI::Mul(xTmp1ZeroFP32, scaleForMulZeroFP32, xTmp1ZeroFP32, dataMaskB32);
                MicroAPI::Mul(xTmp1OneFP32, scaleForMulZeroFP32, xTmp1OneFP32, dataMaskB32);
                ComputeFP4FromHalf(xTmp1ZeroFP32);
                ComputeFP4FromHalf(xTmp1OneFP32);
                MicroAPI::Cast<bfloat16_t, float, castTraitFp32toBF16>(xTmp1ZeroBF16, xTmp1ZeroFP32, dataMaskB32);
                MicroAPI::Cast<bfloat16_t, float, castTraitFp32toBF16>(xTmp1OneBF16, xTmp1OneFP32, dataMaskB32);
                MicroAPI::Pack<uint16_t, uint32_t, MicroAPI::HighLowPart::LOWEST>(
                    (MicroAPI::RegTensor<uint16_t>&)xTmp1ZeroBF16, (MicroAPI::RegTensor<uint32_t>&)xTmp1ZeroBF16);
                MicroAPI::Pack<uint16_t, uint32_t, MicroAPI::HighLowPart::LOWEST>(
                    (MicroAPI::RegTensor<uint16_t>&)xTmp1OneBF16, (MicroAPI::RegTensor<uint32_t>&)xTmp1OneBF16);
                MicroAPI::Interleave(xTmp1ZeroBF16, xTmp1OneBF16, xTmp1ZeroBF16, xTmp1OneBF16);

                // interleave x0 and x1
                MicroAPI::Interleave(xTmp0ZeroBF16, xTmp1ZeroBF16, xTmp0ZeroBF16, xTmp1ZeroBF16);
                MicroAPI::Cast<fp4x2_e2m1_t, bfloat16_t, castTraitBF16toFp4>(y0FP4, xTmp0ZeroBF16, dataMaskB16);
                MicroAPI::Cast<fp4x2_e2m1_t, bfloat16_t, castTraitBF16toFp4>(y1FP4, xTmp1ZeroBF16, dataMaskB16);
            } else {
                MicroAPI::Mul(xTmp0, (MicroAPI::RegTensor<bfloat16_t>&)scaleForMulFP16, xTmp0, dataMaskB16);
                MicroAPI::Mul(xTmp1, (MicroAPI::RegTensor<bfloat16_t>&)scaleForMulFP16, xTmp1, dataMaskB16);

                MicroAPI::Interleave(xTmp0, xTmp1, xTmp0, xTmp1);

                MicroAPI::Cast<fp4x2_e2m1_t, bfloat16_t, castTraitBF16toFp4>(y0FP4, xTmp0, dataMaskB16);
                MicroAPI::Cast<fp4x2_e2m1_t, bfloat16_t, castTraitBF16toFp4>(y1FP4, xTmp1, dataMaskB16);
            }

            // 256个fp4元素作为128个uint8元素搬出
            MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::StoreDist::DIST_PACK4_B32>(
                yAddr, (MicroAPI::RegTensor<uint8_t>&)y0FP4, DIGIT_64, dataMaskB8);
            MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::StoreDist::DIST_PACK4_B32>(
                yAddr, (MicroAPI::RegTensor<uint8_t>&)y1FP4, DIGIT_64, dataMaskB8);
        }
    }
    return;
}

template <typename xDtype, AscendC::RoundMode roundMode>
__aicore__ inline void DynamicDualLevelMxQuantBase<xDtype, roundMode>::ComputeFP4FromHalf(
    MicroAPI::RegTensor<float>& Reg)
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

    // fp4x2_e2m1
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
    MicroAPI::CompareScalar<float, CMPMODE::LT>(specialMask, Reg, 0, pregAll32);
    MicroAPI::Truncate<float, roundMode>(Reg, Reg, pregAll32);
    MicroAPI::Mul(Reg, Reg, (MicroAPI::RegTensor<float>&)exp0FP32, pregAll32);

    MicroAPI::CompareScalar<float, CMPMODE::EQ>(zeroMask, Reg, 0, pregAll32);
    MicroAPI::MaskAnd(zeroMask, specialMask, zeroMask, pregAll32);
    MicroAPI::MaskOr(zeroMask, negInfMask, zeroMask, pregAll32);
    MicroAPI::Select<int32_t>(
        (MicroAPI::RegTensor<int32_t>&)Reg, negZero, (MicroAPI::RegTensor<int32_t>&)Reg, zeroMask);
}

template <typename xDtype, AscendC::RoundMode roundMode>
__aicore__ inline void DynamicDualLevelMxQuantBase<xDtype, roundMode>::CopyOut(
    int64_t yGmAddr, int64_t level0ScaleGmAddr, int64_t level1ScaleGmAddr, int64_t dataLen, int64_t outBurst)
{
    int64_t dstStrideY = 0;
    int64_t dstStrideLevel0Scale = 0;
    int64_t dstStrideLevel1Scale = 0;
    int64_t srcStrideY = (ops::CeilAlign(dataLen, blockSizeRow_) - dataLen) / DIGIT_TWO / UBBlockSize_;
    if (!tilingData_->copyMethod) {
        // 多行搬入，需跳搬，间隔不为0
        dstStrideY = (tilingData_->rowSize - dataLen) / DIGIT_TWO;
        dstStrideLevel0Scale = (tilingData_->rowBlockNum - ops::CeilDiv(dataLen, blockSizeRow_));
        dstStrideLevel1Scale = (level1BlockRowNum_ - ops::CeilDiv(dataLen, DIGIT_64) * DIGIT_TWO);
    }

    DataCopyExtParams yCopyOutParams = {
        static_cast<uint16_t>(outBurst), static_cast<uint32_t>(dataLen / 2), static_cast<uint32_t>(srcStrideY),
        static_cast<uint32_t>(dstStrideY), static_cast<uint32_t>(0)};

    DataCopyExtParams level0ScaleCopyOutParams = {
        static_cast<uint16_t>(outBurst), static_cast<uint32_t>(ops::CeilDiv(dataLen, blockSizeRow_) * sizeof(float)),
        static_cast<uint32_t>(0), static_cast<uint32_t>(dstStrideLevel0Scale * sizeof(float)),
        static_cast<uint32_t>(0)};

    DataCopyExtParams level1ScaleCopyOutParams = {
        static_cast<uint16_t>(outBurst), static_cast<uint32_t>(ops::CeilDiv(dataLen, DIGIT_64) * DIGIT_TWO),
        static_cast<uint32_t>(0), static_cast<uint32_t>(dstStrideLevel1Scale), static_cast<uint32_t>(0)};

    LocalTensor<uint8_t> yLocal = outQueue.template DeQue<uint8_t>();
    DataCopyPad(yGm_[yGmAddr / 2], yLocal, yCopyOutParams);
    outQueue.FreeTensor(yLocal);

    LocalTensor<float> level0ScaleLocal = level0ScaleQueue.template DeQue<float>();
    DataCopyPad(level0ScaleGm_[level0ScaleGmAddr], level0ScaleLocal, level0ScaleCopyOutParams);
    level0ScaleQueue.FreeTensor(level0ScaleLocal);

    LocalTensor<uint8_t> level1ScaleLocal = level1ScaleQueue.template DeQue<uint8_t>();
    DataCopyPad(level1ScaleGm_[level1ScaleGmAddr], level1ScaleLocal, level1ScaleCopyOutParams);
    level1ScaleQueue.FreeTensor(level1ScaleLocal);
}

} // namespace DynamicDualLevelMxQuant
#endif // DYNAMIC_DUAL_LEVEL_MX_QUANT_BASE_H
