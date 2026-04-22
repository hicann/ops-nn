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
 * \file dynamic_mx_quant_with_dual_axis_base.h
 * \brief
 */

#ifndef OPS_NN_DEV_DYNAMIC_MX_QUANT_WITH_DUAL_AXIS_H
#define OPS_NN_DEV_DYNAMIC_MX_QUANT_WITH_DUAL_AXIS_H

#define FLOAT_OVERFLOW_MODE_CTRL 60

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../inc/platform.h"
#include "../inc/kernel_utils.h"
#include "dynamic_mx_quant_with_dual_axis_struct.h"
#include "dynamic_mx_quant_with_dual_axis_tilingdata.h"

namespace DynamicMxQuantWithDualAxis {
using namespace AscendC;

constexpr int64_t DB_BUFFER = 2;
constexpr int64_t DIGIT_TWO = 2;
constexpr int64_t OUT_ELE_NUM_ONE_BLK = 64;
constexpr uint16_t NAN_CUSTOMIZATION = 0x7f81; // 0111 1111 1000 0001
constexpr uint32_t NAN_CUSTOMIZATION_FP32 = 0x7f810000;

constexpr uint32_t MAX_EXP_FOR_FP32 = 0x7f800000;
constexpr uint16_t NAN_FOR_FP8_E8M0 = 0x00ff; // 0000 0000 1111 1111
constexpr uint16_t SPECIAL_VALUE_E2M1 = 0x00ff;
constexpr uint16_t SPECIAL_VALUE_E1M2 = 0x007f;
constexpr uint16_t NEW_MANTISSA = 0x0008;
constexpr uint16_t SPECIAL_EXP_THRESHOLD = 0x0040; // 0000 0000 0100 0000
constexpr int16_t SHR_NUM_FOR_BF16 = 7;
constexpr int16_t SHR_NUM_FOR_FP32 = 23;
constexpr uint16_t FP4_E2M1_BF16_MAX_EXP = 0x0100;
constexpr uint16_t BF16_EXP_BIAS = 0x7f00; // 0111 1111 0000 0000
constexpr int64_t MODE_ROUND = 0;
constexpr int64_t MODE_FLOOR = 1;
constexpr int64_t MODE_RINT = 4;
constexpr uint16_t FP8_E4M3_MAX_EXP = 0x0400; // elem_emax右移7位(BF16E8M7) 0 00001000 0000000
constexpr uint16_t FP8_E5M2_MAX_EXP = 0x0780; // 0 00001111 0000000
constexpr int32_t FP32_BIAS = 127;
constexpr int32_t FP32_BIAS_NEG = -127;
constexpr int32_t NEG_ONE = -1;
constexpr float FOUR = 4.0;
constexpr float ONE_FOURTH = 0.25;
constexpr int32_t NEG_ZERO = 0x80000000;
constexpr uint32_t FP8_E5M2_MAX = 0x37924925; // 1/57344的float32表示 57334是E5M2所能表示的最大值
constexpr uint32_t FP8_E4M3_MAX = 0x3b124925; // 1/448的float32表示 448是E4M3所能表示的最大值
constexpr uint16_t EXP_MASK_BF16 = 0x7f80;    // 0111 1111 1000 0000
constexpr uint16_t EXP_MASK_FP16 = 0x7c00;    // 0111 1100 0000 0000

// CuBALS Scale算法相关常量 (scaleAlg=1, FP8专用)
constexpr uint16_t ABS_MASK_FOR_16BIT = 0x7fff;          // 取绝对值掩码，清除符号位
constexpr uint32_t MAN_MASK_FLOAT = 0x007fffff;          // FP32尾数掩码 (23位尾数)
constexpr uint32_t FP32_EXP_BIAS_CUBLAS = 0x00007f00;    // FP32指数偏移(CuBALS)，左移7位后为BF16的指数偏移
constexpr uint32_t MAX_EXP_FOR_FP8_IN_FP32 = 0x000000ff; // FP8 NAN在E8M0中的表示 (0xFF)
constexpr uint32_t NAN_CUSTOMIZATION_PACK = 0x00007f81;   // NAN的BF16打包表示 (uint32存储)
constexpr uint32_t NUMBER_ZERO_U32 = 0x00000000;          // uint32零常量
constexpr uint32_t NUMBER_TWO_FIVE_FOUR = 0x000000fe;     // 254，FP32指数上界
constexpr uint32_t NUMBER_HALF_U32 = 0x00400000;          // FP32尾数的一半 (2^22)

// DynamicDtypeRange Scale算法相关常量 (scaleAlg=2, FP4_E2M1专用)
constexpr uint16_t ADD_VALUE_FOR_BF16_MAN1 = 0x003f;     // dstTypeMax=0.0/6.0时BF16尾数进位值
constexpr uint16_t ADD_VALUE_FOR_BF16_MAN2 = 0x001f;     // dstTypeMax=7.0时BF16尾数进位值
constexpr uint16_t SUB_NUM_FOR_SCALE_6 = 0x00c1;         // dstTypeMax=0.0/6.0时-2轴减法常量 (FP4_E2M1_BF16_MAX_EXP - addValueBit)
constexpr uint16_t SUB_NUM_FOR_SCALE_7 = 0x00e1;         // dstTypeMax=7.0时-2轴减法常量
constexpr float DIGIT_ZERO_FLOAT = 0.0f;
constexpr float DIGIT_SIX_FLOAT = 6.0f;
constexpr float DIGIT_SEVEN_FLOAT = 7.0f;

template <typename xDtype, typename y1Dtype, typename y2Dtype, AscendC::RoundMode roundMode, uint64_t scaleAlg>
class DynamicMxQuantWithDualAxisBase {
public:
    __aicore__ inline DynamicMxQuantWithDualAxisBase(
        const DynamicMxQuantWithDualAxisTilingData* tilingData, TPipe* pipe)
        : tilingData_(tilingData), pipe_(pipe){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y1, GM_ADDR mxScale1, GM_ADDR y2, GM_ADDR mxScale2);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitParams();
    __aicore__ inline void ProcessOneLoop(
        int64_t calcCol, int64_t calcRow, int64_t xUbOffset, int64_t scale1Offset, int64_t scale2Offset,
        int64_t dimNeg1IsOdd);
    __aicore__ inline void CopyOut(
        int64_t yOffset, int64_t scale1OutOffset, int64_t scale2OutOffset, int64_t blockCount, int64_t dataLen);
    __aicore__ inline void CopyIn(int64_t offset, int64_t blockCount, int64_t dataLen, int64_t dimNeg1IsOdd);
    __aicore__ inline void ComputeAll(int64_t blockCount, int64_t dataLen);
    __aicore__ inline void ComputeScaleOcp(
        uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr, __ubuf__ uint8_t* mxScale1Addr,
        __ubuf__ uint16_t* mxScale1ReciprocalAddr, __ubuf__ uint8_t* mxScale2Addr,
        __ubuf__ uint16_t* mxScale2ReciprocalAddr);
    __aicore__ inline void ComputeScaleCublas(
        uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr, __ubuf__ uint8_t* mxScale1Addr,
        __ubuf__ uint16_t* mxScale1ReciprocalAddr, __ubuf__ uint8_t* mxScale2Addr,
        __ubuf__ uint16_t* mxScale2ReciprocalAddr);
    // DynamicDtypeRange Default: dstTypeMax=0.0/6.0/7.0, 指数域addValueBit进位法 (scaleAlg=2)
    __aicore__ inline void ComputeScaleDynamicDefault(
        uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr, __ubuf__ uint8_t* mxScale1Addr,
        __ubuf__ uint16_t* mxScale1ReciprocalAddr, __ubuf__ uint8_t* mxScale2Addr,
        __ubuf__ uint16_t* mxScale2ReciprocalAddr);
    // DynamicDtypeRange Custom: 自定义dstTypeMax, FP32精度invDstTypeMax乘法法 (scaleAlg=2)
    __aicore__ inline void ComputeScaleDynamicCustom(
        uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr, __ubuf__ uint8_t* mxScale1Addr,
        __ubuf__ uint16_t* mxScale1ReciprocalAddr, __ubuf__ uint8_t* mxScale2Addr,
        __ubuf__ uint16_t* mxScale2ReciprocalAddr);
    __aicore__ inline void ComputeYVf(
        uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr, __ubuf__ uint16_t* mxScale1ReciprocalAddr,
        __ubuf__ uint16_t* mxScale2ReciprocalAddr, __ubuf__ uint8_t* y1Addr, __ubuf__ uint8_t* y2Addr);
    __aicore__ inline void ComputeY1ToFP4(
        uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr, __ubuf__ uint16_t* mxScale1ReciprocalAddr,
        __ubuf__ uint8_t* y1Addr);
    __aicore__ inline void ComputeY1ToFP8(
        uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr, __ubuf__ uint16_t* mxScale1ReciprocalAddr,
        __ubuf__ uint8_t* y1Addr);
    __aicore__ inline void ComputeY2ToFP4(
        uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr, __ubuf__ uint16_t* mxScale2ReciprocalAddr,
        __ubuf__ uint8_t* y2Addr);
    __aicore__ inline void ComputeFP4FromHalf(MicroAPI::RegTensor<float>& Reg);
    __aicore__ inline void ComputeY2ToFP8(
        uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr, __ubuf__ uint16_t* mxScale2ReciprocalAddr,
        __ubuf__ uint8_t* y2Addr);

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
    // DynamicDtypeRange需要CAST_RINT (四舍五入)，与OCP的CAST_TRUNC (截断) 不同
    static constexpr MicroAPI::CastTrait castTraitHalf2BF16Rint = {
        MicroAPI::RegLayout::UNKNOWN, MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING,
        AscendC::RoundMode::CAST_RINT};
    static constexpr MicroAPI::CastTrait castTraitBF16toFp4 = {
        MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT, MicroAPI::MaskMergeMode::ZEROING, roundMode};
    static constexpr MicroAPI::CastTrait castTraitFp32toBF16 = {
        MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT, MicroAPI::MaskMergeMode::ZEROING, roundMode};
    static constexpr MicroAPI::CastTrait castTraitFp32toYdtype = {
        MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT, MicroAPI::MaskMergeMode::ZEROING, roundMode};
    // FP32→FP8 四路RegLayout Cast (参考DynamicMxQuant ComputeData优化模式)
    // 将4组64个FP32值分别Cast到FP8的不同字节位置，通过Add合并后一次Store输出
    static constexpr MicroAPI::CastTrait castTraitFp32toFP8Layout0 = {
        MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT, MicroAPI::MaskMergeMode::ZEROING, roundMode};
    static constexpr MicroAPI::CastTrait castTraitFp32toFP8Layout1 = {
        MicroAPI::RegLayout::ONE, MicroAPI::SatMode::SAT, MicroAPI::MaskMergeMode::ZEROING, roundMode};
    static constexpr MicroAPI::CastTrait castTraitFp32toFP8Layout2 = {
        MicroAPI::RegLayout::TWO, MicroAPI::SatMode::SAT, MicroAPI::MaskMergeMode::ZEROING, roundMode};
    static constexpr MicroAPI::CastTrait castTraitFp32toFP8Layout3 = {
        MicroAPI::RegLayout::THREE, MicroAPI::SatMode::SAT, MicroAPI::MaskMergeMode::ZEROING, roundMode};

private:
    // tiling data
    const DynamicMxQuantWithDualAxisTilingData* tilingData_;

    // pipe & queue & buf
    TPipe* pipe_;
    TQue<QuePosition::VECIN, DB_BUFFER> inQueue;
    TQue<QuePosition::VECOUT, DB_BUFFER> outQueue1;
    TQue<QuePosition::VECOUT, DB_BUFFER> outQueue2;
    TQue<QuePosition::VECOUT, DB_BUFFER> mxScaleQueue1;
    TQue<QuePosition::VECOUT, DB_BUFFER> mxScaleQueue2;
    TBuf<TPosition::VECCALC> mxScale1ReciprocalBuf;
    TBuf<TPosition::VECCALC> mxScale2ReciprocalBuf;
    TBuf<TPosition::VECCALC> tmpScale2Buf;

    // gm
    GlobalTensor<xDtype> xGm1_;
    GlobalTensor<uint8_t> yGm1_;
    GlobalTensor<uint8_t> mxScaleGm1_;
    GlobalTensor<uint8_t> yGm2_;
    GlobalTensor<uint8_t> mxScaleGm2_;

    // base varible
    int64_t blockIdx_ = 0;
    // 当前
    int64_t blockOffset_ = 0;
    int64_t loopPerCore_ = 0;
    int64_t ubRowLen_ = 0;
    int64_t ubRowLenTail_ = 0;
    int64_t ubRowCount_ = 0;
    int64_t ubRowCountTail_ = 0;
    int64_t dimNeg2ScaleNum_ = 0;
    int64_t dimNeg1ScaleNum_ = 0;
    int64_t blockCountPerPage_ = 0;
    uint32_t invDtypeMax_ = 0;
    uint16_t dtypeYMaxExp_ = 0;
    uint16_t fp4SpecialValue_ = 0;
    // DynamicDtypeRange参数 (scaleAlg=2)
    float dstTypeMax_ = 0.0f;
    float invDstTypeMax_ = 0.0f;
    uint16_t addValueBit_ = 0;
    uint16_t subNumForScale_ = 0;
    int64_t blockSize_ = 0;
    // runtime varible
    int64_t mxScale1BufferSize_ = 0;
    int64_t mxScale2BufferSize_ = 0;
    int64_t tmpScale1BufferSize_ = 0;
    int64_t tmpScale2BufferSize_ = 0;
    int64_t inBufferSize_ = 0;

    bool scaleNeedsPad_ = false;
    int64_t vlForHalfNumber_ = platform::GetVRegSize() / sizeof(uint16_t);
    int64_t UBBlockSize_ = platform::GetUbBlockSize();
    int64_t oneBlockCountB16_ = UBBlockSize_ / sizeof(uint16_t);
    int64_t oneBlockCountB8_ = UBBlockSize_ / sizeof(uint8_t);
};

template <typename xDtype, typename y1Dtype, typename y2Dtype, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__aicore__ inline void DynamicMxQuantWithDualAxisBase<xDtype, y1Dtype, y2Dtype, roundMode, scaleAlg>::InitParams()
{
    blockIdx_ = GetBlockIdx();
    int64_t headCoreNum = tilingData_->headCoreNum;
    if (blockIdx_ < headCoreNum) {
        loopPerCore_ = tilingData_->blockPerHeadCore;
        // 切分基本块个数偏移
        blockOffset_ = blockIdx_ * loopPerCore_;
    } else {
        loopPerCore_ = tilingData_->blockPerTailCore;
        blockOffset_ = headCoreNum * tilingData_->blockPerHeadCore + (blockIdx_ - headCoreNum) * loopPerCore_;
    }

    blockSize_ = tilingData_->blockSize;

    // 一次vf计算的行长度，如果是tail后续处理,256
    ubRowLen_ = tilingData_->blockW;
    ubRowLenTail_ = tilingData_->dimNeg1Tail;

    // 一次UB计算的行数，如果是tail后续处理
    ubRowCount_ = tilingData_->splitBlockH;
    ubRowCountTail_ = tilingData_->dimNeg2Tail;

    // 一个batch总共多少个切分基本块
    blockCountPerPage_ = tilingData_->blockCountPerBatch;

    // 一个batch的-2轴scale行数
    dimNeg2ScaleNum_ = tilingData_->scale2RowCountPerBatch;

    // 一个batch的-1轴scale列数
    dimNeg1ScaleNum_ = tilingData_->scale1ColCountPerBatch;

    if constexpr (IsSameType<y1Dtype, fp8_e4m3fn_t>::value) {
        dtypeYMaxExp_ = FP8_E4M3_MAX_EXP;
        invDtypeMax_ = FP8_E4M3_MAX;
    } else if constexpr (IsSameType<y1Dtype, fp8_e5m2_t>::value) {
        dtypeYMaxExp_ = FP8_E5M2_MAX_EXP;
        invDtypeMax_ = FP8_E5M2_MAX;
    } else if constexpr (IsSameType<y1Dtype, fp4x2_e2m1_t>::value) {
        dtypeYMaxExp_ = FP4_E2M1_BF16_MAX_EXP;
        fp4SpecialValue_ = SPECIAL_VALUE_E2M1;
    } else {
        dtypeYMaxExp_ = 0;
        fp4SpecialValue_ = SPECIAL_VALUE_E1M2;
    }

    // DynamicDtypeRange参数初始化 (scaleAlg=2时使用)
    dstTypeMax_ = tilingData_->dstTypeMax;
    invDstTypeMax_ = tilingData_->invDstTypeMax;
    if (dstTypeMax_ == DIGIT_ZERO_FLOAT || dstTypeMax_ == DIGIT_SIX_FLOAT) {
        addValueBit_ = ADD_VALUE_FOR_BF16_MAN1;
        subNumForScale_ = SUB_NUM_FOR_SCALE_6;
    } else if (dstTypeMax_ == DIGIT_SEVEN_FLOAT) {
        addValueBit_ = ADD_VALUE_FOR_BF16_MAN2;
        subNumForScale_ = SUB_NUM_FOR_SCALE_7;
    }
}

template <typename xDtype, typename y1Dtype, typename y2Dtype, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__aicore__ inline void DynamicMxQuantWithDualAxisBase<xDtype, y1Dtype, y2Dtype, roundMode, scaleAlg>::ProcessOneLoop(
    int64_t calcCol, int64_t calcRow, int64_t xUbOffset, int64_t scale1Offset, int64_t scale2Offset,
    int64_t dimNeg1IsOdd)
{
    CopyIn(xUbOffset, calcRow, calcCol, dimNeg1IsOdd);
    ComputeAll(calcRow, calcCol);
    CopyOut(xUbOffset, scale1Offset, scale2Offset, calcRow, calcCol);
}

template <typename xDtype, typename y1Dtype, typename y2Dtype, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__aicore__ inline void DynamicMxQuantWithDualAxisBase<xDtype, y1Dtype, y2Dtype, roundMode, scaleAlg>::ComputeAll(
    int64_t blockCount, int64_t dataLen)
{
    LocalTensor<xDtype> x = inQueue.template DeQue<xDtype>();
    LocalTensor<uint8_t> mxScale1 = mxScaleQueue1.template AllocTensor<uint8_t>();
    LocalTensor<uint8_t> mxScale2 = mxScaleQueue2.template AllocTensor<uint8_t>();
    LocalTensor<uint8_t> y1 = outQueue1.template AllocTensor<uint8_t>();
    LocalTensor<uint8_t> y2 = outQueue2.template AllocTensor<uint8_t>();
    LocalTensor<uint16_t> mxScale1ReciprocalLocal = mxScale1ReciprocalBuf.Get<uint16_t>();
    LocalTensor<uint16_t> mxScale2ReciprocalLocal = mxScale2ReciprocalBuf.Get<uint16_t>();
    LocalTensor<uint8_t> tmpScale2Local = tmpScale2Buf.Get<uint8_t>();

    auto xAddr = (__ubuf__ xDtype*)x.GetPhyAddr();
    auto y1Addr = (__ubuf__ uint8_t*)y1.GetPhyAddr();
    auto y2Addr = (__ubuf__ uint8_t*)y2.GetPhyAddr();
    auto mxScale1Addr = (__ubuf__ uint8_t*)mxScale1.GetPhyAddr();
    auto mxScale2Addr = (__ubuf__ uint8_t*)mxScale2.GetPhyAddr();
    auto tmpScale2Addr = (__ubuf__ uint8_t*)tmpScale2Local.GetPhyAddr();
    // 1/scale
    auto mxScale1ReciprocalAddr = (__ubuf__ uint16_t*)mxScale1ReciprocalLocal.GetPhyAddr();
    auto mxScale2ReciprocalAddr = (__ubuf__ uint16_t*)mxScale2ReciprocalLocal.GetPhyAddr();

    int64_t xOffset = 0;
    int64_t yOffset = 0;
    int64_t scale1UbOffset = 0;
    int64_t scale2UbOffset = 0;
    int64_t scale1ReciprocalOffset = 0;
    int64_t scale2ReciprocalOffset = 0;

    // -2轴有多少个block块循环
    int64_t calcBlockLoop = (blockCount + tilingData_->blockSize - 1) / tilingData_->blockSize;
    int64_t calcBlockTail = blockCount % tilingData_->blockSize;
    int64_t calcLoop = calcBlockTail == 0 ? calcBlockLoop : (calcBlockLoop - 1);
    // block循环
    for (int64_t i = 0; i < calcLoop; i++) {
        xOffset = i * blockSize_ * ubRowLen_;
        if constexpr ((IsSameType<y1Dtype, fp8_e4m3fn_t>::value) || (IsSameType<y1Dtype, fp8_e5m2_t>::value)) {
            yOffset = i * blockSize_ * ubRowLen_;
        } else {
            // 两个fp4合成一个fp8输出，所以要/2
            yOffset = i * blockSize_ * ubRowLen_ / DIGIT_TWO;
        }
        scale1UbOffset = i * blockSize_ * ops::CeilAlign(ubRowLen_ / blockSize_, oneBlockCountB8_);
        scale2UbOffset = i * ubRowLen_;
        scale1ReciprocalOffset = i * blockSize_ * ops::CeilAlign(ubRowLen_ / blockSize_, oneBlockCountB16_);
        scale2ReciprocalOffset = i * ubRowLen_;
        if constexpr (scaleAlg == 0) {
        ComputeScaleOcp(
            dataLen, blockSize_, xAddr + xOffset, mxScale1Addr + scale1UbOffset,
            mxScale1ReciprocalAddr + scale1ReciprocalOffset, tmpScale2Addr + scale2UbOffset,
            mxScale2ReciprocalAddr + scale2ReciprocalOffset);
        } else if constexpr (scaleAlg == 1) {
        // CuBALS Scale算法 (FP8专用)
        ComputeScaleCublas(
            dataLen, blockSize_, xAddr + xOffset, mxScale1Addr + scale1UbOffset,
            mxScale1ReciprocalAddr + scale1ReciprocalOffset, tmpScale2Addr + scale2UbOffset,
            mxScale2ReciprocalAddr + scale2ReciprocalOffset);
        } else if constexpr (scaleAlg == 2) {
        // DynamicDtypeRange Scale算法 (FP4_E2M1专用)
        // 运行时根据dstTypeMax_选择Default (指数域进位法) 或 Custom (FP32精度乘法法)
        if (dstTypeMax_ == DIGIT_ZERO_FLOAT || dstTypeMax_ == DIGIT_SIX_FLOAT ||
            dstTypeMax_ == DIGIT_SEVEN_FLOAT) {
            ComputeScaleDynamicDefault(
                dataLen, blockSize_, xAddr + xOffset, mxScale1Addr + scale1UbOffset,
                mxScale1ReciprocalAddr + scale1ReciprocalOffset, tmpScale2Addr + scale2UbOffset,
                mxScale2ReciprocalAddr + scale2ReciprocalOffset);
        } else {
            ComputeScaleDynamicCustom(
                dataLen, blockSize_, xAddr + xOffset, mxScale1Addr + scale1UbOffset,
                mxScale1ReciprocalAddr + scale1ReciprocalOffset, tmpScale2Addr + scale2UbOffset,
                mxScale2ReciprocalAddr + scale2ReciprocalOffset);
        }
        }

        ComputeYVf(
            dataLen, blockSize_, xAddr + xOffset, mxScale1ReciprocalAddr + scale1ReciprocalOffset,
            mxScale2ReciprocalAddr + scale2ReciprocalOffset, y1Addr + yOffset, y2Addr + yOffset);
    }
    if (calcBlockTail != 0) {
        xOffset = calcLoop * blockSize_ * ubRowLen_;
        if constexpr ((IsSameType<y1Dtype, fp8_e4m3fn_t>::value) || (IsSameType<y1Dtype, fp8_e5m2_t>::value)) {
            yOffset = calcLoop * blockSize_ * ubRowLen_;
        } else {
            yOffset = calcLoop * blockSize_ * ubRowLen_ / DIGIT_TWO;
        }
        scale1UbOffset = calcLoop * blockSize_ * ops::CeilAlign(ubRowLen_ / blockSize_, oneBlockCountB8_);
        scale2UbOffset = calcLoop * ubRowLen_;
        scale1ReciprocalOffset = calcLoop * blockSize_ * ops::CeilAlign(ubRowLen_ / blockSize_, oneBlockCountB16_);
        scale2ReciprocalOffset = calcLoop * ubRowLen_;
        if constexpr (scaleAlg == 0) {
        ComputeScaleOcp(
            dataLen, static_cast<uint16_t>(calcBlockTail), xAddr + xOffset, mxScale1Addr + scale1UbOffset,
            mxScale1ReciprocalAddr + scale1ReciprocalOffset, tmpScale2Addr + scale2UbOffset,
            mxScale2ReciprocalAddr + scale2ReciprocalOffset);
        } else if constexpr (scaleAlg == 1) {
        ComputeScaleCublas(
            dataLen, static_cast<uint16_t>(calcBlockTail), xAddr + xOffset, mxScale1Addr + scale1UbOffset,
            mxScale1ReciprocalAddr + scale1ReciprocalOffset, tmpScale2Addr + scale2UbOffset,
            mxScale2ReciprocalAddr + scale2ReciprocalOffset);
        } else if constexpr (scaleAlg == 2) {
        // DynamicDtypeRange Scale算法 (FP4_E2M1专用) - 尾块处理
        if (dstTypeMax_ == DIGIT_ZERO_FLOAT || dstTypeMax_ == DIGIT_SIX_FLOAT ||
            dstTypeMax_ == DIGIT_SEVEN_FLOAT) {
            ComputeScaleDynamicDefault(
                dataLen, static_cast<uint16_t>(calcBlockTail), xAddr + xOffset, mxScale1Addr + scale1UbOffset,
                mxScale1ReciprocalAddr + scale1ReciprocalOffset, tmpScale2Addr + scale2UbOffset,
                mxScale2ReciprocalAddr + scale2ReciprocalOffset);
        } else {
            ComputeScaleDynamicCustom(
                dataLen, static_cast<uint16_t>(calcBlockTail), xAddr + xOffset, mxScale1Addr + scale1UbOffset,
                mxScale1ReciprocalAddr + scale1ReciprocalOffset, tmpScale2Addr + scale2UbOffset,
                mxScale2ReciprocalAddr + scale2ReciprocalOffset);
        }
        }
        ComputeYVf(
            dataLen, static_cast<uint16_t>(calcBlockTail), xAddr + xOffset,
            mxScale1ReciprocalAddr + scale1ReciprocalOffset, mxScale2ReciprocalAddr + scale2ReciprocalOffset,
            y1Addr + yOffset, y2Addr + yOffset);
    }

    // event_t eventIDVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    // SetFlag<HardEvent::V_S>(eventIDVToS);
    // WaitFlag<HardEvent::V_S>(eventIDVToS);

    // -2轴的scale交织处理
    for (int64_t i = 1; i < ((calcBlockLoop + 1) / DIGIT_TWO * DIGIT_TWO); i = i + 2) {
        Interleave(
            mxScale2[(i - 1) * ubRowLen_], mxScale2[i * ubRowLen_], tmpScale2Local[(i - 1) * ubRowLen_],
            tmpScale2Local[i * ubRowLen_], ubRowLen_);
    }

    mxScaleQueue1.template EnQue(mxScale1);
    outQueue1.template EnQue(y1);
    mxScaleQueue2.template EnQue(mxScale2);
    outQueue2.template EnQue(y2);
    inQueue.template FreeTensor(x);
}

template <typename xDtype, typename y1Dtype, typename y2Dtype, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__aicore__ inline void DynamicMxQuantWithDualAxisBase<xDtype, y1Dtype, y2Dtype, roundMode, scaleAlg>::ComputeScaleOcp(
    uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr, __ubuf__ uint8_t* mxScale1Addr,
    __ubuf__ uint16_t* mxScale1ReciprocalAddr, __ubuf__ uint8_t* mxScale2Addr,
    __ubuf__ uint16_t* mxScale2ReciprocalAddr)
{
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
        // MicroAPI::MaskReg infNanDataMask0;
        // MicroAPI::MaskReg infNanDataMask1;
        MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<xDtype, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg maskB8 = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg maskReduceB8 = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::VL8>();
        MicroAPI::MaskReg maskReduceB16 = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::VL16>();

        MicroAPI::Duplicate(expMaskBF16, EXP_MASK_BF16);
        MicroAPI::Duplicate(expMaskFP16, EXP_MASK_FP16);
        MicroAPI::Duplicate(expMax1Dim2, 0);
        MicroAPI::Duplicate(expMax2Dim2, 0);
        MicroAPI::Duplicate(yMaxExp, dtypeYMaxExp_);
        MicroAPI::Duplicate(nanE8M0, NAN_FOR_FP8_E8M0);
        MicroAPI::Duplicate(biasE8M0, BF16_EXP_BIAS);
        MicroAPI::Duplicate(zero, 0);
        MicroAPI::Duplicate(nanBF16, NAN_CUSTOMIZATION);
        MicroAPI::Duplicate(specialExp, SPECIAL_EXP_THRESHOLD);

        for (uint16_t i = 0; i < blockCount; i++) {
            // 交织搬运，一次搬256个B16
            MicroAPI::DataCopy<xDtype, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_DINTLV_B16>(
                x0, x1, xAddr, vlForHalfNumber_ * DIGIT_TWO);
            if constexpr (IsSameType<xDtype, half>::value) {
                // 原始数据转成bf16
                MicroAPI::Cast<bfloat16_t, xDtype, castTraitHalf2BF16>(x0BF16, x0, maskAll);
                MicroAPI::Cast<bfloat16_t, xDtype, castTraitHalf2BF16>(x1BF16, x1, maskAll);
                // 提取指数位
                MicroAPI::And(x0ExpBF16, (MicroAPI::RegTensor<uint16_t>&)x0BF16, expMaskBF16, maskAll);
                MicroAPI::And(x1ExpBF16, (MicroAPI::RegTensor<uint16_t>&)x1BF16, expMaskBF16, maskAll);
            } else {
                // 提取指数位
                MicroAPI::And(x0ExpBF16, (MicroAPI::RegTensor<uint16_t>&)x0, expMaskBF16, maskAll);
                MicroAPI::And(x1ExpBF16, (MicroAPI::RegTensor<uint16_t>&)x1, expMaskBF16, maskAll);
            }
            // 计算x0和x1的最大值，相当于计算原始相邻两个数据的最大值
            MicroAPI::Max(expMaxDim1, x0ExpBF16, x1ExpBF16, maskAll);
            // ReduceMax一个block，即16个数，配合上一步，可以计算出每32个数的最大值，一共256/32个
            MicroAPI::ReduceMaxWithDataBlock(expMaxDim1, expMaxDim1, maskAll);
            // 二分性能更高，待定
            MicroAPI::Max(expMax1Dim2, expMax1Dim2, x0ExpBF16, maskAll);
            MicroAPI::Max(expMax2Dim2, expMax2Dim2, x1ExpBF16, maskAll);

            // 计算-1轴的scale和1/scale
            // inf/nan值单独处理，结果为E8M0的nan
            MicroAPI::Compare<uint16_t, CMPMODE::NE>(infMask, expMaxDim1, expMaskBF16, maskAll);
            // 0值单独处理，结果为0
            MicroAPI::Compare<uint16_t, CMPMODE::NE>(zeroMask, expMaxDim1, zero, maskAll);
            // 指数位不足被量化类型的ele_max时，为subnormal场景，结果为0
            MicroAPI::Compare<uint16_t, CMPMODE::LE>(invalidDataMask, expMaxDim1, yMaxExp, maskAll);
            MicroAPI::Select<uint16_t>(expMaxDim1, yMaxExp, expMaxDim1, invalidDataMask);
            // 指数位减去expMax，按照BF16的格式处理，例：E5M2的expMax为15，即需要减去0 00001111 0000000
            MicroAPI::Sub(expMaxDim1, expMaxDim1, yMaxExp, maskAll);
            // 右移7位，BF16的指数位移到了末8位
            MicroAPI::ShiftRights(mxScale1B16, expMaxDim1, SHR_NUM_FOR_BF16, maskAll);
            MicroAPI::Select<uint16_t>(mxScale1B16, mxScale1B16, nanE8M0, infMask);
            MicroAPI::Select<uint16_t>(mxScale1B16, mxScale1B16, zero, zeroMask);

            MicroAPI::Pack<uint8_t, uint16_t, MicroAPI::HighLowPart::LOWEST>(mxScale1B8, mxScale1B16);
            MicroAPI::DataCopy<uint8_t>(mxScale1Addr + i * oneBlockCountB8_, mxScale1B8, maskReduceB8);

            // 公式中的1/X
            // 只有在E1M2时，yMaxExp=0，expMaxDim1可能会等于biasE8M0
            MicroAPI::Compare<uint16_t, CMPMODE::EQ>(invalidDataMask, expMaxDim1, biasE8M0, maskAll);

            MicroAPI::Sub(reversedShareExp1, biasE8M0, expMaxDim1, maskAll);
            MicroAPI::Select<uint16_t>(reversedShareExp1, reversedShareExp1, nanBF16, infMask);
            MicroAPI::Select<uint16_t>(reversedShareExp1, reversedShareExp1, zero, zeroMask);
            MicroAPI::Select<uint16_t>(reversedShareExp1, specialExp, reversedShareExp1, invalidDataMask);
            MicroAPI::DataCopy<uint16_t>(mxScale1ReciprocalAddr + i * oneBlockCountB16_, reversedShareExp1, maskReduceB16);
        }
        // 计算-2轴的scale2和1/scale2 交织第一部分
        // inf/nan值单独处理，结果为E8M0的nan
        MicroAPI::Compare<uint16_t, CMPMODE::NE>(infMask, expMax1Dim2, expMaskBF16, maskAll);
        // 0值单独处理，结果为0
        MicroAPI::Compare<uint16_t, CMPMODE::NE>(zeroMask, expMax1Dim2, zero, maskAll);
        // 指数位不足被量化类型的ele_max时，为subnormal场景，结果为0
        MicroAPI::Compare<uint16_t, CMPMODE::LE>(invalidDataMask, expMax1Dim2, yMaxExp, maskAll);
        MicroAPI::Select<uint16_t>(expMax1Dim2, yMaxExp, expMax1Dim2, invalidDataMask);
        // 指数位减去expMax，按照BF16的格式处理，例：E5M2的expMax为15，即需要减去0 00001111 0000000
        MicroAPI::Sub(expMax1Dim2, expMax1Dim2, yMaxExp, maskAll);
        // 右移7位，BF16的指数位移到了末8位
        MicroAPI::ShiftRights(mxScale2ZeroB16, expMax1Dim2, SHR_NUM_FOR_BF16, maskAll);
        MicroAPI::Select<uint16_t>(mxScale2ZeroB16, mxScale2ZeroB16, nanE8M0, infMask);
        MicroAPI::Select<uint16_t>(mxScale2ZeroB16, mxScale2ZeroB16, zero, zeroMask);

        MicroAPI::Pack<uint8_t, uint16_t, MicroAPI::HighLowPart::LOWEST>(mxScale2ZeroB8, mxScale2ZeroB16);

        // 公式中的1/X
        // 只有在E1M2时，yMaxExp=0，expMax1Dim2可能会等于biasE8M0
        MicroAPI::Compare<uint16_t, CMPMODE::EQ>(invalidDataMask, expMax1Dim2, biasE8M0, maskAll);

        MicroAPI::Sub(reversedShareExp2Zero, biasE8M0, expMax1Dim2, maskAll);
        MicroAPI::Select<uint16_t>(reversedShareExp2Zero, reversedShareExp2Zero, nanBF16, infMask);
        MicroAPI::Select<uint16_t>(reversedShareExp2Zero, reversedShareExp2Zero, zero, zeroMask);
        MicroAPI::Select<uint16_t>(reversedShareExp2Zero, specialExp, reversedShareExp2Zero, invalidDataMask);

        // 计算-2轴的scale和1/scale 交织第二部分
        // inf/nan值单独处理，结果为E8M0的nan
        MicroAPI::Compare<uint16_t, CMPMODE::NE>(infMask, expMax2Dim2, expMaskBF16, maskAll);
        // 0值单独处理，结果为0
        MicroAPI::Compare<uint16_t, CMPMODE::NE>(zeroMask, expMax2Dim2, zero, maskAll);
        // 指数位不足被量化类型的ele_max时，为subnormal场景，结果为0
        MicroAPI::Compare<uint16_t, CMPMODE::LE>(invalidDataMask, expMax2Dim2, yMaxExp, maskAll);
        MicroAPI::Select<uint16_t>(expMax2Dim2, yMaxExp, expMax2Dim2, invalidDataMask);
        // 指数位减去expMax，按照BF16的格式处理，例：E5M2的expMax为15，即需要减去0 00001111 0000000
        MicroAPI::Sub(expMax2Dim2, expMax2Dim2, yMaxExp, maskAll);
        // 右移7位，BF16的指数位移到了末8位
        MicroAPI::ShiftRights(mxScale2OneB16, expMax2Dim2, SHR_NUM_FOR_BF16, maskAll);
        MicroAPI::Select<uint16_t>(mxScale2OneB16, mxScale2OneB16, nanE8M0, infMask);
        MicroAPI::Select<uint16_t>(mxScale2OneB16, mxScale2OneB16, zero, zeroMask);

        MicroAPI::Pack<uint8_t, uint16_t, MicroAPI::HighLowPart::LOWEST>(mxScale2OneB8, mxScale2OneB16);
        // 公式中的1/X
        // 只有在E1M2时，yMaxExp=0，expMax2Dim2可能会等于biasE8M0
        MicroAPI::Compare<uint16_t, CMPMODE::EQ>(invalidDataMask, expMax2Dim2, biasE8M0, maskAll);
        MicroAPI::Sub(reversedShareExp2One, biasE8M0, expMax2Dim2, maskAll);
        MicroAPI::Select<uint16_t>(reversedShareExp2One, reversedShareExp2One, nanBF16, infMask);
        MicroAPI::Select<uint16_t>(reversedShareExp2One, reversedShareExp2One, zero, zeroMask);
        MicroAPI::Select<uint16_t>(reversedShareExp2One, specialExp, reversedShareExp2One, invalidDataMask);
        // 交织搬出mxScale和1/scale
        MicroAPI::DataCopy<uint8_t, MicroAPI::StoreDist::DIST_INTLV_B8>(
            mxScale2Addr, mxScale2ZeroB8, mxScale2OneB8, maskB8);
        MicroAPI::DataCopy<uint16_t, MicroAPI::StoreDist::DIST_INTLV_B16>(
            mxScale2ReciprocalAddr, reversedShareExp2Zero, reversedShareExp2One, maskAll);
    }
}

// CuBALS Scale算法实现 (scaleAlg=1, FP8专用)
// 整体框架与ComputeScaleOcp一致：循环blockCount次处理-1轴scale，循环后处理-2轴scale
// 算法差异：OCP使用指数提取法，CuBALS使用 Amax/Amax(DType) + FP32指数尾数条件舍入法
template <typename xDtype, typename y1Dtype, typename y2Dtype, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__aicore__ inline void DynamicMxQuantWithDualAxisBase<xDtype, y1Dtype, y2Dtype, roundMode, scaleAlg>::ComputeScaleCublas(
    uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr, __ubuf__ uint8_t* mxScale1Addr,
    __ubuf__ uint16_t* mxScale1ReciprocalAddr, __ubuf__ uint8_t* mxScale2Addr,
    __ubuf__ uint16_t* mxScale2ReciprocalAddr)
{
    __VEC_SCOPE__
    {
        // ========== 输入数据寄存器 ==========
        MicroAPI::RegTensor<xDtype> x0;
        MicroAPI::RegTensor<xDtype> x1;

        // ========== 绝对值和max寄存器 ==========
        MicroAPI::RegTensor<uint16_t> absMax0;         // x0的绝对值
        MicroAPI::RegTensor<uint16_t> absMax1;         // x1的绝对值
        MicroAPI::RegTensor<uint16_t> absMaxDim1;      // -1轴方向block内绝对值max
        MicroAPI::RegTensor<uint16_t> absMax1Dim2;     // -2轴方向累积max (偶数列, 对应x0)
        MicroAPI::RegTensor<uint16_t> absMax2Dim2;     // -2轴方向累积max (奇数列, 对应x1)
        MicroAPI::RegTensor<uint16_t> zeroB16;         // -1轴Interleave用零寄存器

        // ========== FP32计算寄存器 ==========
        // -1轴: Interleave-with-0后单次Cast Zero处理全部8个值，仅需一组FP32寄存器
        // -2轴: 仍需Zero/One两组独立处理
        MicroAPI::RegTensor<uint32_t> maxFP32_0;       // FP32表示, 链内复用为expPlusOne
        MicroAPI::RegTensor<uint32_t> maxFP32_1;       // -2轴奇数部分FP32表示
        MicroAPI::RegTensor<uint32_t> expFP32_0;       // FP32指数
        MicroAPI::RegTensor<uint32_t> expFP32_1;       // -2轴奇数部分FP32指数
        MicroAPI::RegTensor<uint32_t> manFP32_0;       // FP32尾数, 链内复用为extractExp
        MicroAPI::RegTensor<uint32_t> manFP32_1;       // -2轴奇数部分FP32尾数

        // scale输出寄存器 (循环后复用于-2轴scale输出)
        MicroAPI::RegTensor<uint16_t> scale1B16_0;     // E8M0 uint16, 循环后复用为mxScale2ZeroB16
        MicroAPI::RegTensor<uint16_t> scale1B16_1;     // -2轴奇数部分, 循环后复用为mxScale2OneB16
        MicroAPI::RegTensor<uint16_t> scale1BF16;      // BF16指数格式, 循环后复用为scale2BF16
        MicroAPI::RegTensor<uint8_t> mxScale1B8;       // uint8 scale, 循环后复用为mxScale2ZeroB8
        MicroAPI::RegTensor<uint16_t> reversedShareExp1; // 1/scale BF16, 循环后复用为reversedShareExp2Zero

        // -2轴独立寄存器 (需与复用寄存器同时存活，无法复用)
        MicroAPI::RegTensor<uint8_t> mxScale2OneB8;    // 与mxScale1B8同时存活于最终DataCopy

        // ========== 常量寄存器 ==========
        MicroAPI::RegTensor<uint16_t> absMask;
        MicroAPI::Duplicate(absMask, ABS_MASK_FOR_16BIT);
        MicroAPI::RegTensor<uint32_t> invMax;
        MicroAPI::Duplicate(invMax, invDtypeMax_);            // 1/Amax(DType), FP32表示
        MicroAPI::RegTensor<uint32_t> manMaskFP32;
        MicroAPI::Duplicate(manMaskFP32, MAN_MASK_FLOAT);     // FP32尾数掩码
        MicroAPI::RegTensor<uint32_t> scaleBiasFP32;
        MicroAPI::Duplicate(scaleBiasFP32, FP32_EXP_BIAS_CUBLAS); // BF16偏移在uint32
        MicroAPI::RegTensor<uint32_t> nanPackFP32;
        MicroAPI::Duplicate(nanPackFP32, NAN_CUSTOMIZATION_PACK);

        MicroAPI::RegTensor<uint16_t> nanE8M0;
        MicroAPI::Duplicate(nanE8M0, NAN_FOR_FP8_E8M0);
        MicroAPI::RegTensor<uint16_t> biasE8M0;
        MicroAPI::Duplicate(biasE8M0, BF16_EXP_BIAS);
        MicroAPI::RegTensor<uint16_t> zero;
        MicroAPI::Duplicate(zero, 0);
        MicroAPI::RegTensor<uint16_t> nanBF16;
        MicroAPI::Duplicate(nanBF16, NAN_CUSTOMIZATION);
        MicroAPI::RegTensor<uint16_t> specialExp;
        MicroAPI::Duplicate(specialExp, SPECIAL_EXP_THRESHOLD);
        MicroAPI::RegTensor<uint16_t> maxEleBF16;
        MicroAPI::Duplicate(maxEleBF16, EXP_MASK_BF16);

        MicroAPI::Duplicate(absMax1Dim2, 0);
        MicroAPI::Duplicate(absMax2Dim2, 0);
        MicroAPI::Duplicate(zeroB16, 0);

        // ========== Mask定义 ==========
        MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<xDtype, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg maskB8 = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg maskReduceB8 = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::VL8>();
        MicroAPI::MaskReg maskReduceB16 = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::VL16>();
        MicroAPI::MaskReg maskFP32 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();

        MicroAPI::MaskReg p0;              // 条件舍入mask
        MicroAPI::MaskReg p1;              // subnormal条件mask
        MicroAPI::MaskReg p0Odd;           // -2轴奇数部分条件舍入mask
        MicroAPI::MaskReg p1Odd;           // -2轴奇数部分subnormal条件mask
        MicroAPI::MaskReg infMask;
        MicroAPI::MaskReg invalidDataMask;

        // ========================================================================
        // 循环blockCount次，每次处理一行，计算-1轴scale并累积-2轴max
        // ========================================================================
        for (uint16_t i = 0; i < blockCount; i++) {
            // 1. 交织搬运输入数据: 将256个xDtype按偶奇拆分为x0(偶), x1(奇)
            MicroAPI::DataCopy<xDtype, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_DINTLV_B16>(
                x0, x1, xAddr, vlForHalfNumber_ * DIGIT_TWO);

            // 2. 取绝对值: 清除符号位，保留指数和尾数
            MicroAPI::And(absMax0, (MicroAPI::RegTensor<uint16_t>&)x0, absMask, maskAll);
            MicroAPI::And(absMax1, (MicroAPI::RegTensor<uint16_t>&)x1, absMask, maskAll);

            // 3. -1轴: 先取偶奇max，再ReduceMaxWithDataBlock得到每32个元素的绝对值max
            MicroAPI::Max(absMaxDim1, absMax0, absMax1, maskAll);
            MicroAPI::ReduceMaxWithDataBlock(absMaxDim1, absMaxDim1, maskAll);

            // 4. -2轴: 逐行累积偶数列和奇数列的绝对值max
            MicroAPI::Max(absMax1Dim2, absMax1Dim2, absMax0, maskAll);
            MicroAPI::Max(absMax2Dim2, absMax2Dim2, absMax1, maskAll);

            // ============================================================
            // 5. 计算-1轴CuBALS Scale (FP32精度)
            //    ReduceMaxWithDataBlock后，8个max紧凑存储在寄存器前8个位置
            //    与0交织后，Cast Zero一次处理全部8个值转为FP32
            //    (与原始DynamicMxQuant的ComputeCuBLAS一致)
            // ============================================================

            // 与0交织: [v0,0,v1,0,...,v7,0,...] → Cast Zero可一次取出全部8个有效值
            MicroAPI::Interleave(absMaxDim1, zeroB16, absMaxDim1, zeroB16);
            MicroAPI::Cast<float, xDtype, castTraitXdtypetoFp32Zero>(
                (MicroAPI::RegTensor<float>&)maxFP32_0,
                (MicroAPI::RegTensor<xDtype>&)absMaxDim1, maskAll);
            // 乘以 1/Amax(DType): max * invDtypeMax
            MicroAPI::Mul(
                (MicroAPI::RegTensor<float>&)maxFP32_0,
                (MicroAPI::RegTensor<float>&)maxFP32_0,
                (MicroAPI::RegTensor<float>&)invMax, maskFP32);
            // 提取FP32指数: 右移23位
            MicroAPI::ShiftRights(expFP32_0, maxFP32_0, SHR_NUM_FOR_FP32, maskFP32);
            // 提取FP32尾数: 与尾数掩码
            MicroAPI::And(manFP32_0, maxFP32_0, manMaskFP32, maskFP32);
            // 条件舍入: normal场景 (exp>0 && exp<254 && man>0) → exp+1
            MicroAPI::CompareScalar<uint32_t, CMPMODE::GT>(p0, expFP32_0, NUMBER_ZERO_U32, maskFP32);
            MicroAPI::CompareScalar<uint32_t, CMPMODE::LT>(p0, expFP32_0, NUMBER_TWO_FIVE_FOUR, p0);
            MicroAPI::CompareScalar<uint32_t, CMPMODE::GT>(p0, manFP32_0, NUMBER_ZERO_U32, p0);
            // 条件舍入: subnormal场景 (exp==0 && man>HALF) → exp+1
            MicroAPI::CompareScalar<uint32_t, CMPMODE::EQ>(p1, expFP32_0, NUMBER_ZERO_U32, maskFP32);
            MicroAPI::CompareScalar<uint32_t, CMPMODE::GT>(p1, manFP32_0, NUMBER_HALF_U32, p1);
            MicroAPI::MaskOr(p0, p0, p1, maskFP32);
            // 执行条件加1
            MicroAPI::Adds(maxFP32_0, expFP32_0, 1, maskFP32);
            MicroAPI::Select(manFP32_0, maxFP32_0, expFP32_0, p0);
            // Pack到uint16 (INF/NAN→0xFF, zero→0 自然通过条件舍入保持, 在BF16域1/scale中统一处理)
            MicroAPI::Pack<uint16_t, uint32_t, MicroAPI::HighLowPart::LOWEST>(scale1B16_0, manFP32_0);

            // 左移7位，将E8M0值定位到BF16指数域 (用于计算1/scale)
            MicroAPI::ShiftLefts(scale1BF16, scale1B16_0, SHR_NUM_FOR_BF16, maskAll);

            // --- 输出-1轴scale (uint8) ---
            MicroAPI::Pack<uint8_t, uint16_t, MicroAPI::HighLowPart::LOWEST>(mxScale1B8, scale1B16_0);
            MicroAPI::DataCopy<uint8_t>(mxScale1Addr + i * oneBlockCountB8_, mxScale1B8, maskReduceB8);

            // --- 计算并输出-1轴 1/scale (与原始DynamicMxQuant一致: inf→nan, special→specialExp, 无零值检查) ---
            MicroAPI::Compare<uint16_t, CMPMODE::NE>(infMask, scale1BF16, maxEleBF16, maskAll);
            MicroAPI::Compare<uint16_t, CMPMODE::EQ>(invalidDataMask, scale1BF16, biasE8M0, maskAll);
            MicroAPI::Sub(reversedShareExp1, biasE8M0, scale1BF16, maskAll);
            MicroAPI::Select<uint16_t>(reversedShareExp1, reversedShareExp1, nanBF16, infMask);
            MicroAPI::Select<uint16_t>(reversedShareExp1, specialExp, reversedShareExp1, invalidDataMask);
            MicroAPI::DataCopy<uint16_t>(
                mxScale1ReciprocalAddr + i * oneBlockCountB16_, reversedShareExp1, maskReduceB16);
            // 恢复zeroB16 (Interleave会修改dst1)
            MicroAPI::Duplicate(zeroB16, 0);
        }

        // ========================================================================
        // 循环结束后，计算-2轴CuBALS Scale
        // absMax1Dim2: 128个偶数列的累积绝对值max
        // absMax2Dim2: 128个奇数列的累积绝对值max
        // 每个需要拆分为Zero/One两半分别做FP32计算，再合并
        // ========================================================================

        // ---------- 处理absMax1Dim2 (偶数列, -2轴scale的交织第一部分) ----------
        // Zero半 (偶数位) — 复用-1轴寄存器: maxFP32_0, expFP32_0, manFP32_0
        MicroAPI::Cast<float, xDtype, castTraitXdtypetoFp32Zero>(
            (MicroAPI::RegTensor<float>&)maxFP32_0,
            (MicroAPI::RegTensor<xDtype>&)absMax1Dim2, maskAll);
        MicroAPI::Mul(
            (MicroAPI::RegTensor<float>&)maxFP32_0,
            (MicroAPI::RegTensor<float>&)maxFP32_0,
            (MicroAPI::RegTensor<float>&)invMax, maskFP32);
        MicroAPI::ShiftRights(expFP32_0, maxFP32_0, SHR_NUM_FOR_FP32, maskFP32);
        MicroAPI::And(manFP32_0, maxFP32_0, manMaskFP32, maskFP32);
        MicroAPI::CompareScalar<uint32_t, CMPMODE::GT>(p0, expFP32_0, NUMBER_ZERO_U32, maskFP32);
        MicroAPI::CompareScalar<uint32_t, CMPMODE::LT>(p0, expFP32_0, NUMBER_TWO_FIVE_FOUR, p0);
        MicroAPI::CompareScalar<uint32_t, CMPMODE::GT>(p0, manFP32_0, NUMBER_ZERO_U32, p0);
        MicroAPI::CompareScalar<uint32_t, CMPMODE::EQ>(p1, expFP32_0, NUMBER_ZERO_U32, maskFP32);
        MicroAPI::CompareScalar<uint32_t, CMPMODE::GT>(p1, manFP32_0, NUMBER_HALF_U32, p1);
        MicroAPI::MaskOr(p0, p0, p1, maskFP32);
        // 链内复用: maxFP32_0→expPlusOne (maxFP32_0已死亡@And)
        MicroAPI::Adds(maxFP32_0, expFP32_0, 1, maskFP32);
        // 链内复用: manFP32_0→extractExp (manFP32_0已死亡@CompareScalar)
        MicroAPI::Select(manFP32_0, maxFP32_0, expFP32_0, p0);
        MicroAPI::Pack<uint16_t, uint32_t, MicroAPI::HighLowPart::LOWEST>(scale1B16_0, manFP32_0);

        // One半 (奇数位) - 使用独立的p0Odd/p1Odd，与Zero半并行
        MicroAPI::Cast<float, xDtype, castTraitXdtypetoFp32One>(
            (MicroAPI::RegTensor<float>&)maxFP32_1,
            (MicroAPI::RegTensor<xDtype>&)absMax1Dim2, maskAll);
        MicroAPI::Mul(
            (MicroAPI::RegTensor<float>&)maxFP32_1,
            (MicroAPI::RegTensor<float>&)maxFP32_1,
            (MicroAPI::RegTensor<float>&)invMax, maskFP32);
        MicroAPI::ShiftRights(expFP32_1, maxFP32_1, SHR_NUM_FOR_FP32, maskFP32);
        MicroAPI::And(manFP32_1, maxFP32_1, manMaskFP32, maskFP32);
        MicroAPI::CompareScalar<uint32_t, CMPMODE::GT>(p0Odd, expFP32_1, NUMBER_ZERO_U32, maskFP32);
        MicroAPI::CompareScalar<uint32_t, CMPMODE::LT>(p0Odd, expFP32_1, NUMBER_TWO_FIVE_FOUR, p0Odd);
        MicroAPI::CompareScalar<uint32_t, CMPMODE::GT>(p0Odd, manFP32_1, NUMBER_ZERO_U32, p0Odd);
        MicroAPI::CompareScalar<uint32_t, CMPMODE::EQ>(p1Odd, expFP32_1, NUMBER_ZERO_U32, maskFP32);
        MicroAPI::CompareScalar<uint32_t, CMPMODE::GT>(p1Odd, manFP32_1, NUMBER_HALF_U32, p1Odd);
        MicroAPI::MaskOr(p0Odd, p0Odd, p1Odd, maskFP32);
        // 链内复用: maxFP32_1→expPlusOne, manFP32_1→extractExp
        MicroAPI::Adds(maxFP32_1, expFP32_1, 1, maskFP32);
        MicroAPI::Select(manFP32_1, maxFP32_1, expFP32_1, p0Odd);
        MicroAPI::Pack<uint16_t, uint32_t, MicroAPI::HighLowPart::LOWEST>(scale1B16_1, manFP32_1);

        // 合并Zero和One，恢复原始列顺序
        MicroAPI::Interleave(scale1B16_0, scale1B16_1, scale1B16_0, scale1B16_1);
        // 左移7位得到BF16指数格式 (用于计算1/scale), 复用scale1BF16
        MicroAPI::ShiftLefts(scale1BF16, scale1B16_0, SHR_NUM_FOR_BF16, maskAll);
        // 输出scale (uint8), 复用mxScale1B8
        MicroAPI::Pack<uint8_t, uint16_t, MicroAPI::HighLowPart::LOWEST>(mxScale1B8, scale1B16_0);

        // 计算1/scale, 复用reversedShareExp1 (与原始DynamicMxQuant一致: 无零值检查)
        MicroAPI::Compare<uint16_t, CMPMODE::NE>(infMask, scale1BF16, maxEleBF16, maskAll);
        MicroAPI::Compare<uint16_t, CMPMODE::EQ>(invalidDataMask, scale1BF16, biasE8M0, maskAll);
        MicroAPI::Sub(reversedShareExp1, biasE8M0, scale1BF16, maskAll);
        MicroAPI::Select<uint16_t>(reversedShareExp1, reversedShareExp1, nanBF16, infMask);
        MicroAPI::Select<uint16_t>(reversedShareExp1, specialExp, reversedShareExp1, invalidDataMask);

        // ---------- 处理absMax2Dim2 (奇数列, -2轴scale的交织第二部分) ----------
        // Zero半 — 再次复用maxFP32_0, expFP32_0, manFP32_0
        MicroAPI::Cast<float, xDtype, castTraitXdtypetoFp32Zero>(
            (MicroAPI::RegTensor<float>&)maxFP32_0,
            (MicroAPI::RegTensor<xDtype>&)absMax2Dim2, maskAll);
        MicroAPI::Mul(
            (MicroAPI::RegTensor<float>&)maxFP32_0,
            (MicroAPI::RegTensor<float>&)maxFP32_0,
            (MicroAPI::RegTensor<float>&)invMax, maskFP32);
        MicroAPI::ShiftRights(expFP32_0, maxFP32_0, SHR_NUM_FOR_FP32, maskFP32);
        MicroAPI::And(manFP32_0, maxFP32_0, manMaskFP32, maskFP32);
        MicroAPI::CompareScalar<uint32_t, CMPMODE::GT>(p0, expFP32_0, NUMBER_ZERO_U32, maskFP32);
        MicroAPI::CompareScalar<uint32_t, CMPMODE::LT>(p0, expFP32_0, NUMBER_TWO_FIVE_FOUR, p0);
        MicroAPI::CompareScalar<uint32_t, CMPMODE::GT>(p0, manFP32_0, NUMBER_ZERO_U32, p0);
        MicroAPI::CompareScalar<uint32_t, CMPMODE::EQ>(p1, expFP32_0, NUMBER_ZERO_U32, maskFP32);
        MicroAPI::CompareScalar<uint32_t, CMPMODE::GT>(p1, manFP32_0, NUMBER_HALF_U32, p1);
        MicroAPI::MaskOr(p0, p0, p1, maskFP32);
        // 链内复用: maxFP32_0→expPlusOne, manFP32_0→extractExp
        MicroAPI::Adds(maxFP32_0, expFP32_0, 1, maskFP32);
        MicroAPI::Select(manFP32_0, maxFP32_0, expFP32_0, p0);
        MicroAPI::Pack<uint16_t, uint32_t, MicroAPI::HighLowPart::LOWEST>(scale1B16_0, manFP32_0);

        // One半 - 使用独立的p0Odd/p1Odd，与Zero半并行
        MicroAPI::Cast<float, xDtype, castTraitXdtypetoFp32One>(
            (MicroAPI::RegTensor<float>&)maxFP32_1,
            (MicroAPI::RegTensor<xDtype>&)absMax2Dim2, maskAll);
        MicroAPI::Mul(
            (MicroAPI::RegTensor<float>&)maxFP32_1,
            (MicroAPI::RegTensor<float>&)maxFP32_1,
            (MicroAPI::RegTensor<float>&)invMax, maskFP32);
        MicroAPI::ShiftRights(expFP32_1, maxFP32_1, SHR_NUM_FOR_FP32, maskFP32);
        MicroAPI::And(manFP32_1, maxFP32_1, manMaskFP32, maskFP32);
        MicroAPI::CompareScalar<uint32_t, CMPMODE::GT>(p0Odd, expFP32_1, NUMBER_ZERO_U32, maskFP32);
        MicroAPI::CompareScalar<uint32_t, CMPMODE::LT>(p0Odd, expFP32_1, NUMBER_TWO_FIVE_FOUR, p0Odd);
        MicroAPI::CompareScalar<uint32_t, CMPMODE::GT>(p0Odd, manFP32_1, NUMBER_ZERO_U32, p0Odd);
        MicroAPI::CompareScalar<uint32_t, CMPMODE::EQ>(p1Odd, expFP32_1, NUMBER_ZERO_U32, maskFP32);
        MicroAPI::CompareScalar<uint32_t, CMPMODE::GT>(p1Odd, manFP32_1, NUMBER_HALF_U32, p1Odd);
        MicroAPI::MaskOr(p0Odd, p0Odd, p1Odd, maskFP32);
        // 链内复用: maxFP32_1→expPlusOne, manFP32_1→extractExp
        MicroAPI::Adds(maxFP32_1, expFP32_1, 1, maskFP32);
        MicroAPI::Select(manFP32_1, maxFP32_1, expFP32_1, p0Odd);
        MicroAPI::Pack<uint16_t, uint32_t, MicroAPI::HighLowPart::LOWEST>(scale1B16_1, manFP32_1);

        // 合并Zero和One
        MicroAPI::Interleave(scale1B16_0, scale1B16_1, scale1B16_0, scale1B16_1);
        // 复用scale1BF16
        MicroAPI::ShiftLefts(scale1BF16, scale1B16_0, SHR_NUM_FOR_BF16, maskAll);
        MicroAPI::Pack<uint8_t, uint16_t, MicroAPI::HighLowPart::LOWEST>(mxScale2OneB8, scale1B16_0);

        // 计算1/scale, 复用absMax0 (循环后已死亡) (与原始DynamicMxQuant一致: 无零值检查)
        MicroAPI::Compare<uint16_t, CMPMODE::NE>(infMask, scale1BF16, maxEleBF16, maskAll);
        MicroAPI::Compare<uint16_t, CMPMODE::EQ>(invalidDataMask, scale1BF16, biasE8M0, maskAll);
        MicroAPI::Sub(absMax0, biasE8M0, scale1BF16, maskAll);
        MicroAPI::Select<uint16_t>(absMax0, absMax0, nanBF16, infMask);
        MicroAPI::Select<uint16_t>(absMax0, specialExp, absMax0, invalidDataMask);

        // 交织搬出-2轴的mxScale和1/scale
        MicroAPI::DataCopy<uint8_t, MicroAPI::StoreDist::DIST_INTLV_B8>(
            mxScale2Addr, mxScale1B8, mxScale2OneB8, maskB8);
        MicroAPI::DataCopy<uint16_t, MicroAPI::StoreDist::DIST_INTLV_B16>(
            mxScale2ReciprocalAddr, reversedShareExp1, absMax0, maskAll);
    }
}

// DynamicDtypeRange Default Scale算法实现 (scaleAlg=2, dstTypeMax=0.0/6.0/7.0, FP4_E2M1专用)
// 整体框架与ComputeScaleOcp一致：循环blockCount次处理-1轴scale，循环后处理-2轴scale
// -1轴算法：取BF16绝对值max → addValueBit进位取指数法
// -2轴算法：累积BF16绝对值max → SUB_NUM_FOR_SCALE减法取指数法
template <typename xDtype, typename y1Dtype, typename y2Dtype, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__aicore__ inline void
DynamicMxQuantWithDualAxisBase<xDtype, y1Dtype, y2Dtype, roundMode, scaleAlg>::ComputeScaleDynamicDefault(
    uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr, __ubuf__ uint8_t* mxScale1Addr,
    __ubuf__ uint16_t* mxScale1ReciprocalAddr, __ubuf__ uint8_t* mxScale2Addr,
    __ubuf__ uint16_t* mxScale2ReciprocalAddr)
{
    __VEC_SCOPE__
    {
        // ========== 输入数据寄存器 ==========
        MicroAPI::RegTensor<xDtype> x0;
        MicroAPI::RegTensor<xDtype> x1;
        MicroAPI::RegTensor<bfloat16_t> x0BF16;
        MicroAPI::RegTensor<bfloat16_t> x1BF16;

        // ========== 绝对值和max寄存器 ==========
        MicroAPI::RegTensor<uint16_t> absVal0;         // x0的BF16绝对值
        MicroAPI::RegTensor<uint16_t> absVal1;         // x1的BF16绝对值
        MicroAPI::RegTensor<uint16_t> absMaxDim1;      // -1轴block内绝对值max (ReduceMax后)
        MicroAPI::RegTensor<uint16_t> absMax1Dim2;     // -2轴累积绝对值max (偶数列, 对应x0)
        MicroAPI::RegTensor<uint16_t> absMax2Dim2;     // -2轴累积绝对值max (奇数列, 对应x1)

        // ========== -1轴scale计算寄存器 (循环后复用于-2轴) ==========
        MicroAPI::RegTensor<uint16_t> expOnly;         // 提取的指数位, 循环后复用为dim2ExpOnly
        MicroAPI::RegTensor<uint16_t> addedVal;        // addValueBit进位后的值, 循环后复用为dim2SubResult
        MicroAPI::RegTensor<uint16_t> sharedExp;       // 指数差值, 循环后复用为dim2ExpExtract
        MicroAPI::RegTensor<uint16_t> scaleValue;      // E8M0 scale值, 循环后复用为mxScale2B16
        MicroAPI::RegTensor<uint8_t> mxScale1B8;       // -1轴scale输出, 循环后复用为mxScale2ZeroB8
        MicroAPI::RegTensor<uint16_t> reversedShareExp1; // -1轴1/scale, 循环后复用为reversedShareExp2Zero

        // ========== -2轴独立寄存器 (需与复用寄存器同时存活，无法复用) ==========
        MicroAPI::RegTensor<uint8_t> mxScale2OneB8;    // 与mxScale1B8同时存活于最终DataCopy

        // ========== 常量寄存器 ==========
        MicroAPI::RegTensor<uint16_t> absMask;
        MicroAPI::Duplicate(absMask, ABS_MASK_FOR_16BIT);        // 绝对值掩码 0x7fff
        MicroAPI::RegTensor<uint16_t> expMaskBF16;
        MicroAPI::Duplicate(expMaskBF16, EXP_MASK_BF16);         // BF16指数掩码 0x7f80
        MicroAPI::RegTensor<uint16_t> expMaskFP16;
        MicroAPI::Duplicate(expMaskFP16, EXP_MASK_FP16);         // FP16指数掩码 0x7c00 (INF/NAN检测)
        MicroAPI::RegTensor<uint16_t> addValue;
        MicroAPI::Duplicate(addValue, addValueBit_);              // BF16尾数进位值
        MicroAPI::RegTensor<uint16_t> maxExpValue;
        MicroAPI::Duplicate(maxExpValue, FP4_E2M1_BF16_MAX_EXP); // FP4_E2M1的emax在BF16中的表示
        MicroAPI::RegTensor<uint16_t> subNumForScale;
        MicroAPI::Duplicate(subNumForScale, subNumForScale_);     // -2轴减法常量
        MicroAPI::RegTensor<uint16_t> nanE8M0;
        MicroAPI::Duplicate(nanE8M0, NAN_FOR_FP8_E8M0);          // E8M0的NAN值 0xFF
        MicroAPI::RegTensor<uint16_t> biasE8M0;
        MicroAPI::Duplicate(biasE8M0, BF16_EXP_BIAS);            // BF16指数偏移 0x7f00
        MicroAPI::RegTensor<uint16_t> zero;
        MicroAPI::Duplicate(zero, 0);
        MicroAPI::RegTensor<uint16_t> nanBF16;
        MicroAPI::Duplicate(nanBF16, NAN_CUSTOMIZATION);          // NAN_CUSTOMIZATION 0x7f81
        MicroAPI::RegTensor<uint16_t> specialExp;
        MicroAPI::Duplicate(specialExp, SPECIAL_EXP_THRESHOLD);   // 特殊指数阈值 0x0040

        MicroAPI::Duplicate(absMax1Dim2, 0);
        MicroAPI::Duplicate(absMax2Dim2, 0);

        // ========== Mask定义 ==========
        MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<xDtype, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg maskB8 = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg maskReduceB8 = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::VL8>();
        MicroAPI::MaskReg maskReduceB16 = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::VL16>();

        MicroAPI::MaskReg infMask;
        MicroAPI::MaskReg zeroMask;
        MicroAPI::MaskReg invalidDataMask;
        MicroAPI::MaskReg infNanDataMask0;
        MicroAPI::MaskReg infNanDataMask1;

        // ========================================================================
        // 循环blockCount次: 计算-1轴scale并累积-2轴BF16绝对值max
        // DynamicDtypeRange需要完整的BF16绝对值 (不仅仅是指数)
        // ========================================================================
        for (uint16_t i = 0; i < blockCount; i++) {
            // 1. 交织搬运输入数据: 将256个xDtype按偶奇拆分为x0(偶), x1(奇)
            MicroAPI::DataCopy<xDtype, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_DINTLV_B16>(
                x0, x1, xAddr, vlForHalfNumber_ * DIGIT_TWO);

            // 2. 获取BF16绝对值 (区分half和bf16输入)
            if constexpr (IsSameType<xDtype, half>::value) {
                // FP16输入: 先检查INF/NAN，再转BF16(RINT)，取绝对值，INF/NAN替换为BF16 INF
                MicroAPI::And(expOnly, (MicroAPI::RegTensor<uint16_t>&)x0, expMaskFP16, maskAll);
                MicroAPI::Compare<uint16_t, CMPMODE::NE>(infNanDataMask0, expOnly, expMaskFP16, maskAll);
                MicroAPI::And(expOnly, (MicroAPI::RegTensor<uint16_t>&)x1, expMaskFP16, maskAll);
                MicroAPI::Compare<uint16_t, CMPMODE::NE>(infNanDataMask1, expOnly, expMaskFP16, maskAll);
                // 转BF16 (使用CAST_RINT四舍五入，不同于OCP的CAST_TRUNC截断)
                MicroAPI::Cast<bfloat16_t, xDtype, castTraitHalf2BF16Rint>(x0BF16, x0, maskAll);
                MicroAPI::Cast<bfloat16_t, xDtype, castTraitHalf2BF16Rint>(x1BF16, x1, maskAll);
                // 取绝对值
                MicroAPI::And(absVal0, (MicroAPI::RegTensor<uint16_t>&)x0BF16, absMask, maskAll);
                MicroAPI::And(absVal1, (MicroAPI::RegTensor<uint16_t>&)x1BF16, absMask, maskAll);
                // INF/NAN位置替换为BF16的INF (0x7f80)
                MicroAPI::Select<uint16_t>(absVal0, absVal0, expMaskBF16, infNanDataMask0);
                MicroAPI::Select<uint16_t>(absVal1, absVal1, expMaskBF16, infNanDataMask1);
            } else {
                // BF16输入: 直接取绝对值
                MicroAPI::And(absVal0, (MicroAPI::RegTensor<uint16_t>&)x0, absMask, maskAll);
                MicroAPI::And(absVal1, (MicroAPI::RegTensor<uint16_t>&)x1, absMask, maskAll);
            }

            // 3. -1轴: 偶奇Max + ReduceMaxWithDataBlock，得到每32个元素的绝对值max
            MicroAPI::Max(absMaxDim1, absVal0, absVal1, maskAll);
            MicroAPI::ReduceMaxWithDataBlock(absMaxDim1, absMaxDim1, maskAll);

            // 4. -2轴: 逐行累积偶数列和奇数列的BF16绝对值max
            MicroAPI::Max(absMax1Dim2, absMax1Dim2, absVal0, maskAll);
            MicroAPI::Max(absMax2Dim2, absMax2Dim2, absVal1, maskAll);

            // ============================================================
            // 5. 计算-1轴DynamicDtypeRange Default Scale
            //    ReduceMax后8个绝对值max紧凑存储在位置0-7
            //    使用addValueBit进位法: 将addValueBit加到完整BF16绝对值上，
            //    通过尾数进位自动实现指数的四舍五入
            // ============================================================

            // 提取指数位 (仅用于INF/NAN、零值、subnormal检查)
            MicroAPI::And(expOnly, absMaxDim1, expMaskBF16, maskAll);
            // INF/NAN检查: 指数全1
            MicroAPI::Compare<uint16_t, CMPMODE::NE>(infMask, expOnly, expMaskBF16, maskAll);
            // 零值检查
            MicroAPI::Compare<uint16_t, CMPMODE::NE>(zeroMask, expOnly, zero, maskAll);
            // subnormal检查: 指数 < FP4_E2M1_BF16_MAX_EXP (注意使用LT，不是LE)
            MicroAPI::Compare<uint16_t, CMPMODE::LT>(invalidDataMask, expOnly, maxExpValue, maskAll);

            // addValueBit进位: 将addValueBit加到完整BF16绝对值上
            MicroAPI::Add(addedVal, absMaxDim1, addValue, maskAll);
            // 从进位后的结果中提取指数
            MicroAPI::And(addedVal, addedVal, expMaskBF16, maskAll);
            // subnormal场景: 使用maxExpValue (FP4_E2M1_BF16_MAX_EXP)
            MicroAPI::Select<uint16_t>(addedVal, maxExpValue, addedVal, invalidDataMask);
            // 减去FP4_E2M1_BF16_MAX_EXP得到指数差值
            MicroAPI::Sub(sharedExp, addedVal, maxExpValue, maskAll);
            // 右移7位，将BF16指数移到低8位 → E8M0 scale
            MicroAPI::ShiftRights(scaleValue, sharedExp, SHR_NUM_FOR_BF16, maskAll);
            // INF/NAN → NAN_FOR_FP8_E8M0 (0xFF)
            MicroAPI::Select<uint16_t>(scaleValue, scaleValue, nanE8M0, infMask);
            // 零值 → 0
            MicroAPI::Select<uint16_t>(scaleValue, scaleValue, zero, zeroMask);

            // 输出-1轴scale (uint8)
            MicroAPI::Pack<uint8_t, uint16_t, MicroAPI::HighLowPart::LOWEST>(mxScale1B8, scaleValue);
            MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                mxScale1Addr, mxScale1B8, oneBlockCountB8_, maskReduceB8);

            // 计算-1轴1/scale
            // sharedExp是左移7位前的指数差值，可直接用于BF16域1/scale计算
            MicroAPI::Compare<uint16_t, CMPMODE::EQ>(invalidDataMask, sharedExp, biasE8M0, maskAll);
            MicroAPI::Sub(reversedShareExp1, biasE8M0, sharedExp, maskAll);
            MicroAPI::Select<uint16_t>(reversedShareExp1, reversedShareExp1, nanBF16, infMask);
            MicroAPI::Select<uint16_t>(reversedShareExp1, reversedShareExp1, zero, zeroMask);
            MicroAPI::Select<uint16_t>(reversedShareExp1, specialExp, reversedShareExp1, invalidDataMask);
            MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                mxScale1ReciprocalAddr, reversedShareExp1, oneBlockCountB16_, maskReduceB16);
        }

        // ========================================================================
        // 循环结束后，计算-2轴DynamicDtypeRange Default Scale
        // absMax1Dim2: 128个偶数列的累积BF16绝对值max
        // absMax2Dim2: 128个奇数列的累积BF16绝对值max
        // 使用SUB_NUM_FOR_SCALE减法: 直接从完整BF16绝对值减去
        // (FP4_E2M1_BF16_MAX_EXP - addValueBit)，等效于addValueBit进位法
        // ========================================================================

        // ---------- 处理absMax1Dim2 (偶数列, -2轴scale的交织第一部分) ----------
        // 复用expOnly为dim2ExpOnly
        MicroAPI::And(expOnly, absMax1Dim2, expMaskBF16, maskAll);
        // INF/NAN检查
        MicroAPI::Compare<uint16_t, CMPMODE::NE>(infMask, expOnly, expMaskBF16, maskAll);
        // 零值检查
        MicroAPI::Compare<uint16_t, CMPMODE::NE>(zeroMask, expOnly, zero, maskAll);
        // subnormal检查: 指数 < FP4_E2M1_BF16_MAX_EXP
        MicroAPI::Compare<uint16_t, CMPMODE::LT>(invalidDataMask, expOnly, maxExpValue, maskAll);

        // 复用addedVal为dim2SubResult
        MicroAPI::Sub(addedVal, absMax1Dim2, subNumForScale, maskAll);
        // subnormal → 0
        MicroAPI::Select<uint16_t>(addedVal, zero, addedVal, invalidDataMask);
        // 右移7位 → E8M0 scale, 复用scaleValue为mxScale2B16
        MicroAPI::ShiftRights(scaleValue, addedVal, SHR_NUM_FOR_BF16, maskAll);
        // INF/NAN → NAN
        MicroAPI::Select<uint16_t>(scaleValue, scaleValue, nanE8M0, infMask);
        // 零值 → 0
        MicroAPI::Select<uint16_t>(scaleValue, scaleValue, zero, zeroMask);

        // 输出scale (uint8) — mxScale1B8复用为mxScale2ZeroB8
        MicroAPI::Pack<uint8_t, uint16_t, MicroAPI::HighLowPart::LOWEST>(mxScale1B8, scaleValue);

        // 计算-2轴1/scale — reversedShareExp1复用为reversedShareExp2Zero
        // 复用sharedExp为dim2ExpExtract
        MicroAPI::And(sharedExp, addedVal, expMaskBF16, maskAll);
        MicroAPI::Compare<uint16_t, CMPMODE::EQ>(invalidDataMask, sharedExp, biasE8M0, maskAll);
        MicroAPI::Sub(reversedShareExp1, biasE8M0, sharedExp, maskAll);
        MicroAPI::Select<uint16_t>(reversedShareExp1, reversedShareExp1, nanBF16, infMask);
        MicroAPI::Select<uint16_t>(reversedShareExp1, reversedShareExp1, zero, zeroMask);
        MicroAPI::Select<uint16_t>(reversedShareExp1, specialExp, reversedShareExp1, invalidDataMask);

        // ---------- 处理absMax2Dim2 (奇数列, -2轴scale的交织第二部分) ----------
        // 再次复用expOnly, addedVal, sharedExp, scaleValue
        MicroAPI::And(expOnly, absMax2Dim2, expMaskBF16, maskAll);
        MicroAPI::Compare<uint16_t, CMPMODE::NE>(infMask, expOnly, expMaskBF16, maskAll);
        MicroAPI::Compare<uint16_t, CMPMODE::NE>(zeroMask, expOnly, zero, maskAll);
        MicroAPI::Compare<uint16_t, CMPMODE::LT>(invalidDataMask, expOnly, maxExpValue, maskAll);

        MicroAPI::Sub(addedVal, absMax2Dim2, subNumForScale, maskAll);
        MicroAPI::Select<uint16_t>(addedVal, zero, addedVal, invalidDataMask);
        // 复用scaleValue为mxScale2OneB16
        MicroAPI::ShiftRights(scaleValue, addedVal, SHR_NUM_FOR_BF16, maskAll);
        MicroAPI::Select<uint16_t>(scaleValue, scaleValue, nanE8M0, infMask);
        MicroAPI::Select<uint16_t>(scaleValue, scaleValue, zero, zeroMask);

        MicroAPI::Pack<uint8_t, uint16_t, MicroAPI::HighLowPart::LOWEST>(mxScale2OneB8, scaleValue);

        // 计算-2轴1/scale — absVal0复用为reversedShareExp2One (循环后死亡，与reversedShareExp1不冲突)
        MicroAPI::And(sharedExp, addedVal, expMaskBF16, maskAll);
        MicroAPI::Compare<uint16_t, CMPMODE::EQ>(invalidDataMask, sharedExp, biasE8M0, maskAll);
        MicroAPI::Sub(absVal0, biasE8M0, sharedExp, maskAll);
        MicroAPI::Select<uint16_t>(absVal0, absVal0, nanBF16, infMask);
        MicroAPI::Select<uint16_t>(absVal0, absVal0, zero, zeroMask);
        MicroAPI::Select<uint16_t>(absVal0, specialExp, absVal0, invalidDataMask);

        // 交织搬出-2轴的mxScale和1/scale
        MicroAPI::DataCopy<uint8_t, MicroAPI::StoreDist::DIST_INTLV_B8>(
            mxScale2Addr, mxScale1B8, mxScale2OneB8, maskB8);
        MicroAPI::DataCopy<uint16_t, MicroAPI::StoreDist::DIST_INTLV_B16>(
            mxScale2ReciprocalAddr, reversedShareExp1, absVal0, maskAll);
    }
}

// DynamicDtypeRange Custom Scale算法实现 (scaleAlg=2, 自定义dstTypeMax, FP4_E2M1专用)
// 与CuBALS (scaleAlg=1) 类似，使用FP32精度乘以invDstTypeMax_，但：
// 1. 乘法因子为invDstTypeMax_ (1/dstTypeMax) 而非invDtypeMax_ (1/AMax(DType))
// 2. 条件舍入仅处理normal场景 (exp>0 && exp<254 && man>0)，不处理subnormal场景
template <typename xDtype, typename y1Dtype, typename y2Dtype, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__aicore__ inline void
DynamicMxQuantWithDualAxisBase<xDtype, y1Dtype, y2Dtype, roundMode, scaleAlg>::ComputeScaleDynamicCustom(
    uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr, __ubuf__ uint8_t* mxScale1Addr,
    __ubuf__ uint16_t* mxScale1ReciprocalAddr, __ubuf__ uint8_t* mxScale2Addr,
    __ubuf__ uint16_t* mxScale2ReciprocalAddr)
{
    __VEC_SCOPE__
    {
        // ========== 输入数据寄存器 ==========
        MicroAPI::RegTensor<xDtype> x0;
        MicroAPI::RegTensor<xDtype> x1;

        // ========== 绝对值和max寄存器 ==========
        MicroAPI::RegTensor<uint16_t> absMax0;         // x0的绝对值
        MicroAPI::RegTensor<uint16_t> absMax1;         // x1的绝对值
        MicroAPI::RegTensor<uint16_t> absMaxDim1;      // -1轴方向block内绝对值max
        MicroAPI::RegTensor<uint16_t> absMax1Dim2;     // -2轴方向累积max (偶数列, 对应x0)
        MicroAPI::RegTensor<uint16_t> absMax2Dim2;     // -2轴方向累积max (奇数列, 对应x1)
        MicroAPI::RegTensor<uint16_t> zeroB16;         // -1轴Interleave用零寄存器

        // ========== FP32计算寄存器 ==========
        // -1轴: Interleave-with-0后单次Cast Zero处理全部8个值，仅需一组FP32寄存器
        // -2轴: 仍需Zero/One两组独立处理
        MicroAPI::RegTensor<uint32_t> maxFP32_0;       // FP32表示, 链内复用为expPlusOne
        MicroAPI::RegTensor<uint32_t> maxFP32_1;       // -2轴奇数部分FP32表示
        MicroAPI::RegTensor<uint32_t> expFP32_0;       // FP32指数
        MicroAPI::RegTensor<uint32_t> expFP32_1;       // -2轴奇数部分FP32指数
        MicroAPI::RegTensor<uint32_t> manFP32_0;       // FP32尾数, 链内复用为extractExp
        MicroAPI::RegTensor<uint32_t> manFP32_1;       // -2轴奇数部分FP32尾数

        // scale输出寄存器 (循环后复用于-2轴scale输出)
        MicroAPI::RegTensor<uint16_t> scale1B16_0;     // E8M0 uint16偶数, 循环后复用为mxScale2ZeroB16
        MicroAPI::RegTensor<uint16_t> scale1B16_1;     // E8M0 uint16奇数, 循环后复用为mxScale2OneB16
        MicroAPI::RegTensor<uint16_t> scale1BF16;      // BF16指数格式, 循环后复用为scale2BF16
        MicroAPI::RegTensor<uint8_t> mxScale1B8;       // uint8 scale, 循环后复用为mxScale2ZeroB8
        MicroAPI::RegTensor<uint16_t> reversedShareExp1; // 1/scale BF16, 循环后复用为reversedShareExp2Zero

        // -2轴独立寄存器 (需与复用寄存器同时存活，无法复用)
        MicroAPI::RegTensor<uint8_t> mxScale2OneB8;    // 与mxScale1B8同时存活于最终DataCopy

        // ========== 常量寄存器 ==========
        MicroAPI::RegTensor<uint16_t> absMask;
        MicroAPI::Duplicate(absMask, ABS_MASK_FOR_16BIT);
        MicroAPI::RegTensor<float> invDstTypeMaxReg;
        MicroAPI::Duplicate(invDstTypeMaxReg, invDstTypeMax_);     // 1/dstTypeMax, FP32表示
        MicroAPI::RegTensor<uint32_t> manMaskFP32;
        MicroAPI::Duplicate(manMaskFP32, MAN_MASK_FLOAT);          // FP32尾数掩码
        MicroAPI::RegTensor<uint32_t> scaleBiasFP32;
        MicroAPI::Duplicate(scaleBiasFP32, FP32_EXP_BIAS_CUBLAS);  // BF16偏移在uint32

        MicroAPI::RegTensor<uint16_t> nanE8M0;
        MicroAPI::Duplicate(nanE8M0, NAN_FOR_FP8_E8M0);
        MicroAPI::RegTensor<uint16_t> biasE8M0;
        MicroAPI::Duplicate(biasE8M0, BF16_EXP_BIAS);
        MicroAPI::RegTensor<uint16_t> zero;
        MicroAPI::Duplicate(zero, 0);
        MicroAPI::RegTensor<uint16_t> nanBF16;
        MicroAPI::Duplicate(nanBF16, NAN_CUSTOMIZATION);
        MicroAPI::RegTensor<uint16_t> specialExp;
        MicroAPI::Duplicate(specialExp, SPECIAL_EXP_THRESHOLD);
        MicroAPI::RegTensor<uint16_t> maxEleBF16;
        MicroAPI::Duplicate(maxEleBF16, EXP_MASK_BF16);

        MicroAPI::Duplicate(absMax1Dim2, 0);
        MicroAPI::Duplicate(absMax2Dim2, 0);
        MicroAPI::Duplicate(zeroB16, 0);

        // ========== Mask定义 ==========
        MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<xDtype, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg maskB8 = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg maskReduceB8 = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::VL8>();
        MicroAPI::MaskReg maskReduceB16 = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::VL16>();
        MicroAPI::MaskReg maskFP32 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();

        MicroAPI::MaskReg p0;              // 条件舍入: normal场景掩码
        MicroAPI::MaskReg infMask;
        MicroAPI::MaskReg invalidDataMask;

        // ========================================================================
        // 循环blockCount次，每次处理一行，计算-1轴scale并累积-2轴max
        // 与CuBALS一致: 取绝对值 → Max → ReduceMax → FP32域scale计算
        // ========================================================================
        for (uint16_t i = 0; i < blockCount; i++) {
            // 1. 交织搬运输入数据: 将256个xDtype按偶奇拆分为x0(偶), x1(奇)
            MicroAPI::DataCopy<xDtype, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_DINTLV_B16>(
                x0, x1, xAddr, vlForHalfNumber_ * DIGIT_TWO);

            // 2. 取绝对值: 清除符号位，保留指数和尾数
            MicroAPI::And(absMax0, (MicroAPI::RegTensor<uint16_t>&)x0, absMask, maskAll);
            MicroAPI::And(absMax1, (MicroAPI::RegTensor<uint16_t>&)x1, absMask, maskAll);

            // 3. -1轴: 先取偶奇max，再ReduceMaxWithDataBlock得到每32个元素的绝对值max
            MicroAPI::Max(absMaxDim1, absMax0, absMax1, maskAll);
            MicroAPI::ReduceMaxWithDataBlock(absMaxDim1, absMaxDim1, maskAll);

            // 4. -2轴: 逐行累积偶数列和奇数列的绝对值max
            MicroAPI::Max(absMax1Dim2, absMax1Dim2, absMax0, maskAll);
            MicroAPI::Max(absMax2Dim2, absMax2Dim2, absMax1, maskAll);

            // ============================================================
            // 5. 计算-1轴Custom Scale (FP32精度)
            //    ReduceMaxWithDataBlock后，8个max紧凑存储在寄存器前8个位置
            //    与0交织后，Cast Zero一次处理全部8个值转为FP32
            //    乘以invDstTypeMax_ (1/dstTypeMax)
            //    仅normal场景条件舍入 (无subnormal舍入)
            //    (与原始DynamicMxQuant一致)
            // ============================================================

            // 与0交织: [v0,0,v1,0,...,v7,0,...] → Cast Zero可一次取出全部8个有效值
            MicroAPI::Interleave(absMaxDim1, zeroB16, absMaxDim1, zeroB16);
            MicroAPI::Cast<float, xDtype, castTraitXdtypetoFp32Zero>(
                (MicroAPI::RegTensor<float>&)maxFP32_0,
                (MicroAPI::RegTensor<xDtype>&)absMaxDim1, maskAll);
            // 乘以 1/dstTypeMax
            MicroAPI::Mul(
                (MicroAPI::RegTensor<float>&)maxFP32_0,
                (MicroAPI::RegTensor<float>&)maxFP32_0,
                invDstTypeMaxReg, maskFP32);
            // 提取FP32指数: 右移23位
            MicroAPI::ShiftRights(expFP32_0, maxFP32_0, SHR_NUM_FOR_FP32, maskFP32);
            // 提取FP32尾数: 与尾数掩码
            MicroAPI::And(manFP32_0, maxFP32_0, manMaskFP32, maskFP32);
            // 条件舍入: 仅normal场景 (exp>0 && exp<254 && man>0) → exp+1
            // 注意: 与CuBALS不同，DynamicDtypeRange Custom不处理subnormal场景
            MicroAPI::CompareScalar<uint32_t, CMPMODE::GT>(p0, expFP32_0, NUMBER_ZERO_U32, maskFP32);
            MicroAPI::CompareScalar<uint32_t, CMPMODE::LT>(p0, expFP32_0, NUMBER_TWO_FIVE_FOUR, p0);
            MicroAPI::CompareScalar<uint32_t, CMPMODE::GT>(p0, manFP32_0, NUMBER_ZERO_U32, p0);
            // 执行条件加1
            MicroAPI::Adds(maxFP32_0, expFP32_0, 1, maskFP32);
            MicroAPI::Select(manFP32_0, maxFP32_0, expFP32_0, p0);
            // Pack到uint16 (INF/NAN→0xFF, zero→0 自然通过条件舍入保持, 在BF16域1/scale中统一处理)
            MicroAPI::Pack<uint16_t, uint32_t, MicroAPI::HighLowPart::LOWEST>(scale1B16_0, manFP32_0);

            // 左移7位，将E8M0值定位到BF16指数域 (用于计算1/scale)
            MicroAPI::ShiftLefts(scale1BF16, scale1B16_0, SHR_NUM_FOR_BF16, maskAll);

            // --- 输出-1轴scale (uint8) ---
            MicroAPI::Pack<uint8_t, uint16_t, MicroAPI::HighLowPart::LOWEST>(mxScale1B8, scale1B16_0);
            MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                mxScale1Addr, mxScale1B8, oneBlockCountB8_, maskReduceB8);

            // --- 计算并输出-1轴 1/scale (与原始DynamicMxQuant一致: inf→nan, special→specialExp, 无零值检查) ---
            MicroAPI::Compare<uint16_t, CMPMODE::NE>(infMask, scale1BF16, maxEleBF16, maskAll);
            MicroAPI::Compare<uint16_t, CMPMODE::EQ>(invalidDataMask, scale1BF16, biasE8M0, maskAll);
            MicroAPI::Sub(reversedShareExp1, biasE8M0, scale1BF16, maskAll);
            MicroAPI::Select<uint16_t>(reversedShareExp1, reversedShareExp1, nanBF16, infMask);
            MicroAPI::Select<uint16_t>(reversedShareExp1, specialExp, reversedShareExp1, invalidDataMask);
            MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                mxScale1ReciprocalAddr, reversedShareExp1, oneBlockCountB16_, maskReduceB16);
            // 恢复zeroB16 (Interleave会修改dst1)
            MicroAPI::Duplicate(zeroB16, 0);
        }

        // ========================================================================
        // 循环结束后，计算-2轴Custom Scale
        // absMax1Dim2: 128个偶数列的累积绝对值max
        // absMax2Dim2: 128个奇数列的累积绝对值max
        // 每个需要拆分为Zero/One两半分别做FP32计算，再合并
        // ========================================================================

        // ---------- 处理absMax1Dim2 (偶数列, -2轴scale的交织第一部分) ----------
        // Zero半 (偶数位) — 复用循环体偶数部分寄存器
        MicroAPI::Cast<float, xDtype, castTraitXdtypetoFp32Zero>(
            (MicroAPI::RegTensor<float>&)maxFP32_0,
            (MicroAPI::RegTensor<xDtype>&)absMax1Dim2, maskAll);
        MicroAPI::Mul(
            (MicroAPI::RegTensor<float>&)maxFP32_0,
            (MicroAPI::RegTensor<float>&)maxFP32_0,
            invDstTypeMaxReg, maskFP32);
        MicroAPI::ShiftRights(expFP32_0, maxFP32_0, SHR_NUM_FOR_FP32, maskFP32);
        MicroAPI::And(manFP32_0, maxFP32_0, manMaskFP32, maskFP32);
        // 条件舍入: 仅normal场景 (无subnormal)
        MicroAPI::CompareScalar<uint32_t, CMPMODE::GT>(p0, expFP32_0, NUMBER_ZERO_U32, maskFP32);
        MicroAPI::CompareScalar<uint32_t, CMPMODE::LT>(p0, expFP32_0, NUMBER_TWO_FIVE_FOUR, p0);
        MicroAPI::CompareScalar<uint32_t, CMPMODE::GT>(p0, manFP32_0, NUMBER_ZERO_U32, p0);
        // 链内复用: maxFP32_0→expPlusOne, expFP32_0同时作为exp和最终结果
        MicroAPI::Adds(maxFP32_0, expFP32_0, 1, maskFP32);
        MicroAPI::Select(expFP32_0, maxFP32_0, expFP32_0, p0);
        MicroAPI::Pack<uint16_t, uint32_t, MicroAPI::HighLowPart::LOWEST>(scale1B16_0, expFP32_0);

        // One半 (奇数位) — 复用循环体奇数部分寄存器
        MicroAPI::Cast<float, xDtype, castTraitXdtypetoFp32One>(
            (MicroAPI::RegTensor<float>&)maxFP32_1,
            (MicroAPI::RegTensor<xDtype>&)absMax1Dim2, maskAll);
        MicroAPI::Mul(
            (MicroAPI::RegTensor<float>&)maxFP32_1,
            (MicroAPI::RegTensor<float>&)maxFP32_1,
            invDstTypeMaxReg, maskFP32);
        MicroAPI::ShiftRights(expFP32_1, maxFP32_1, SHR_NUM_FOR_FP32, maskFP32);
        MicroAPI::And(manFP32_1, maxFP32_1, manMaskFP32, maskFP32);
        MicroAPI::CompareScalar<uint32_t, CMPMODE::GT>(p0, expFP32_1, NUMBER_ZERO_U32, maskFP32);
        MicroAPI::CompareScalar<uint32_t, CMPMODE::LT>(p0, expFP32_1, NUMBER_TWO_FIVE_FOUR, p0);
        MicroAPI::CompareScalar<uint32_t, CMPMODE::GT>(p0, manFP32_1, NUMBER_ZERO_U32, p0);
        MicroAPI::Adds(maxFP32_1, expFP32_1, 1, maskFP32);
        MicroAPI::Select(expFP32_1, maxFP32_1, expFP32_1, p0);
        MicroAPI::Pack<uint16_t, uint32_t, MicroAPI::HighLowPart::LOWEST>(scale1B16_1, expFP32_1);

        // 合并Zero和One，恢复原始列顺序 — scale输出寄存器复用
        MicroAPI::Interleave(scale1B16_0, scale1B16_1, scale1B16_0, scale1B16_1);
        // 左移7位得到BF16指数格式 — scale1BF16复用
        MicroAPI::ShiftLefts(scale1BF16, scale1B16_0, SHR_NUM_FOR_BF16, maskAll);
        // 输出scale (uint8) — mxScale1B8复用为mxScale2ZeroB8
        MicroAPI::Pack<uint8_t, uint16_t, MicroAPI::HighLowPart::LOWEST>(mxScale1B8, scale1B16_0);

        // 计算1/scale — reversedShareExp1复用为reversedShareExp2Zero (与原始DynamicMxQuant一致: 无零值检查)
        MicroAPI::Compare<uint16_t, CMPMODE::NE>(infMask, scale1BF16, maxEleBF16, maskAll);
        MicroAPI::Compare<uint16_t, CMPMODE::EQ>(invalidDataMask, scale1BF16, biasE8M0, maskAll);
        MicroAPI::Sub(reversedShareExp1, biasE8M0, scale1BF16, maskAll);
        MicroAPI::Select<uint16_t>(reversedShareExp1, reversedShareExp1, nanBF16, infMask);
        MicroAPI::Select<uint16_t>(reversedShareExp1, specialExp, reversedShareExp1, invalidDataMask);

        // ---------- 处理absMax2Dim2 (奇数列, -2轴scale的交织第二部分) ----------
        // Zero半 — 复用循环体偶数部分寄存器
        MicroAPI::Cast<float, xDtype, castTraitXdtypetoFp32Zero>(
            (MicroAPI::RegTensor<float>&)maxFP32_0,
            (MicroAPI::RegTensor<xDtype>&)absMax2Dim2, maskAll);
        MicroAPI::Mul(
            (MicroAPI::RegTensor<float>&)maxFP32_0,
            (MicroAPI::RegTensor<float>&)maxFP32_0,
            invDstTypeMaxReg, maskFP32);
        MicroAPI::ShiftRights(expFP32_0, maxFP32_0, SHR_NUM_FOR_FP32, maskFP32);
        MicroAPI::And(manFP32_0, maxFP32_0, manMaskFP32, maskFP32);
        MicroAPI::CompareScalar<uint32_t, CMPMODE::GT>(p0, expFP32_0, NUMBER_ZERO_U32, maskFP32);
        MicroAPI::CompareScalar<uint32_t, CMPMODE::LT>(p0, expFP32_0, NUMBER_TWO_FIVE_FOUR, p0);
        MicroAPI::CompareScalar<uint32_t, CMPMODE::GT>(p0, manFP32_0, NUMBER_ZERO_U32, p0);
        MicroAPI::Adds(maxFP32_0, expFP32_0, 1, maskFP32);
        MicroAPI::Select(expFP32_0, maxFP32_0, expFP32_0, p0);
        MicroAPI::Pack<uint16_t, uint32_t, MicroAPI::HighLowPart::LOWEST>(scale1B16_0, expFP32_0);

        // One半 — 复用循环体奇数部分寄存器
        MicroAPI::Cast<float, xDtype, castTraitXdtypetoFp32One>(
            (MicroAPI::RegTensor<float>&)maxFP32_1,
            (MicroAPI::RegTensor<xDtype>&)absMax2Dim2, maskAll);
        MicroAPI::Mul(
            (MicroAPI::RegTensor<float>&)maxFP32_1,
            (MicroAPI::RegTensor<float>&)maxFP32_1,
            invDstTypeMaxReg, maskFP32);
        MicroAPI::ShiftRights(expFP32_1, maxFP32_1, SHR_NUM_FOR_FP32, maskFP32);
        MicroAPI::And(manFP32_1, maxFP32_1, manMaskFP32, maskFP32);
        MicroAPI::CompareScalar<uint32_t, CMPMODE::GT>(p0, expFP32_1, NUMBER_ZERO_U32, maskFP32);
        MicroAPI::CompareScalar<uint32_t, CMPMODE::LT>(p0, expFP32_1, NUMBER_TWO_FIVE_FOUR, p0);
        MicroAPI::CompareScalar<uint32_t, CMPMODE::GT>(p0, manFP32_1, NUMBER_ZERO_U32, p0);
        MicroAPI::Adds(maxFP32_1, expFP32_1, 1, maskFP32);
        MicroAPI::Select(expFP32_1, maxFP32_1, expFP32_1, p0);
        MicroAPI::Pack<uint16_t, uint32_t, MicroAPI::HighLowPart::LOWEST>(scale1B16_1, expFP32_1);

        // 合并Zero和One — scale输出寄存器复用
        MicroAPI::Interleave(scale1B16_0, scale1B16_1, scale1B16_0, scale1B16_1);
        // scale1BF16复用
        MicroAPI::ShiftLefts(scale1BF16, scale1B16_0, SHR_NUM_FOR_BF16, maskAll);
        MicroAPI::Pack<uint8_t, uint16_t, MicroAPI::HighLowPart::LOWEST>(mxScale2OneB8, scale1B16_0);

        // 计算1/scale — absMax0复用为reversedShareExp2One (循环后死亡，与reversedShareExp1不冲突) (与原始DynamicMxQuant一致: 无零值检查)
        MicroAPI::Compare<uint16_t, CMPMODE::NE>(infMask, scale1BF16, maxEleBF16, maskAll);
        MicroAPI::Compare<uint16_t, CMPMODE::EQ>(invalidDataMask, scale1BF16, biasE8M0, maskAll);
        MicroAPI::Sub(absMax0, biasE8M0, scale1BF16, maskAll);
        MicroAPI::Select<uint16_t>(absMax0, absMax0, nanBF16, infMask);
        MicroAPI::Select<uint16_t>(absMax0, specialExp, absMax0, invalidDataMask);

        // 交织搬出-2轴的mxScale和1/scale
        MicroAPI::DataCopy<uint8_t, MicroAPI::StoreDist::DIST_INTLV_B8>(
            mxScale2Addr, mxScale1B8, mxScale2OneB8, maskB8);
        MicroAPI::DataCopy<uint16_t, MicroAPI::StoreDist::DIST_INTLV_B16>(
            mxScale2ReciprocalAddr, reversedShareExp1, absMax0, maskAll);
    }
}

template <typename xDtype, typename y1Dtype, typename y2Dtype, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__aicore__ inline void DynamicMxQuantWithDualAxisBase<xDtype, y1Dtype, y2Dtype, roundMode, scaleAlg>::ComputeYVf(
    uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr, __ubuf__ uint16_t* mxScale1ReciprocalAddr,
    __ubuf__ uint16_t* mxScale2ReciprocalAddr, __ubuf__ uint8_t* y1Addr, __ubuf__ uint8_t* y2Addr)
{
    if constexpr (IsSameType<y1Dtype, fp4x2_e2m1_t>::value || IsSameType<y1Dtype, fp4x2_e1m2_t>::value) {
        // 算Y1是交织处理
        ComputeY1ToFP4(dataLen, blockCount, xAddr, mxScale1ReciprocalAddr, y1Addr);
        // 算y2是按单VF处理，基本块是两个VF长度，所以需要算两次
        ComputeY2ToFP4(dataLen, blockCount, xAddr, mxScale2ReciprocalAddr, y2Addr);
        ComputeY2ToFP4(
            dataLen, blockCount, xAddr + vlForHalfNumber_, mxScale2ReciprocalAddr + vlForHalfNumber_,
            y2Addr + vlForHalfNumber_ / 2);
    } else {
        ComputeY1ToFP8(dataLen, blockCount, xAddr, mxScale1ReciprocalAddr, y1Addr);
        ComputeY2ToFP8(dataLen, blockCount, xAddr, mxScale2ReciprocalAddr, y2Addr);
        ComputeY2ToFP8(
            dataLen, blockCount, xAddr + vlForHalfNumber_, mxScale2ReciprocalAddr + vlForHalfNumber_,
            y2Addr + vlForHalfNumber_);
    }
}

template <typename xDtype, typename y1Dtype, typename y2Dtype, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__aicore__ inline void DynamicMxQuantWithDualAxisBase<xDtype, y1Dtype, y2Dtype, roundMode, scaleAlg>::ComputeY1ToFP4(
    uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr, __ubuf__ uint16_t* mxScale1ReciprocalAddr,
    __ubuf__ uint8_t* y1Addr)
{
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
            MicroAPI::DataCopy<xDtype, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_DINTLV_B16>(
                x0, x1, xAddr, vlForHalfNumber_ * DIGIT_TWO);
            MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_E2B_B16>(
                scaleForMulFP16, mxScale1ReciprocalAddr, oneBlockCountB16_);

            if constexpr (IsSameType<xDtype, half>::value) {
                MicroAPI::Cast<float, bfloat16_t, castTraitXdtypetoFp32Zero>(
                    scaleForMulZeroFP32, (MicroAPI::RegTensor<bfloat16_t>&)scaleForMulFP16, dataMaskB16);

                // x0 cast to bf16
                MicroAPI::Cast<float, xDtype, castTraitXdtypetoFp32Zero>(x0ZeroFP32, x0, dataMaskB16);
                MicroAPI::Cast<float, xDtype, castTraitXdtypetoFp32One>(x0OneFP32, x0, dataMaskB16);

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

                // x1 cast to bf16
                MicroAPI::Cast<float, xDtype, castTraitXdtypetoFp32Zero>(x1ZeroFP32, x1, dataMaskB16);
                MicroAPI::Cast<float, xDtype, castTraitXdtypetoFp32One>(x1OneFP32, x1, dataMaskB16);

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

                // interleave x0 and x1
                MicroAPI::Interleave(x0ZeroBF16, x1ZeroBF16, x0ZeroBF16, x1ZeroBF16);
                MicroAPI::Cast<y1Dtype, bfloat16_t, castTraitBF16toFp4>(x0FP4, x0ZeroBF16, dataMaskB16);
                MicroAPI::Cast<y1Dtype, bfloat16_t, castTraitBF16toFp4>(x1FP4, x1ZeroBF16, dataMaskB16);
            } else {
                MicroAPI::Mul(x0, x0, (MicroAPI::RegTensor<xDtype>&)scaleForMulFP16, dataMaskB16);
                MicroAPI::Mul(x1, x1, (MicroAPI::RegTensor<xDtype>&)scaleForMulFP16, dataMaskB16);
                MicroAPI::Interleave(x0, x1, x0, x1);
                MicroAPI::Cast<y1Dtype, xDtype, castTraitBF16toFp4>(x0FP4, x0, dataMaskB16);
                MicroAPI::Cast<y1Dtype, xDtype, castTraitBF16toFp4>(x1FP4, x1, dataMaskB16);
            }

            // copy to ub
            MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::StoreDist::DIST_PACK4_B32>(
                y1Addr, (MicroAPI::RegTensor<uint8_t>&)x0FP4, OUT_ELE_NUM_ONE_BLK, dataMaskB8);
            MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::StoreDist::DIST_PACK4_B32>(
                y1Addr, (MicroAPI::RegTensor<uint8_t>&)x1FP4, OUT_ELE_NUM_ONE_BLK, dataMaskB8);
        }
    }
    return;
}

// 优化后的ComputeY1ToFP8: 参考DynamicMxQuant ComputeData的4路RegLayout Cast+Add模式
// 消除多次Interleave操作，使用ZERO/ONE/TWO/THREE四个RegLayout将FP32→FP8的结果
// 分别放到uint32的4个字节位置，通过Add合并后一次StoreAlign输出
template <typename xDtype, typename y1Dtype, typename y2Dtype, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__aicore__ inline void DynamicMxQuantWithDualAxisBase<xDtype, y1Dtype, y2Dtype, roundMode, scaleAlg>::ComputeY1ToFP8(
    uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr, __ubuf__ uint16_t* mxScale1ReciprocalAddr,
    __ubuf__ uint8_t* y1Addr)
{
    __VEC_SCOPE__
    {
        MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<uint16_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg maskFP8 = MicroAPI::CreateMask<y1Dtype>();
        MicroAPI::RegTensor<uint16_t> scaleForMulFP16;
        MicroAPI::RegTensor<float> scaleForMulFP32;
        MicroAPI::RegTensor<xDtype> x0;
        MicroAPI::RegTensor<xDtype> x1;
        MicroAPI::RegTensor<float> x0ZeroFP32;
        MicroAPI::RegTensor<float> x0OneFP32;
        MicroAPI::RegTensor<float> x1ZeroFP32;
        MicroAPI::RegTensor<float> x1OneFP32;
        // 4路FP8寄存器，分别对应uint32中的4个字节位置
        MicroAPI::RegTensor<y1Dtype> fp8Layout0;  // x0 Zero → byte 0
        MicroAPI::RegTensor<y1Dtype> fp8Layout1;  // x1 Zero → byte 1
        MicroAPI::RegTensor<y1Dtype> fp8Layout2;  // x0 One  → byte 2
        MicroAPI::RegTensor<y1Dtype> fp8Layout3;  // x1 One  → byte 3

        for (uint16_t i = 0; i < blockCount; i++) {
            // 交织搬运: 256个xDtype按偶奇拆分为x0(偶128), x1(奇128)
            MicroAPI::DataCopy<xDtype, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_DINTLV_B16>(
                x0, x1, xAddr, vlForHalfNumber_ * DIGIT_TWO);
            // 搬运1/scale: 8个scale广播到128个位置
            MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_E2B_B16>(
                scaleForMulFP16, mxScale1ReciprocalAddr, oneBlockCountB16_);
            if constexpr (IsSameType<xDtype, half>::value) {
                // half输入: 先Cast到FP32再乘scale (避免half精度损失)
                MicroAPI::Cast<float, xDtype, castTraitXdtypetoFp32Zero>(x0ZeroFP32, x0, maskAll);
                MicroAPI::Cast<float, xDtype, castTraitXdtypetoFp32One>(x0OneFP32, x0, maskAll);
                MicroAPI::Cast<float, bfloat16_t, castTraitXdtypetoFp32Zero>(
                    scaleForMulFP32, (MicroAPI::RegTensor<bfloat16_t>&)scaleForMulFP16, maskAll);
                MicroAPI::Mul(x0ZeroFP32, x0ZeroFP32, scaleForMulFP32, maskAll);
                MicroAPI::Mul(x0OneFP32, x0OneFP32, scaleForMulFP32, maskAll);
                MicroAPI::Cast<float, xDtype, castTraitXdtypetoFp32Zero>(x1ZeroFP32, x1, maskAll);
                MicroAPI::Cast<float, xDtype, castTraitXdtypetoFp32One>(x1OneFP32, x1, maskAll);
                MicroAPI::Mul(x1ZeroFP32, x1ZeroFP32, scaleForMulFP32, maskAll);
                MicroAPI::Mul(x1OneFP32, x1OneFP32, scaleForMulFP32, maskAll);
            } else {
                // bf16输入: 直接在bf16域乘scale，再Cast到FP32
                MicroAPI::Mul(x0, x0, (MicroAPI::RegTensor<xDtype>&)scaleForMulFP16, maskAll);
                MicroAPI::Mul(x1, x1, (MicroAPI::RegTensor<xDtype>&)scaleForMulFP16, maskAll);
                MicroAPI::Cast<float, xDtype, castTraitXdtypetoFp32Zero>(x0ZeroFP32, x0, maskAll);
                MicroAPI::Cast<float, xDtype, castTraitXdtypetoFp32One>(x0OneFP32, x0, maskAll);
                MicroAPI::Cast<float, xDtype, castTraitXdtypetoFp32Zero>(x1ZeroFP32, x1, maskAll);
                MicroAPI::Cast<float, xDtype, castTraitXdtypetoFp32One>(x1OneFP32, x1, maskAll);
            }
            // 4路RegLayout Cast: 将4组64个FP32值分别Cast到FP8的不同字节位置
            // Layout0(byte0): x0 Zero (偶数列偶数位)
            // Layout2(byte2): x0 One  (偶数列奇数位)
            // Layout1(byte1): x1 Zero (奇数列偶数位)
            // Layout3(byte3): x1 One  (奇数列奇数位)
            MicroAPI::Cast<y1Dtype, float, castTraitFp32toFP8Layout0>(fp8Layout0, x0ZeroFP32, maskAll);
            MicroAPI::Cast<y1Dtype, float, castTraitFp32toFP8Layout2>(fp8Layout2, x0OneFP32, maskAll);
            MicroAPI::Cast<y1Dtype, float, castTraitFp32toFP8Layout1>(fp8Layout1, x1ZeroFP32, maskAll);
            MicroAPI::Cast<y1Dtype, float, castTraitFp32toFP8Layout3>(fp8Layout3, x1OneFP32, maskAll);
            // Add合并: 4个字节位置的FP8值合并到一个寄存器
            MicroAPI::Add(
                (MicroAPI::RegTensor<uint8_t>&)fp8Layout0,
                (MicroAPI::RegTensor<uint8_t>&)fp8Layout0,
                (MicroAPI::RegTensor<uint8_t>&)fp8Layout2, maskFP8);
            MicroAPI::Add(
                (MicroAPI::RegTensor<uint8_t>&)fp8Layout1,
                (MicroAPI::RegTensor<uint8_t>&)fp8Layout1,
                (MicroAPI::RegTensor<uint8_t>&)fp8Layout3, maskFP8);
            MicroAPI::Add(
                (MicroAPI::RegTensor<uint8_t>&)fp8Layout0,
                (MicroAPI::RegTensor<uint8_t>&)fp8Layout0,
                (MicroAPI::RegTensor<uint8_t>&)fp8Layout1, maskFP8);
            // 一次性输出256个FP8值
            MicroAPI::DataCopy<uint8_t, MicroAPI::StoreDist::DIST_NORM_B8>(
                y1Addr + i * vlForHalfNumber_ * DIGIT_TWO, (MicroAPI::RegTensor<uint8_t>&)fp8Layout0, maskFP8);
        }
    }
    return;
}

template <typename xDtype, typename y1Dtype, typename y2Dtype, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__aicore__ inline void DynamicMxQuantWithDualAxisBase<xDtype, y1Dtype, y2Dtype, roundMode, scaleAlg>::ComputeY2ToFP4(
    uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr, __ubuf__ uint16_t* mxScale2ReciprocalAddr,
    __ubuf__ uint8_t* y2Addr)
{
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
        MicroAPI::RegTensor<y1Dtype> yZeroFP8;
        MicroAPI::RegTensor<y1Dtype> yOneFP8;
        MicroAPI::RegTensor<y1Dtype> yZeroFP4;
        MicroAPI::MaskReg zeroMask;
        MicroAPI::MaskReg specialMask;
        MicroAPI::MaskReg negInfMask;

        MicroAPI::RegTensor<int32_t> negZero;
        MicroAPI::RegTensor<int32_t> maxExpFP32;
        MicroAPI::RegTensor<int32_t> exp0FP32;
        MicroAPI::RegTensor<int32_t> exp1FP32;

        MicroAPI::Duplicate(negZero, NEG_ZERO);

        MicroAPI::MaskReg pregAll8 = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg pregAll16 = MicroAPI::CreateMask<uint16_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg pregAll32 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();

        MicroAPI::DataCopy<uint16_t, MicroAPI::LoadDist::DIST_NORM>(reversedShareExp, mxScale2ReciprocalAddr);

        for (uint16_t j = 0; j < blockCount; j++) {
            MicroAPI::DataCopy<xDtype, MicroAPI::LoadDist::DIST_NORM>(x, xAddr + j * ubRowLen_);
            if constexpr (IsSameType<xDtype, half>::value) {
                MicroAPI::Cast<float, xDtype, castTraitXdtypetoFp32Zero>(x0FP32, x, pregAll16);
                MicroAPI::Cast<float, xDtype, castTraitXdtypetoFp32One>(x1FP32, x, pregAll16);
                MicroAPI::Cast<float, bfloat16_t, castTraitXdtypetoFp32Zero>(
                    reversedShareExp0FP32, (MicroAPI::RegTensor<bfloat16_t>&)reversedShareExp, pregAll16);
                MicroAPI::Cast<float, bfloat16_t, castTraitXdtypetoFp32One>(
                    reversedShareExp1FP32, (MicroAPI::RegTensor<bfloat16_t>&)reversedShareExp, pregAll16);
                MicroAPI::Mul(x0FP32, x0FP32, reversedShareExp0FP32, pregAll32);
                MicroAPI::Mul(x1FP32, x1FP32, reversedShareExp1FP32, pregAll32);

                ComputeFP4FromHalf(x0FP32);
                ComputeFP4FromHalf(x1FP32);

                MicroAPI::Cast<bfloat16_t, float, castTraitFp32toBF16>(
                    (MicroAPI::RegTensor<bfloat16_t>&)x0BF16, x0FP32, pregAll32);
                MicroAPI::Pack<uint16_t, uint32_t, MicroAPI::HighLowPart::LOWEST>(
                    (MicroAPI::RegTensor<uint16_t>&)x0BF16, (MicroAPI::RegTensor<uint32_t>&)x0BF16);

                MicroAPI::Cast<bfloat16_t, float, castTraitFp32toBF16>(
                    (MicroAPI::RegTensor<bfloat16_t>&)x1BF16, x1FP32, pregAll32);
                MicroAPI::Pack<uint16_t, uint32_t, MicroAPI::HighLowPart::LOWEST>(
                    (MicroAPI::RegTensor<uint16_t>&)x1BF16, (MicroAPI::RegTensor<uint32_t>&)x1BF16);
                MicroAPI::Interleave(x0BF16, x1BF16, x0BF16, x1BF16);
                MicroAPI::Cast<y1Dtype, bfloat16_t, castTraitBF16toFp4>(
                    yZeroFP4, (MicroAPI::RegTensor<bfloat16_t>&)x0BF16, pregAll16);
            } else {
                MicroAPI::Mul(xBF16, x, (MicroAPI::RegTensor<bfloat16_t>&)reversedShareExp, pregAll16);
                MicroAPI::Cast<y1Dtype, bfloat16_t, castTraitBF16toFp4>(yZeroFP4, xBF16, pregAll16);
            }
            DataCopy<uint8_t, MicroAPI::StoreDist::DIST_PACK4_B32>(
                y2Addr + (j * ubRowLen_ / DIGIT_TWO), (MicroAPI::RegTensor<uint8_t>&)yZeroFP4, pregAll8);
        }
    }
}

// 优化后的ComputeY2ToFP8: 参考DynamicMxQuant ComputeData的RegLayout优化模式
// 使用ZERO/ONE两个RegLayout将FP32→FP8的结果分别放到uint32的byte0和byte1位置
// 通过Add合并后Pack+DataCopy输出，消除Interleave和多余的Pack操作
template <typename xDtype, typename y1Dtype, typename y2Dtype, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__aicore__ inline void DynamicMxQuantWithDualAxisBase<xDtype, y1Dtype, y2Dtype, roundMode, scaleAlg>::ComputeY2ToFP8(
    uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr, __ubuf__ uint16_t* mxScale2ReciprocalAddr,
    __ubuf__ uint8_t* y2Addr)
{
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<xDtype> x;
        MicroAPI::RegTensor<float> x0FP32;
        MicroAPI::RegTensor<float> x1FP32;
        MicroAPI::RegTensor<uint16_t> reversedShareExp;
        MicroAPI::RegTensor<float> reversedShareExp0FP32;
        MicroAPI::RegTensor<float> reversedShareExp1FP32;
        // 2路FP8寄存器，分别对应uint32中的byte0和byte1位置
        MicroAPI::RegTensor<y1Dtype> fp8Layout0;  // x0 (偶数位) → byte 0
        MicroAPI::RegTensor<y1Dtype> fp8Layout1;  // x1 (奇数位) → byte 1

        MicroAPI::MaskReg pregAll8 = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::H>();
        MicroAPI::MaskReg pregAll16 = MicroAPI::CreateMask<uint16_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg pregAll32 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg maskFP8 = MicroAPI::CreateMask<y1Dtype>();

        MicroAPI::DataCopy<uint16_t, MicroAPI::LoadDist::DIST_NORM>(reversedShareExp, mxScale2ReciprocalAddr);
        MicroAPI::Cast<float, bfloat16_t, castTraitXdtypetoFp32Zero>(
            reversedShareExp0FP32, (MicroAPI::RegTensor<bfloat16_t>&)reversedShareExp, pregAll16);
        MicroAPI::Cast<float, bfloat16_t, castTraitXdtypetoFp32One>(
            reversedShareExp1FP32, (MicroAPI::RegTensor<bfloat16_t>&)reversedShareExp, pregAll16);
        for (uint16_t j = 0; j < blockCount; j++) {
            MicroAPI::DataCopy<xDtype, MicroAPI::LoadDist::DIST_NORM>(x, xAddr + j * ubRowLen_);
            MicroAPI::Cast<float, xDtype, castTraitXdtypetoFp32Zero>(x0FP32, x, pregAll16);
            MicroAPI::Cast<float, xDtype, castTraitXdtypetoFp32One>(x1FP32, x, pregAll16);

            MicroAPI::Mul(x0FP32, x0FP32, reversedShareExp0FP32, pregAll32);
            MicroAPI::Mul(x1FP32, x1FP32, reversedShareExp1FP32, pregAll32);

            // 2路RegLayout Cast: 将2组64个FP32值分别Cast到FP8的不同字节位置
            // Layout0(byte0): x0 (偶数位元素)
            // Layout1(byte1): x1 (奇数位元素)
            MicroAPI::Cast<y1Dtype, float, castTraitFp32toFP8Layout0>(fp8Layout0, x0FP32, pregAll32);
            MicroAPI::Cast<y1Dtype, float, castTraitFp32toFP8Layout1>(fp8Layout1, x1FP32, pregAll32);
            // Add合并: byte0和byte1位置的FP8值合并到一个寄存器
            MicroAPI::Add(
                (MicroAPI::RegTensor<uint8_t>&)fp8Layout0,
                (MicroAPI::RegTensor<uint8_t>&)fp8Layout0,
                (MicroAPI::RegTensor<uint8_t>&)fp8Layout1, maskFP8);
            // Pack: 提取每个uint32的低16位(包含2个FP8值)，紧凑为128个FP8
            MicroAPI::Pack<uint16_t, uint32_t, MicroAPI::HighLowPart::LOWEST>(
                (MicroAPI::RegTensor<uint16_t>&)fp8Layout0, (MicroAPI::RegTensor<uint32_t>&)fp8Layout0);

            DataCopy(y2Addr + (j * ubRowLen_), (MicroAPI::RegTensor<uint8_t>&)fp8Layout0, pregAll8);
        }
    }
}

template <typename xDtype, typename y1Dtype, typename y2Dtype, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__aicore__ inline void
DynamicMxQuantWithDualAxisBase<xDtype, y1Dtype, y2Dtype, roundMode, scaleAlg>::ComputeFP4FromHalf(
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
    if constexpr (IsSameType<y1Dtype, fp4x2_e1m2_t>::value) {
        MicroAPI::Muls(Reg, Reg, FOUR, pregAll32);
        MicroAPI::CompareScalar<float, CMPMODE::LT>(specialMask, Reg, 0, pregAll32);
        MicroAPI::Truncate<float, roundMode>(Reg, Reg, pregAll32);
        MicroAPI::Muls(Reg, Reg, ONE_FOURTH, pregAll32);
    } else {
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
    }
    MicroAPI::CompareScalar<float, CMPMODE::EQ>(zeroMask, Reg, 0, pregAll32);
    MicroAPI::MaskAnd(zeroMask, specialMask, zeroMask, pregAll32);
    MicroAPI::MaskOr(zeroMask, negInfMask, zeroMask, pregAll32);
    MicroAPI::Select<int32_t>(
        (MicroAPI::RegTensor<int32_t>&)Reg, negZero, (MicroAPI::RegTensor<int32_t>&)Reg, zeroMask);
}

template <typename xDtype, typename y1Dtype, typename y2Dtype, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__aicore__ inline void DynamicMxQuantWithDualAxisBase<xDtype, y1Dtype, y2Dtype, roundMode, scaleAlg>::CopyIn(
    int64_t offset, int64_t blockCount, int64_t dataLen, int64_t dimNeg1IsOdd)
{
    // 第一行第一块到第二行的第一块，间隔长度
    int64_t rightPadding =
        ops::CeilAlign(static_cast<int64_t>(dataLen * sizeof(xDtype)), UBBlockSize_) / sizeof(xDtype) - dataLen;
    DataCopyExtParams copyInParams = {
        static_cast<uint16_t>(blockCount), static_cast<uint32_t>(dataLen * sizeof(xDtype)),
        static_cast<uint32_t>((tilingData_->dimNeg1 - dataLen) * sizeof(xDtype)),
        static_cast<uint32_t>((ubRowLen_ - dataLen) * sizeof(xDtype) / UBBlockSize_), static_cast<uint32_t>(0)};
    DataCopyPadExtParams<xDtype> padParams{true, 0, static_cast<uint8_t>(rightPadding), 0};

    LocalTensor<xDtype> xLocal = inQueue.template AllocTensor<xDtype>();
    if (dimNeg1IsOdd) {
        Duplicate<xDtype>(xLocal, static_cast<xDtype>(0), inBufferSize_ / sizeof(xDtype));
        event_t eventIDVToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(eventIDVToMTE2);
        WaitFlag<HardEvent::V_MTE2>(eventIDVToMTE2);
    }
    DataCopyPad(xLocal, xGm1_[offset], copyInParams, padParams);
    inQueue.template EnQue(xLocal);
}

template <typename xDtype, typename y1Dtype, typename y2Dtype, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__aicore__ inline void DynamicMxQuantWithDualAxisBase<xDtype, y1Dtype, y2Dtype, roundMode, scaleAlg>::CopyOut(
    int64_t yOffset, int64_t scale1OutOffset, int64_t scale2OutOffset, int64_t blockCount, int64_t dataLen)
{
    uint16_t outBurst = 0;
    uint32_t outBlockLen = 0;
    uint32_t srcStride = 0;
    uint32_t dstStride = 0;
    int64_t YOffset = yOffset;
    // -2轴两行交织搬，考虑32对齐,计算偏移
    uint32_t scaleSrcStride =
        DIGIT_TWO * ops::CeilDiv(dataLen, UBBlockSize_) - ops::CeilDiv(DIGIT_TWO * dataLen, UBBlockSize_);

    if constexpr (IsSameType<y1Dtype, fp4x2_e2m1_t>::value || IsSameType<y1Dtype, fp4x2_e1m2_t>::value) {
        outBurst = blockCount;
        outBlockLen = dataLen / DIGIT_TWO * sizeof(uint8_t);
        srcStride = ((ubRowLen_ - dataLen) / DIGIT_TWO * sizeof(uint8_t) / UBBlockSize_);
        dstStride = (tilingData_->dimNeg1 - dataLen) / DIGIT_TWO * sizeof(uint8_t);
        YOffset = yOffset / DIGIT_TWO;
    } else {
        outBurst = blockCount;
        outBlockLen = dataLen * sizeof(uint8_t);
        srcStride = ((ubRowLen_ - dataLen) * sizeof(y1Dtype) / UBBlockSize_);
        dstStride = (tilingData_->dimNeg1 - dataLen) * sizeof(uint8_t);
        YOffset = yOffset;
    }
    DataCopyExtParams yCopyOutParams = {
        static_cast<uint16_t>(outBurst), static_cast<uint32_t>(outBlockLen), static_cast<uint32_t>(srcStride),
        static_cast<uint32_t>(dstStride), static_cast<uint32_t>(0)};

    uint32_t dataLenReduce = static_cast<uint32_t>(ops::CeilDiv(dataLen, blockSize_));
    uint32_t scale1OutLen = dataLenReduce % 2 == 1 ? dataLenReduce + 1 : dataLenReduce;

    DataCopyExtParams scale1CopyOutParams = {
        static_cast<uint16_t>(outBurst), static_cast<uint32_t>(scale1OutLen * sizeof(uint8_t)),
        static_cast<uint32_t>(0),
        static_cast<uint32_t>(
            ops::CeilAlign(tilingData_->dimNeg1, blockSize_ * DIGIT_TWO) / blockSize_ -
            ops::CeilAlign(dataLen, blockSize_ * DIGIT_TWO) / blockSize_),
        static_cast<uint32_t>(0)};

    DataCopyExtParams scale2CopyOutParams = {
        static_cast<uint16_t>(ops::CeilDiv(blockCount, DIGIT_TWO * blockSize_)),
        static_cast<uint32_t>(dataLen * DIGIT_TWO * sizeof(uint8_t)), static_cast<uint32_t>(scaleSrcStride),
        static_cast<uint32_t>(DIGIT_TWO * (tilingData_->dimNeg1 - dataLen) * sizeof(uint8_t)),
        static_cast<uint32_t>(0)};

    LocalTensor<uint8_t> y1Local = outQueue1.template DeQue<uint8_t>();
    DataCopyPad(yGm1_[YOffset], y1Local, yCopyOutParams);
    outQueue1.FreeTensor(y1Local);

    LocalTensor<uint8_t> y2Local = outQueue2.template DeQue<uint8_t>();
    DataCopyPad(yGm2_[YOffset], y2Local, yCopyOutParams);
    outQueue2.FreeTensor(y2Local);

    LocalTensor<uint8_t> mxScale1Local = mxScaleQueue1.template DeQue<uint8_t>();
    DataCopyPad(mxScaleGm1_[scale1OutOffset], mxScale1Local, scale1CopyOutParams);
    mxScaleQueue1.FreeTensor(mxScale1Local);

    LocalTensor<uint8_t> mxScale2Local = mxScaleQueue2.template DeQue<uint8_t>();
    DataCopyPad(mxScaleGm2_[scale2OutOffset], mxScale2Local, scale2CopyOutParams);
    mxScaleQueue2.FreeTensor(mxScale2Local);
}

template <typename xDtype, typename y1Dtype, typename y2Dtype, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__aicore__ inline void DynamicMxQuantWithDualAxisBase<xDtype, y1Dtype, y2Dtype, roundMode, scaleAlg>::Init(
    GM_ADDR x, GM_ADDR y1, GM_ADDR mxScale1, GM_ADDR y2, GM_ADDR mxScale2)
{
#if (__NPU_ARCH__ == 3510)
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
#endif
    // init block params
    InitParams();

    xGm1_.SetGlobalBuffer((__gm__ xDtype*)(x));
    yGm1_.SetGlobalBuffer((__gm__ uint8_t*)(y1));
    mxScaleGm1_.SetGlobalBuffer((__gm__ uint8_t*)(mxScale1));
    yGm2_.SetGlobalBuffer((__gm__ uint8_t*)(y2));
    mxScaleGm2_.SetGlobalBuffer((__gm__ uint8_t*)(mxScale2));

    inBufferSize_ = ubRowLen_ * ubRowCount_ * sizeof(xDtype);
    // -2轴scalebuffersize
    mxScale2BufferSize_ = ubRowLen_ * (ops::CeilDiv(ubRowCount_, DIGIT_TWO * blockSize_) * DIGIT_TWO);

    // -1轴 scalebuffersize
    mxScale1BufferSize_ = ubRowCount_ * UBBlockSize_;
    // -1，-2轴 y的buffersize一致
    int64_t outBufferSize = ubRowLen_ * ubRowCount_;

    // -2轴 1/scale
    tmpScale2BufferSize_ = ubRowLen_ * (ops::CeilDiv(ubRowCount_, DIGIT_TWO * blockSize_) * DIGIT_TWO) * sizeof(xDtype);

    // -1轴 1/scale
    tmpScale1BufferSize_ = ubRowCount_ * UBBlockSize_;

    pipe_->InitBuffer(inQueue, DB_BUFFER, inBufferSize_);
    pipe_->InitBuffer(mxScaleQueue1, DB_BUFFER, mxScale1BufferSize_);
    pipe_->InitBuffer(mxScaleQueue2, DB_BUFFER, mxScale2BufferSize_);
    pipe_->InitBuffer(outQueue1, DB_BUFFER, outBufferSize);
    pipe_->InitBuffer(outQueue2, DB_BUFFER, outBufferSize);
    pipe_->InitBuffer(mxScale1ReciprocalBuf, tmpScale1BufferSize_);
    pipe_->InitBuffer(mxScale2ReciprocalBuf, tmpScale2BufferSize_);
    pipe_->InitBuffer(tmpScale2Buf, mxScale2BufferSize_);
}

template <typename xDtype, typename y1Dtype, typename y2Dtype, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__aicore__ inline void DynamicMxQuantWithDualAxisBase<xDtype, y1Dtype, y2Dtype, roundMode, scaleAlg>::Process()
{
    if (blockIdx_ >= tilingData_->usedCoreNum) {
        return;
    }

    for (int64_t i = 0; i < loopPerCore_; i++) {
        // 由本次ub计算的block块数推导列数
        int64_t calcCol = ((blockOffset_ + i) % tilingData_->dimNeg1BlockNum == tilingData_->dimNeg1BlockNum - 1) ?
                              ubRowLenTail_ :
                              ubRowLen_;
        // 本次ub计算的行数
        int64_t calcRow = ((blockOffset_ + i) / tilingData_->dimNeg1BlockNum % tilingData_->dimNeg2SplitBlockNum) ==
                                  (tilingData_->dimNeg2SplitBlockNum - 1) ?
                              ubRowCountTail_ :
                              ubRowCount_;
        // 单batch偏移+单行偏移+单列偏移
        int64_t xUbOffset = (blockOffset_ + i) / blockCountPerPage_ * tilingData_->dimNeg1 * tilingData_->dimNeg2 +
                            (blockOffset_ + i) % blockCountPerPage_ / tilingData_->dimNeg1BlockNum * ubRowCount_ *
                                tilingData_->dimNeg1 +
                            (blockOffset_ + i) % blockCountPerPage_ % tilingData_->dimNeg1BlockNum * ubRowLen_;
        // -2轴偏移
        int64_t scale2Offset =
            (blockOffset_ + i) / blockCountPerPage_ * dimNeg2ScaleNum_ * tilingData_->dimNeg1 +
            (blockOffset_ + i) % blockCountPerPage_ / tilingData_->dimNeg1BlockNum * tilingData_->splitBlockH /
                tilingData_->blockSize * tilingData_->dimNeg1 +
            (blockOffset_ + i) % blockCountPerPage_ % tilingData_->dimNeg1BlockNum * ubRowLen_ * DIGIT_TWO;
        // -1轴偏移
        int64_t scale1Offset =
            (blockOffset_ + i) / blockCountPerPage_ * dimNeg1ScaleNum_ * tilingData_->dimNeg2 +
            (blockOffset_ + i) % blockCountPerPage_ / tilingData_->dimNeg1BlockNum * tilingData_->splitBlockH *
                dimNeg1ScaleNum_ +
            (blockOffset_ + i) % blockCountPerPage_ % tilingData_->dimNeg1BlockNum * ubRowLen_ / tilingData_->blockSize;

        // 尾轴reduce后是否是奇数
        int64_t dimNeg1IsOdd = ubRowLenTail_ < ubRowLen_;
        ProcessOneLoop(calcCol, calcRow, xUbOffset, scale1Offset, scale2Offset, dimNeg1IsOdd);
    }
}

} // namespace DynamicMxQuantWithDualAxis
#endif // OPS_NN_DEV_DYNAMIC_MX_QUANT_WITH_DUAL_AXIS_H
