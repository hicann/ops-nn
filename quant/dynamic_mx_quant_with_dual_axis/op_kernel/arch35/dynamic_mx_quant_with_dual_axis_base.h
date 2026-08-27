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
constexpr uint16_t ABS_MASK_FOR_16BIT = 0x7fff;       // 取绝对值掩码，清除符号位
constexpr uint32_t MAN_MASK_FLOAT = 0x007fffff;       // FP32尾数掩码 (23位尾数)
constexpr uint32_t FP32_EXP_BIAS_CUBLAS = 0x00007f00; // FP32指数偏移(CuBALS)，左移7位后为BF16的指数偏移
constexpr uint32_t MAX_EXP_FOR_FP8_IN_FP32 = 0x000000ff; // FP8 NAN在E8M0中的表示 (0xFF)
constexpr uint32_t NAN_CUSTOMIZATION_PACK = 0x00007f81;  // NAN的BF16打包表示 (uint32存储)
constexpr uint32_t NUMBER_ZERO_U32 = 0x00000000;         // uint32零常量
constexpr uint32_t NUMBER_TWO_FIVE_FOUR = 0x000000fe;    // 254，FP32指数上界
constexpr uint32_t NUMBER_HALF_U32 = 0x00400000;         // FP32尾数的一半 (2^22)

// DynamicDtypeRange Scale算法相关常量 (scaleAlg=2, FP4_E2M1专用)
constexpr uint16_t ADD_VALUE_FOR_BF16_MAN1 = 0x003f; // dstTypeMax=0.0/6.0时BF16尾数进位值
constexpr uint16_t ADD_VALUE_FOR_BF16_MAN2 = 0x001f; // dstTypeMax=7.0时BF16尾数进位值
constexpr uint16_t
    SUB_NUM_FOR_SCALE_6 = 0x00c1; // dstTypeMax=0.0/6.0时-2轴减法常量 (FP4_E2M1_BF16_MAX_EXP - addValueBit)
constexpr uint16_t SUB_NUM_FOR_SCALE_7 = 0x00e1; // dstTypeMax=7.0时-2轴减法常量
constexpr float DIGIT_ZERO_FLOAT = 0.0f;
constexpr float DIGIT_SIX_FLOAT = 6.0f;
constexpr float DIGIT_SEVEN_FLOAT = 7.0f;
constexpr uint16_t elementAfterReduce = platform::GetVRegSize() / platform::GetUbBlockSize();

template <typename xDtype, typename y1Dtype, typename y2Dtype, AscendC::RoundMode roundMode, uint64_t scaleAlg>
class DynamicMxQuantWithDualAxisBase {
public:
    __aicore__ inline DynamicMxQuantWithDualAxisBase(const DynamicMxQuantWithDualAxisTilingData* tilingData,
                                                     TPipe* pipe)
        : tilingData_(tilingData), pipe_(pipe){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y1, GM_ADDR mxScale1, GM_ADDR y2, GM_ADDR mxScale2);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitParams();
    __aicore__ inline void ProcessOneLoop(int64_t calcCol, int64_t calcRow, int64_t xUbOffset, int64_t scale1Offset,
                                          int64_t scale2Offset, int64_t dimNeg1IsOdd);
    __aicore__ inline void CopyOut(int64_t yOffset, int64_t scale1OutOffset, int64_t scale2OutOffset,
                                   int64_t blockCount, int64_t dataLen);
    __aicore__ inline void CopyIn(int64_t offset, int64_t blockCount, int64_t dataLen, int64_t dimNeg1IsOdd);
    __aicore__ inline void ComputeAll(int64_t blockCount, int64_t dataLen);
    __aicore__ inline void ComputeScaleOcp(uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr,
                                           __ubuf__ uint8_t* mxScale1Addr, __ubuf__ uint16_t* mxScale1ReciprocalAddr,
                                           __ubuf__ uint8_t* mxScale2Addr, __ubuf__ uint16_t* mxScale2ReciprocalAddr);
    __aicore__ inline void ComputeScaleCublas(uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr,
                                              __ubuf__ uint8_t* mxScale1Addr, __ubuf__ uint16_t* mxScale1ReciprocalAddr,
                                              __ubuf__ uint8_t* mxScale2Addr,
                                              __ubuf__ uint16_t* mxScale2ReciprocalAddr);
    // DynamicDtypeRange Default: dstTypeMax=0.0/6.0/7.0, 指数域addValueBit进位法 (scaleAlg=2)
    __aicore__ inline void ComputeScaleDynamicDefault(uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr,
                                                      __ubuf__ uint8_t* mxScale1Addr,
                                                      __ubuf__ uint16_t* mxScale1ReciprocalAddr,
                                                      __ubuf__ uint8_t* mxScale2Addr,
                                                      __ubuf__ uint16_t* mxScale2ReciprocalAddr);
    // DynamicDtypeRange Custom: 自定义dstTypeMax, FP32精度invDstTypeMax乘法法 (scaleAlg=2)
    __aicore__ inline void ComputeScaleDynamicCustom(uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr,
                                                     __ubuf__ uint8_t* mxScale1Addr,
                                                     __ubuf__ uint16_t* mxScale1ReciprocalAddr,
                                                     __ubuf__ uint8_t* mxScale2Addr,
                                                     __ubuf__ uint16_t* mxScale2ReciprocalAddr);
    __aicore__ inline void ComputeYVf(uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr,
                                      __ubuf__ uint16_t* mxScale1ReciprocalAddr,
                                      __ubuf__ uint16_t* mxScale2ReciprocalAddr, __ubuf__ uint8_t* y1Addr,
                                      __ubuf__ uint8_t* y2Addr);
    __aicore__ inline void ComputeYBF16ToFP4(uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr,
                                             __ubuf__ uint16_t* mxScale1ReciprocalAddr, __ubuf__ uint8_t* y1Addr,
                                             __ubuf__ uint16_t* mxScale2ReciprocalAddr, __ubuf__ uint8_t* y2Addr);
    __aicore__ inline void ComputeYFP16ToFP4(uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr,
                                             __ubuf__ uint16_t* mxScale1ReciprocalAddr, __ubuf__ uint8_t* y1Addr,
                                             __ubuf__ uint16_t* mxScale2ReciprocalAddr, __ubuf__ uint8_t* y2Addr);
    __aicore__ inline void ComputeY1ToFP8(uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr,
                                          __ubuf__ uint16_t* mxScale1ReciprocalAddr, __ubuf__ uint8_t* y1Addr);
    __aicore__ inline void ComputeFP4FromHalf(Reg::RegTensor<float>& Reg);
    __aicore__ inline void ComputeY2ToFP8(uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr,
                                          __ubuf__ uint16_t* mxScale2ReciprocalAddr, __ubuf__ uint8_t* y2Addr);

protected:
    static constexpr Reg::CastTrait castTraitXdtypetoFp32Zero = {
        Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN, Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
    static constexpr Reg::CastTrait castTraitXdtypetoFp32One = {
        Reg::RegLayout::ONE, Reg::SatMode::UNKNOWN, Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
    static constexpr Reg::CastTrait castTraitHalf2BF16 = {Reg::RegLayout::UNKNOWN, Reg::SatMode::UNKNOWN,
                                                          Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_TRUNC};
    // DynamicDtypeRange需要CAST_RINT (四舍五入)，与OCP的CAST_TRUNC (截断) 不同
    static constexpr Reg::CastTrait castTraitHalf2BF16Rint = {
        Reg::RegLayout::UNKNOWN, Reg::SatMode::UNKNOWN, Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
    static constexpr Reg::CastTrait castTraitBF16toFp4 = {Reg::RegLayout::ZERO, Reg::SatMode::SAT,
                                                          Reg::MaskMergeMode::ZEROING, roundMode};
    static constexpr Reg::CastTrait castTraitFp32toBF16 = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                           Reg::MaskMergeMode::ZEROING, roundMode};
    static constexpr Reg::CastTrait castTraitFp32toYdtype = {Reg::RegLayout::ZERO, Reg::SatMode::SAT,
                                                             Reg::MaskMergeMode::ZEROING, roundMode};
    // FP32→FP8 四路RegLayout Cast (参考DynamicMxQuant ComputeData优化模式)
    // 将4组64个FP32值分别Cast到FP8的不同字节位置，通过Add合并后一次Store输出
    static constexpr Reg::CastTrait castTraitFp32toFP8Layout0 = {Reg::RegLayout::ZERO, Reg::SatMode::SAT,
                                                                 Reg::MaskMergeMode::ZEROING, roundMode};
    static constexpr Reg::CastTrait castTraitFp32toFP8Layout1 = {Reg::RegLayout::ONE, Reg::SatMode::SAT,
                                                                 Reg::MaskMergeMode::ZEROING, roundMode};
    static constexpr Reg::CastTrait castTraitFp32toFP8Layout2 = {Reg::RegLayout::TWO, Reg::SatMode::SAT,
                                                                 Reg::MaskMergeMode::ZEROING, roundMode};
    static constexpr Reg::CastTrait castTraitFp32toFP8Layout3 = {Reg::RegLayout::THREE, Reg::SatMode::SAT,
                                                                 Reg::MaskMergeMode::ZEROING, roundMode};

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
            ComputeScaleOcp(dataLen, blockSize_, xAddr + xOffset, mxScale1Addr + scale1UbOffset,
                            mxScale1ReciprocalAddr + scale1ReciprocalOffset, tmpScale2Addr + scale2UbOffset,
                            mxScale2ReciprocalAddr + scale2ReciprocalOffset);
        } else if constexpr (scaleAlg == 1) {
            // CuBALS Scale算法 (FP8专用)
            ComputeScaleCublas(dataLen, blockSize_, xAddr + xOffset, mxScale1Addr + scale1UbOffset,
                               mxScale1ReciprocalAddr + scale1ReciprocalOffset, tmpScale2Addr + scale2UbOffset,
                               mxScale2ReciprocalAddr + scale2ReciprocalOffset);
        } else if constexpr (scaleAlg == 2) {
            // DynamicDtypeRange Scale算法 (FP4_E2M1专用)
            // 运行时根据dstTypeMax_选择Default (指数域进位法) 或 Custom (FP32精度乘法法)
            if (dstTypeMax_ == DIGIT_ZERO_FLOAT || dstTypeMax_ == DIGIT_SIX_FLOAT || dstTypeMax_ == DIGIT_SEVEN_FLOAT) {
                ComputeScaleDynamicDefault(dataLen, blockSize_, xAddr + xOffset, mxScale1Addr + scale1UbOffset,
                                           mxScale1ReciprocalAddr + scale1ReciprocalOffset,
                                           tmpScale2Addr + scale2UbOffset,
                                           mxScale2ReciprocalAddr + scale2ReciprocalOffset);
            } else {
                ComputeScaleDynamicCustom(dataLen, blockSize_, xAddr + xOffset, mxScale1Addr + scale1UbOffset,
                                          mxScale1ReciprocalAddr + scale1ReciprocalOffset,
                                          tmpScale2Addr + scale2UbOffset,
                                          mxScale2ReciprocalAddr + scale2ReciprocalOffset);
            }
        }

        ComputeYVf(dataLen, blockSize_, xAddr + xOffset, mxScale1ReciprocalAddr + scale1ReciprocalOffset,
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
            ComputeScaleOcp(dataLen, static_cast<uint16_t>(calcBlockTail), xAddr + xOffset,
                            mxScale1Addr + scale1UbOffset, mxScale1ReciprocalAddr + scale1ReciprocalOffset,
                            tmpScale2Addr + scale2UbOffset, mxScale2ReciprocalAddr + scale2ReciprocalOffset);
        } else if constexpr (scaleAlg == 1) {
            ComputeScaleCublas(dataLen, static_cast<uint16_t>(calcBlockTail), xAddr + xOffset,
                               mxScale1Addr + scale1UbOffset, mxScale1ReciprocalAddr + scale1ReciprocalOffset,
                               tmpScale2Addr + scale2UbOffset, mxScale2ReciprocalAddr + scale2ReciprocalOffset);
        } else if constexpr (scaleAlg == 2) {
            // DynamicDtypeRange Scale算法 (FP4_E2M1专用) - 尾块处理
            if (dstTypeMax_ == DIGIT_ZERO_FLOAT || dstTypeMax_ == DIGIT_SIX_FLOAT || dstTypeMax_ == DIGIT_SEVEN_FLOAT) {
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
        ComputeYVf(dataLen, static_cast<uint16_t>(calcBlockTail), xAddr + xOffset,
                   mxScale1ReciprocalAddr + scale1ReciprocalOffset, mxScale2ReciprocalAddr + scale2ReciprocalOffset,
                   y1Addr + yOffset, y2Addr + yOffset);
    }

    // event_t eventIDVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    // SetFlag<HardEvent::V_S>(eventIDVToS);
    // WaitFlag<HardEvent::V_S>(eventIDVToS);

    // -2轴的scale交织处理
    for (int64_t i = 1; i < ((calcBlockLoop + 1) / DIGIT_TWO * DIGIT_TWO); i = i + 2) {
        Interleave(mxScale2[(i - 1) * ubRowLen_], mxScale2[i * ubRowLen_], tmpScale2Local[(i - 1) * ubRowLen_],
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
#ifndef ASCENDC_CPU_DEBUG
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
        // Reg::MaskReg infNanDataMask0;
        // Reg::MaskReg infNanDataMask1;
        Reg::MaskReg maskAll = Reg::CreateMask<xDtype, Reg::MaskPattern::ALL>();
        Reg::MaskReg maskB8 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg maskReduceB8 = Reg::CreateMask<uint8_t, Reg::MaskPattern::VL8>();
        Reg::MaskReg maskReduceB16 = Reg::CreateMask<uint8_t, Reg::MaskPattern::VL16>();

        Reg::Duplicate(expMaskBF16, EXP_MASK_BF16);
        Reg::Duplicate(expMaskFP16, EXP_MASK_FP16);
        Reg::Duplicate(expMax1Dim2, 0);
        Reg::Duplicate(expMax2Dim2, 0);
        Reg::Duplicate(yMaxExp, dtypeYMaxExp_);
        Reg::Duplicate(nanE8M0, NAN_FOR_FP8_E8M0);
        Reg::Duplicate(biasE8M0, BF16_EXP_BIAS);
        Reg::Duplicate(zero, 0);
        Reg::Duplicate(nanBF16, NAN_CUSTOMIZATION);
        Reg::Duplicate(specialExp, SPECIAL_EXP_THRESHOLD);

        for (uint16_t i = 0; i < blockCount; i++) {
            // 交织搬运，一次搬256个B16
            Reg::LoadAlign<xDtype, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B16>(
                x0, x1, xAddr, vlForHalfNumber_ * DIGIT_TWO);
            if constexpr (IsSameType<xDtype, half>::value) {
                // 原始数据转成bf16
                Reg::Cast<bfloat16_t, xDtype, castTraitHalf2BF16>(x0BF16, x0, maskAll);
                Reg::Cast<bfloat16_t, xDtype, castTraitHalf2BF16>(x1BF16, x1, maskAll);
                // 提取指数位
                Reg::And(x0ExpBF16, (Reg::RegTensor<uint16_t>&)x0BF16, expMaskBF16, maskAll);
                Reg::And(x1ExpBF16, (Reg::RegTensor<uint16_t>&)x1BF16, expMaskBF16, maskAll);
            } else {
                // 提取指数位
                Reg::And(x0ExpBF16, (Reg::RegTensor<uint16_t>&)x0, expMaskBF16, maskAll);
                Reg::And(x1ExpBF16, (Reg::RegTensor<uint16_t>&)x1, expMaskBF16, maskAll);
            }
            // 计算x0和x1的最大值，相当于计算原始相邻两个数据的最大值
            Reg::Max(expMaxDim1, x0ExpBF16, x1ExpBF16, maskAll);
            // ReduceMax一个block，即16个数，配合上一步，可以计算出每32个数的最大值，一共256/32个
            Reg::ReduceDataBlock<Reg::ReduceType::MAX>(expMaxDim1, expMaxDim1, maskAll);
            // 二分性能更高，待定
            Reg::Max(expMax1Dim2, expMax1Dim2, x0ExpBF16, maskAll);
            Reg::Max(expMax2Dim2, expMax2Dim2, x1ExpBF16, maskAll);

            // 计算-1轴的scale和1/scale
            // inf/nan值单独处理，结果为E8M0的nan
            Reg::Compare<uint16_t, CMPMODE::NE>(infMask, expMaxDim1, expMaskBF16, maskAll);
            // 0值单独处理，结果为0
            Reg::Compare<uint16_t, CMPMODE::NE>(zeroMask, expMaxDim1, zero, maskAll);
            // 指数位不足被量化类型的ele_max时，为subnormal场景，结果为0
            Reg::Compare<uint16_t, CMPMODE::LE>(invalidDataMask, expMaxDim1, yMaxExp, maskAll);
            Reg::Select<uint16_t>(expMaxDim1, yMaxExp, expMaxDim1, invalidDataMask);
            // 指数位减去expMax，按照BF16的格式处理，例：E5M2的expMax为15，即需要减去0 00001111 0000000
            Reg::Sub(expMaxDim1, expMaxDim1, yMaxExp, maskAll);
            // 右移7位，BF16的指数位移到了末8位
            Reg::ShiftRights(mxScale1B16, expMaxDim1, SHR_NUM_FOR_BF16, maskAll);
            Reg::Select<uint16_t>(mxScale1B16, mxScale1B16, nanE8M0, infMask);
            Reg::Select<uint16_t>(mxScale1B16, mxScale1B16, zero, zeroMask);

            Reg::Pack<uint8_t, uint16_t, Reg::HighLowPart::LOWEST>(mxScale1B8, mxScale1B16);
            Reg::StoreAlign<uint8_t>(mxScale1Addr + i * oneBlockCountB8_, mxScale1B8, maskReduceB8);

            // 公式中的1/X
            // 只有在E1M2时，yMaxExp=0，expMaxDim1可能会等于biasE8M0
            Reg::Compare<uint16_t, CMPMODE::EQ>(invalidDataMask, expMaxDim1, biasE8M0, maskAll);

            Reg::Sub(reversedShareExp1, biasE8M0, expMaxDim1, maskAll);
            Reg::Select<uint16_t>(reversedShareExp1, reversedShareExp1, nanBF16, infMask);
            Reg::Select<uint16_t>(reversedShareExp1, reversedShareExp1, zero, zeroMask);
            Reg::Select<uint16_t>(reversedShareExp1, specialExp, reversedShareExp1, invalidDataMask);
            Reg::StoreAlign<uint16_t>(mxScale1ReciprocalAddr + i * oneBlockCountB16_, reversedShareExp1, maskReduceB16);
        }
        // 计算-2轴的scale2和1/scale2 交织第一部分
        // inf/nan值单独处理，结果为E8M0的nan
        Reg::Compare<uint16_t, CMPMODE::NE>(infMask, expMax1Dim2, expMaskBF16, maskAll);
        // 0值单独处理，结果为0
        Reg::Compare<uint16_t, CMPMODE::NE>(zeroMask, expMax1Dim2, zero, maskAll);
        // 指数位不足被量化类型的ele_max时，为subnormal场景，结果为0
        Reg::Compare<uint16_t, CMPMODE::LE>(invalidDataMask, expMax1Dim2, yMaxExp, maskAll);
        Reg::Select<uint16_t>(expMax1Dim2, yMaxExp, expMax1Dim2, invalidDataMask);
        // 指数位减去expMax，按照BF16的格式处理，例：E5M2的expMax为15，即需要减去0 00001111 0000000
        Reg::Sub(expMax1Dim2, expMax1Dim2, yMaxExp, maskAll);
        // 右移7位，BF16的指数位移到了末8位
        Reg::ShiftRights(mxScale2ZeroB16, expMax1Dim2, SHR_NUM_FOR_BF16, maskAll);
        Reg::Select<uint16_t>(mxScale2ZeroB16, mxScale2ZeroB16, nanE8M0, infMask);
        Reg::Select<uint16_t>(mxScale2ZeroB16, mxScale2ZeroB16, zero, zeroMask);

        Reg::Pack<uint8_t, uint16_t, Reg::HighLowPart::LOWEST>(mxScale2ZeroB8, mxScale2ZeroB16);

        // 公式中的1/X
        // 只有在E1M2时，yMaxExp=0，expMax1Dim2可能会等于biasE8M0
        Reg::Compare<uint16_t, CMPMODE::EQ>(invalidDataMask, expMax1Dim2, biasE8M0, maskAll);

        Reg::Sub(reversedShareExp2Zero, biasE8M0, expMax1Dim2, maskAll);
        Reg::Select<uint16_t>(reversedShareExp2Zero, reversedShareExp2Zero, nanBF16, infMask);
        Reg::Select<uint16_t>(reversedShareExp2Zero, reversedShareExp2Zero, zero, zeroMask);
        Reg::Select<uint16_t>(reversedShareExp2Zero, specialExp, reversedShareExp2Zero, invalidDataMask);

        // 计算-2轴的scale和1/scale 交织第二部分
        // inf/nan值单独处理，结果为E8M0的nan
        Reg::Compare<uint16_t, CMPMODE::NE>(infMask, expMax2Dim2, expMaskBF16, maskAll);
        // 0值单独处理，结果为0
        Reg::Compare<uint16_t, CMPMODE::NE>(zeroMask, expMax2Dim2, zero, maskAll);
        // 指数位不足被量化类型的ele_max时，为subnormal场景，结果为0
        Reg::Compare<uint16_t, CMPMODE::LE>(invalidDataMask, expMax2Dim2, yMaxExp, maskAll);
        Reg::Select<uint16_t>(expMax2Dim2, yMaxExp, expMax2Dim2, invalidDataMask);
        // 指数位减去expMax，按照BF16的格式处理，例：E5M2的expMax为15，即需要减去0 00001111 0000000
        Reg::Sub(expMax2Dim2, expMax2Dim2, yMaxExp, maskAll);
        // 右移7位，BF16的指数位移到了末8位
        Reg::ShiftRights(mxScale2OneB16, expMax2Dim2, SHR_NUM_FOR_BF16, maskAll);
        Reg::Select<uint16_t>(mxScale2OneB16, mxScale2OneB16, nanE8M0, infMask);
        Reg::Select<uint16_t>(mxScale2OneB16, mxScale2OneB16, zero, zeroMask);

        Reg::Pack<uint8_t, uint16_t, Reg::HighLowPart::LOWEST>(mxScale2OneB8, mxScale2OneB16);
        // 公式中的1/X
        // 只有在E1M2时，yMaxExp=0，expMax2Dim2可能会等于biasE8M0
        Reg::Compare<uint16_t, CMPMODE::EQ>(invalidDataMask, expMax2Dim2, biasE8M0, maskAll);
        Reg::Sub(reversedShareExp2One, biasE8M0, expMax2Dim2, maskAll);
        Reg::Select<uint16_t>(reversedShareExp2One, reversedShareExp2One, nanBF16, infMask);
        Reg::Select<uint16_t>(reversedShareExp2One, reversedShareExp2One, zero, zeroMask);
        Reg::Select<uint16_t>(reversedShareExp2One, specialExp, reversedShareExp2One, invalidDataMask);
        // 交织搬出mxScale和1/scale
        Reg::StoreAlign<uint8_t, Reg::StoreDist::DIST_INTLV_B8>(mxScale2Addr, mxScale2ZeroB8, mxScale2OneB8, maskB8);
        Reg::StoreAlign<uint16_t, Reg::StoreDist::DIST_INTLV_B16>(mxScale2ReciprocalAddr, reversedShareExp2Zero,
                                                                  reversedShareExp2One, maskAll);
    }
#endif
}

// CuBALS Scale算法实现 (scaleAlg=1, FP8专用)
// 整体框架与ComputeScaleOcp一致：循环blockCount次处理-1轴scale，循环后处理-2轴scale
// 算法差异：OCP使用指数提取法，CuBALS使用 Amax/Amax(DType) + FP32指数尾数条件舍入法
template <typename xDtype, typename y1Dtype, typename y2Dtype, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__aicore__ inline void
DynamicMxQuantWithDualAxisBase<xDtype, y1Dtype, y2Dtype, roundMode, scaleAlg>::ComputeScaleCublas(
    uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr, __ubuf__ uint8_t* mxScale1Addr,
    __ubuf__ uint16_t* mxScale1ReciprocalAddr, __ubuf__ uint8_t* mxScale2Addr,
    __ubuf__ uint16_t* mxScale2ReciprocalAddr)
{
#ifndef ASCENDC_CPU_DEBUG
    __VEC_SCOPE__
    {
        // ========== 输入数据寄存器 ==========
        Reg::RegTensor<xDtype> x0;
        Reg::RegTensor<xDtype> x1;

        // ========== 绝对值和max寄存器 ==========
        Reg::RegTensor<uint16_t> absMax0;     // x0的绝对值
        Reg::RegTensor<uint16_t> absMax1;     // x1的绝对值
        Reg::RegTensor<uint16_t> absMaxDim1;  // -1轴方向block内绝对值max
        Reg::RegTensor<uint16_t> absMax1Dim2; // -2轴方向累积max (偶数列, 对应x0)
        Reg::RegTensor<uint16_t> absMax2Dim2; // -2轴方向累积max (奇数列, 对应x1)
        Reg::RegTensor<uint16_t> zeroB16;     // -1轴Interleave用零寄存器

        // ========== FP32计算寄存器 ==========
        // -1轴: Interleave-with-0后单次Cast Zero处理全部8个值，仅需一组FP32寄存器
        // -2轴: 仍需Zero/One两组独立处理
        Reg::RegTensor<uint32_t> maxFP32_0; // FP32表示, 链内复用为expPlusOne
        Reg::RegTensor<uint32_t> maxFP32_1; // -2轴奇数部分FP32表示
        Reg::RegTensor<uint32_t> expFP32_0; // FP32指数
        Reg::RegTensor<uint32_t> expFP32_1; // -2轴奇数部分FP32指数
        Reg::RegTensor<uint32_t> manFP32_0; // FP32尾数, 链内复用为extractExp
        Reg::RegTensor<uint32_t> manFP32_1; // -2轴奇数部分FP32尾数

        // scale输出寄存器 (循环后复用于-2轴scale输出)
        Reg::RegTensor<uint16_t> scale1B16_0;       // E8M0 uint16, 循环后复用为mxScale2ZeroB16
        Reg::RegTensor<uint16_t> scale1B16_1;       // -2轴奇数部分, 循环后复用为mxScale2OneB16
        Reg::RegTensor<uint16_t> scale1BF16;        // BF16指数格式, 循环后复用为scale2BF16
        Reg::RegTensor<uint8_t> mxScale1B8;         // uint8 scale, 循环后复用为mxScale2ZeroB8
        Reg::RegTensor<uint16_t> reversedShareExp1; // 1/scale BF16, 循环后复用为reversedShareExp2Zero

        // -2轴独立寄存器 (需与复用寄存器同时存活，无法复用)
        Reg::RegTensor<uint8_t> mxScale2OneB8; // 与mxScale1B8同时存活于最终DataCopy

        // ========== 常量寄存器 ==========
        Reg::RegTensor<uint16_t> absMask;
        Reg::Duplicate(absMask, ABS_MASK_FOR_16BIT);
        Reg::RegTensor<uint32_t> invMax;
        Reg::Duplicate(invMax, invDtypeMax_); // 1/Amax(DType), FP32表示
        Reg::RegTensor<uint32_t> manMaskFP32;
        Reg::Duplicate(manMaskFP32, MAN_MASK_FLOAT); // FP32尾数掩码
        Reg::RegTensor<uint32_t> scaleBiasFP32;
        Reg::Duplicate(scaleBiasFP32, FP32_EXP_BIAS_CUBLAS); // BF16偏移在uint32
        Reg::RegTensor<uint32_t> nanPackFP32;
        Reg::Duplicate(nanPackFP32, NAN_CUSTOMIZATION_PACK);

        Reg::RegTensor<uint16_t> nanE8M0;
        Reg::Duplicate(nanE8M0, NAN_FOR_FP8_E8M0);
        Reg::RegTensor<uint16_t> biasE8M0;
        Reg::Duplicate(biasE8M0, BF16_EXP_BIAS);
        Reg::RegTensor<uint16_t> zero;
        Reg::Duplicate(zero, 0);
        Reg::RegTensor<uint16_t> nanBF16;
        Reg::Duplicate(nanBF16, NAN_CUSTOMIZATION);
        Reg::RegTensor<uint16_t> specialExp;
        Reg::Duplicate(specialExp, SPECIAL_EXP_THRESHOLD);
        Reg::RegTensor<uint16_t> maxEleBF16;
        Reg::Duplicate(maxEleBF16, EXP_MASK_BF16);

        Reg::Duplicate(absMax1Dim2, 0);
        Reg::Duplicate(absMax2Dim2, 0);
        Reg::Duplicate(zeroB16, 0);

        // ========== Mask定义 ==========
        Reg::MaskReg maskAll = Reg::CreateMask<xDtype, Reg::MaskPattern::ALL>();
        Reg::MaskReg maskB8 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg maskReduceB8 = Reg::CreateMask<uint8_t, Reg::MaskPattern::VL8>();
        Reg::MaskReg maskReduceB16 = Reg::CreateMask<uint8_t, Reg::MaskPattern::VL16>();
        Reg::MaskReg maskFP32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();

        Reg::MaskReg p0;    // 条件舍入mask
        Reg::MaskReg p1;    // subnormal条件mask
        Reg::MaskReg p0Odd; // -2轴奇数部分条件舍入mask
        Reg::MaskReg p1Odd; // -2轴奇数部分subnormal条件mask
        Reg::MaskReg infMask;
        Reg::MaskReg invalidDataMask;

        // ========================================================================
        // 循环blockCount次，每次处理一行，计算-1轴scale并累积-2轴max
        // ========================================================================
        for (uint16_t i = 0; i < blockCount; i++) {
            // 1. 交织搬运输入数据: 将256个xDtype按偶奇拆分为x0(偶), x1(奇)
            Reg::LoadAlign<xDtype, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B16>(
                x0, x1, xAddr, vlForHalfNumber_ * DIGIT_TWO);

            // 2. 取绝对值: 清除符号位，保留指数和尾数
            Reg::And(absMax0, (Reg::RegTensor<uint16_t>&)x0, absMask, maskAll);
            Reg::And(absMax1, (Reg::RegTensor<uint16_t>&)x1, absMask, maskAll);

            // 3. -1轴: 先取偶奇max，再ReduceMaxWithDataBlock得到每32个元素的绝对值max
            Reg::Max(absMaxDim1, absMax0, absMax1, maskAll);
            Reg::ReduceDataBlock<Reg::ReduceType::MAX>(absMaxDim1, absMaxDim1, maskAll);

            // 4. -2轴: 逐行累积偶数列和奇数列的绝对值max
            Reg::Max(absMax1Dim2, absMax1Dim2, absMax0, maskAll);
            Reg::Max(absMax2Dim2, absMax2Dim2, absMax1, maskAll);

            // ============================================================
            // 5. 计算-1轴CuBALS Scale (FP32精度)
            //    ReduceMaxWithDataBlock后，8个max紧凑存储在寄存器前8个位置
            //    与0交织后，Cast Zero一次处理全部8个值转为FP32
            //    (与原始DynamicMxQuant的ComputeCuBLAS一致)
            // ============================================================

            // 与0交织: [v0,0,v1,0,...,v7,0,...] → Cast Zero可一次取出全部8个有效值
            Reg::Interleave(absMaxDim1, zeroB16, absMaxDim1, zeroB16);
            Reg::Cast<float, xDtype, castTraitXdtypetoFp32Zero>((Reg::RegTensor<float>&)maxFP32_0,
                                                                (Reg::RegTensor<xDtype>&)absMaxDim1, maskAll);
            // 乘以 1/Amax(DType): max * invDtypeMax
            Reg::Mul((Reg::RegTensor<float>&)maxFP32_0, (Reg::RegTensor<float>&)maxFP32_0,
                     (Reg::RegTensor<float>&)invMax, maskFP32);
            // 提取FP32指数: 右移23位
            Reg::ShiftRights(expFP32_0, maxFP32_0, SHR_NUM_FOR_FP32, maskFP32);
            // 提取FP32尾数: 与尾数掩码
            Reg::And(manFP32_0, maxFP32_0, manMaskFP32, maskFP32);
            // 条件舍入: normal场景 (exp>0 && exp<254 && man>0) → exp+1
            Reg::Compares<uint32_t, CMPMODE::GT>(p0, expFP32_0, NUMBER_ZERO_U32, maskFP32);
            Reg::Compares<uint32_t, CMPMODE::LT>(p0, expFP32_0, NUMBER_TWO_FIVE_FOUR, p0);
            Reg::Compares<uint32_t, CMPMODE::GT>(p0, manFP32_0, NUMBER_ZERO_U32, p0);
            // 条件舍入: subnormal场景 (exp==0 && man>HALF) → exp+1
            Reg::Compares<uint32_t, CMPMODE::EQ>(p1, expFP32_0, NUMBER_ZERO_U32, maskFP32);
            Reg::Compares<uint32_t, CMPMODE::GT>(p1, manFP32_0, NUMBER_HALF_U32, p1);
            Reg::Or(p0, p0, p1, maskFP32);
            // 执行条件加1
            Reg::Adds(maxFP32_0, expFP32_0, 1, maskFP32);
            Reg::Select(manFP32_0, maxFP32_0, expFP32_0, p0);
            // Pack到uint16 (INF/NAN→0xFF, zero→0 自然通过条件舍入保持, 在BF16域1/scale中统一处理)
            Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>(scale1B16_0, manFP32_0);

            // 左移7位，将E8M0值定位到BF16指数域 (用于计算1/scale)
            Reg::ShiftLefts(scale1BF16, scale1B16_0, SHR_NUM_FOR_BF16, maskAll);

            // --- 输出-1轴scale (uint8) ---
            Reg::Pack<uint8_t, uint16_t, Reg::HighLowPart::LOWEST>(mxScale1B8, scale1B16_0);
            Reg::StoreAlign<uint8_t>(mxScale1Addr + i * oneBlockCountB8_, mxScale1B8, maskReduceB8);

            // --- 计算并输出-1轴 1/scale (与原始DynamicMxQuant一致: inf→nan, special→specialExp, 无零值检查) ---
            Reg::Compare<uint16_t, CMPMODE::NE>(infMask, scale1BF16, maxEleBF16, maskAll);
            Reg::Compare<uint16_t, CMPMODE::EQ>(invalidDataMask, scale1BF16, biasE8M0, maskAll);
            Reg::Sub(reversedShareExp1, biasE8M0, scale1BF16, maskAll);
            Reg::Select<uint16_t>(reversedShareExp1, reversedShareExp1, nanBF16, infMask);
            Reg::Select<uint16_t>(reversedShareExp1, specialExp, reversedShareExp1, invalidDataMask);
            Reg::StoreAlign<uint16_t>(mxScale1ReciprocalAddr + i * oneBlockCountB16_, reversedShareExp1, maskReduceB16);
            // 恢复zeroB16 (Interleave会修改dst1)
            Reg::Duplicate(zeroB16, 0);
        }

        // ========================================================================
        // 循环结束后，计算-2轴CuBALS Scale
        // absMax1Dim2: 128个偶数列的累积绝对值max
        // absMax2Dim2: 128个奇数列的累积绝对值max
        // 每个需要拆分为Zero/One两半分别做FP32计算，再合并
        // ========================================================================

        // ---------- 处理absMax1Dim2 (偶数列, -2轴scale的交织第一部分) ----------
        // Zero半 (偶数位) — 复用-1轴寄存器: maxFP32_0, expFP32_0, manFP32_0
        Reg::Cast<float, xDtype, castTraitXdtypetoFp32Zero>((Reg::RegTensor<float>&)maxFP32_0,
                                                            (Reg::RegTensor<xDtype>&)absMax1Dim2, maskAll);
        Reg::Mul((Reg::RegTensor<float>&)maxFP32_0, (Reg::RegTensor<float>&)maxFP32_0, (Reg::RegTensor<float>&)invMax,
                 maskFP32);
        Reg::ShiftRights(expFP32_0, maxFP32_0, SHR_NUM_FOR_FP32, maskFP32);
        Reg::And(manFP32_0, maxFP32_0, manMaskFP32, maskFP32);
        Reg::Compares<uint32_t, CMPMODE::GT>(p0, expFP32_0, NUMBER_ZERO_U32, maskFP32);
        Reg::Compares<uint32_t, CMPMODE::LT>(p0, expFP32_0, NUMBER_TWO_FIVE_FOUR, p0);
        Reg::Compares<uint32_t, CMPMODE::GT>(p0, manFP32_0, NUMBER_ZERO_U32, p0);
        Reg::Compares<uint32_t, CMPMODE::EQ>(p1, expFP32_0, NUMBER_ZERO_U32, maskFP32);
        Reg::Compares<uint32_t, CMPMODE::GT>(p1, manFP32_0, NUMBER_HALF_U32, p1);
        Reg::Or(p0, p0, p1, maskFP32);
        // 链内复用: maxFP32_0→expPlusOne (maxFP32_0已死亡@And)
        Reg::Adds(maxFP32_0, expFP32_0, 1, maskFP32);
        // 链内复用: manFP32_0→extractExp (manFP32_0已死亡@Compares)
        Reg::Select(manFP32_0, maxFP32_0, expFP32_0, p0);
        Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>(scale1B16_0, manFP32_0);

        // One半 (奇数位) - 使用独立的p0Odd/p1Odd，与Zero半并行
        Reg::Cast<float, xDtype, castTraitXdtypetoFp32One>((Reg::RegTensor<float>&)maxFP32_1,
                                                           (Reg::RegTensor<xDtype>&)absMax1Dim2, maskAll);
        Reg::Mul((Reg::RegTensor<float>&)maxFP32_1, (Reg::RegTensor<float>&)maxFP32_1, (Reg::RegTensor<float>&)invMax,
                 maskFP32);
        Reg::ShiftRights(expFP32_1, maxFP32_1, SHR_NUM_FOR_FP32, maskFP32);
        Reg::And(manFP32_1, maxFP32_1, manMaskFP32, maskFP32);
        Reg::Compares<uint32_t, CMPMODE::GT>(p0Odd, expFP32_1, NUMBER_ZERO_U32, maskFP32);
        Reg::Compares<uint32_t, CMPMODE::LT>(p0Odd, expFP32_1, NUMBER_TWO_FIVE_FOUR, p0Odd);
        Reg::Compares<uint32_t, CMPMODE::GT>(p0Odd, manFP32_1, NUMBER_ZERO_U32, p0Odd);
        Reg::Compares<uint32_t, CMPMODE::EQ>(p1Odd, expFP32_1, NUMBER_ZERO_U32, maskFP32);
        Reg::Compares<uint32_t, CMPMODE::GT>(p1Odd, manFP32_1, NUMBER_HALF_U32, p1Odd);
        Reg::Or(p0Odd, p0Odd, p1Odd, maskFP32);
        // 链内复用: maxFP32_1→expPlusOne, manFP32_1→extractExp
        Reg::Adds(maxFP32_1, expFP32_1, 1, maskFP32);
        Reg::Select(manFP32_1, maxFP32_1, expFP32_1, p0Odd);
        Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>(scale1B16_1, manFP32_1);

        // 合并Zero和One，恢复原始列顺序
        Reg::Interleave(scale1B16_0, scale1B16_1, scale1B16_0, scale1B16_1);
        // 左移7位得到BF16指数格式 (用于计算1/scale), 复用scale1BF16
        Reg::ShiftLefts(scale1BF16, scale1B16_0, SHR_NUM_FOR_BF16, maskAll);
        // 输出scale (uint8), 复用mxScale1B8
        Reg::Pack<uint8_t, uint16_t, Reg::HighLowPart::LOWEST>(mxScale1B8, scale1B16_0);

        // 计算1/scale, 复用reversedShareExp1 (与原始DynamicMxQuant一致: 无零值检查)
        Reg::Compare<uint16_t, CMPMODE::NE>(infMask, scale1BF16, maxEleBF16, maskAll);
        Reg::Compare<uint16_t, CMPMODE::EQ>(invalidDataMask, scale1BF16, biasE8M0, maskAll);
        Reg::Sub(reversedShareExp1, biasE8M0, scale1BF16, maskAll);
        Reg::Select<uint16_t>(reversedShareExp1, reversedShareExp1, nanBF16, infMask);
        Reg::Select<uint16_t>(reversedShareExp1, specialExp, reversedShareExp1, invalidDataMask);

        // ---------- 处理absMax2Dim2 (奇数列, -2轴scale的交织第二部分) ----------
        // Zero半 — 再次复用maxFP32_0, expFP32_0, manFP32_0
        Reg::Cast<float, xDtype, castTraitXdtypetoFp32Zero>((Reg::RegTensor<float>&)maxFP32_0,
                                                            (Reg::RegTensor<xDtype>&)absMax2Dim2, maskAll);
        Reg::Mul((Reg::RegTensor<float>&)maxFP32_0, (Reg::RegTensor<float>&)maxFP32_0, (Reg::RegTensor<float>&)invMax,
                 maskFP32);
        Reg::ShiftRights(expFP32_0, maxFP32_0, SHR_NUM_FOR_FP32, maskFP32);
        Reg::And(manFP32_0, maxFP32_0, manMaskFP32, maskFP32);
        Reg::Compares<uint32_t, CMPMODE::GT>(p0, expFP32_0, NUMBER_ZERO_U32, maskFP32);
        Reg::Compares<uint32_t, CMPMODE::LT>(p0, expFP32_0, NUMBER_TWO_FIVE_FOUR, p0);
        Reg::Compares<uint32_t, CMPMODE::GT>(p0, manFP32_0, NUMBER_ZERO_U32, p0);
        Reg::Compares<uint32_t, CMPMODE::EQ>(p1, expFP32_0, NUMBER_ZERO_U32, maskFP32);
        Reg::Compares<uint32_t, CMPMODE::GT>(p1, manFP32_0, NUMBER_HALF_U32, p1);
        Reg::Or(p0, p0, p1, maskFP32);
        // 链内复用: maxFP32_0→expPlusOne, manFP32_0→extractExp
        Reg::Adds(maxFP32_0, expFP32_0, 1, maskFP32);
        Reg::Select(manFP32_0, maxFP32_0, expFP32_0, p0);
        Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>(scale1B16_0, manFP32_0);

        // One半 - 使用独立的p0Odd/p1Odd，与Zero半并行
        Reg::Cast<float, xDtype, castTraitXdtypetoFp32One>((Reg::RegTensor<float>&)maxFP32_1,
                                                           (Reg::RegTensor<xDtype>&)absMax2Dim2, maskAll);
        Reg::Mul((Reg::RegTensor<float>&)maxFP32_1, (Reg::RegTensor<float>&)maxFP32_1, (Reg::RegTensor<float>&)invMax,
                 maskFP32);
        Reg::ShiftRights(expFP32_1, maxFP32_1, SHR_NUM_FOR_FP32, maskFP32);
        Reg::And(manFP32_1, maxFP32_1, manMaskFP32, maskFP32);
        Reg::Compares<uint32_t, CMPMODE::GT>(p0Odd, expFP32_1, NUMBER_ZERO_U32, maskFP32);
        Reg::Compares<uint32_t, CMPMODE::LT>(p0Odd, expFP32_1, NUMBER_TWO_FIVE_FOUR, p0Odd);
        Reg::Compares<uint32_t, CMPMODE::GT>(p0Odd, manFP32_1, NUMBER_ZERO_U32, p0Odd);
        Reg::Compares<uint32_t, CMPMODE::EQ>(p1Odd, expFP32_1, NUMBER_ZERO_U32, maskFP32);
        Reg::Compares<uint32_t, CMPMODE::GT>(p1Odd, manFP32_1, NUMBER_HALF_U32, p1Odd);
        Reg::Or(p0Odd, p0Odd, p1Odd, maskFP32);
        // 链内复用: maxFP32_1→expPlusOne, manFP32_1→extractExp
        Reg::Adds(maxFP32_1, expFP32_1, 1, maskFP32);
        Reg::Select(manFP32_1, maxFP32_1, expFP32_1, p0Odd);
        Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>(scale1B16_1, manFP32_1);

        // 合并Zero和One
        Reg::Interleave(scale1B16_0, scale1B16_1, scale1B16_0, scale1B16_1);
        // 复用scale1BF16
        Reg::ShiftLefts(scale1BF16, scale1B16_0, SHR_NUM_FOR_BF16, maskAll);
        Reg::Pack<uint8_t, uint16_t, Reg::HighLowPart::LOWEST>(mxScale2OneB8, scale1B16_0);

        // 计算1/scale, 复用absMax0 (循环后已死亡) (与原始DynamicMxQuant一致: 无零值检查)
        Reg::Compare<uint16_t, CMPMODE::NE>(infMask, scale1BF16, maxEleBF16, maskAll);
        Reg::Compare<uint16_t, CMPMODE::EQ>(invalidDataMask, scale1BF16, biasE8M0, maskAll);
        Reg::Sub(absMax0, biasE8M0, scale1BF16, maskAll);
        Reg::Select<uint16_t>(absMax0, absMax0, nanBF16, infMask);
        Reg::Select<uint16_t>(absMax0, specialExp, absMax0, invalidDataMask);

        // 交织搬出-2轴的mxScale和1/scale
        Reg::StoreAlign<uint8_t, Reg::StoreDist::DIST_INTLV_B8>(mxScale2Addr, mxScale1B8, mxScale2OneB8, maskB8);
        Reg::StoreAlign<uint16_t, Reg::StoreDist::DIST_INTLV_B16>(mxScale2ReciprocalAddr, reversedShareExp1, absMax0,
                                                                  maskAll);
    }
#endif
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
        Reg::RegTensor<xDtype> x0;
        Reg::RegTensor<xDtype> x1;
        Reg::RegTensor<bfloat16_t> x0BF16;
        Reg::RegTensor<bfloat16_t> x1BF16;

        // ========== 绝对值和max寄存器 ==========
        Reg::RegTensor<uint16_t> absVal0;     // x0的BF16绝对值
        Reg::RegTensor<uint16_t> absVal1;     // x1的BF16绝对值
        Reg::RegTensor<uint16_t> absMaxDim1;  // -1轴block内绝对值max (ReduceMax后)
        Reg::RegTensor<uint16_t> absMax1Dim2; // -2轴累积绝对值max (偶数列, 对应x0)
        Reg::RegTensor<uint16_t> absMax2Dim2; // -2轴累积绝对值max (奇数列, 对应x1)

        // ========== -1轴scale计算寄存器 (循环后复用于-2轴) ==========
        Reg::RegTensor<uint16_t> expOnly;           // 提取的指数位, 循环后复用为dim2ExpOnly
        Reg::RegTensor<uint16_t> addedVal;          // addValueBit进位后的值, 循环后复用为dim2SubResult
        Reg::RegTensor<uint16_t> sharedExp;         // 指数差值, 循环后复用为dim2ExpExtract
        Reg::RegTensor<uint16_t> scaleValue;        // E8M0 scale值, 循环后复用为mxScale2B16
        Reg::RegTensor<uint8_t> mxScale1B8;         // -1轴scale输出, 循环后复用为mxScale2ZeroB8
        Reg::RegTensor<uint16_t> reversedShareExp1; // -1轴1/scale, 循环后复用为reversedShareExp2Zero

        // ========== -2轴独立寄存器 (需与复用寄存器同时存活，无法复用) ==========
        Reg::RegTensor<uint8_t> mxScale2OneB8; // 与mxScale1B8同时存活于最终DataCopy

        // ========== 常量寄存器 ==========
        Reg::RegTensor<uint16_t> absMask;
        Reg::Duplicate(absMask, ABS_MASK_FOR_16BIT); // 绝对值掩码 0x7fff
        Reg::RegTensor<uint16_t> expMaskBF16;
        Reg::Duplicate(expMaskBF16, EXP_MASK_BF16); // BF16指数掩码 0x7f80
        Reg::RegTensor<uint16_t> expMaskFP16;
        Reg::Duplicate(expMaskFP16, EXP_MASK_FP16); // FP16指数掩码 0x7c00 (INF/NAN检测)
        Reg::RegTensor<uint16_t> addValue;
        Reg::Duplicate(addValue, addValueBit_); // BF16尾数进位值
        Reg::RegTensor<uint16_t> maxExpValue;
        Reg::Duplicate(maxExpValue, FP4_E2M1_BF16_MAX_EXP); // FP4_E2M1的emax在BF16中的表示
        Reg::RegTensor<uint16_t> subNumForScale;
        Reg::Duplicate(subNumForScale, subNumForScale_); // -2轴减法常量
        Reg::RegTensor<uint16_t> nanE8M0;
        Reg::Duplicate(nanE8M0, NAN_FOR_FP8_E8M0); // E8M0的NAN值 0xFF
        Reg::RegTensor<uint16_t> biasE8M0;
        Reg::Duplicate(biasE8M0, BF16_EXP_BIAS); // BF16指数偏移 0x7f00
        Reg::RegTensor<uint16_t> zero;
        Reg::Duplicate(zero, 0);
        Reg::RegTensor<uint16_t> nanBF16;
        Reg::Duplicate(nanBF16, NAN_CUSTOMIZATION); // NAN_CUSTOMIZATION 0x7f81
        Reg::RegTensor<uint16_t> specialExp;
        Reg::Duplicate(specialExp, SPECIAL_EXP_THRESHOLD); // 特殊指数阈值 0x0040

        Reg::Duplicate(absMax1Dim2, 0);
        Reg::Duplicate(absMax2Dim2, 0);

        // ========== Mask定义 ==========
        Reg::MaskReg maskAll = Reg::CreateMask<xDtype, Reg::MaskPattern::ALL>();
        Reg::MaskReg maskB8 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg maskReduceB8 = Reg::CreateMask<uint8_t, Reg::MaskPattern::VL8>();
        Reg::MaskReg maskReduceB16 = Reg::CreateMask<uint8_t, Reg::MaskPattern::VL16>();

        Reg::MaskReg infMask;
        Reg::MaskReg zeroMask;
        Reg::MaskReg invalidDataMask;
        Reg::MaskReg infNanDataMask0;
        Reg::MaskReg infNanDataMask1;

        // ========================================================================
        // 循环blockCount次: 计算-1轴scale并累积-2轴BF16绝对值max
        // DynamicDtypeRange需要完整的BF16绝对值 (不仅仅是指数)
        // ========================================================================
        for (uint16_t i = 0; i < blockCount; i++) {
            // 1. 交织搬运输入数据: 将256个xDtype按偶奇拆分为x0(偶), x1(奇)
            Reg::LoadAlign<xDtype, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B16>(
                x0, x1, xAddr, vlForHalfNumber_ * DIGIT_TWO);

            // 2. 获取BF16绝对值 (区分half和bf16输入)
            if constexpr (IsSameType<xDtype, half>::value) {
                // FP16输入: 先检查INF/NAN，再转BF16(RINT)，取绝对值，INF/NAN替换为BF16 INF
                Reg::And(expOnly, (Reg::RegTensor<uint16_t>&)x0, expMaskFP16, maskAll);
                Reg::Compare<uint16_t, CMPMODE::NE>(infNanDataMask0, expOnly, expMaskFP16, maskAll);
                Reg::And(expOnly, (Reg::RegTensor<uint16_t>&)x1, expMaskFP16, maskAll);
                Reg::Compare<uint16_t, CMPMODE::NE>(infNanDataMask1, expOnly, expMaskFP16, maskAll);
                // 转BF16 (使用CAST_RINT四舍五入，不同于OCP的CAST_TRUNC截断)
                Reg::Cast<bfloat16_t, xDtype, castTraitHalf2BF16Rint>(x0BF16, x0, maskAll);
                Reg::Cast<bfloat16_t, xDtype, castTraitHalf2BF16Rint>(x1BF16, x1, maskAll);
                // 取绝对值
                Reg::And(absVal0, (Reg::RegTensor<uint16_t>&)x0BF16, absMask, maskAll);
                Reg::And(absVal1, (Reg::RegTensor<uint16_t>&)x1BF16, absMask, maskAll);
                // INF/NAN位置替换为BF16的INF (0x7f80)
                Reg::Select<uint16_t>(absVal0, absVal0, expMaskBF16, infNanDataMask0);
                Reg::Select<uint16_t>(absVal1, absVal1, expMaskBF16, infNanDataMask1);
            } else {
                // BF16输入: 直接取绝对值
                Reg::And(absVal0, (Reg::RegTensor<uint16_t>&)x0, absMask, maskAll);
                Reg::And(absVal1, (Reg::RegTensor<uint16_t>&)x1, absMask, maskAll);
            }

            // 3. -1轴: 偶奇Max + ReduceMaxWithDataBlock，得到每32个元素的绝对值max
            Reg::Max(absMaxDim1, absVal0, absVal1, maskAll);
            Reg::ReduceDataBlock<Reg::ReduceType::MAX>(absMaxDim1, absMaxDim1, maskAll);

            // 4. -2轴: 逐行累积偶数列和奇数列的BF16绝对值max
            Reg::Max(absMax1Dim2, absMax1Dim2, absVal0, maskAll);
            Reg::Max(absMax2Dim2, absMax2Dim2, absVal1, maskAll);

            // ============================================================
            // 5. 计算-1轴DynamicDtypeRange Default Scale
            //    ReduceMax后8个绝对值max紧凑存储在位置0-7
            //    使用addValueBit进位法: 将addValueBit加到完整BF16绝对值上，
            //    通过尾数进位自动实现指数的四舍五入
            // ============================================================

            // 提取指数位 (仅用于INF/NAN、零值、subnormal检查)
            Reg::And(expOnly, absMaxDim1, expMaskBF16, maskAll);
            // INF/NAN检查: 指数全1
            Reg::Compare<uint16_t, CMPMODE::NE>(infMask, expOnly, expMaskBF16, maskAll);
            // 零值检查
            Reg::Compare<uint16_t, CMPMODE::NE>(zeroMask, expOnly, zero, maskAll);
            // subnormal检查: 指数 < FP4_E2M1_BF16_MAX_EXP (注意使用LT，不是LE)
            Reg::Compare<uint16_t, CMPMODE::LT>(invalidDataMask, expOnly, maxExpValue, maskAll);

            // addValueBit进位: 将addValueBit加到完整BF16绝对值上
            Reg::Add(addedVal, absMaxDim1, addValue, maskAll);
            // 从进位后的结果中提取指数
            Reg::And(addedVal, addedVal, expMaskBF16, maskAll);
            // subnormal场景: 使用maxExpValue (FP4_E2M1_BF16_MAX_EXP)
            Reg::Select<uint16_t>(addedVal, maxExpValue, addedVal, invalidDataMask);
            // 减去FP4_E2M1_BF16_MAX_EXP得到指数差值
            Reg::Sub(sharedExp, addedVal, maxExpValue, maskAll);
            // 右移7位，将BF16指数移到低8位 → E8M0 scale
            Reg::ShiftRights(scaleValue, sharedExp, SHR_NUM_FOR_BF16, maskAll);
            // INF/NAN → NAN_FOR_FP8_E8M0 (0xFF)
            Reg::Select<uint16_t>(scaleValue, scaleValue, nanE8M0, infMask);
            // 零值 → 0
            Reg::Select<uint16_t>(scaleValue, scaleValue, zero, zeroMask);

            // 输出-1轴scale (uint8)
            Reg::Pack<uint8_t, uint16_t, Reg::HighLowPart::LOWEST>(mxScale1B8, scaleValue);
            Reg::StoreAlign<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE>(mxScale1Addr, mxScale1B8, oneBlockCountB8_,
                                                                         maskReduceB8);

            // 计算-1轴1/scale
            // sharedExp是左移7位前的指数差值，可直接用于BF16域1/scale计算
            Reg::Compare<uint16_t, CMPMODE::EQ>(invalidDataMask, sharedExp, biasE8M0, maskAll);
            Reg::Sub(reversedShareExp1, biasE8M0, sharedExp, maskAll);
            Reg::Select<uint16_t>(reversedShareExp1, reversedShareExp1, nanBF16, infMask);
            Reg::Select<uint16_t>(reversedShareExp1, reversedShareExp1, zero, zeroMask);
            Reg::Select<uint16_t>(reversedShareExp1, specialExp, reversedShareExp1, invalidDataMask);
            Reg::StoreAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(mxScale1ReciprocalAddr, reversedShareExp1,
                                                                          oneBlockCountB16_, maskReduceB16);
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
        Reg::And(expOnly, absMax1Dim2, expMaskBF16, maskAll);
        // INF/NAN检查
        Reg::Compare<uint16_t, CMPMODE::NE>(infMask, expOnly, expMaskBF16, maskAll);
        // 零值检查
        Reg::Compare<uint16_t, CMPMODE::NE>(zeroMask, expOnly, zero, maskAll);
        // subnormal检查: 指数 < FP4_E2M1_BF16_MAX_EXP
        Reg::Compare<uint16_t, CMPMODE::LT>(invalidDataMask, expOnly, maxExpValue, maskAll);

        // 复用addedVal为dim2SubResult
        Reg::Sub(addedVal, absMax1Dim2, subNumForScale, maskAll);
        // subnormal → 0
        Reg::Select<uint16_t>(addedVal, zero, addedVal, invalidDataMask);
        // 右移7位 → E8M0 scale, 复用scaleValue为mxScale2B16
        Reg::ShiftRights(scaleValue, addedVal, SHR_NUM_FOR_BF16, maskAll);
        // INF/NAN → NAN
        Reg::Select<uint16_t>(scaleValue, scaleValue, nanE8M0, infMask);
        // 零值 → 0
        Reg::Select<uint16_t>(scaleValue, scaleValue, zero, zeroMask);

        // 输出scale (uint8) — mxScale1B8复用为mxScale2ZeroB8
        Reg::Pack<uint8_t, uint16_t, Reg::HighLowPart::LOWEST>(mxScale1B8, scaleValue);

        // 计算-2轴1/scale — reversedShareExp1复用为reversedShareExp2Zero
        // 复用sharedExp为dim2ExpExtract
        Reg::And(sharedExp, addedVal, expMaskBF16, maskAll);
        Reg::Compare<uint16_t, CMPMODE::EQ>(invalidDataMask, sharedExp, biasE8M0, maskAll);
        Reg::Sub(reversedShareExp1, biasE8M0, sharedExp, maskAll);
        Reg::Select<uint16_t>(reversedShareExp1, reversedShareExp1, nanBF16, infMask);
        Reg::Select<uint16_t>(reversedShareExp1, reversedShareExp1, zero, zeroMask);
        Reg::Select<uint16_t>(reversedShareExp1, specialExp, reversedShareExp1, invalidDataMask);

        // ---------- 处理absMax2Dim2 (奇数列, -2轴scale的交织第二部分) ----------
        // 再次复用expOnly, addedVal, sharedExp, scaleValue
        Reg::And(expOnly, absMax2Dim2, expMaskBF16, maskAll);
        Reg::Compare<uint16_t, CMPMODE::NE>(infMask, expOnly, expMaskBF16, maskAll);
        Reg::Compare<uint16_t, CMPMODE::NE>(zeroMask, expOnly, zero, maskAll);
        Reg::Compare<uint16_t, CMPMODE::LT>(invalidDataMask, expOnly, maxExpValue, maskAll);

        Reg::Sub(addedVal, absMax2Dim2, subNumForScale, maskAll);
        Reg::Select<uint16_t>(addedVal, zero, addedVal, invalidDataMask);
        // 复用scaleValue为mxScale2OneB16
        Reg::ShiftRights(scaleValue, addedVal, SHR_NUM_FOR_BF16, maskAll);
        Reg::Select<uint16_t>(scaleValue, scaleValue, nanE8M0, infMask);
        Reg::Select<uint16_t>(scaleValue, scaleValue, zero, zeroMask);

        Reg::Pack<uint8_t, uint16_t, Reg::HighLowPart::LOWEST>(mxScale2OneB8, scaleValue);

        // 计算-2轴1/scale — absVal0复用为reversedShareExp2One (循环后死亡，与reversedShareExp1不冲突)
        Reg::And(sharedExp, addedVal, expMaskBF16, maskAll);
        Reg::Compare<uint16_t, CMPMODE::EQ>(invalidDataMask, sharedExp, biasE8M0, maskAll);
        Reg::Sub(absVal0, biasE8M0, sharedExp, maskAll);
        Reg::Select<uint16_t>(absVal0, absVal0, nanBF16, infMask);
        Reg::Select<uint16_t>(absVal0, absVal0, zero, zeroMask);
        Reg::Select<uint16_t>(absVal0, specialExp, absVal0, invalidDataMask);

        // 交织搬出-2轴的mxScale和1/scale
        Reg::StoreAlign<uint8_t, Reg::StoreDist::DIST_INTLV_B8>(mxScale2Addr, mxScale1B8, mxScale2OneB8, maskB8);
        Reg::StoreAlign<uint16_t, Reg::StoreDist::DIST_INTLV_B16>(mxScale2ReciprocalAddr, reversedShareExp1, absVal0,
                                                                  maskAll);
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
        Reg::RegTensor<xDtype> x0;
        Reg::RegTensor<xDtype> x1;

        // ========== 绝对值和max寄存器 ==========
        Reg::RegTensor<uint16_t> absMax0;     // x0的绝对值
        Reg::RegTensor<uint16_t> absMax1;     // x1的绝对值
        Reg::RegTensor<uint16_t> absMaxDim1;  // -1轴方向block内绝对值max
        Reg::RegTensor<uint16_t> absMax1Dim2; // -2轴方向累积max (偶数列, 对应x0)
        Reg::RegTensor<uint16_t> absMax2Dim2; // -2轴方向累积max (奇数列, 对应x1)
        Reg::RegTensor<uint16_t> zeroB16;     // -1轴Interleave用零寄存器

        // ========== FP32计算寄存器 ==========
        // -1轴: Interleave-with-0后单次Cast Zero处理全部8个值，仅需一组FP32寄存器
        // -2轴: 仍需Zero/One两组独立处理
        Reg::RegTensor<uint32_t> maxFP32_0; // FP32表示, 链内复用为expPlusOne
        Reg::RegTensor<uint32_t> maxFP32_1; // -2轴奇数部分FP32表示
        Reg::RegTensor<uint32_t> expFP32_0; // FP32指数
        Reg::RegTensor<uint32_t> expFP32_1; // -2轴奇数部分FP32指数
        Reg::RegTensor<uint32_t> manFP32_0; // FP32尾数, 链内复用为extractExp
        Reg::RegTensor<uint32_t> manFP32_1; // -2轴奇数部分FP32尾数

        // scale输出寄存器 (循环后复用于-2轴scale输出)
        Reg::RegTensor<uint16_t> scale1B16_0;       // E8M0 uint16偶数, 循环后复用为mxScale2ZeroB16
        Reg::RegTensor<uint16_t> scale1B16_1;       // E8M0 uint16奇数, 循环后复用为mxScale2OneB16
        Reg::RegTensor<uint16_t> scale1BF16;        // BF16指数格式, 循环后复用为scale2BF16
        Reg::RegTensor<uint8_t> mxScale1B8;         // uint8 scale, 循环后复用为mxScale2ZeroB8
        Reg::RegTensor<uint16_t> reversedShareExp1; // 1/scale BF16, 循环后复用为reversedShareExp2Zero

        // -2轴独立寄存器 (需与复用寄存器同时存活，无法复用)
        Reg::RegTensor<uint8_t> mxScale2OneB8; // 与mxScale1B8同时存活于最终DataCopy

        // ========== 常量寄存器 ==========
        Reg::RegTensor<uint16_t> absMask;
        Reg::Duplicate(absMask, ABS_MASK_FOR_16BIT);
        Reg::RegTensor<float> invDstTypeMaxReg;
        Reg::Duplicate(invDstTypeMaxReg, invDstTypeMax_); // 1/dstTypeMax, FP32表示
        Reg::RegTensor<uint32_t> manMaskFP32;
        Reg::Duplicate(manMaskFP32, MAN_MASK_FLOAT); // FP32尾数掩码
        Reg::RegTensor<uint32_t> scaleBiasFP32;
        Reg::Duplicate(scaleBiasFP32, FP32_EXP_BIAS_CUBLAS); // BF16偏移在uint32

        Reg::RegTensor<uint16_t> nanE8M0;
        Reg::Duplicate(nanE8M0, NAN_FOR_FP8_E8M0);
        Reg::RegTensor<uint16_t> biasE8M0;
        Reg::Duplicate(biasE8M0, BF16_EXP_BIAS);
        Reg::RegTensor<uint16_t> zero;
        Reg::Duplicate(zero, 0);
        Reg::RegTensor<uint16_t> nanBF16;
        Reg::Duplicate(nanBF16, NAN_CUSTOMIZATION);
        Reg::RegTensor<uint16_t> specialExp;
        Reg::Duplicate(specialExp, SPECIAL_EXP_THRESHOLD);
        Reg::RegTensor<uint16_t> maxEleBF16;
        Reg::Duplicate(maxEleBF16, EXP_MASK_BF16);

        Reg::Duplicate(absMax1Dim2, 0);
        Reg::Duplicate(absMax2Dim2, 0);
        Reg::Duplicate(zeroB16, 0);

        // ========== Mask定义 ==========
        Reg::MaskReg maskAll = Reg::CreateMask<xDtype, Reg::MaskPattern::ALL>();
        Reg::MaskReg maskB8 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg maskReduceB8 = Reg::CreateMask<uint8_t, Reg::MaskPattern::VL8>();
        Reg::MaskReg maskReduceB16 = Reg::CreateMask<uint8_t, Reg::MaskPattern::VL16>();
        Reg::MaskReg maskFP32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();

        Reg::MaskReg p0; // 条件舍入: normal场景掩码
        Reg::MaskReg infMask;
        Reg::MaskReg invalidDataMask;

        // ========================================================================
        // 循环blockCount次，每次处理一行，计算-1轴scale并累积-2轴max
        // 与CuBALS一致: 取绝对值 → Max → ReduceMax → FP32域scale计算
        // ========================================================================
        for (uint16_t i = 0; i < blockCount; i++) {
            // 1. 交织搬运输入数据: 将256个xDtype按偶奇拆分为x0(偶), x1(奇)
            Reg::LoadAlign<xDtype, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B16>(
                x0, x1, xAddr, vlForHalfNumber_ * DIGIT_TWO);

            // 2. 取绝对值: 清除符号位，保留指数和尾数
            Reg::And(absMax0, (Reg::RegTensor<uint16_t>&)x0, absMask, maskAll);
            Reg::And(absMax1, (Reg::RegTensor<uint16_t>&)x1, absMask, maskAll);

            // 3. -1轴: 先取偶奇max，再ReduceMaxWithDataBlock得到每32个元素的绝对值max
            Reg::Max(absMaxDim1, absMax0, absMax1, maskAll);
            Reg::ReduceDataBlock<Reg::ReduceType::MAX>(absMaxDim1, absMaxDim1, maskAll);

            // 4. -2轴: 逐行累积偶数列和奇数列的绝对值max
            Reg::Max(absMax1Dim2, absMax1Dim2, absMax0, maskAll);
            Reg::Max(absMax2Dim2, absMax2Dim2, absMax1, maskAll);

            // ============================================================
            // 5. 计算-1轴Custom Scale (FP32精度)
            //    ReduceMaxWithDataBlock后，8个max紧凑存储在寄存器前8个位置
            //    与0交织后，Cast Zero一次处理全部8个值转为FP32
            //    乘以invDstTypeMax_ (1/dstTypeMax)
            //    仅normal场景条件舍入 (无subnormal舍入)
            //    (与原始DynamicMxQuant一致)
            // ============================================================

            // 与0交织: [v0,0,v1,0,...,v7,0,...] → Cast Zero可一次取出全部8个有效值
            Reg::Interleave(absMaxDim1, zeroB16, absMaxDim1, zeroB16);
            Reg::Cast<float, xDtype, castTraitXdtypetoFp32Zero>((Reg::RegTensor<float>&)maxFP32_0,
                                                                (Reg::RegTensor<xDtype>&)absMaxDim1, maskAll);
            // 乘以 1/dstTypeMax
            Reg::Mul((Reg::RegTensor<float>&)maxFP32_0, (Reg::RegTensor<float>&)maxFP32_0, invDstTypeMaxReg, maskFP32);
            // 提取FP32指数: 右移23位
            Reg::ShiftRights(expFP32_0, maxFP32_0, SHR_NUM_FOR_FP32, maskFP32);
            // 提取FP32尾数: 与尾数掩码
            Reg::And(manFP32_0, maxFP32_0, manMaskFP32, maskFP32);
            // 条件舍入: 仅normal场景 (exp>0 && exp<254 && man>0) → exp+1
            // 注意: 与CuBALS不同，DynamicDtypeRange Custom不处理subnormal场景
            Reg::Compares<uint32_t, CMPMODE::GT>(p0, expFP32_0, NUMBER_ZERO_U32, maskFP32);
            Reg::Compares<uint32_t, CMPMODE::LT>(p0, expFP32_0, NUMBER_TWO_FIVE_FOUR, p0);
            Reg::Compares<uint32_t, CMPMODE::GT>(p0, manFP32_0, NUMBER_ZERO_U32, p0);
            // 执行条件加1
            Reg::Adds(maxFP32_0, expFP32_0, 1, maskFP32);
            Reg::Select(manFP32_0, maxFP32_0, expFP32_0, p0);
            // Pack到uint16 (INF/NAN→0xFF, zero→0 自然通过条件舍入保持, 在BF16域1/scale中统一处理)
            Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>(scale1B16_0, manFP32_0);

            // 左移7位，将E8M0值定位到BF16指数域 (用于计算1/scale)
            Reg::ShiftLefts(scale1BF16, scale1B16_0, SHR_NUM_FOR_BF16, maskAll);

            // --- 输出-1轴scale (uint8) ---
            Reg::Pack<uint8_t, uint16_t, Reg::HighLowPart::LOWEST>(mxScale1B8, scale1B16_0);
            Reg::StoreAlign<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE>(mxScale1Addr, mxScale1B8, oneBlockCountB8_,
                                                                         maskReduceB8);

            // --- 计算并输出-1轴 1/scale (与原始DynamicMxQuant一致: inf→nan, special→specialExp, 无零值检查) ---
            Reg::Compare<uint16_t, CMPMODE::NE>(infMask, scale1BF16, maxEleBF16, maskAll);
            Reg::Compare<uint16_t, CMPMODE::EQ>(invalidDataMask, scale1BF16, biasE8M0, maskAll);
            Reg::Sub(reversedShareExp1, biasE8M0, scale1BF16, maskAll);
            Reg::Select<uint16_t>(reversedShareExp1, reversedShareExp1, nanBF16, infMask);
            Reg::Select<uint16_t>(reversedShareExp1, specialExp, reversedShareExp1, invalidDataMask);
            Reg::StoreAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(mxScale1ReciprocalAddr, reversedShareExp1,
                                                                          oneBlockCountB16_, maskReduceB16);
            // 恢复zeroB16 (Interleave会修改dst1)
            Reg::Duplicate(zeroB16, 0);
        }

        // ========================================================================
        // 循环结束后，计算-2轴Custom Scale
        // absMax1Dim2: 128个偶数列的累积绝对值max
        // absMax2Dim2: 128个奇数列的累积绝对值max
        // 每个需要拆分为Zero/One两半分别做FP32计算，再合并
        // ========================================================================

        // ---------- 处理absMax1Dim2 (偶数列, -2轴scale的交织第一部分) ----------
        // Zero半 (偶数位) — 复用循环体偶数部分寄存器
        Reg::Cast<float, xDtype, castTraitXdtypetoFp32Zero>((Reg::RegTensor<float>&)maxFP32_0,
                                                            (Reg::RegTensor<xDtype>&)absMax1Dim2, maskAll);
        Reg::Mul((Reg::RegTensor<float>&)maxFP32_0, (Reg::RegTensor<float>&)maxFP32_0, invDstTypeMaxReg, maskFP32);
        Reg::ShiftRights(expFP32_0, maxFP32_0, SHR_NUM_FOR_FP32, maskFP32);
        Reg::And(manFP32_0, maxFP32_0, manMaskFP32, maskFP32);
        // 条件舍入: 仅normal场景 (无subnormal)
        Reg::Compares<uint32_t, CMPMODE::GT>(p0, expFP32_0, NUMBER_ZERO_U32, maskFP32);
        Reg::Compares<uint32_t, CMPMODE::LT>(p0, expFP32_0, NUMBER_TWO_FIVE_FOUR, p0);
        Reg::Compares<uint32_t, CMPMODE::GT>(p0, manFP32_0, NUMBER_ZERO_U32, p0);
        // 链内复用: maxFP32_0→expPlusOne, expFP32_0同时作为exp和最终结果
        Reg::Adds(maxFP32_0, expFP32_0, 1, maskFP32);
        Reg::Select(expFP32_0, maxFP32_0, expFP32_0, p0);
        Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>(scale1B16_0, expFP32_0);

        // One半 (奇数位) — 复用循环体奇数部分寄存器
        Reg::Cast<float, xDtype, castTraitXdtypetoFp32One>((Reg::RegTensor<float>&)maxFP32_1,
                                                           (Reg::RegTensor<xDtype>&)absMax1Dim2, maskAll);
        Reg::Mul((Reg::RegTensor<float>&)maxFP32_1, (Reg::RegTensor<float>&)maxFP32_1, invDstTypeMaxReg, maskFP32);
        Reg::ShiftRights(expFP32_1, maxFP32_1, SHR_NUM_FOR_FP32, maskFP32);
        Reg::And(manFP32_1, maxFP32_1, manMaskFP32, maskFP32);
        Reg::Compares<uint32_t, CMPMODE::GT>(p0, expFP32_1, NUMBER_ZERO_U32, maskFP32);
        Reg::Compares<uint32_t, CMPMODE::LT>(p0, expFP32_1, NUMBER_TWO_FIVE_FOUR, p0);
        Reg::Compares<uint32_t, CMPMODE::GT>(p0, manFP32_1, NUMBER_ZERO_U32, p0);
        Reg::Adds(maxFP32_1, expFP32_1, 1, maskFP32);
        Reg::Select(expFP32_1, maxFP32_1, expFP32_1, p0);
        Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>(scale1B16_1, expFP32_1);

        // 合并Zero和One，恢复原始列顺序 — scale输出寄存器复用
        Reg::Interleave(scale1B16_0, scale1B16_1, scale1B16_0, scale1B16_1);
        // 左移7位得到BF16指数格式 — scale1BF16复用
        Reg::ShiftLefts(scale1BF16, scale1B16_0, SHR_NUM_FOR_BF16, maskAll);
        // 输出scale (uint8) — mxScale1B8复用为mxScale2ZeroB8
        Reg::Pack<uint8_t, uint16_t, Reg::HighLowPart::LOWEST>(mxScale1B8, scale1B16_0);

        // 计算1/scale — reversedShareExp1复用为reversedShareExp2Zero (与原始DynamicMxQuant一致: 无零值检查)
        Reg::Compare<uint16_t, CMPMODE::NE>(infMask, scale1BF16, maxEleBF16, maskAll);
        Reg::Compare<uint16_t, CMPMODE::EQ>(invalidDataMask, scale1BF16, biasE8M0, maskAll);
        Reg::Sub(reversedShareExp1, biasE8M0, scale1BF16, maskAll);
        Reg::Select<uint16_t>(reversedShareExp1, reversedShareExp1, nanBF16, infMask);
        Reg::Select<uint16_t>(reversedShareExp1, specialExp, reversedShareExp1, invalidDataMask);

        // ---------- 处理absMax2Dim2 (奇数列, -2轴scale的交织第二部分) ----------
        // Zero半 — 复用循环体偶数部分寄存器
        Reg::Cast<float, xDtype, castTraitXdtypetoFp32Zero>((Reg::RegTensor<float>&)maxFP32_0,
                                                            (Reg::RegTensor<xDtype>&)absMax2Dim2, maskAll);
        Reg::Mul((Reg::RegTensor<float>&)maxFP32_0, (Reg::RegTensor<float>&)maxFP32_0, invDstTypeMaxReg, maskFP32);
        Reg::ShiftRights(expFP32_0, maxFP32_0, SHR_NUM_FOR_FP32, maskFP32);
        Reg::And(manFP32_0, maxFP32_0, manMaskFP32, maskFP32);
        Reg::Compares<uint32_t, CMPMODE::GT>(p0, expFP32_0, NUMBER_ZERO_U32, maskFP32);
        Reg::Compares<uint32_t, CMPMODE::LT>(p0, expFP32_0, NUMBER_TWO_FIVE_FOUR, p0);
        Reg::Compares<uint32_t, CMPMODE::GT>(p0, manFP32_0, NUMBER_ZERO_U32, p0);
        Reg::Adds(maxFP32_0, expFP32_0, 1, maskFP32);
        Reg::Select(expFP32_0, maxFP32_0, expFP32_0, p0);
        Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>(scale1B16_0, expFP32_0);

        // One半 — 复用循环体奇数部分寄存器
        Reg::Cast<float, xDtype, castTraitXdtypetoFp32One>((Reg::RegTensor<float>&)maxFP32_1,
                                                           (Reg::RegTensor<xDtype>&)absMax2Dim2, maskAll);
        Reg::Mul((Reg::RegTensor<float>&)maxFP32_1, (Reg::RegTensor<float>&)maxFP32_1, invDstTypeMaxReg, maskFP32);
        Reg::ShiftRights(expFP32_1, maxFP32_1, SHR_NUM_FOR_FP32, maskFP32);
        Reg::And(manFP32_1, maxFP32_1, manMaskFP32, maskFP32);
        Reg::Compares<uint32_t, CMPMODE::GT>(p0, expFP32_1, NUMBER_ZERO_U32, maskFP32);
        Reg::Compares<uint32_t, CMPMODE::LT>(p0, expFP32_1, NUMBER_TWO_FIVE_FOUR, p0);
        Reg::Compares<uint32_t, CMPMODE::GT>(p0, manFP32_1, NUMBER_ZERO_U32, p0);
        Reg::Adds(maxFP32_1, expFP32_1, 1, maskFP32);
        Reg::Select(expFP32_1, maxFP32_1, expFP32_1, p0);
        Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>(scale1B16_1, expFP32_1);

        // 合并Zero和One — scale输出寄存器复用
        Reg::Interleave(scale1B16_0, scale1B16_1, scale1B16_0, scale1B16_1);
        // scale1BF16复用
        Reg::ShiftLefts(scale1BF16, scale1B16_0, SHR_NUM_FOR_BF16, maskAll);
        Reg::Pack<uint8_t, uint16_t, Reg::HighLowPart::LOWEST>(mxScale2OneB8, scale1B16_0);

        // 计算1/scale — absMax0复用为reversedShareExp2One (循环后死亡，与reversedShareExp1不冲突)
        // (与原始DynamicMxQuant一致: 无零值检查)
        Reg::Compare<uint16_t, CMPMODE::NE>(infMask, scale1BF16, maxEleBF16, maskAll);
        Reg::Compare<uint16_t, CMPMODE::EQ>(invalidDataMask, scale1BF16, biasE8M0, maskAll);
        Reg::Sub(absMax0, biasE8M0, scale1BF16, maskAll);
        Reg::Select<uint16_t>(absMax0, absMax0, nanBF16, infMask);
        Reg::Select<uint16_t>(absMax0, specialExp, absMax0, invalidDataMask);

        // 交织搬出-2轴的mxScale和1/scale
        Reg::StoreAlign<uint8_t, Reg::StoreDist::DIST_INTLV_B8>(mxScale2Addr, mxScale1B8, mxScale2OneB8, maskB8);
        Reg::StoreAlign<uint16_t, Reg::StoreDist::DIST_INTLV_B16>(mxScale2ReciprocalAddr, reversedShareExp1, absMax0,
                                                                  maskAll);
    }
}

template <typename xDtype, typename y1Dtype, typename y2Dtype, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__aicore__ inline void DynamicMxQuantWithDualAxisBase<xDtype, y1Dtype, y2Dtype, roundMode, scaleAlg>::ComputeYVf(
    uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr, __ubuf__ uint16_t* mxScale1ReciprocalAddr,
    __ubuf__ uint16_t* mxScale2ReciprocalAddr, __ubuf__ uint8_t* y1Addr, __ubuf__ uint8_t* y2Addr)
{
    if constexpr (IsSameType<y1Dtype, fp4x2_e2m1_t>::value || IsSameType<y1Dtype, fp4x2_e1m2_t>::value) {
        if constexpr (IsSameType<xDtype, half>::value) {
            ComputeYFP16ToFP4(dataLen, blockCount, xAddr, mxScale1ReciprocalAddr, y1Addr, mxScale2ReciprocalAddr,
                              y2Addr);
        } else {
            ComputeYBF16ToFP4(dataLen, blockCount, xAddr, mxScale1ReciprocalAddr, y1Addr, mxScale2ReciprocalAddr,
                              y2Addr);
        }
    } else {
        ComputeY1ToFP8(dataLen, blockCount, xAddr, mxScale1ReciprocalAddr, y1Addr);
        ComputeY2ToFP8(dataLen, blockCount, xAddr, mxScale2ReciprocalAddr, y2Addr);
        ComputeY2ToFP8(dataLen, blockCount, xAddr + vlForHalfNumber_, mxScale2ReciprocalAddr + vlForHalfNumber_,
                       y2Addr + vlForHalfNumber_);
    }
}

template <typename xDtype, typename y1Dtype, typename y2Dtype, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__aicore__ inline void DynamicMxQuantWithDualAxisBase<xDtype, y1Dtype, y2Dtype, roundMode, scaleAlg>::ComputeYBF16ToFP4(
    uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr, __ubuf__ uint16_t* mxScale1ReciprocalAddr,
    __ubuf__ uint8_t* y1Addr, __ubuf__ uint16_t* mxScale2ReciprocalAddr, __ubuf__ uint8_t* y2Addr)
{
    __VEC_SCOPE__
    {
        Reg::MaskReg dataMaskB8 = Reg::CreateMask<uint8_t>();
        Reg::MaskReg dataMaskB16 = Reg::CreateMask<half>();
        Reg::RegTensor<uint16_t> scaleForMulFP16;
        Reg::RegTensor<xDtype> x0;
        Reg::RegTensor<xDtype> x1;

        Reg::RegTensor<uint16_t> reversedShareExp0;
        Reg::RegTensor<uint16_t> reversedShareExp1;
        Reg::RegTensor<bfloat16_t> dim0x0;
        Reg::RegTensor<bfloat16_t> dim0x1;
        Reg::RegTensor<bfloat16_t> dim1x0;
        Reg::RegTensor<bfloat16_t> dim1x1;

        Reg::RegTensor<y1Dtype> dim0x0FP4;
        Reg::RegTensor<y1Dtype> dim0x1FP4;
        Reg::RegTensor<y1Dtype> dim1x0FP4;
        Reg::RegTensor<y1Dtype> dim1x1FP4;

        Reg::LoadAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B16>(
            reversedShareExp0, reversedShareExp1, mxScale2ReciprocalAddr, vlForHalfNumber_ * DIGIT_TWO);

        for (uint16_t i = 0; i < blockCount; i++) {
            Reg::LoadAlign<xDtype, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B16>(
                x0, x1, xAddr, vlForHalfNumber_ * DIGIT_TWO);
            Reg::LoadAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_E2B_B16>(
                scaleForMulFP16, mxScale1ReciprocalAddr, oneBlockCountB16_);

            Reg::Mul(dim0x0, x0, (Reg::RegTensor<xDtype>&)reversedShareExp0, dataMaskB16);
            Reg::Mul(dim0x1, x1, (Reg::RegTensor<xDtype>&)reversedShareExp1, dataMaskB16);
            Reg::Mul(dim1x0, x0, (Reg::RegTensor<xDtype>&)scaleForMulFP16, dataMaskB16);
            Reg::Mul(dim1x1, x1, (Reg::RegTensor<xDtype>&)scaleForMulFP16, dataMaskB16);
            Reg::Interleave(dim0x0, dim0x1, dim0x0, dim0x1);
            Reg::Interleave(dim1x0, dim1x1, dim1x0, dim1x1);
            Reg::Cast<y1Dtype, xDtype, castTraitBF16toFp4>(dim0x0FP4, dim0x0, dataMaskB16);
            Reg::Cast<y1Dtype, xDtype, castTraitBF16toFp4>(dim0x1FP4, dim0x1, dataMaskB16);
            Reg::Cast<y1Dtype, xDtype, castTraitBF16toFp4>(dim1x0FP4, dim1x0, dataMaskB16);
            Reg::Cast<y1Dtype, xDtype, castTraitBF16toFp4>(dim1x1FP4, dim1x1, dataMaskB16);

            // copy to ub
            Reg::StoreAlign<uint8_t, Reg::StoreDist::DIST_PACK4_B32>(y2Addr + (i * ubRowLen_ / DIGIT_TWO),
                                                                     (Reg::RegTensor<uint8_t>&)dim0x0FP4, dataMaskB8);
            Reg::StoreAlign<uint8_t, Reg::StoreDist::DIST_PACK4_B32>(
                y2Addr + OUT_ELE_NUM_ONE_BLK + (i * ubRowLen_ / DIGIT_TWO), (Reg::RegTensor<uint8_t>&)dim0x1FP4,
                dataMaskB8);
            Reg::StoreAlign<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_PACK4_B32>(
                y1Addr, (Reg::RegTensor<uint8_t>&)dim1x0FP4, OUT_ELE_NUM_ONE_BLK, dataMaskB8);
            Reg::StoreAlign<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_PACK4_B32>(
                y1Addr, (Reg::RegTensor<uint8_t>&)dim1x1FP4, OUT_ELE_NUM_ONE_BLK, dataMaskB8);
        }
    }
    return;
}

template <typename xDtype, typename y1Dtype, typename y2Dtype, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__aicore__ inline void DynamicMxQuantWithDualAxisBase<xDtype, y1Dtype, y2Dtype, roundMode, scaleAlg>::ComputeYFP16ToFP4(
    uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr, __ubuf__ uint16_t* mxScale1ReciprocalAddr,
    __ubuf__ uint8_t* y1Addr, __ubuf__ uint16_t* mxScale2ReciprocalAddr, __ubuf__ uint8_t* y2Addr)
{
    __VEC_SCOPE__
    {
        Reg::MaskReg dataMaskB8 = Reg::CreateMask<uint8_t>();
        Reg::MaskReg dataMaskB16 = Reg::CreateMask<half>();
        Reg::MaskReg dataMaskB32 = Reg::CreateMask<float>();

        Reg::RegTensor<xDtype> x0;
        Reg::RegTensor<xDtype> x1;

        Reg::RegTensor<float> x0ZeroFP32;
        Reg::RegTensor<float> x0OneFP32;
        Reg::RegTensor<float> x1ZeroFP32;
        Reg::RegTensor<float> x1OneFP32;
        Reg::RegTensor<float> scaleForMulZeroFP32;
        Reg::RegTensor<float> scaleForMulOneFP32;
        Reg::RegTensor<float> reversedShareExp0ZeroFP32;
        Reg::RegTensor<float> reversedShareExp0OneFP32;
        Reg::RegTensor<float> reversedShareExp1ZeroFP32;
        Reg::RegTensor<float> reversedShareExp1OneFP32;

        Reg::RegTensor<float> dim0x0ZeroFP32;
        Reg::RegTensor<float> dim0x0OneFP32;
        Reg::RegTensor<float> dim0x1ZeroFP32;
        Reg::RegTensor<float> dim0x1OneFP32;
        Reg::RegTensor<float> dim1x0ZeroFP32;
        Reg::RegTensor<float> dim1x0OneFP32;
        Reg::RegTensor<float> dim1x1ZeroFP32;
        Reg::RegTensor<float> dim1x1OneFP32;

        Reg::RegTensor<bfloat16_t> dim0x0ZeroBF16;
        Reg::RegTensor<bfloat16_t> dim0x0OneBF16;
        Reg::RegTensor<bfloat16_t> dim0x1ZeroBF16;
        Reg::RegTensor<bfloat16_t> dim0x1OneBF16;
        Reg::RegTensor<bfloat16_t> dim1x0ZeroBF16;
        Reg::RegTensor<bfloat16_t> dim1x0OneBF16;
        Reg::RegTensor<bfloat16_t> dim1x1ZeroBF16;
        Reg::RegTensor<bfloat16_t> dim1x1OneBF16;
        //
        Reg::RegTensor<uint16_t> scaleForMulFP16;
        Reg::RegTensor<uint16_t> reversedShareExp0;
        Reg::RegTensor<uint16_t> reversedShareExp1;

        Reg::RegTensor<y1Dtype> dim0x0FP4;
        Reg::RegTensor<y1Dtype> dim0x1FP4;
        Reg::RegTensor<y1Dtype> dim1x0FP4;
        Reg::RegTensor<y1Dtype> dim1x1FP4;
        //
        Reg::LoadAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B16>(
            reversedShareExp0, reversedShareExp1, mxScale2ReciprocalAddr, vlForHalfNumber_ * DIGIT_TWO);
        Reg::Cast<float, bfloat16_t, castTraitXdtypetoFp32Zero>(
            reversedShareExp0ZeroFP32, (Reg::RegTensor<bfloat16_t>&)reversedShareExp0, dataMaskB16);
        Reg::Cast<float, bfloat16_t, castTraitXdtypetoFp32One>(
            reversedShareExp0OneFP32, (Reg::RegTensor<bfloat16_t>&)reversedShareExp0, dataMaskB16);
        Reg::Cast<float, bfloat16_t, castTraitXdtypetoFp32Zero>(
            reversedShareExp1ZeroFP32, (Reg::RegTensor<bfloat16_t>&)reversedShareExp1, dataMaskB16);
        Reg::Cast<float, bfloat16_t, castTraitXdtypetoFp32One>(
            reversedShareExp1OneFP32, (Reg::RegTensor<bfloat16_t>&)reversedShareExp1, dataMaskB16);

        for (uint16_t i = 0; i < blockCount; i++) {
            Reg::LoadAlign<xDtype, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B16>(
                x0, x1, xAddr, vlForHalfNumber_ * DIGIT_TWO);
            Reg::LoadAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_E2B_B16>(
                scaleForMulFP16, mxScale1ReciprocalAddr, oneBlockCountB16_);
            Reg::Cast<float, bfloat16_t, castTraitXdtypetoFp32Zero>(
                scaleForMulZeroFP32, (Reg::RegTensor<bfloat16_t>&)scaleForMulFP16, dataMaskB16);

            Reg::Cast<float, xDtype, castTraitXdtypetoFp32Zero>(x0ZeroFP32, x0, dataMaskB16);
            Reg::Cast<float, xDtype, castTraitXdtypetoFp32One>(x0OneFP32, x0, dataMaskB16);
            Reg::Cast<float, xDtype, castTraitXdtypetoFp32Zero>(x1ZeroFP32, x1, dataMaskB16);
            Reg::Cast<float, xDtype, castTraitXdtypetoFp32One>(x1OneFP32, x1, dataMaskB16);

            Reg::Mul(dim0x0ZeroFP32, reversedShareExp0ZeroFP32, x0ZeroFP32, dataMaskB32);
            Reg::Mul(dim0x0OneFP32, reversedShareExp0OneFP32, x0OneFP32, dataMaskB32);
            Reg::Mul(dim1x0ZeroFP32, scaleForMulZeroFP32, x0ZeroFP32, dataMaskB32);
            Reg::Mul(dim1x0OneFP32, scaleForMulZeroFP32, x0OneFP32, dataMaskB32);

            ComputeFP4FromHalf(dim0x0ZeroFP32);
            ComputeFP4FromHalf(dim0x0OneFP32);
            ComputeFP4FromHalf(dim1x0ZeroFP32);
            ComputeFP4FromHalf(dim1x0OneFP32);

            Reg::Cast<bfloat16_t, float, castTraitFp32toBF16>(dim0x0ZeroBF16, dim0x0ZeroFP32, dataMaskB32);
            Reg::Cast<bfloat16_t, float, castTraitFp32toBF16>(dim0x0OneBF16, dim0x0OneFP32, dataMaskB32);
            Reg::Cast<bfloat16_t, float, castTraitFp32toBF16>(dim1x0ZeroBF16, dim1x0ZeroFP32, dataMaskB32);
            Reg::Cast<bfloat16_t, float, castTraitFp32toBF16>(dim1x0OneBF16, dim1x0OneFP32, dataMaskB32);

            Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>((Reg::RegTensor<uint16_t>&)dim0x0ZeroBF16,
                                                                    (Reg::RegTensor<uint32_t>&)dim0x0ZeroBF16);
            Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>((Reg::RegTensor<uint16_t>&)dim0x0OneBF16,
                                                                    (Reg::RegTensor<uint32_t>&)dim0x0OneBF16);
            Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>((Reg::RegTensor<uint16_t>&)dim1x0ZeroBF16,
                                                                    (Reg::RegTensor<uint32_t>&)dim1x0ZeroBF16);
            Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>((Reg::RegTensor<uint16_t>&)dim1x0OneBF16,
                                                                    (Reg::RegTensor<uint32_t>&)dim1x0OneBF16);
            Reg::Interleave(dim0x0ZeroBF16, dim0x0OneBF16, dim0x0ZeroBF16, dim0x0OneBF16);
            Reg::Interleave(dim1x0ZeroBF16, dim1x0OneBF16, dim1x0ZeroBF16, dim1x0OneBF16);

            Reg::Mul(dim0x1ZeroFP32, reversedShareExp1ZeroFP32, x1ZeroFP32, dataMaskB32);
            Reg::Mul(dim0x1OneFP32, reversedShareExp1OneFP32, x1OneFP32, dataMaskB32);
            Reg::Mul(dim1x1ZeroFP32, scaleForMulZeroFP32, x1ZeroFP32, dataMaskB32);
            Reg::Mul(dim1x1OneFP32, scaleForMulZeroFP32, x1OneFP32, dataMaskB32);

            ComputeFP4FromHalf(dim0x1ZeroFP32);
            ComputeFP4FromHalf(dim0x1OneFP32);
            ComputeFP4FromHalf(dim1x1ZeroFP32);
            ComputeFP4FromHalf(dim1x1OneFP32);

            Reg::Cast<bfloat16_t, float, castTraitFp32toBF16>(dim0x1ZeroBF16, dim0x1ZeroFP32, dataMaskB32);
            Reg::Cast<bfloat16_t, float, castTraitFp32toBF16>(dim0x1OneBF16, dim0x1OneFP32, dataMaskB32);
            Reg::Cast<bfloat16_t, float, castTraitFp32toBF16>(dim1x1ZeroBF16, dim1x1ZeroFP32, dataMaskB32);
            Reg::Cast<bfloat16_t, float, castTraitFp32toBF16>(dim1x1OneBF16, dim1x1OneFP32, dataMaskB32);

            Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>((Reg::RegTensor<uint16_t>&)dim0x1ZeroBF16,
                                                                    (Reg::RegTensor<uint32_t>&)dim0x1ZeroBF16);
            Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>((Reg::RegTensor<uint16_t>&)dim0x1OneBF16,
                                                                    (Reg::RegTensor<uint32_t>&)dim0x1OneBF16);
            Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>((Reg::RegTensor<uint16_t>&)dim1x1ZeroBF16,
                                                                    (Reg::RegTensor<uint32_t>&)dim1x1ZeroBF16);
            Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>((Reg::RegTensor<uint16_t>&)dim1x1OneBF16,
                                                                    (Reg::RegTensor<uint32_t>&)dim1x1OneBF16);
            Reg::Interleave(dim0x1ZeroBF16, dim0x1OneBF16, dim0x1ZeroBF16, dim0x1OneBF16);
            Reg::Interleave(dim1x1ZeroBF16, dim1x1OneBF16, dim1x1ZeroBF16, dim1x1OneBF16);

            // interleave x0 and x1
            Reg::Interleave(dim0x0ZeroBF16, dim0x1ZeroBF16, dim0x0ZeroBF16, dim0x1ZeroBF16);
            Reg::Interleave(dim1x0ZeroBF16, dim1x1ZeroBF16, dim1x0ZeroBF16, dim1x1ZeroBF16);
            Reg::Cast<y1Dtype, bfloat16_t, castTraitBF16toFp4>(dim0x0FP4, dim0x0ZeroBF16, dataMaskB16);
            Reg::Cast<y1Dtype, bfloat16_t, castTraitBF16toFp4>(dim0x1FP4, dim0x1ZeroBF16, dataMaskB16);
            Reg::Cast<y1Dtype, bfloat16_t, castTraitBF16toFp4>(dim1x0FP4, dim1x0ZeroBF16, dataMaskB16);
            Reg::Cast<y1Dtype, bfloat16_t, castTraitBF16toFp4>(dim1x1FP4, dim1x1ZeroBF16, dataMaskB16);

            // copy to ub
            Reg::StoreAlign<uint8_t, Reg::StoreDist::DIST_PACK4_B32>(y2Addr + (i * ubRowLen_ / DIGIT_TWO),
                                                                     (Reg::RegTensor<uint8_t>&)dim0x0FP4, dataMaskB8);
            Reg::StoreAlign<uint8_t, Reg::StoreDist::DIST_PACK4_B32>(
                y2Addr + OUT_ELE_NUM_ONE_BLK + (i * ubRowLen_ / DIGIT_TWO), (Reg::RegTensor<uint8_t>&)dim0x1FP4,
                dataMaskB8);
            Reg::StoreAlign<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_PACK4_B32>(
                y1Addr, (Reg::RegTensor<uint8_t>&)dim1x0FP4, OUT_ELE_NUM_ONE_BLK, dataMaskB8);
            Reg::StoreAlign<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_PACK4_B32>(
                y1Addr, (Reg::RegTensor<uint8_t>&)dim1x1FP4, OUT_ELE_NUM_ONE_BLK, dataMaskB8);
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
#ifndef ASCENDC_CPU_DEBUG
    __VEC_SCOPE__
    {
        Reg::MaskReg maskAll = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg maskFP8 = Reg::CreateMask<y1Dtype>();
        Reg::RegTensor<uint16_t> scaleForMulFP16;
        Reg::RegTensor<float> scaleForMulFP32;
        Reg::RegTensor<xDtype> x0;
        Reg::RegTensor<xDtype> x1;
        Reg::RegTensor<float> x0ZeroFP32;
        Reg::RegTensor<float> x0OneFP32;
        Reg::RegTensor<float> x1ZeroFP32;
        Reg::RegTensor<float> x1OneFP32;
        // 4路FP8寄存器，分别对应uint32中的4个字节位置
        Reg::RegTensor<y1Dtype> fp8Layout0; // x0 Zero → byte 0
        Reg::RegTensor<y1Dtype> fp8Layout1; // x1 Zero → byte 1
        Reg::RegTensor<y1Dtype> fp8Layout2; // x0 One  → byte 2
        Reg::RegTensor<y1Dtype> fp8Layout3; // x1 One  → byte 3

        for (uint16_t i = 0; i < blockCount; i++) {
            // 交织搬运: 256个xDtype按偶奇拆分为x0(偶128), x1(奇128)
            Reg::LoadAlign<xDtype, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B16>(
                x0, x1, xAddr, vlForHalfNumber_ * DIGIT_TWO);
            // 搬运1/scale: 8个scale广播到128个位置
            Reg::LoadAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_E2B_B16>(
                scaleForMulFP16, mxScale1ReciprocalAddr, oneBlockCountB16_);
            if constexpr (IsSameType<xDtype, half>::value) {
                // half输入: 先Cast到FP32再乘scale (避免half精度损失)
                Reg::Cast<float, xDtype, castTraitXdtypetoFp32Zero>(x0ZeroFP32, x0, maskAll);
                Reg::Cast<float, xDtype, castTraitXdtypetoFp32One>(x0OneFP32, x0, maskAll);
                Reg::Cast<float, bfloat16_t, castTraitXdtypetoFp32Zero>(
                    scaleForMulFP32, (Reg::RegTensor<bfloat16_t>&)scaleForMulFP16, maskAll);
                Reg::Mul(x0ZeroFP32, x0ZeroFP32, scaleForMulFP32, maskAll);
                Reg::Mul(x0OneFP32, x0OneFP32, scaleForMulFP32, maskAll);
                Reg::Cast<float, xDtype, castTraitXdtypetoFp32Zero>(x1ZeroFP32, x1, maskAll);
                Reg::Cast<float, xDtype, castTraitXdtypetoFp32One>(x1OneFP32, x1, maskAll);
                Reg::Mul(x1ZeroFP32, x1ZeroFP32, scaleForMulFP32, maskAll);
                Reg::Mul(x1OneFP32, x1OneFP32, scaleForMulFP32, maskAll);
            } else {
                // bf16输入: 直接在bf16域乘scale，再Cast到FP32
                Reg::Mul(x0, x0, (Reg::RegTensor<xDtype>&)scaleForMulFP16, maskAll);
                Reg::Mul(x1, x1, (Reg::RegTensor<xDtype>&)scaleForMulFP16, maskAll);
                Reg::Cast<float, xDtype, castTraitXdtypetoFp32Zero>(x0ZeroFP32, x0, maskAll);
                Reg::Cast<float, xDtype, castTraitXdtypetoFp32One>(x0OneFP32, x0, maskAll);
                Reg::Cast<float, xDtype, castTraitXdtypetoFp32Zero>(x1ZeroFP32, x1, maskAll);
                Reg::Cast<float, xDtype, castTraitXdtypetoFp32One>(x1OneFP32, x1, maskAll);
            }
            // 4路RegLayout Cast: 将4组64个FP32值分别Cast到FP8的不同字节位置
            // Layout0(byte0): x0 Zero (偶数列偶数位)
            // Layout2(byte2): x0 One  (偶数列奇数位)
            // Layout1(byte1): x1 Zero (奇数列偶数位)
            // Layout3(byte3): x1 One  (奇数列奇数位)
            Reg::Cast<y1Dtype, float, castTraitFp32toFP8Layout0>(fp8Layout0, x0ZeroFP32, maskAll);
            Reg::Cast<y1Dtype, float, castTraitFp32toFP8Layout2>(fp8Layout2, x0OneFP32, maskAll);
            Reg::Cast<y1Dtype, float, castTraitFp32toFP8Layout1>(fp8Layout1, x1ZeroFP32, maskAll);
            Reg::Cast<y1Dtype, float, castTraitFp32toFP8Layout3>(fp8Layout3, x1OneFP32, maskAll);
            // Add合并: 4个字节位置的FP8值合并到一个寄存器
            Reg::Add((Reg::RegTensor<uint8_t>&)fp8Layout0, (Reg::RegTensor<uint8_t>&)fp8Layout0,
                     (Reg::RegTensor<uint8_t>&)fp8Layout2, maskFP8);
            Reg::Add((Reg::RegTensor<uint8_t>&)fp8Layout1, (Reg::RegTensor<uint8_t>&)fp8Layout1,
                     (Reg::RegTensor<uint8_t>&)fp8Layout3, maskFP8);
            Reg::Add((Reg::RegTensor<uint8_t>&)fp8Layout0, (Reg::RegTensor<uint8_t>&)fp8Layout0,
                     (Reg::RegTensor<uint8_t>&)fp8Layout1, maskFP8);
            // 一次性输出256个FP8值
            Reg::StoreAlign<uint8_t, Reg::StoreDist::DIST_NORM_B8>(y1Addr + i * vlForHalfNumber_ * DIGIT_TWO,
                                                                   (Reg::RegTensor<uint8_t>&)fp8Layout0, maskFP8);
        }
    }
#endif
    return;
}

// 优化后的ComputeY2ToFP8: 参考DynamicMxQuant ComputeData的RegLayout优化模式
// 使用ZERO/ONE两个RegLayout将FP32→FP8的结果分别放到uint32的byte0和byte1位置
// 通过Add合并后Pack+DataCopy输出，消除Interleave和多余的Pack操作
template <typename xDtype, typename y1Dtype, typename y2Dtype, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__aicore__ inline void DynamicMxQuantWithDualAxisBase<xDtype, y1Dtype, y2Dtype, roundMode, scaleAlg>::ComputeY2ToFP8(
    uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr, __ubuf__ uint16_t* mxScale2ReciprocalAddr,
    __ubuf__ uint8_t* y2Addr)
{
#ifndef ASCENDC_CPU_DEBUG
    __VEC_SCOPE__
    {
        Reg::RegTensor<xDtype> x;
        Reg::RegTensor<float> x0FP32;
        Reg::RegTensor<float> x1FP32;
        Reg::RegTensor<uint16_t> reversedShareExp;
        Reg::RegTensor<float> reversedShareExp0FP32;
        Reg::RegTensor<float> reversedShareExp1FP32;
        // 2路FP8寄存器，分别对应uint32中的byte0和byte1位置
        Reg::RegTensor<y1Dtype> fp8Layout0; // x0 (偶数位) → byte 0
        Reg::RegTensor<y1Dtype> fp8Layout1; // x1 (奇数位) → byte 1

        Reg::MaskReg pregAll8 = Reg::CreateMask<uint8_t, Reg::MaskPattern::H>();
        Reg::MaskReg pregAll16 = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg pregAll32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg maskFP8 = Reg::CreateMask<y1Dtype>();

        Reg::LoadAlign<uint16_t, Reg::LoadDist::DIST_NORM>(reversedShareExp, mxScale2ReciprocalAddr);
        Reg::Cast<float, bfloat16_t, castTraitXdtypetoFp32Zero>(
            reversedShareExp0FP32, (Reg::RegTensor<bfloat16_t>&)reversedShareExp, pregAll16);
        Reg::Cast<float, bfloat16_t, castTraitXdtypetoFp32One>(
            reversedShareExp1FP32, (Reg::RegTensor<bfloat16_t>&)reversedShareExp, pregAll16);
        for (uint16_t j = 0; j < blockCount; j++) {
            Reg::LoadAlign<xDtype, Reg::LoadDist::DIST_NORM>(x, xAddr + j * ubRowLen_);
            Reg::Cast<float, xDtype, castTraitXdtypetoFp32Zero>(x0FP32, x, pregAll16);
            Reg::Cast<float, xDtype, castTraitXdtypetoFp32One>(x1FP32, x, pregAll16);

            Reg::Mul(x0FP32, x0FP32, reversedShareExp0FP32, pregAll32);
            Reg::Mul(x1FP32, x1FP32, reversedShareExp1FP32, pregAll32);

            // 2路RegLayout Cast: 将2组64个FP32值分别Cast到FP8的不同字节位置
            // Layout0(byte0): x0 (偶数位元素)
            // Layout1(byte1): x1 (奇数位元素)
            Reg::Cast<y1Dtype, float, castTraitFp32toFP8Layout0>(fp8Layout0, x0FP32, pregAll32);
            Reg::Cast<y1Dtype, float, castTraitFp32toFP8Layout1>(fp8Layout1, x1FP32, pregAll32);
            // Add合并: byte0和byte1位置的FP8值合并到一个寄存器
            Reg::Add((Reg::RegTensor<uint8_t>&)fp8Layout0, (Reg::RegTensor<uint8_t>&)fp8Layout0,
                     (Reg::RegTensor<uint8_t>&)fp8Layout1, maskFP8);
            // Pack: 提取每个uint32的低16位(包含2个FP8值)，紧凑为128个FP8
            Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>((Reg::RegTensor<uint16_t>&)fp8Layout0,
                                                                    (Reg::RegTensor<uint32_t>&)fp8Layout0);

            Reg::StoreAlign(y2Addr + (j * ubRowLen_), (Reg::RegTensor<uint8_t>&)fp8Layout0, pregAll8);
        }
    }
#endif
}

template <typename xDtype, typename y1Dtype, typename y2Dtype, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__aicore__ inline void DynamicMxQuantWithDualAxisBase<xDtype, y1Dtype, y2Dtype, roundMode,
                                                      scaleAlg>::ComputeFP4FromHalf(Reg::RegTensor<float>& Reg)
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
        Reg::Muls(Reg, Reg, FOUR, pregAll32);
        Reg::Compares<float, CMPMODE::LT>(specialMask, Reg, 0, pregAll32);
        Reg::Truncate<float, roundMode>(Reg, Reg, pregAll32);
        Reg::Muls(Reg, Reg, ONE_FOURTH, pregAll32);
    } else {
        // fp4x2_e2m1
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
    Reg::Compares<float, CMPMODE::EQ>(zeroMask, Reg, 0, pregAll32);
    Reg::And(zeroMask, specialMask, zeroMask, pregAll32);
    Reg::Or(zeroMask, negInfMask, zeroMask, pregAll32);
    Reg::Select<int32_t>((Reg::RegTensor<int32_t>&)Reg, negZero, (Reg::RegTensor<int32_t>&)Reg, zeroMask);
}

template <typename xDtype, typename y1Dtype, typename y2Dtype, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__aicore__ inline void DynamicMxQuantWithDualAxisBase<xDtype, y1Dtype, y2Dtype, roundMode, scaleAlg>::CopyIn(
    int64_t offset, int64_t blockCount, int64_t dataLen, int64_t dimNeg1IsOdd)
{
    // 第一行第一块到第二行的第一块，间隔长度
    int64_t rightPadding = ops::CeilAlign(static_cast<int64_t>(dataLen * sizeof(xDtype)), UBBlockSize_) /
                               sizeof(xDtype) -
                           dataLen;
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
    uint32_t scaleSrcStride = DIGIT_TWO * ops::CeilDiv(dataLen, UBBlockSize_) -
                              ops::CeilDiv(DIGIT_TWO * dataLen, UBBlockSize_);

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
    DataCopyExtParams yCopyOutParams = {static_cast<uint16_t>(outBurst), static_cast<uint32_t>(outBlockLen),
                                        static_cast<uint32_t>(srcStride), static_cast<uint32_t>(dstStride),
                                        static_cast<uint32_t>(0)};

    uint32_t dataLenReduce = static_cast<uint32_t>(ops::CeilDiv(dataLen, blockSize_));
    uint32_t scale1OutLen = dataLenReduce % 2 == 1 ? dataLenReduce + 1 : dataLenReduce;

    DataCopyExtParams scale1CopyOutParams = {
        static_cast<uint16_t>(outBurst), static_cast<uint32_t>(scale1OutLen * sizeof(uint8_t)),
        static_cast<uint32_t>(0),
        static_cast<uint32_t>(ops::CeilAlign(tilingData_->dimNeg1, blockSize_ * DIGIT_TWO) / blockSize_ -
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
        int64_t scale2Offset = (blockOffset_ + i) / blockCountPerPage_ * dimNeg2ScaleNum_ * tilingData_->dimNeg1 +
                               (blockOffset_ + i) % blockCountPerPage_ / tilingData_->dimNeg1BlockNum *
                                   tilingData_->splitBlockH / tilingData_->blockSize * tilingData_->dimNeg1 +
                               (blockOffset_ + i) % blockCountPerPage_ % tilingData_->dimNeg1BlockNum * ubRowLen_ *
                                   DIGIT_TWO;
        // -1轴偏移
        int64_t scale1Offset = (blockOffset_ + i) / blockCountPerPage_ * dimNeg1ScaleNum_ * tilingData_->dimNeg2 +
                               (blockOffset_ + i) % blockCountPerPage_ / tilingData_->dimNeg1BlockNum *
                                   tilingData_->splitBlockH * dimNeg1ScaleNum_ +
                               (blockOffset_ + i) % blockCountPerPage_ % tilingData_->dimNeg1BlockNum * ubRowLen_ /
                                   tilingData_->blockSize;

        // 尾轴reduce后是否是奇数
        int64_t dimNeg1IsOdd = ubRowLenTail_ < ubRowLen_;
        ProcessOneLoop(calcCol, calcRow, xUbOffset, scale1Offset, scale2Offset, dimNeg1IsOdd);
    }
}

} // namespace DynamicMxQuantWithDualAxis
#endif // OPS_NN_DEV_DYNAMIC_MX_QUANT_WITH_DUAL_AXIS_H
