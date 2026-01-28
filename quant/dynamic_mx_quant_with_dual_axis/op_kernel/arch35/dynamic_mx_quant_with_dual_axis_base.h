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
    static constexpr MicroAPI::CastTrait castTraitBF16toFp4 = {
        MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT, MicroAPI::MaskMergeMode::ZEROING, roundMode};
    static constexpr MicroAPI::CastTrait castTraitFp32toBF16 = {
        MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT, MicroAPI::MaskMergeMode::ZEROING, roundMode};
    static constexpr MicroAPI::CastTrait castTraitFp32toYdtype = {
        MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT, MicroAPI::MaskMergeMode::ZEROING, roundMode};

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
    int64_t oneBlockCountB16_ = UBBlockSize_ / sizeof(xDtype);
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

    auto xAddr = (__ubuf__ xDtype*)x.GetPhyAddr();
    auto y1Addr = (__ubuf__ uint8_t*)y1.GetPhyAddr();
    auto y2Addr = (__ubuf__ uint8_t*)y2.GetPhyAddr();
    auto mxScale1Addr = (__ubuf__ uint8_t*)mxScale1.GetPhyAddr();
    auto mxScale2Addr = (__ubuf__ uint8_t*)mxScale2.GetPhyAddr();
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
        ComputeScaleOcp(
            dataLen, blockSize_, xAddr + xOffset, mxScale1Addr + scale1UbOffset,
            mxScale1ReciprocalAddr + scale1ReciprocalOffset, mxScale2Addr + scale2UbOffset,
            mxScale2ReciprocalAddr + scale2ReciprocalOffset);

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
        ComputeScaleOcp(
            dataLen, static_cast<uint16_t>(calcBlockTail), xAddr + xOffset, mxScale1Addr + scale1UbOffset,
            mxScale1ReciprocalAddr + scale1ReciprocalOffset, mxScale2Addr + scale2UbOffset,
            mxScale2ReciprocalAddr + scale2ReciprocalOffset);
        ComputeYVf(
            dataLen, static_cast<uint16_t>(calcBlockTail), xAddr + xOffset,
            mxScale1ReciprocalAddr + scale1ReciprocalOffset, mxScale2ReciprocalAddr + scale2ReciprocalOffset,
            y1Addr + yOffset, y2Addr + yOffset);
    }

    event_t eventIDVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIDVToS);
    WaitFlag<HardEvent::V_S>(eventIDVToS);

    // -2轴的scale交织处理
    for (int64_t i = 1; i < ((calcBlockLoop + 1) / DIGIT_TWO * DIGIT_TWO); i = i + 2) {
        Interleave(
            mxScale2[(i - 1) * ubRowLen_], mxScale2[i * ubRowLen_], mxScale2[(i - 1) * ubRowLen_],
            mxScale2[i * ubRowLen_], ubRowLen_);
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
        MicroAPI::MaskReg infNanDataMask0;
        MicroAPI::MaskReg infNanDataMask1;
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
                // 提取指数位
                MicroAPI::And(x0ExpFP16, (MicroAPI::RegTensor<uint16_t>&)x0, expMaskFP16, maskAll);
                MicroAPI::And(x1ExpFP16, (MicroAPI::RegTensor<uint16_t>&)x1, expMaskFP16, maskAll);
                // 比较INF/NAN数据
                MicroAPI::Compare<uint16_t, CMPMODE::NE>(infNanDataMask0, x0ExpFP16, expMaskFP16, maskAll);
                MicroAPI::Compare<uint16_t, CMPMODE::NE>(infNanDataMask1, x1ExpFP16, expMaskFP16, maskAll);
                // 原始数据转成bf16
                MicroAPI::Cast<bfloat16_t, xDtype, castTraitHalf2BF16>(x0BF16, x0, maskAll);
                MicroAPI::Cast<bfloat16_t, xDtype, castTraitHalf2BF16>(x1BF16, x1, maskAll);
                // 提取指数位
                MicroAPI::And(x0ExpBF16, (MicroAPI::RegTensor<uint16_t>&)x0BF16, expMaskBF16, maskAll);
                MicroAPI::And(x1ExpBF16, (MicroAPI::RegTensor<uint16_t>&)x1BF16, expMaskBF16, maskAll);
                // 选择数据，INF/NAN数据时设成BF的INF/NAN
                MicroAPI::Select<uint16_t>(x0ExpBF16, x0ExpBF16, expMaskBF16, infNanDataMask0);
                MicroAPI::Select<uint16_t>(x1ExpBF16, x1ExpBF16, expMaskBF16, infNanDataMask1);
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
            MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                mxScale1Addr, mxScale1B8, UBBlockSize_ / sizeof(uint8_t), maskReduceB8);

            // 公式中的1/X
            // 只有在E1M2时，yMaxExp=0，expMaxDim1可能会等于biasE8M0
            MicroAPI::Compare<uint16_t, CMPMODE::EQ>(invalidDataMask, expMaxDim1, biasE8M0, maskAll);

            MicroAPI::Sub(reversedShareExp1, biasE8M0, expMaxDim1, maskAll);
            MicroAPI::Select<uint16_t>(reversedShareExp1, reversedShareExp1, nanBF16, infMask);
            MicroAPI::Select<uint16_t>(reversedShareExp1, reversedShareExp1, zero, zeroMask);
            MicroAPI::Select<uint16_t>(reversedShareExp1, specialExp, reversedShareExp1, invalidDataMask);
            MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                mxScale1ReciprocalAddr, reversedShareExp1, UBBlockSize_ / sizeof(uint16_t), maskReduceB16);
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
                scaleForMulFP16, mxScale1ReciprocalAddr, UBBlockSize_ / sizeof(uint16_t));

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

template <typename xDtype, typename y1Dtype, typename y2Dtype, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__aicore__ inline void DynamicMxQuantWithDualAxisBase<xDtype, y1Dtype, y2Dtype, roundMode, scaleAlg>::ComputeY1ToFP8(
    uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr, __ubuf__ uint16_t* mxScale1ReciprocalAddr,
    __ubuf__ uint8_t* y1Addr)
{
    __VEC_SCOPE__
    {
        MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<uint16_t, MicroAPI::MaskPattern::ALL>();
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
            MicroAPI::DataCopy<xDtype, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_DINTLV_B16>(
                x0, x1, xAddr, vlForHalfNumber_ * DIGIT_TWO);
            MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_E2B_B16>(
                scaleForMulFP16, mxScale1ReciprocalAddr, UBBlockSize_ / sizeof(uint16_t));
            if constexpr (IsSameType<xDtype, half>::value) {
                MicroAPI::Cast<float, xDtype, DynamicMxQuantWithDualAxisBase::castTraitXdtypetoFp32Zero>(
                    x0ZeroFP32, x0, maskAll);
                MicroAPI::Cast<float, xDtype, castTraitXdtypetoFp32One>(x0OneFP32, x0, maskAll);
                MicroAPI::Cast<float, bfloat16_t, castTraitXdtypetoFp32Zero>(
                    scaleForMulFP32, (MicroAPI::RegTensor<bfloat16_t>&)scaleForMulFP16, maskAll);
                MicroAPI::Mul(x0ZeroFP32, x0ZeroFP32, scaleForMulFP32, maskAll);
                MicroAPI::Mul(x0OneFP32, x0OneFP32, scaleForMulFP32, maskAll);
                MicroAPI::Interleave(x0ZeroFP32, x0OneFP32, x0ZeroFP32, x0OneFP32);
                MicroAPI::Cast<float, xDtype, castTraitXdtypetoFp32Zero>(x1ZeroFP32, x1, maskAll);
                MicroAPI::Cast<float, xDtype, castTraitXdtypetoFp32One>(x1OneFP32, x1, maskAll);
                MicroAPI::Mul(x1ZeroFP32, x1ZeroFP32, scaleForMulFP32, maskAll);
                MicroAPI::Mul(x1OneFP32, x1OneFP32, scaleForMulFP32, maskAll);
                MicroAPI::Interleave(x1ZeroFP32, x1OneFP32, x1ZeroFP32, x1OneFP32);
                MicroAPI::Interleave(x0ZeroFP32, x1ZeroFP32, x0ZeroFP32, x1ZeroFP32);
                MicroAPI::Interleave(x0OneFP32, x1OneFP32, x0OneFP32, x1OneFP32);
                MicroAPI::Cast<y1Dtype, float, castTraitFp32toYdtype>(x0ZeroFP8, x0ZeroFP32, maskAll);
                MicroAPI::Cast<y1Dtype, float, castTraitFp32toYdtype>(x0OneFP8, x1ZeroFP32, maskAll);
                MicroAPI::Cast<y1Dtype, float, castTraitFp32toYdtype>(x1ZeroFP8, x0OneFP32, maskAll);
                MicroAPI::Cast<y1Dtype, float, castTraitFp32toYdtype>(x1OneFP8, x1OneFP32, maskAll);
            } else {
                MicroAPI::Mul(x0, x0, (MicroAPI::RegTensor<xDtype>&)scaleForMulFP16, maskAll);
                MicroAPI::Mul(x1, x1, (MicroAPI::RegTensor<xDtype>&)scaleForMulFP16, maskAll);
                MicroAPI::Interleave(x0, x1, x0, x1);
                MicroAPI::Cast<float, xDtype, castTraitXdtypetoFp32Zero>(x0ZeroFP32, x0, maskAll);
                MicroAPI::Cast<float, xDtype, castTraitXdtypetoFp32One>(x0OneFP32, x0, maskAll);
                MicroAPI::Interleave(x0ZeroFP32, x0OneFP32, x0ZeroFP32, x0OneFP32);
                MicroAPI::Cast<y1Dtype, float, castTraitFp32toYdtype>(x0ZeroFP8, x0ZeroFP32, maskAll);
                MicroAPI::Cast<y1Dtype, float, castTraitFp32toYdtype>(x0OneFP8, x0OneFP32, maskAll);
                MicroAPI::Cast<float, xDtype, castTraitXdtypetoFp32Zero>(x1ZeroFP32, x1, maskAll);
                MicroAPI::Cast<float, xDtype, castTraitXdtypetoFp32One>(x1OneFP32, x1, maskAll);
                MicroAPI::Interleave(x1ZeroFP32, x1OneFP32, x1ZeroFP32, x1OneFP32);
                MicroAPI::Cast<y1Dtype, float, castTraitFp32toYdtype>(x1ZeroFP8, x1ZeroFP32, maskAll);
                MicroAPI::Cast<y1Dtype, float, castTraitFp32toYdtype>(x1OneFP8, x1OneFP32, maskAll);
            }
            MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::StoreDist::DIST_PACK4_B32>(
                y1Addr, (MicroAPI::RegTensor<uint8_t>&)x0ZeroFP8, OUT_ELE_NUM_ONE_BLK, maskAll);
            MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::StoreDist::DIST_PACK4_B32>(
                y1Addr, (MicroAPI::RegTensor<uint8_t>&)x0OneFP8, OUT_ELE_NUM_ONE_BLK, maskAll);
            MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::StoreDist::DIST_PACK4_B32>(
                y1Addr, (MicroAPI::RegTensor<uint8_t>&)x1ZeroFP8, OUT_ELE_NUM_ONE_BLK, maskAll);
            MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::StoreDist::DIST_PACK4_B32>(
                y1Addr, (MicroAPI::RegTensor<uint8_t>&)x1OneFP8, OUT_ELE_NUM_ONE_BLK, maskAll);
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
        MicroAPI::RegTensor<y1Dtype> yZeroFP8;
        MicroAPI::RegTensor<y1Dtype> yOneFP8;

        MicroAPI::MaskReg pregAll8 = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg pregAll16 = MicroAPI::CreateMask<uint16_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg pregAll32 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();

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

            MicroAPI::Cast<y1Dtype, float, castTraitFp32toYdtype>(
                yZeroFP8, (MicroAPI::RegTensor<float>&)x0FP32, pregAll32);
            MicroAPI::Pack<uint16_t, uint32_t, MicroAPI::HighLowPart::LOWEST>(
                (MicroAPI::RegTensor<uint16_t>&)yZeroFP8, (MicroAPI::RegTensor<uint32_t>&)yZeroFP8);

            MicroAPI::Cast<y1Dtype, float, castTraitFp32toYdtype>(
                yOneFP8, (MicroAPI::RegTensor<float>&)x1FP32, pregAll32);
            MicroAPI::Pack<uint16_t, uint32_t, MicroAPI::HighLowPart::LOWEST>(
                (MicroAPI::RegTensor<uint16_t>&)yOneFP8, (MicroAPI::RegTensor<uint32_t>&)yOneFP8);

            MicroAPI::Interleave(
                (MicroAPI::RegTensor<uint16_t>&)yZeroFP8, (MicroAPI::RegTensor<uint16_t>&)yOneFP8,
                (MicroAPI::RegTensor<uint16_t>&)yZeroFP8, (MicroAPI::RegTensor<uint16_t>&)yOneFP8);

            MicroAPI::Pack<uint8_t, uint16_t, MicroAPI::HighLowPart::LOWEST>(
                (MicroAPI::RegTensor<uint8_t>&)yZeroFP8, (MicroAPI::RegTensor<uint16_t>&)yZeroFP8);

            DataCopy(y2Addr + (j * ubRowLen_), (MicroAPI::RegTensor<uint8_t>&)yZeroFP8, pregAll8);
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
#if (__NPU_ARCH__ == 3101)
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
