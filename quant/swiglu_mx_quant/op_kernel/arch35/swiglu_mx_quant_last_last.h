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
 * \file swiglu_mx_quant_last_last.h
 * \brief Regbase implementation for Swiglu + MX quantization (activate_dim=-1, axis=-1)
 */

#ifndef SWIGLU_MX_QUANT_LAST_LAST_H
#define SWIGLU_MX_QUANT_LAST_LAST_H

#include "swiglu_mx_quant_common.h"
#include "kernel_operator.h"
#include "op_kernel/math_util.h"
#include "op_kernel/platform_util.h"
#include "kernel_tiling/kernel_tiling.h"

namespace SwigluMxQuant {
using namespace AscendC;
// Regbase class for SwiGLU + MX Quantization
// T: Input data type (half, bfloat16_t)
// ComputeT: Computation data type (float)
// QuantT: Quantized output type (fp8_e4m3fn_t, fp8_e5m2_t, etc.)
// IsTailAxis: Whether axis is the last axis (true for axis=-1)
template <typename T, typename U, typename T_IDX, bool isGroupIndex>
class SwigluMxQuantLastLast {
public:
    __aicore__ inline SwigluMxQuantLastLast(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR group_index, GM_ADDR y, GM_ADDR mxscale, GM_ADDR workspace,
        const SwigluMxQuantTilingData *__restrict tilingData, AscendC::TPipe *pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitGroupIndex();
    __aicore__ inline void Compute(int64_t dim0Size, int64_t dim1Size, int64_t dim1AlignSize, bool isTailDim1);
    __aicore__ inline void ComputeF8(__local_mem__ T *swigluUbAddr, __local_mem__ uint16_t *maxExpUbAddr,
        __local_mem__ uint16_t *halfScaleLocalAddr, int64_t dim0Size, int64_t dim1Size, int64_t dim1AlignSize);
    __aicore__ inline void ComputeScale(__local_mem__ uint16_t *maxExpAddr, __local_mem__ uint16_t *mxScaleLocalAddr,
        __local_mem__ uint16_t *halfScaleLocalAddr, int64_t dim0OnceSize, int64_t dim1OnceSize, int64_t alignDim1Size);
    __aicore__ inline void ComputeScaleBLAS(__local_mem__ uint16_t *maxExpAddr,
        __local_mem__ uint16_t *mxScaleLocalAddr, __local_mem__ uint16_t *halfScaleLocalAddr, int64_t dim0OnceSize,
        int64_t dim1OnceSize, int64_t alignDim1Size);
    __aicore__ inline void ComputeVfMaxExpVf(__local_mem__ T *srcAddr, __local_mem__ uint16_t *maxExpAddr,
        int64_t dim0OnceSize, int64_t dim1OnceSize, int64_t alignDim1Size);
    __aicore__ inline void ComputeVfMaxExpVfBLAS(__local_mem__ T *srcAddr, __local_mem__ uint16_t *maxExpAddr,
        int64_t dim0OnceSize, int64_t dim1OnceSize, int64_t alignDim1Size);
    __aicore__ inline void ComputeVfSwigluV2(__local_mem__ T *x1UbAddr, __local_mem__ T *x2UbAddr,
        __local_mem__ T *swigluUbAddr, int64_t dim0OnceSize, int64_t dim1OnceSize, int64_t dim1AlignSize,
        bool isTailDim1);
    __aicore__ inline void ComputeVfSwigluV1(__local_mem__ T *x1UbAddr, __local_mem__ T *x2UbAddr,
        __local_mem__ T *swigluUbAddr, int64_t dim0OnceSize, int64_t dim1OnceSize, int64_t dim1AlignSize,
        bool isTailDim1);
    __aicore__ inline void CopyIn(int64_t dim0LoopIdx, int64_t dim1LoopIdx, int64_t dim0OnceSize, int64_t dim1OnceSize);
    __aicore__ inline void CopyOut(int64_t dim0LoopIdx, int64_t dim1LoopIdx, int64_t dim0OnceSize, int64_t dim1OnceSize,
        int64_t dim1OnceSizeAlgin);

private:
    GlobalTensor<T> xGm_;
    GlobalTensor<T_IDX> groupIndexGm_;
    GlobalTensor<uint8_t> yGm_;
    GlobalTensor<uint8_t> scaleGm_;
    const SwigluMxQuantTilingData *tiling_;
    AscendC::TPipe *pipe_;

    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQuex_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outQuey_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outQueScale_;

    // Tmp buffers for UB allocation
    TBuf<QuePosition::VECCALC> reduceSumBuffer_;
    TBuf<QuePosition::VECCALC> swigluBuffer_;
    TBuf<QuePosition::VECCALC> maxExpBuffer_;
    TBuf<QuePosition::VECCALC> maxhalfScaleBuffer_;
    int32_t blockIdx_ = 0;
    // tilingData
    int64_t realCoreNum_ = 0;  // 运行的总核数
    int64_t swigluMode_ = 0;   // swigluMode 0 是左右激活, 1是奇偶激活
    int64_t activateLeft_ = 0; // 0 表示false，右半部分激活, 1表示true，左半部分激活
    float clampLimit_ = 0.0f;  // swiglu 上限值
    float gluBias_ = 0.0f;     // 变体glu的bias
    float gluAlpha_ = 0.0f;    // 变体glu的alpha
    int64_t scaleAlg_ = 0;     // 0表示ocp算法 1 cublas算法

    int64_t dim0Size_ = 0;       // 合轴为2维, 和轴后0维度大小  和轴后的shape是 M,2N,  dim0Size_ = M
    int64_t dim1Size_ = 0;       // 和轴后1维度大小 dim1Size_ = 2N
    int64_t factorDim0Size_ = 0; // 和轴后0维基本块大小 基本块大小是按照1* 256个数来的
    int64_t factorDim1Size_ = 0; // 和轴后1维基本块大小  注意基本块是按照N计算的,不是2N,也就是可以放多少个256个数
    int64_t blockFactor_ = 0; // 对dim0分核，每个核处理多少行,例如行数172，核数realCoreNum_=64，blockFactor_ = 172/ 64
    int64_t tailBlock_ = 0;      // 取余的结果，例如核数64,行数是172,那么blockFactor = 172/ 64
                                 // =2,前44行每个核处理3行，后面20个核每个核处理2行,tailBlock_=20
    int64_t loopTimesBDim0_ = 0; // 前面的核0维循环次数，也就是tailBlock_个核循环次数
    int64_t tailBDim0_ = 0; // 前面的核0维循环尾块大小，也就是tailBlock_个核循环最后一次处理多少行
    int64_t loopTimesTDim0_ = 0; // 后面的核0维循环次数，也就是大于tailBlock_的核循环次数
    int64_t tailTDim0_ = 0; // 后面的核0维循环尾块，也就是大于tailBlock_的核循环最后一次处理多少数
    int64_t loopTimesDim1_ = 0; // 1维循环次数
    int64_t tailDim1_ = 0; // 1维最后一次循环的尾块大小，肯定小于256 注意这个也是swiglu后的一般的个数 这里也是代表一半
    int64_t groupIndexNum_ = 0; // groupIndex的shape
    int64_t roundMode_ = 0; // 1==rint  2==floor  4==round
    int64_t halfInput_ = 0;

    int64_t blockOffset_ = 0; // 核间offset
    int64_t blockFactorNum_ = 0;
    int64_t loopTimesDim0_ = 0;
    int64_t tailDim0_ = 0;
    uint16_t f4Emax_ = 0;
    uint32_t dtypeMax = 0;
    uint16_t f8Emax_ = 0;
    int64_t outputScaleRowBytes_ = 0;
    uint32_t vfLenT_ = Ops::Base::GetVRegSize() / sizeof(T);
    uint32_t oneBlockUb_ = Ops::Base::GetUbBlockSize();
    uint32_t oneBlockNum_ = oneBlockUb_ / sizeof(T);
    uint32_t vfLenFp32_ = Ops::Base::GetVRegSize() / sizeof(float);
};

template <typename U, AscendC::RoundMode toBf16RoundMode, AscendC::RoundMode roundMode>
__aicore__ inline void ComputeFP4FromHalf(MicroAPI::RegTensor<float> &reg, MicroAPI::RegTensor<int32_t> &negZero,
    MicroAPI::MaskReg &pregAll32)
{
    MicroAPI::MaskReg zeroMask;
    MicroAPI::MaskReg specialMask;
    MicroAPI::MaskReg negInfMask;

    MicroAPI::RegTensor<int32_t> maxExpFP32;
    MicroAPI::RegTensor<int32_t> exp0FP32;
    MicroAPI::RegTensor<int32_t> exp1FP32;
    MicroAPI::Compare<int32_t, CMPMODE::EQ>(negInfMask, (MicroAPI::RegTensor<int32_t> &)reg, negZero, pregAll32);
    if constexpr (IsSameType<U, fp4x2_e1m2_t>::value) {
        MicroAPI::Muls(reg, reg, FOUR, pregAll32);
        MicroAPI::CompareScalar<float, CMPMODE::LT>(specialMask, reg, 0, pregAll32);
        MicroAPI::Truncate<float, roundMode>(reg, reg, pregAll32);
        MicroAPI::Muls(reg, reg, ONE_FOURTH, pregAll32);
    } else {
        MicroAPI::Duplicate(maxExpFP32, MAX_EXP_FOR_FP32);
        MicroAPI::And(exp0FP32, (MicroAPI::RegTensor<int32_t> &)reg, maxExpFP32, pregAll32);
        MicroAPI::ShiftRights(exp0FP32, exp0FP32, SHR_NUM_FOR_FP32, pregAll32);
        MicroAPI::Adds(exp0FP32, exp0FP32, FP32_BIAS_NEG, pregAll32);
        MicroAPI::Maxs(exp0FP32, exp0FP32, 0, pregAll32);
        MicroAPI::Adds(exp0FP32, exp0FP32, NEG_ONE, pregAll32);
        MicroAPI::Muls(exp1FP32, exp0FP32, NEG_ONE, pregAll32);
        MicroAPI::Adds(exp1FP32, exp1FP32, FP32_BIAS, pregAll32);
        MicroAPI::ShiftLefts(exp1FP32, exp1FP32, SHR_NUM_FOR_FP32, pregAll32);

        MicroAPI::Mul(reg, reg, (MicroAPI::RegTensor<float> &)exp1FP32, pregAll32);
        MicroAPI::Adds(exp0FP32, exp0FP32, FP32_BIAS, pregAll32);
        MicroAPI::ShiftLefts(exp0FP32, exp0FP32, SHR_NUM_FOR_FP32, pregAll32);
        MicroAPI::CompareScalar<float, CMPMODE::LT>(specialMask, reg, 0, pregAll32);
        MicroAPI::Truncate<float, roundMode>(reg, reg, pregAll32);
        MicroAPI::Mul(reg, reg, (MicroAPI::RegTensor<float> &)exp0FP32, pregAll32);
    }
    MicroAPI::CompareScalar<float, CMPMODE::EQ>(zeroMask, reg, 0, pregAll32);
    MicroAPI::MaskAnd(zeroMask, specialMask, zeroMask, pregAll32);
    MicroAPI::MaskOr(zeroMask, negInfMask, zeroMask, pregAll32);
    MicroAPI::Select<int32_t>((MicroAPI::RegTensor<int32_t> &)reg, negZero, (MicroAPI::RegTensor<int32_t> &)reg,
        zeroMask);
}

template <typename T_IDX>
__aicore__ inline void ReduceAllVf(LocalTensor<T_IDX> &reduceSumUb, LocalTensor<T_IDX> &groupIndexUb,
    int64_t groupIndexNum_)
{
    uint32_t vfTidx = Ops::Base::GetVRegSize() / sizeof(T_IDX);
    uint16_t times = groupIndexNum_ / vfTidx;
    uint32_t tailNum = groupIndexNum_ % vfTidx;
    uint16_t tailTimes = tailNum != 0 ? 1 : 0;
    auto dstAddr = (__ubuf__ T_IDX *)reduceSumUb.GetPhyAddr();
    auto srcAddr = (__ubuf__ T_IDX *)groupIndexUb.GetPhyAddr();
    auto srcAddr1 = (__ubuf__ T_IDX *)groupIndexUb[times * vfTidx].GetPhyAddr();
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<T_IDX> addReg;
        AscendC::MicroAPI::RegTensor<T_IDX> reduceSumReg;
        AscendC::MicroAPI::RegTensor<T_IDX> reduceSumTReg;
        AscendC::MicroAPI::RegTensor<T_IDX> srcReg;
        AscendC::MicroAPI::Duplicate(addReg, 0);
        AscendC::MicroAPI::MaskReg mask = AscendC::MicroAPI::CreateMask<T_IDX, MicroAPI::MaskPattern::ALL>();
        for (uint16_t i = 0; i < times; i++) {
            AscendC::MicroAPI::AddrReg srcIdxOffset = AscendC::MicroAPI::CreateAddrReg<T_IDX>(i, vfTidx);
            AscendC::MicroAPI::DataCopy(srcReg, srcAddr, srcIdxOffset);
            AscendC::MicroAPI::Add(addReg, addReg, srcReg, mask);
        }
        AscendC::MicroAPI::ReduceSum(reduceSumReg, addReg, mask);
        for (uint16_t j = 0; j < tailTimes; j++) {
            AscendC::MicroAPI::MaskReg maskT = AscendC::MicroAPI::UpdateMask<T_IDX>(tailNum);
            AscendC::MicroAPI::DataCopy(srcReg, srcAddr1);
            AscendC::MicroAPI::ReduceSum(reduceSumTReg, srcReg, maskT);
            AscendC::MicroAPI::Add(reduceSumReg, reduceSumTReg, reduceSumReg, maskT);
        }
        AscendC::MicroAPI::MaskReg maskOne = AscendC::MicroAPI::CreateMask<T_IDX, MicroAPI::MaskPattern::VL1>();
        AscendC::MicroAPI::DataCopy(dstAddr, reduceSumReg, maskOne);
    }
}

template <typename T, typename U, typename T_IDX, bool isGroupIndex>
__aicore__ inline void SwigluMxQuantLastLast<T, U, T_IDX, isGroupIndex>::InitGroupIndex()
{
    groupIndexNum_ = tiling_->groupIndexNum;
    // groupindex存在，核内计算tiling, 先allreduce来计算总行数
    LocalTensor<T_IDX> reduceSumUb = reduceSumBuffer_.Get<T_IDX>();
    LocalTensor<T_IDX> groupIndexUb = inQuex_.AllocTensor<T_IDX>();
    DataCopyExtParams copyInParam = { 1, 0, 0, 0, 0 };
    DataCopyPadExtParams<T_IDX> copyPadParams = { false, 0, 0, 0 };
    copyInParam.blockLen = groupIndexNum_ * sizeof(T_IDX);
    DataCopyPad(groupIndexUb, groupIndexGm_[0], copyInParam, copyPadParams);
    inQuex_.EnQue(groupIndexUb);
    groupIndexUb = inQuex_.DeQue<T_IDX>();
    ReduceAllVf<T_IDX>(reduceSumUb, groupIndexUb, groupIndexNum_);
    inQuex_.FreeTensor(groupIndexUb);
    event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventVS);
    WaitFlag<HardEvent::V_S>(eventVS);
    int64_t allNum = static_cast<int64_t>(reduceSumUb.GetValue(0));
    // 开始计算分核
    blockFactor_ = allNum / realCoreNum_;
    tailBlock_ = allNum % realCoreNum_;
    int64_t bBlockNum = blockFactor_ + 1;
    int64_t tBlockNum = blockFactor_;
    if (allNum < realCoreNum_) {
        realCoreNum_ = allNum;
        blockFactor_ = 1;
        tailBlock_ = 0;
    }
    if (tailBlock_ == 0) {
        bBlockNum = blockFactor_;
        tBlockNum = blockFactor_;
    }
    loopTimesBDim0_ =
        (bBlockNum + factorDim0Size_ - 1) / factorDim0Size_; // 前面的核0维循环次数，也就是tailBlock_个核循环次数
    tailBDim0_ = bBlockNum - (loopTimesBDim0_ - 1) *
        factorDim0Size_; // 前面的核0维循环尾块大小，也就是tailBlock_个核循环最后一次处理多少行
    loopTimesTDim0_ = (tBlockNum + factorDim0Size_ - 1) /
        factorDim0Size_; // 后面的核0维循环次数，也就是大于tailBlock_的核循环次数
    tailTDim0_ = tBlockNum - (loopTimesTDim0_ - 1) *
        factorDim0Size_; // 后面的核0维循环尾块，也就是大于tailBlock_的核循环最后一次处理多少数
}

template <typename T, typename U, typename T_IDX, bool isGroupIndex>
__aicore__ inline void SwigluMxQuantLastLast<T, U, T_IDX, isGroupIndex>::Init(GM_ADDR x, GM_ADDR group_index, GM_ADDR y,
    GM_ADDR mxscale, GM_ADDR workspace, const SwigluMxQuantTilingData *__restrict tilingData, AscendC::TPipe *pipe)
{
#if (__NPU_ARCH__ == 3510)
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
#endif
    tiling_ = tilingData;
    pipe_ = pipe;
    blockIdx_ = GetBlockIdx();
    xGm_.SetGlobalBuffer((__gm__ T *)x);
    yGm_.SetGlobalBuffer((__gm__ uint8_t *)y);
    scaleGm_.SetGlobalBuffer((__gm__ uint8_t *)mxscale);
    if constexpr (isGroupIndex) {
        groupIndexGm_.SetGlobalBuffer((__gm__ T_IDX *)group_index);
    }
    // 开始get tilingData
    dim0Size_ = tiling_->inputDim1;
    dim1Size_ = tiling_->inputDim2;
    halfInput_ = dim1Size_ / CONST_2;
    realCoreNum_ = tiling_->usedCoreNum;
    factorDim0Size_ = tiling_->maxBasicNumUbDim1;
    factorDim1Size_ = tiling_->maxBasicNumUbDim2;
    roundMode_ = tiling_->roundMode;

    // Initialize pipe buffers
    int32_t factorSize = factorDim0Size_ * factorDim1Size_;
    pipe_->InitBuffer(inQuex_, CONST_2, factorSize * X_ONCE_NUM * sizeof(T));
    if constexpr (IsSame<U, fp4x2_e2m1_t>::value || IsSame<U, fp4x2_e1m2_t>::value) {
        pipe_->InitBuffer(outQuey_, CONST_2, factorSize * QUANT_ONCE_NUM_FP4 * sizeof(uint8_t));
    } else {
        pipe_->InitBuffer(outQuey_, CONST_2, (factorSize * QUANT_ONCE_NUM) * sizeof(uint8_t));
    }
    int32_t scaleUbSize = factorSize * SCALE_ONCE_NUM;
    scaleUbSize = ((scaleUbSize + CONST_64 - 1) / CONST_64) * CONST_64;
    pipe_->InitBuffer(outQueScale_, CONST_2, scaleUbSize);

    // Initialize tmp buffers according to new UB allocation scheme
    // group_reserve: 32 bytes (fixed size, pre-reserved for future inputs)
    pipe_->InitBuffer(swigluBuffer_, factorSize * QUANT_ONCE_NUM * sizeof(T));
    int32_t maxExpUbSize = factorSize * SCALE_ONCE_NUM * sizeof(uint16_t);
    maxExpUbSize = ((maxExpUbSize + oneBlockUb_ - 1) / oneBlockUb_) * oneBlockUb_;
    // reciprocal_scale: 8 × sizeof(half) = 16 bytes (8 quantization blocks, stored as FP16)
    pipe_->InitBuffer(maxExpBuffer_, maxExpUbSize);

    // max_exp: 8 × sizeof(int16_t) = 16 bytes (max exponent for OCP algorithm)
    pipe_->InitBuffer(maxhalfScaleBuffer_, maxExpUbSize);
    if constexpr (isGroupIndex) {
        pipe_->InitBuffer(reduceSumBuffer_, oneBlockUb_);
    }

    if constexpr (isGroupIndex) {
        InitGroupIndex();
    } else {
        blockFactor_ = tiling_->tailCoreBasicNumDim1;
        tailBlock_ = tiling_->frontCoreNum;
        loopTimesBDim0_ = tiling_->frontCoreLoopTimes;
        tailBDim0_ = tiling_->frontCoreLastLoopBasicNum;
        loopTimesTDim0_ = tiling_->tailCoreLoopTimes;
        tailTDim0_ = tiling_->tailCoreLastLoopBasicNum;
    }
    swigluMode_ = tiling_->swigluMode;
    activateLeft_ = tiling_->activateLeft;
    clampLimit_ = tiling_->clampLimit;
    gluBias_ = tiling_->gluBias;
    gluAlpha_ = tiling_->gluAlpha;
    scaleAlg_ = tiling_->scaleAlg;
    loopTimesDim1_ = tiling_->ubLoopPerRow;
    tailDim1_ = tiling_->ubTailPerRow; // 注意这个是一半的数
    if (blockIdx_ < tailBlock_) {
        blockFactorNum_ = blockFactor_ + 1;
        blockOffset_ = blockIdx_ * blockFactorNum_;
        loopTimesDim0_ = loopTimesBDim0_;
        tailDim0_ = tailBDim0_;
    } else {
        blockFactorNum_ = blockFactor_;
        blockOffset_ = tailBlock_ * (blockFactor_ + 1) + (blockIdx_ - tailBlock_) * blockFactorNum_;
        loopTimesDim0_ = loopTimesTDim0_;
        tailDim0_ = tailTDim0_;
    }
    outputScaleRowBytes_ = (halfInput_ + CONST_32 - 1) / CONST_32;
    if (outputScaleRowBytes_ % CONST_2 != 0) {
        outputScaleRowBytes_ = outputScaleRowBytes_ + 1;
    }
    if constexpr (IsSame<U, fp4x2_e2m1_t>::value) {
        f4Emax_ = FP4_E2M1_BF16_MAX_EXP;
    }
    if constexpr (IsSame<U, fp4x2_e1m2_t>::value) {
        f4Emax_ = FP4_E1M2_MAX_EXP;
    }
    if constexpr (IsSame<U, fp8_e4m3fn_t>::value) {
        f8Emax_ = FP8_E4M3_MAX_EXP;
        dtypeMax = FP8_E4M3_MAX;
    }
    if constexpr (IsSame<U, fp8_e5m2_t>::value) {
        f8Emax_ = FP8_E5M2_MAX_EXP;
        dtypeMax = FP8_E5M2_MAX;
    }
}

template <typename T, typename U, typename T_IDX, bool isGroupIndex>
__aicore__ inline void SwigluMxQuantLastLast<T, U, T_IDX, isGroupIndex>::Process()
{
    if (blockIdx_ > realCoreNum_) {
        return;
    }
    int64_t dim1Size = factorDim1Size_ * QUANT_ONCE_NUM;
    int64_t scaleNum = (tailDim1_ + CONST_32 - 1) / CONST_32;
    int64_t dim1AlignSize = scaleNum * CONST_32;
    if (scaleNum % CONST_2 != 0) {
        dim1AlignSize = dim1AlignSize + CONST_32;
    }
    for (int64_t dim0LoopIdx = 0; dim0LoopIdx < loopTimesDim0_; dim0LoopIdx++) {
        int64_t dim0Size = dim0LoopIdx == loopTimesDim0_ - 1 ? tailDim0_ : factorDim0Size_;
        for (int64_t dim1LoopIdx = 0; dim1LoopIdx < loopTimesDim1_; dim1LoopIdx++) {
            int64_t dim1SizeNow = dim1LoopIdx == loopTimesDim1_ - 1 ? tailDim1_ : dim1Size;
            bool isTailDim1 = dim1LoopIdx == loopTimesDim1_ - 1 ? true : false;
            int64_t dim1AlignSizeNow = dim1LoopIdx == loopTimesDim1_ - 1 ? dim1AlignSize : dim1Size;
            CopyIn(dim0LoopIdx, dim1LoopIdx, dim0Size, dim1SizeNow);
            Compute(dim0Size, dim1SizeNow, dim1AlignSizeNow, isTailDim1);
            CopyOut(dim0LoopIdx, dim1LoopIdx, dim0Size, dim1SizeNow, dim1AlignSizeNow);
        }
    }
}

template <typename T, typename U, AscendC::RoundMode toBf16RoundMode, AscendC::RoundMode roundMode>
__aicore__ inline void ComputeData(__ubuf__ T *srcAddr, __ubuf__ uint16_t *halfScaleLocalAddr,
    __ubuf__ int8_t *outLocalAddr, int64_t dim0OnceSize, int64_t dim1OnceSize, int64_t dim1AlignSize)
{
    uint32_t totalCountInUB = dim0OnceSize * dim1AlignSize;
    uint16_t loopNum = CeilDivision(totalCountInUB, QUANT_ONCE_NUM);
    int64_t elementAfterReduce = SCALE_ONCE_NUM;
    int64_t onceXNum = QUANT_ONCE_NUM;
    int64_t onceYNum = OUT_ELE_NUM_ONE_BLK;

    static constexpr AscendC::MicroAPI::CastTrait castTrait = { AscendC::MicroAPI::RegLayout::ZERO,
        AscendC::MicroAPI::SatMode::UNKNOWN, AscendC::MicroAPI::MaskMergeMode::ZEROING, roundMode };
    static constexpr AscendC::MicroAPI::CastTrait castTraitHalf2Bf16 = { AscendC::MicroAPI::RegLayout::UNKNOWN,
        AscendC::MicroAPI::SatMode::UNKNOWN, AscendC::MicroAPI::MaskMergeMode::ZEROING, toBf16RoundMode };
    static constexpr MicroAPI::CastTrait castTraitFp32toBF16 = { MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
        MicroAPI::MaskMergeMode::ZEROING, roundMode };

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::MaskReg dataMask1;
        AscendC::MicroAPI::RegTensor<uint16_t> halfScaleForMul;
        AscendC::MicroAPI::RegTensor<T> vdExp0;
        AscendC::MicroAPI::RegTensor<T> vdExp1;
        AscendC::MicroAPI::RegTensor<U> vdExp0FP4;
        AscendC::MicroAPI::RegTensor<U> vdExp1FP4;
        AscendC::MicroAPI::RegTensor<int32_t> negZero;
        MicroAPI::RegTensor<float> halfScaleForMulFP32;
        MicroAPI::RegTensor<float> vdExp0ZeroFP32;
        MicroAPI::RegTensor<float> vdExp0OneFP32;
        MicroAPI::RegTensor<float> vdExp1ZeroFP32;
        MicroAPI::RegTensor<float> vdExp1OneFP32;
        MicroAPI::RegTensor<bfloat16_t> vdExp0ZeroBF16;
        MicroAPI::RegTensor<bfloat16_t> vdExp0OneBF16;
        MicroAPI::RegTensor<bfloat16_t> vdExp1ZeroBF16;
        MicroAPI::RegTensor<bfloat16_t> vdExp1OneBF16;

        AscendC::MicroAPI::RegTensor<bfloat16_t> vdExp0BF16;
        AscendC::MicroAPI::RegTensor<bfloat16_t> vdExp1BF16;
        MicroAPI::MaskReg dataMaskB16 = MicroAPI::CreateMask<half>();
        MicroAPI::MaskReg dataMaskB32 = MicroAPI::CreateMask<float>();
        if constexpr (IsSame<T, half>::value && roundMode != RoundMode::CAST_FLOOR) {
            MicroAPI::Duplicate(negZero, NEG_ZERO);
        }
        for (uint16_t i = 0; i < loopNum; i++) {
            dataMask1 = AscendC::MicroAPI::UpdateMask<T>(totalCountInUB);
            AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                AscendC::MicroAPI::LoadDist::DIST_DINTLV_B16>(vdExp0, vdExp1, srcAddr, onceXNum);
            AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                AscendC::MicroAPI::LoadDist::DIST_E2B_B16>(halfScaleForMul, halfScaleLocalAddr, elementAfterReduce);

            if constexpr (IsSame<T, half>::value && roundMode == RoundMode::CAST_FLOOR) {
                AscendC::MicroAPI::RegTensor<bfloat16_t> vdExp0BF16;
                AscendC::MicroAPI::RegTensor<bfloat16_t> vdExp1BF16;
                AscendC::MicroAPI::Cast<bfloat16_t, T, castTraitHalf2Bf16>(vdExp0BF16, vdExp0, dataMask1);
                AscendC::MicroAPI::Cast<bfloat16_t, T, castTraitHalf2Bf16>(vdExp1BF16, vdExp1, dataMask1);
                AscendC::MicroAPI::Mul(vdExp0BF16, vdExp0BF16,
                    (AscendC::MicroAPI::RegTensor<bfloat16_t> &)halfScaleForMul, dataMask1);
                AscendC::MicroAPI::Mul(vdExp1BF16, vdExp1BF16,
                    (AscendC::MicroAPI::RegTensor<bfloat16_t> &)halfScaleForMul, dataMask1);
                AscendC::MicroAPI::Interleave(vdExp0BF16, vdExp1BF16, vdExp0BF16, vdExp1BF16);
                AscendC::MicroAPI::Cast<U, bfloat16_t, castTrait>(vdExp0FP4, vdExp0BF16, dataMask1);
                AscendC::MicroAPI::Cast<U, bfloat16_t, castTrait>(vdExp1FP4, vdExp1BF16, dataMask1);
            } else if constexpr (IsSame<T, half>::value && roundMode != RoundMode::CAST_FLOOR) {
                // 需要升fp32计算
                MicroAPI::Cast<float, bfloat16_t, CAST_BF16_FP16_TO_FP32>(halfScaleForMulFP32,
                    (MicroAPI::RegTensor<bfloat16_t> &)halfScaleForMul, dataMaskB16);

                // vdExp0
                MicroAPI::Cast<float, T, CAST_BF16_FP16_TO_FP32>(vdExp0ZeroFP32, vdExp0, dataMaskB16);
                MicroAPI::Cast<float, T, CAST_BF16_FP16_TO_FP32_ONE>(vdExp0OneFP32, vdExp0, dataMaskB16);

                MicroAPI::Mul(vdExp0ZeroFP32, vdExp0ZeroFP32, halfScaleForMulFP32, dataMaskB32);
                MicroAPI::Mul(vdExp0OneFP32, vdExp0OneFP32, halfScaleForMulFP32, dataMaskB32);
                ComputeFP4FromHalf<U, toBf16RoundMode, roundMode>(vdExp0ZeroFP32, negZero, dataMaskB32);
                ComputeFP4FromHalf<U, toBf16RoundMode, roundMode>(vdExp0OneFP32, negZero, dataMaskB32);
                AscendC::MicroAPI::Cast<bfloat16_t, float, castTraitFp32toBF16>(vdExp0ZeroBF16, vdExp0ZeroFP32,
                    dataMaskB32);
                AscendC::MicroAPI::Cast<bfloat16_t, float, castTraitFp32toBF16>(vdExp0OneBF16, vdExp0OneFP32,
                    dataMaskB32);
                MicroAPI::Pack<uint16_t, uint32_t, MicroAPI::HighLowPart::LOWEST>(
                    (MicroAPI::RegTensor<uint16_t> &)vdExp0ZeroBF16, (MicroAPI::RegTensor<uint32_t> &)vdExp0ZeroBF16);
                MicroAPI::Pack<uint16_t, uint32_t, MicroAPI::HighLowPart::LOWEST>(
                    (MicroAPI::RegTensor<uint16_t> &)vdExp0OneBF16, (MicroAPI::RegTensor<uint32_t> &)vdExp0OneBF16);
                MicroAPI::Interleave(vdExp0ZeroBF16, vdExp0OneBF16, vdExp0ZeroBF16, vdExp0OneBF16);

                // vdExp1
                MicroAPI::Cast<float, T, CAST_BF16_FP16_TO_FP32>(vdExp1ZeroFP32, vdExp1, dataMaskB16);
                MicroAPI::Cast<float, T, CAST_BF16_FP16_TO_FP32_ONE>(vdExp1OneFP32, vdExp1, dataMaskB16);

                MicroAPI::Mul(vdExp1ZeroFP32, vdExp1ZeroFP32, halfScaleForMulFP32, dataMaskB32);
                MicroAPI::Mul(vdExp1OneFP32, vdExp1OneFP32, halfScaleForMulFP32, dataMaskB32);
                ComputeFP4FromHalf<U, toBf16RoundMode, roundMode>(vdExp1ZeroFP32, negZero, dataMaskB32);
                ComputeFP4FromHalf<U, toBf16RoundMode, roundMode>(vdExp1OneFP32, negZero, dataMaskB32);
                AscendC::MicroAPI::Cast<bfloat16_t, float, castTraitFp32toBF16>(vdExp1ZeroBF16, vdExp1ZeroFP32,
                    dataMaskB32);
                AscendC::MicroAPI::Cast<bfloat16_t, float, castTraitFp32toBF16>(vdExp1OneBF16, vdExp1OneFP32,
                    dataMaskB32);
                MicroAPI::Pack<uint16_t, uint32_t, MicroAPI::HighLowPart::LOWEST>(
                    (MicroAPI::RegTensor<uint16_t> &)vdExp1ZeroBF16, (MicroAPI::RegTensor<uint32_t> &)vdExp1ZeroBF16);
                MicroAPI::Pack<uint16_t, uint32_t, MicroAPI::HighLowPart::LOWEST>(
                    (MicroAPI::RegTensor<uint16_t> &)vdExp1OneBF16, (MicroAPI::RegTensor<uint32_t> &)vdExp1OneBF16);
                MicroAPI::Interleave(vdExp1ZeroBF16, vdExp1OneBF16, vdExp1ZeroBF16, vdExp1OneBF16);

                AscendC::MicroAPI::Interleave(vdExp0ZeroBF16, vdExp1ZeroBF16, vdExp0ZeroBF16, vdExp1ZeroBF16);
                AscendC::MicroAPI::Cast<U, bfloat16_t, castTrait>(vdExp0FP4, vdExp0ZeroBF16, dataMask1);
                AscendC::MicroAPI::Cast<U, bfloat16_t, castTrait>(vdExp1FP4, vdExp1ZeroBF16, dataMask1);
            } else {
                AscendC::MicroAPI::Mul(vdExp0, vdExp0, (AscendC::MicroAPI::RegTensor<T> &)halfScaleForMul, dataMask1);
                AscendC::MicroAPI::Mul(vdExp1, vdExp1, (AscendC::MicroAPI::RegTensor<T> &)halfScaleForMul, dataMask1);
                AscendC::MicroAPI::Interleave(vdExp0, vdExp1, vdExp0, vdExp1);
                AscendC::MicroAPI::Cast<U, T, castTrait>(vdExp0FP4, vdExp0, dataMask1);
                AscendC::MicroAPI::Cast<U, T, castTrait>(vdExp1FP4, vdExp1, dataMask1);
            }
            AscendC::MicroAPI::DataCopy<int8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(outLocalAddr,
                (AscendC::MicroAPI::RegTensor<int8_t> &)vdExp0FP4, onceYNum, dataMask1);
            AscendC::MicroAPI::DataCopy<int8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(outLocalAddr,
                (AscendC::MicroAPI::RegTensor<int8_t> &)vdExp1FP4, onceYNum, dataMask1);
        }
    }
}

template <typename T, typename U>
__aicore__ inline void ComputeDataF8(__ubuf__ T *srcAddr, __ubuf__ uint16_t *halfScaleLocalAddr,
    __ubuf__ int8_t *outLocalAddr, int64_t dim0OnceSize, int64_t dim1OnceSize, int64_t dim1AlignSize)
{
    uint32_t totalCountInUB = dim0OnceSize * dim1AlignSize;
    uint16_t loopNum = CeilDivision(totalCountInUB, QUANT_ONCE_NUM);
    int64_t elementAfterReduce = SCALE_ONCE_NUM;
    int64_t onceXNum = QUANT_ONCE_NUM;
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<uint16_t> halfScaleForMul;
        AscendC::MicroAPI::RegTensor<float> floatScaleForMul;
        AscendC::MicroAPI::RegTensor<T> vdExp0;
        AscendC::MicroAPI::RegTensor<T> vdExp1;
        AscendC::MicroAPI::RegTensor<float> vdExp0FP32Zero;
        AscendC::MicroAPI::RegTensor<float> vdExp0FP32One;
        AscendC::MicroAPI::RegTensor<float> vdExp1FP32Zero;
        AscendC::MicroAPI::RegTensor<float> vdExp1FP32One;
        AscendC::MicroAPI::RegTensor<U> vdExp0FP8Zero;
        AscendC::MicroAPI::RegTensor<U> vdExp0FP8One;
        AscendC::MicroAPI::RegTensor<U> vdExp1FP8Zero;
        AscendC::MicroAPI::RegTensor<U> vdExp1FP8One;
        AscendC::MicroAPI::MaskReg maskAll =
            AscendC::MicroAPI::CreateMask<uint16_t, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::MaskReg maskAllB8 = 
            AscendC::MicroAPI::CreateMask<uint8_t, AscendC::MicroAPI::MaskPattern::ALL>();
        for (uint16_t i = 0; i < loopNum; i++) {
            AscendC::MicroAPI::DataCopy<
                T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE, AscendC::MicroAPI::LoadDist::DIST_DINTLV_B16>(
                vdExp0, vdExp1, srcAddr, onceXNum);
            AscendC::MicroAPI::DataCopy<
                uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE, AscendC::MicroAPI::LoadDist::DIST_E2B_B16>(
                halfScaleForMul, halfScaleLocalAddr, elementAfterReduce);
            if constexpr (IsSame<T, half>::value) {
                AscendC::MicroAPI::Cast<float, T, CAST_ZERO>(vdExp0FP32Zero, vdExp0, maskAll);
                AscendC::MicroAPI::Cast<float, T, CAST_ONE>(vdExp0FP32One, vdExp0, maskAll);
                AscendC::MicroAPI::Cast<float, bfloat16_t, CAST_ZERO>(
                    floatScaleForMul, (AscendC::MicroAPI::RegTensor<bfloat16_t>&)halfScaleForMul, maskAll);
                AscendC::MicroAPI::Mul(vdExp0FP32Zero, vdExp0FP32Zero, floatScaleForMul, maskAll);
                AscendC::MicroAPI::Mul(vdExp0FP32One, vdExp0FP32One, floatScaleForMul, maskAll);
            
                AscendC::MicroAPI::Cast<float, T, CAST_ZERO>(vdExp1FP32Zero, vdExp1, maskAll);
                AscendC::MicroAPI::Cast<float, T, CAST_ONE>(vdExp1FP32One, vdExp1, maskAll);
                AscendC::MicroAPI::Mul(vdExp1FP32Zero, vdExp1FP32Zero, floatScaleForMul, maskAll);
                AscendC::MicroAPI::Mul(vdExp1FP32One, vdExp1FP32One, floatScaleForMul, maskAll);
            
            } else {
                AscendC::MicroAPI::Mul(vdExp0, vdExp0, (AscendC::MicroAPI::RegTensor<T>&)halfScaleForMul, maskAll);
                AscendC::MicroAPI::Mul(vdExp1, vdExp1, (AscendC::MicroAPI::RegTensor<T>&)halfScaleForMul, maskAll);
                AscendC::MicroAPI::Cast<float, T, CAST_ZERO>(vdExp0FP32Zero, vdExp0, maskAll);
                AscendC::MicroAPI::Cast<float, T, CAST_ONE>(vdExp0FP32One, vdExp0, maskAll);
                AscendC::MicroAPI::Cast<float, T, CAST_ZERO>(vdExp1FP32Zero, vdExp1, maskAll);
                AscendC::MicroAPI::Cast<float, T, CAST_ONE>(vdExp1FP32One, vdExp1, maskAll);
            }
            AscendC::MicroAPI::Cast<U, float, CAST_32_TO_80>(vdExp0FP8Zero, vdExp0FP32Zero, maskAll);
            AscendC::MicroAPI::Cast<U, float, CAST_32_TO_82>(vdExp0FP8One, vdExp0FP32One, maskAll);
            AscendC::MicroAPI::Cast<U, float, CAST_32_TO_81>(vdExp1FP8Zero, vdExp1FP32Zero, maskAll);
            AscendC::MicroAPI::Cast<U, float, CAST_32_TO_83>(vdExp1FP8One, vdExp1FP32One, maskAll);
        
            AscendC::MicroAPI::Add((AscendC::MicroAPI::RegTensor<uint8_t>&)vdExp0FP8Zero,
                (AscendC::MicroAPI::RegTensor<uint8_t>&)vdExp0FP8Zero,
                    (AscendC::MicroAPI::RegTensor<uint8_t>&)vdExp0FP8One, maskAllB8);
            AscendC::MicroAPI::Add((AscendC::MicroAPI::RegTensor<uint8_t>&)vdExp0FP8Zero,
                (AscendC::MicroAPI::RegTensor<uint8_t>&)vdExp0FP8Zero,
                    (AscendC::MicroAPI::RegTensor<uint8_t>&)vdExp1FP8Zero, maskAllB8);
            AscendC::MicroAPI::Add((AscendC::MicroAPI::RegTensor<uint8_t>&)vdExp0FP8Zero,
                (AscendC::MicroAPI::RegTensor<uint8_t>&)vdExp0FP8Zero,
                    (AscendC::MicroAPI::RegTensor<uint8_t>&)vdExp1FP8One, maskAllB8);
            AscendC::MicroAPI::DataCopy<
            int8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE, AscendC::MicroAPI::StoreDist::DIST_NORM_B8>(
            outLocalAddr, (AscendC::MicroAPI::RegTensor<int8_t>&)vdExp0FP8Zero, onceXNum, maskAllB8);
        }
    }
}

template <typename T, typename U, typename T_IDX, bool isGroupIndex>
__aicore__ inline void SwigluMxQuantLastLast<T, U, T_IDX, isGroupIndex>::ComputeF8(__local_mem__ T *swigluUbAddr,
    __local_mem__ uint16_t *maxExpUbAddr, __local_mem__ uint16_t *halfScaleLocalAddr,
        int64_t dim0OnceSize, int64_t dim1OnceSize, int64_t dim1AlignSize)
{
    if (scaleAlg_ == 0) {
        ComputeVfMaxExpVf(swigluUbAddr, maxExpUbAddr, dim0OnceSize, dim1OnceSize, dim1AlignSize);
        LocalTensor<uint16_t> mxScaleLocal = outQueScale_.AllocTensor<uint16_t>();
        auto mxScaleLocalAddr = (__ubuf__ uint16_t *)mxScaleLocal.GetPhyAddr();
        ComputeScale(maxExpUbAddr, mxScaleLocalAddr, halfScaleLocalAddr, dim0OnceSize, dim1OnceSize, dim1AlignSize);
        outQueScale_.EnQue(mxScaleLocal);
    } else {
        ComputeVfMaxExpVfBLAS(swigluUbAddr, maxExpUbAddr, dim0OnceSize, dim1OnceSize, dim1AlignSize);
        LocalTensor<uint16_t> mxScaleLocal = outQueScale_.AllocTensor<uint16_t>();
        auto mxScaleLocalAddr = (__ubuf__ uint16_t *)mxScaleLocal.GetPhyAddr();
        ComputeScaleBLAS(maxExpUbAddr, mxScaleLocalAddr, halfScaleLocalAddr, dim0OnceSize, dim1OnceSize,
            dim1AlignSize);
        outQueScale_.EnQue(mxScaleLocal);
    }
}

template <typename T, typename U, typename T_IDX, bool isGroupIndex>
__aicore__ inline void SwigluMxQuantLastLast<T, U, T_IDX, isGroupIndex>::Compute(int64_t dim0OnceSize,
    int64_t dim1OnceSize, int64_t dim1AlignSize, bool isTailDim1)
{
    LocalTensor<T> xlocal = inQuex_.DeQue<T>();
    auto x1UbAddr = (__ubuf__ T *)xlocal.GetPhyAddr();
    auto x2UbAddr = (__ubuf__ T *)xlocal[factorDim0Size_ * factorDim1Size_ * QUANT_ONCE_NUM].GetPhyAddr();
    LocalTensor<T> swigluUb = swigluBuffer_.Get<T>();
    auto swigluUbAddr = (__ubuf__ T *)swigluUb.GetPhyAddr();
    if (swigluMode_ == 0 && activateLeft_ == 0) {
        x1UbAddr = (__ubuf__ T *)xlocal[factorDim0Size_ * factorDim1Size_ * QUANT_ONCE_NUM].GetPhyAddr();
        x2UbAddr = (__ubuf__ T *)xlocal.GetPhyAddr();
    }
    if (swigluMode_ == 1) {
        x1UbAddr = (__ubuf__ T *)xlocal.GetPhyAddr();
        x2UbAddr = (__ubuf__ T *)xlocal[CONST_64].GetPhyAddr();
    }
    if (swigluMode_ == 0) {
        ComputeVfSwigluV1(x1UbAddr, x2UbAddr, swigluUbAddr, dim0OnceSize, dim1OnceSize, dim1AlignSize, isTailDim1);
    } else {
        ComputeVfSwigluV2(x1UbAddr, x2UbAddr, swigluUbAddr, dim0OnceSize, dim1OnceSize, dim1AlignSize, isTailDim1);
    }
    inQuex_.FreeTensor(xlocal);
    LocalTensor<uint16_t> maxExpUb = maxExpBuffer_.Get<uint16_t>();
    auto maxExpUbAddr = (__ubuf__ uint16_t *)maxExpUb.GetPhyAddr();
    LocalTensor<uint16_t> halfScaleLocal = maxhalfScaleBuffer_.Get<uint16_t>();
    auto halfScaleLocalAddr = reinterpret_cast<__ubuf__ uint16_t *>(halfScaleLocal.GetPhyAddr());
    if constexpr (IsSame<U, fp4x2_e2m1_t>::value || IsSame<U, fp4x2_e1m2_t>::value) {
        ComputeVfMaxExpVf(swigluUbAddr, maxExpUbAddr, dim0OnceSize, dim1OnceSize, dim1AlignSize);
        LocalTensor<uint16_t> mxScaleLocal = outQueScale_.AllocTensor<uint16_t>();
        auto mxScaleLocalAddr = (__ubuf__ uint16_t *)mxScaleLocal.GetPhyAddr();
        ComputeScale(maxExpUbAddr, mxScaleLocalAddr, halfScaleLocalAddr, dim0OnceSize, dim1OnceSize, dim1AlignSize);
        outQueScale_.EnQue(mxScaleLocal);
    } else {
        ComputeF8(swigluUbAddr, maxExpUbAddr, halfScaleLocalAddr, dim0OnceSize,
            dim1OnceSize, dim1AlignSize);
    }
    LocalTensor<int8_t> outLocal = outQuey_.AllocTensor<int8_t>();
    auto outLocalAddr = (__ubuf__ int8_t *)outLocal.GetPhyAddr();
    if constexpr (IsSame<U, fp4x2_e2m1_t>::value || IsSame<U, fp4x2_e1m2_t>::value) {
        if (roundMode_ == 1) { // RINT
            ComputeData<T, U, RoundMode::CAST_TRUNC, RoundMode::CAST_RINT>(swigluUbAddr, halfScaleLocalAddr,
                outLocalAddr, dim0OnceSize, dim1OnceSize, dim1AlignSize);
        } else if (roundMode_ == CONST_4) { // ROUND
            ComputeData<T, U, RoundMode::CAST_TRUNC, RoundMode::CAST_ROUND>(swigluUbAddr, halfScaleLocalAddr,
                outLocalAddr, dim0OnceSize, dim1OnceSize, dim1AlignSize);
        } else { // FLOOR
            ComputeData<T, U, RoundMode::CAST_FLOOR, RoundMode::CAST_FLOOR>(swigluUbAddr, halfScaleLocalAddr,
                outLocalAddr, dim0OnceSize, dim1OnceSize, dim1AlignSize);
        }
    } else {
        ComputeDataF8<T, U>(swigluUbAddr, halfScaleLocalAddr, outLocalAddr,
            dim0OnceSize, dim1OnceSize, dim1AlignSize);
    }
    outQuey_.EnQue(outLocal);
}

template <typename T, typename U, typename T_IDX, bool isGroupIndex>
__aicore__ inline void SwigluMxQuantLastLast<T, U, T_IDX, isGroupIndex>::ComputeScale(
    __local_mem__ uint16_t *maxExpAddr, __local_mem__ uint16_t *mxScaleLocalAddr,
    __local_mem__ uint16_t *halfScaleLocalAddr, int64_t dim0OnceSize, int64_t dim1OnceSize, int64_t alignDim1Size)
{
    uint32_t totalScaleInUB = dim0OnceSize * (alignDim1Size / CONST_32);
    uint16_t loopNumScale = CeilDivision(totalScaleInUB, QUANT_ONCE_NUM_FP4);
    uint16_t maxExpBf16 = MAX_EXP_FOR_BF16;
    uint16_t fEmax = f4Emax_;
    int64_t onceNum = QUANT_ONCE_NUM_FP4;
    int64_t onceNumMxScale = CONST_64;
    if constexpr (IsSame<U, fp8_e4m3fn_t>::value || IsSame<U, fp8_e5m2_t>::value) {
        fEmax = f8Emax_;
    }
    uint16_t bf16ExpBias = BF16_EXP_BIAS;
    uint16_t maxExpFp8 = MAX_EXP_FOR_FP8;
    uint16_t nanCustomZation = NAN_CUSTOMIZATION;
    uint16_t specailExpThreshold = SPECIAL_EXP_THRESHOLD;
    // scale输出固定fp8
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<uint16_t> expMask;
        AscendC::MicroAPI::Duplicate(expMask, maxExpBf16);
        AscendC::MicroAPI::RegTensor<uint16_t> vdMaxExp;

        AscendC::MicroAPI::RegTensor<T> vdExp0;
        AscendC::MicroAPI::RegTensor<T> vdExp1;

        AscendC::MicroAPI::MaskReg cmpResult;
        AscendC::MicroAPI::MaskReg zeroMask;
        AscendC::MicroAPI::MaskReg cmpResultSub;
        AscendC::MicroAPI::MaskReg preMaskScale;
        AscendC::MicroAPI::RegTensor<uint16_t> maxExpValue;
        AscendC::MicroAPI::Duplicate(maxExpValue, fEmax);
        AscendC::MicroAPI::RegTensor<uint16_t> sharedExp;
        AscendC::MicroAPI::RegTensor<uint16_t> scaleValue;
        AscendC::MicroAPI::RegTensor<uint16_t> scaleBias;
        AscendC::MicroAPI::Duplicate(scaleBias, bf16ExpBias);
        AscendC::MicroAPI::RegTensor<uint16_t> halfScale;
        AscendC::MicroAPI::RegTensor<uint16_t> fp8NanRegTensor;
        AscendC::MicroAPI::Duplicate(fp8NanRegTensor, maxExpFp8);
        AscendC::MicroAPI::RegTensor<uint16_t> zeroRegTensor;
        AscendC::MicroAPI::Duplicate(zeroRegTensor, 0);
        AscendC::MicroAPI::RegTensor<uint16_t> nanRegTensor;
        AscendC::MicroAPI::Duplicate(nanRegTensor, nanCustomZation);
        AscendC::MicroAPI::MaskReg invalidDataMask;
        AscendC::MicroAPI::MaskReg specialDataMask;
        AscendC::MicroAPI::RegTensor<uint16_t> specialExpRegTensor;
        AscendC::MicroAPI::Duplicate(specialExpRegTensor, specailExpThreshold);
        for (uint16_t i = 0; i < loopNumScale; i++) {
            preMaskScale = AscendC::MicroAPI::UpdateMask<uint16_t>(totalScaleInUB);
            AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(vdMaxExp,
                maxExpAddr, onceNum);
            AscendC::MicroAPI::Compare<uint16_t, CMPMODE::NE>(cmpResult, vdMaxExp, expMask, preMaskScale); // INF/NAN
            AscendC::MicroAPI::Compare<uint16_t, CMPMODE::NE>(zeroMask, vdMaxExp, zeroRegTensor, preMaskScale);
            AscendC::MicroAPI::Compare<uint16_t, CMPMODE::LE>(invalidDataMask, vdMaxExp, maxExpValue, preMaskScale);

            AscendC::MicroAPI::Select<uint16_t>(vdMaxExp, maxExpValue, vdMaxExp, invalidDataMask);

            AscendC::MicroAPI::Sub(sharedExp, vdMaxExp, maxExpValue, preMaskScale);
            AscendC::MicroAPI::ShiftRights(scaleValue, sharedExp, SHR_NUM_FOR_BF16, preMaskScale);

            AscendC::MicroAPI::Select<uint16_t>(scaleValue, scaleValue, fp8NanRegTensor, cmpResult);
            AscendC::MicroAPI::Select<uint16_t>(scaleValue, scaleValue, zeroRegTensor, zeroMask);

            AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                AscendC::MicroAPI::StoreDist::DIST_PACK_B16>(mxScaleLocalAddr, scaleValue, onceNumMxScale, preMaskScale);

            AscendC::MicroAPI::Compare<uint16_t, CMPMODE::EQ>(specialDataMask, sharedExp, scaleBias, preMaskScale);
            AscendC::MicroAPI::Sub(halfScale, scaleBias, sharedExp, preMaskScale);
            AscendC::MicroAPI::Select<uint16_t>(halfScale, halfScale, nanRegTensor, cmpResult);
            AscendC::MicroAPI::Select<uint16_t>(halfScale, halfScale, zeroRegTensor, zeroMask);
            AscendC::MicroAPI::Select<uint16_t>(halfScale, specialExpRegTensor, halfScale, specialDataMask);
            AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(halfScaleLocalAddr,
                halfScale, onceNum, preMaskScale);
        }
    }
}

template <typename T, typename U, typename T_IDX, bool isGroupIndex>
__aicore__ inline void SwigluMxQuantLastLast<T, U, T_IDX, isGroupIndex>::ComputeScaleBLAS(
    __local_mem__ uint16_t *maxExpAddr, __local_mem__ uint16_t *mxScaleLocalAddr,
    __local_mem__ uint16_t *halfScaleLocalAddr, int64_t dim0OnceSize, int64_t dim1OnceSize, int64_t alignDim1Size)
{
    uint32_t totalScaleInUB = dim0OnceSize * (alignDim1Size / CONST_32);
    uint16_t loopNumScale = CeilDivision(totalScaleInUB, CONST_64);
    uint32_t manMaskFloat = MAN_MASK_FLOAT;
    uint32_t maxExpForFp32 = MAX_EXP_FOR_FP32;
    uint32_t fp32ExpBiasCublas = FP32_EXP_BIAS_CUBLAS;
    uint32_t nanCustomiZation = NAN_CUSTOMIZATION_PACK;
    uint32_t maxExpForFp8InFp32 = MAX_EXP_FOR_FP8_IN_FP32;
    uint32_t zeroForAll = ZERO_FOR_ALL;
    uint32_t Exp254 = EXP_254;
    uint32_t halfForMan = HALF_FOR_MAN;
    int64_t onceNum = CONST_64;
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<uint16_t> max16;
        AscendC::MicroAPI::RegTensor<uint32_t> max32;
        AscendC::MicroAPI::RegTensor<uint32_t> exp32;
        AscendC::MicroAPI::RegTensor<uint32_t> man32;
        AscendC::MicroAPI::RegTensor<uint32_t> normalExp32;
        AscendC::MicroAPI::RegTensor<uint32_t> expAddOne32;
        AscendC::MicroAPI::RegTensor<uint32_t> extractExp;
        AscendC::MicroAPI::RegTensor<uint16_t> expOut;
        AscendC::MicroAPI::RegTensor<uint32_t> halfScale;
        AscendC::MicroAPI::RegTensor<uint16_t> recExpOut;

        AscendC::MicroAPI::RegTensor<uint32_t> invMax;
        AscendC::MicroAPI::Duplicate(invMax, dtypeMax);
        AscendC::MicroAPI::RegTensor<uint32_t> manMaskFP32;
        AscendC::MicroAPI::Duplicate(manMaskFP32, manMaskFloat);
        AscendC::MicroAPI::RegTensor<uint32_t> expMask;
        AscendC::MicroAPI::Duplicate(expMask, maxExpForFp32);
        AscendC::MicroAPI::RegTensor<uint32_t> zeroRegTensor32;
        AscendC::MicroAPI::Duplicate(zeroRegTensor32, 0);
        AscendC::MicroAPI::RegTensor<uint32_t> scaleBias;
        AscendC::MicroAPI::Duplicate(scaleBias, fp32ExpBiasCublas);
        AscendC::MicroAPI::RegTensor<uint32_t> nanRegTensor;
        AscendC::MicroAPI::Duplicate(nanRegTensor, nanCustomiZation);
        AscendC::MicroAPI::RegTensor<uint32_t> fp8NanRegTensor;
        AscendC::MicroAPI::Duplicate(fp8NanRegTensor, maxExpForFp8InFp32);

        AscendC::MicroAPI::MaskReg cmpResult;
        AscendC::MicroAPI::MaskReg zeroMask;
        AscendC::MicroAPI::MaskReg p0;
        AscendC::MicroAPI::MaskReg p1;
        AscendC::MicroAPI::MaskReg p2;
        AscendC::MicroAPI::MaskReg maskHalf =
            AscendC::MicroAPI::CreateMask<uint16_t, AscendC::MicroAPI::MaskPattern::VL64>();
        AscendC::MicroAPI::MaskReg preMaskScale = AscendC::MicroAPI::CreateMask<uint32_t>();
        AscendC::MicroAPI::MaskReg mask = AscendC::MicroAPI::CreateMask<uint16_t>();

        for (uint16_t i = 0; i < loopNumScale; i++) {
            AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(max16, maxExpAddr, onceNum);

            AscendC::MicroAPI::Cast<float, T, CAST_BF16_FP16_TO_FP32>((AscendC::MicroAPI::RegTensor<float> &)max32,
                (AscendC::MicroAPI::RegTensor<T> &)max16, preMaskScale);
            AscendC::MicroAPI::Compare<uint32_t, CMPMODE::LT>(cmpResult, max32, expMask, preMaskScale);
            AscendC::MicroAPI::Compare<uint32_t, CMPMODE::NE>(zeroMask, max32, zeroRegTensor32, preMaskScale);

            AscendC::MicroAPI::Mul((AscendC::MicroAPI::RegTensor<float> &)max32,
                (AscendC::MicroAPI::RegTensor<float> &)max32, (AscendC::MicroAPI::RegTensor<float> &)invMax,
                preMaskScale);
            AscendC::MicroAPI::ShiftRights(exp32, max32, SHR_NUM_FOR_FP32, preMaskScale);
            AscendC::MicroAPI::And(man32, max32, manMaskFP32, preMaskScale);

            AscendC::MicroAPI::CompareScalar<uint32_t, CMPMODE::GT>(p0, exp32, zeroForAll, preMaskScale);
            AscendC::MicroAPI::CompareScalar<uint32_t, CMPMODE::LT>(p1, exp32, Exp254, preMaskScale);
            AscendC::MicroAPI::CompareScalar<uint32_t, CMPMODE::GT>(p2, man32, zeroForAll, preMaskScale);
            AscendC::MicroAPI::MaskAnd(p0, p0, p1, preMaskScale);
            AscendC::MicroAPI::MaskAnd(p0, p0, p2, preMaskScale);

            AscendC::MicroAPI::CompareScalar<uint32_t, CMPMODE::EQ>(p1, exp32, zeroForAll, preMaskScale);
            AscendC::MicroAPI::CompareScalar<uint32_t, CMPMODE::GT>(p2, man32, halfForMan, preMaskScale);
            AscendC::MicroAPI::MaskAnd(p1, p1, p2, preMaskScale);
            AscendC::MicroAPI::MaskOr(p0, p0, p1, preMaskScale);

            AscendC::MicroAPI::Adds(expAddOne32, exp32, 1, preMaskScale);
            AscendC::MicroAPI::Select(extractExp, expAddOne32, exp32, p0);
            AscendC::MicroAPI::Select<uint32_t>(extractExp, extractExp, fp8NanRegTensor, cmpResult);
            AscendC::MicroAPI::Select<uint32_t>(extractExp, extractExp, zeroRegTensor32, zeroMask);
            AscendC::MicroAPI::Pack<uint16_t, uint32_t, AscendC::MicroAPI::HighLowPart::LOWEST>(expOut, extractExp);

            AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                AscendC::MicroAPI::StoreDist::DIST_PACK_B16>(mxScaleLocalAddr, expOut, CONST_32, maskHalf);

            AscendC::MicroAPI::ShiftLefts(extractExp, extractExp, SHR_NUM_FOR_BF16, preMaskScale);
            AscendC::MicroAPI::Sub(halfScale, scaleBias, extractExp, preMaskScale);
            AscendC::MicroAPI::Select<uint32_t>(halfScale, halfScale, nanRegTensor, cmpResult);
            AscendC::MicroAPI::Select<uint32_t>(halfScale, halfScale, zeroRegTensor32, zeroMask);
            AscendC::MicroAPI::Pack<uint16_t, uint32_t, AscendC::MicroAPI::HighLowPart::LOWEST>(recExpOut, halfScale);

            AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(halfScaleLocalAddr,
                recExpOut, onceNum, mask);
        }
    }
}

template <typename T, typename U, typename T_IDX, bool isGroupIndex>
__aicore__ inline void SwigluMxQuantLastLast<T, U, T_IDX, isGroupIndex>::ComputeVfMaxExpVf(__local_mem__ T *srcAddr,
    __local_mem__ uint16_t *maxExpAddr, int64_t dim0OnceSize, int64_t dim1OnceSize, int64_t alignDim1Size)
{
    uint32_t totalCountInUB = dim0OnceSize * alignDim1Size;
    uint16_t loopNum = CeilDivision(totalCountInUB, QUANT_ONCE_NUM);
    uint16_t invalidFp16 = INVALID_FLOAT16;
    uint16_t maxExpbf16 = MAX_EXP_FOR_BF16;
    int64_t onceNum = QUANT_ONCE_NUM;
    int64_t scaleNum = SCALE_ONCE_NUM;
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
        AscendC::MicroAPI::Duplicate(expMaskBF16, maxExpbf16);
        AscendC::MicroAPI::RegTensor<uint16_t> invalidmaskfp16;
        AscendC::MicroAPI::Duplicate(invalidmaskfp16, invalidFp16);
        AscendC::MicroAPI::RegTensor<uint16_t> vdMaxExp;
        AscendC::MicroAPI::MaskReg scaleMask1;
        AscendC::MicroAPI::MaskReg scaleMask2;
        AscendC::MicroAPI::MaskReg invalidDataMask0;
        AscendC::MicroAPI::MaskReg invalidDataMask1;
        AscendC::MicroAPI::UnalignReg u1;
        for (uint16_t i = 0; i < loopNum; i++) {
            scaleMask1 = AscendC::MicroAPI::UpdateMask<T>(totalCountInUB);
            scaleMask2 = AscendC::MicroAPI::UpdateMask<T>(totalCountInUB);
            AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                AscendC::MicroAPI::LoadDist::DIST_DINTLV_B16>(vdExp0, vdExp1, srcAddr, onceNum);

            if constexpr (IsSame<T, half>::value) {
                AscendC::MicroAPI::And(vdExpSelect0, (AscendC::MicroAPI::RegTensor<uint16_t> &)vdExp0, invalidmaskfp16,
                    scaleMask1);
                AscendC::MicroAPI::And(vdExpSelect1, (AscendC::MicroAPI::RegTensor<uint16_t> &)vdExp1, invalidmaskfp16,
                    scaleMask1);
                AscendC::MicroAPI::Compare<uint16_t, CMPMODE::NE>(invalidDataMask0, vdExpSelect0, invalidmaskfp16,
                    scaleMask1);
                AscendC::MicroAPI::Compare<uint16_t, CMPMODE::NE>(invalidDataMask1, vdExpSelect1, invalidmaskfp16,
                    scaleMask1);
                AscendC::MicroAPI::Cast<bfloat16_t, T, CAST_HALF_TO_BF16>(vdExp0BF16, vdExp0, scaleMask1);
                AscendC::MicroAPI::Cast<bfloat16_t, T, CAST_HALF_TO_BF16>(vdExp1BF16, vdExp1, scaleMask1);
                AscendC::MicroAPI::And(vdExpExtract0, (AscendC::MicroAPI::RegTensor<uint16_t> &)vdExp0BF16, expMaskBF16,
                    scaleMask1);
                AscendC::MicroAPI::And(vdExpExtract1, (AscendC::MicroAPI::RegTensor<uint16_t> &)vdExp1BF16, expMaskBF16,
                    scaleMask1);
                AscendC::MicroAPI::Select<uint16_t>(vdExpExtract0, vdExpExtract0, expMaskBF16, invalidDataMask0);
                AscendC::MicroAPI::Select<uint16_t>(vdExpExtract1, vdExpExtract1, expMaskBF16, invalidDataMask1);
            } else {
                AscendC::MicroAPI::And(vdExpExtract0, (AscendC::MicroAPI::RegTensor<uint16_t> &)vdExp0, expMaskBF16,
                    scaleMask1);
                AscendC::MicroAPI::And(vdExpExtract1, (AscendC::MicroAPI::RegTensor<uint16_t> &)vdExp1, expMaskBF16,
                    scaleMask1);
            }

            AscendC::MicroAPI::Max(vdMaxExp, vdExpExtract0, vdExpExtract1, scaleMask1);
            AscendC::MicroAPI::ReduceMaxWithDataBlock(vdMaxExp, vdMaxExp, scaleMask1);
            AscendC::MicroAPI::DataCopyUnAlign<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(maxExpAddr,
                vdMaxExp, u1, scaleNum);
        }
        AscendC::MicroAPI::DataCopyUnAlignPost(maxExpAddr, u1, 0);
    }
}

template <typename T, typename U, typename T_IDX, bool isGroupIndex>
__aicore__ inline void SwigluMxQuantLastLast<T, U, T_IDX, isGroupIndex>::ComputeVfMaxExpVfBLAS(__local_mem__ T *srcAddr,
    __local_mem__ uint16_t *maxExpAddr, int64_t dim0OnceSize, int64_t dim1OnceSize, int64_t alignDim1Size)
{
    uint32_t totalCountInUB = dim0OnceSize * alignDim1Size;
    uint16_t loopNum = CeilDivision(totalCountInUB, QUANT_ONCE_NUM);
    int64_t onceNum = QUANT_ONCE_NUM;
    int64_t scaleNum = SCALE_ONCE_NUM;
    uint16_t absMaskFor16Bit = ABS_MASK_FOR_16BIT;
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<T> vdExp0;
        AscendC::MicroAPI::RegTensor<T> vdExp1;
        AscendC::MicroAPI::RegTensor<uint16_t> absMask16Bit;
        AscendC::MicroAPI::Duplicate(absMask16Bit, absMaskFor16Bit);
        AscendC::MicroAPI::RegTensor<uint16_t> vdMaxExp;
        AscendC::MicroAPI::MaskReg scaleMask1;
        AscendC::MicroAPI::MaskReg scaleMask2;
        AscendC::MicroAPI::UnalignReg u1;
        for (uint16_t i = 0; i < loopNum; i++) {
            scaleMask1 = AscendC::MicroAPI::UpdateMask<T>(totalCountInUB);
            scaleMask2 = AscendC::MicroAPI::UpdateMask<T>(totalCountInUB);
            AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                AscendC::MicroAPI::LoadDist::DIST_DINTLV_B16>(vdExp0, vdExp1, srcAddr, onceNum);
            AscendC::MicroAPI::And((AscendC::MicroAPI::RegTensor<uint16_t> &)vdExp0,
                (AscendC::MicroAPI::RegTensor<uint16_t> &)vdExp0, absMask16Bit, scaleMask1);
            AscendC::MicroAPI::And((AscendC::MicroAPI::RegTensor<uint16_t> &)vdExp1,
                (AscendC::MicroAPI::RegTensor<uint16_t> &)vdExp1, absMask16Bit, scaleMask1);
            AscendC::MicroAPI::Max(vdMaxExp, (AscendC::MicroAPI::RegTensor<uint16_t> &)vdExp0,
                (AscendC::MicroAPI::RegTensor<uint16_t> &)vdExp1, scaleMask1);
            AscendC::MicroAPI::ReduceMaxWithDataBlock(vdMaxExp, vdMaxExp, scaleMask1);
            AscendC::MicroAPI::DataCopyUnAlign<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(maxExpAddr,
                vdMaxExp, u1, scaleNum);
        }
        AscendC::MicroAPI::DataCopyUnAlignPost(maxExpAddr, u1, 0);
    }
}

template <typename T, typename U, typename T_IDX, bool isGroupIndex>
__aicore__ inline void SwigluMxQuantLastLast<T, U, T_IDX, isGroupIndex>::ComputeVfSwigluV2(__local_mem__ T *x1UbAddr,
    __local_mem__ T *x2UbAddr, __local_mem__ T *swigluUbAddr, int64_t dim0OnceSize, int64_t dim1OnceSize,
    int64_t dim1AlignSize, bool isTailDim1)
{
    // 在计算swiglu时需把pad 0做了
    uint32_t alignDim1In = dim1OnceSize * CONST_2;
    uint32_t alignDim1Out = dim1AlignSize;
    uint16_t dim1VfTimes = alignDim1In / vfLenT_;
    uint16_t dim0VfTimes = static_cast<uint16_t>(dim0OnceSize);
    uint32_t dim1Tail = alignDim1In % vfLenT_; // 17
    uint16_t dim1TailTimes = 0;
    uint16_t dim1Tail2 = 0;
    uint32_t mask1Num = 0;
    uint32_t mask2Num = 0;
    uint32_t mask3Num = 0;
    T numZero = 0;
    float negScalarOne = -1.0f;
    float clampLimit = clampLimit_;
    float negClampLimit = -clampLimit_;
    float negAplha = -gluAlpha_;
    float scalarOne = 1.0f;
    float gluBias = gluBias_;
    auto x1UbAddr1 = x1UbAddr;
    auto x2UbAddr1 = x2UbAddr;
    auto swigluUbAddr1 = swigluUbAddr;
    auto swigluUbAddr2 = swigluUbAddr;
    if (isTailDim1 && dim1Tail > 0) {
        alignDim1In = ((alignDim1In + oneBlockNum_ - 1) / oneBlockNum_) * oneBlockNum_;
        dim1TailTimes = 1;
        mask1Num = dim1Tail / CONST_2; //  搬出只有一半的数
        uint32_t padNum = alignDim1Out - dim1VfTimes * vfLenFp32_;
        if (padNum <= vfLenFp32_) {
            mask2Num = padNum;
        } else {
            dim1Tail2 = 1;
            mask2Num = vfLenFp32_;
            mask3Num = padNum - vfLenFp32_;
        }
        x1UbAddr1 = x1UbAddr + dim1VfTimes * vfLenT_;
        x2UbAddr1 = x2UbAddr + dim1VfTimes * vfLenT_;
        swigluUbAddr1 = swigluUbAddr + dim1VfTimes * vfLenFp32_;
        swigluUbAddr2 = swigluUbAddr + dim1VfTimes * vfLenFp32_ + dim1TailTimes * vfLenFp32_;
    }
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
        AscendC::MicroAPI::MaskReg mask1 = AscendC::MicroAPI::UpdateMask<float>(mask1Num);
        AscendC::MicroAPI::MaskReg mask2 = AscendC::MicroAPI::UpdateMask<float>(mask2Num);
        AscendC::MicroAPI::MaskReg mask3 = AscendC::MicroAPI::UpdateMask<T>(mask3Num);
        for (uint16_t dim0vfLoopIdx = 0; dim0vfLoopIdx < dim0VfTimes; dim0vfLoopIdx++) {
            for (uint16_t dim1vfLoopIdx = 0; dim1vfLoopIdx < dim1VfTimes; dim1vfLoopIdx++) {
                AscendC::MicroAPI::AddrReg srcIdxOffset =
                    AscendC::MicroAPI::CreateAddrReg<T>(dim0vfLoopIdx, alignDim1In, dim1vfLoopIdx, 128);
                AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(vregX1, x1UbAddr,
                    srcIdxOffset);
                AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(vregX2, x2UbAddr,
                    srcIdxOffset);
                AscendC::MicroAPI::Cast<float, T, CAST_BF16_FP16_TO_FP32>(vregX1F, vregX1, mask); // 64
                AscendC::MicroAPI::Cast<float, T, CAST_BF16_FP16_TO_FP32>(vregX2F, vregX2, mask); // // 64

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
                AscendC::MicroAPI::AddrReg outOffset =
                    AscendC::MicroAPI::CreateAddrReg<T>(dim0vfLoopIdx, alignDim1Out, dim1vfLoopIdx, 64);
                DataCopy<T, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(swigluUbAddr, outTReg, outOffset, mask);
            }
            AscendC::MicroAPI::AddrReg srcIdxOffset1 = AscendC::MicroAPI::CreateAddrReg<T>(dim0vfLoopIdx, alignDim1In);
            AscendC::MicroAPI::AddrReg outOffset1 = AscendC::MicroAPI::CreateAddrReg<T>(dim0vfLoopIdx, alignDim1Out);
            for (uint16_t aa = 0; aa < dim1TailTimes; aa++) {
                AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(vregX1, x1UbAddr1,
                    srcIdxOffset1);
                AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(vregX2, x2UbAddr1,
                    srcIdxOffset1);
                AscendC::MicroAPI::Cast<float, T, CAST_BF16_FP16_TO_FP32>(vregX1F, vregX1, mask); // 64
                AscendC::MicroAPI::Cast<float, T, CAST_BF16_FP16_TO_FP32>(vregX2F, vregX2, mask); // // 64

                AscendC::MicroAPI::DeInterleave(vregX1DeF, vregX2DeF, vregX1F, vregX2F);
                AscendC::MicroAPI::Mins(minsReg, vregX1DeF, clampLimit, mask1);
                AscendC::MicroAPI::Muls(mulsReg, minsReg, negAplha, mask1);
                AscendC::MicroAPI::Exp(expReg, mulsReg, mask1);
                AscendC::MicroAPI::Adds(addsReg, expReg, scalarOne, mask1);
                AscendC::MicroAPI::Div(sigmoidReg, minsReg, addsReg, mask1);

                AscendC::MicroAPI::Mins(vregX2DeF, vregX2DeF, clampLimit, mask1);
                AscendC::MicroAPI::Maxs(vregX2DeF, vregX2DeF, negClampLimit, mask1);
                AscendC::MicroAPI::Adds(vregX2DeF, vregX2DeF, gluBias, mask1);

                AscendC::MicroAPI::Mul(outFReg, sigmoidReg, vregX2DeF, mask1);
                AscendC::MicroAPI::Cast<T, float, CAST_FP32_TO_FP16_BF16>(outTReg, outFReg, mask1);
                DataCopy<T, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(swigluUbAddr1, outTReg, outOffset1, mask2);
            }
            for (uint16_t cc = 0; cc < dim1Tail2; cc++) {
                Duplicate<T>(vregX1, numZero);
                DataCopy<T>(swigluUbAddr2, vregX1, outOffset1, mask3);
            }
        }
    }
}

template <typename T, typename U, typename T_IDX, bool isGroupIndex>
__aicore__ inline void SwigluMxQuantLastLast<T, U, T_IDX, isGroupIndex>::ComputeVfSwigluV1(__local_mem__ T *x1UbAddr,
    __local_mem__ T *x2UbAddr, __local_mem__ T *swigluUbAddr, int64_t dim0OnceSize, int64_t dim1OnceSize,
    int64_t dim1AlignSize, bool isTailDim1)
{
    // 在计算swiglu时需把pad 0做了
    uint16_t dim0VfTimes = dim0OnceSize;
    uint16_t dim1VfTimes = dim1OnceSize / vfLenFp32_;
    uint32_t dim1Tail = dim1OnceSize % vfLenFp32_;
    uint16_t dim1TailTimes = 0;
    uint16_t dim1Tail2 = 0;
    uint32_t mask1Num = 0;
    uint32_t mask2Num = 0;
    uint32_t mask3Num = 0;
    uint32_t alignDim1In = dim1OnceSize;
    uint32_t alignDim1Out = dim1AlignSize;
    auto x1UbAddr1 = x1UbAddr;
    auto x2UbAddr1 = x2UbAddr;
    auto swigluUbAddr1 = swigluUbAddr;
    auto swigluUbAddr2 = swigluUbAddr;
    T numZero = 0;
    if (isTailDim1 && dim1Tail > 0) {
        mask1Num = dim1Tail;
        dim1TailTimes = 1;
        alignDim1In = ((dim1OnceSize + oneBlockNum_ - 1) / oneBlockNum_) * oneBlockNum_;
        uint32_t padNum = alignDim1Out - dim1VfTimes * vfLenFp32_;
        if (padNum <= vfLenFp32_) {
            mask2Num = padNum;
        } else {
            dim1Tail2 = 1;
            mask2Num = vfLenFp32_;
            mask3Num = padNum - vfLenFp32_;
        }
        int32_t offsetAlgin = dim1VfTimes * vfLenFp32_;
        x1UbAddr1 = x1UbAddr + offsetAlgin;
        x2UbAddr1 = x2UbAddr + offsetAlgin;
        swigluUbAddr1 = swigluUbAddr + offsetAlgin;
        swigluUbAddr2 = swigluUbAddr + offsetAlgin + dim1TailTimes * vfLenFp32_;
    }
    float scalarOne = 1.0f;
    float negScalarOne = -1.0f;
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<T> vregX1;
        AscendC::MicroAPI::RegTensor<T> vregX2;
        AscendC::MicroAPI::RegTensor<float> vregX1F;
        AscendC::MicroAPI::RegTensor<float> vregX2F;
        AscendC::MicroAPI::RegTensor<float> negReg;
        AscendC::MicroAPI::RegTensor<float> expReg;
        AscendC::MicroAPI::RegTensor<float> addsReg;
        AscendC::MicroAPI::RegTensor<float> sigmoidReg;
        AscendC::MicroAPI::RegTensor<float> outFReg;
        AscendC::MicroAPI::RegTensor<T> outTReg;
        AscendC::MicroAPI::MaskReg mask = AscendC::MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::MaskReg mask1 = AscendC::MicroAPI::UpdateMask<float>(mask1Num);
        AscendC::MicroAPI::MaskReg mask2 = AscendC::MicroAPI::UpdateMask<float>(mask2Num);
        AscendC::MicroAPI::MaskReg mask3 = AscendC::MicroAPI::UpdateMask<T>(mask3Num);
        for (uint16_t dim0vfLoopIdx = 0; dim0vfLoopIdx < dim0VfTimes; dim0vfLoopIdx++) {
            for (uint16_t dim1vfLoopIdx = 0; dim1vfLoopIdx < dim1VfTimes; dim1vfLoopIdx++) {
                AscendC::MicroAPI::AddrReg srcIdxOffset =
                    AscendC::MicroAPI::CreateAddrReg<T>(dim0vfLoopIdx, alignDim1In, dim1vfLoopIdx, 64);
                AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(vregX1, x1UbAddr,
                    srcIdxOffset);
                AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(vregX2, x2UbAddr,
                    srcIdxOffset);
                AscendC::MicroAPI::Cast<float, T, CAST_BF16_FP16_TO_FP32>(vregX1F, vregX1, mask);
                AscendC::MicroAPI::Cast<float, T, CAST_BF16_FP16_TO_FP32>(vregX2F, vregX2, mask);

                AscendC::MicroAPI::Muls(negReg, vregX1F, negScalarOne, mask);
                AscendC::MicroAPI::Exp(expReg, negReg, mask);
                AscendC::MicroAPI::Adds(addsReg, expReg, scalarOne, mask);
                AscendC::MicroAPI::Div(sigmoidReg, vregX1F, addsReg, mask);
                AscendC::MicroAPI::Mul(outFReg, sigmoidReg, vregX2F, mask);

                AscendC::MicroAPI::Cast<T, float, CAST_FP32_TO_FP16_BF16>(outTReg, outFReg, mask);
                AscendC::MicroAPI::AddrReg outOffset =
                    AscendC::MicroAPI::CreateAddrReg<T>(dim0vfLoopIdx, alignDim1Out, dim1vfLoopIdx, 64);
                DataCopy<T, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(swigluUbAddr, outTReg, outOffset, mask);
            }
            AscendC::MicroAPI::AddrReg srcIdxOffset1 = AscendC::MicroAPI::CreateAddrReg<T>(dim0vfLoopIdx, alignDim1In);
            AscendC::MicroAPI::AddrReg outOffset1 = AscendC::MicroAPI::CreateAddrReg<T>(dim0vfLoopIdx, alignDim1Out);
            for (uint16_t aa = 0; aa < dim1TailTimes; aa++) {
                AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(vregX1, x1UbAddr1,
                    srcIdxOffset1);
                AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(vregX2, x2UbAddr1,
                    srcIdxOffset1);
                AscendC::MicroAPI::Cast<float, T, CAST_BF16_FP16_TO_FP32>(vregX1F, vregX1, mask1);
                AscendC::MicroAPI::Cast<float, T, CAST_BF16_FP16_TO_FP32>(vregX2F, vregX2, mask1);

                AscendC::MicroAPI::Muls(negReg, vregX1F, negScalarOne, mask1);
                AscendC::MicroAPI::Exp(expReg, negReg, mask1);
                AscendC::MicroAPI::Adds(addsReg, expReg, scalarOne, mask1);
                AscendC::MicroAPI::Div(sigmoidReg, vregX1F, addsReg, mask1);
                AscendC::MicroAPI::Mul(outFReg, sigmoidReg, vregX2F, mask1);

                AscendC::MicroAPI::Cast<T, float, CAST_FP32_TO_FP16_BF16>(outTReg, outFReg, mask1);
                DataCopy<T, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(swigluUbAddr1, outTReg, outOffset1, mask2);
            }
            for (uint16_t cc = 0; cc < dim1Tail2; cc++) {
                Duplicate<T>(vregX1, numZero);
                DataCopy<T>(swigluUbAddr2, vregX1, outOffset1, mask3);
            }
        }
    }
}

template <typename T, typename U, typename T_IDX, bool isGroupIndex>
__aicore__ inline void SwigluMxQuantLastLast<T, U, T_IDX, isGroupIndex>::CopyIn(int64_t dim0LoopIdx,
    int64_t dim1LoopIdx, int64_t dim0OnceSize, int64_t dim1OnceSize)
{
    LocalTensor<T> xlocal = inQuex_.AllocTensor<T>();
    DataCopyExtParams copyInParam = { 0, 0, 0, 0, 0 };
    DataCopyPadExtParams<T> copyPadParams = { false, 0, 0, 0 };
    copyInParam.blockCount = dim0OnceSize;
    copyInParam.blockLen = dim1OnceSize * sizeof(T);
    if (swigluMode_ == 0) {
        int64_t offset = (blockOffset_ + dim0LoopIdx * factorDim0Size_) * dim1Size_ + dim1LoopIdx * factorDim1Size_ * QUANT_ONCE_NUM;
        copyInParam.srcStride = (dim1Size_ - dim1OnceSize) * sizeof(T);
        DataCopyPad(xlocal, xGm_[offset], copyInParam, copyPadParams);
        DataCopyPad(xlocal[factorDim0Size_ * factorDim1Size_ * QUANT_ONCE_NUM], xGm_[offset + halfInput_], copyInParam,
            copyPadParams);
    } else {
        int64_t offset = (blockOffset_ + dim0LoopIdx * factorDim0Size_) * dim1Size_ + dim1LoopIdx * factorDim1Size_ * X_ONCE_NUM;
        copyInParam.blockCount = dim0OnceSize;
        copyInParam.blockLen = dim1OnceSize * CONST_2 * sizeof(T);
        copyInParam.srcStride = (dim1Size_ - dim1OnceSize * CONST_2) * sizeof(T);
        DataCopyPad(xlocal, xGm_[offset], copyInParam, copyPadParams);
    }
    inQuex_.EnQue(xlocal);
}

template <typename T, typename U, typename T_IDX, bool isGroupIndex>
__aicore__ inline void SwigluMxQuantLastLast<T, U, T_IDX, isGroupIndex>::CopyOut(int64_t dim0LoopIdx,
    int64_t dim1LoopIdx, int64_t dim0OnceSize, int64_t dim1OnceSize, int64_t dim1OnceSizeAlgin)
{
    LocalTensor<uint8_t> mxScaleLocal = outQueScale_.DeQue<uint8_t>();
    LocalTensor<uint8_t> outLocal = outQuey_.DeQue<uint8_t>();
    DataCopyExtParams copyOutParamData = { 0, 0, 0, 0, 0 };
    copyOutParamData.blockCount = dim0OnceSize;
    int64_t offset = 0;
    if constexpr (IsSame<U, fp4x2_e2m1_t>::value || IsSame<U, fp4x2_e1m2_t>::value) {
        copyOutParamData.blockLen = dim1OnceSize / CONST_2;
        copyOutParamData.srcStride = (dim1OnceSizeAlgin / CONST_2 - copyOutParamData.blockLen) /
            oneBlockUb_; // ub需要一个block对齐,除以32就是为了计算有多少个block
        copyOutParamData.dstStride = dim1Size_ / CONST_4 - copyOutParamData.blockLen;
        offset =
            (blockOffset_ + dim0LoopIdx * factorDim0Size_) * dim1Size_ / CONST_4 + dim1LoopIdx * factorDim1Size_ * QUANT_ONCE_NUM / CONST_2;
    } else {
        copyOutParamData.blockLen = dim1OnceSize;
        copyOutParamData.srcStride = (dim1OnceSizeAlgin - copyOutParamData.blockLen) /
            oneBlockUb_; // ub需要一个block对齐,除以32就是为了计算有多少个block
        copyOutParamData.dstStride = halfInput_ - copyOutParamData.blockLen;
        offset = (blockOffset_ + dim0LoopIdx * factorDim0Size_) * halfInput_ + dim1LoopIdx * factorDim1Size_ * QUANT_ONCE_NUM;
    }
    DataCopyPad(yGm_[offset], outLocal, copyOutParamData);

    DataCopyExtParams copyOutParamScale = { 0, 0, 0, 0, 0 };
    uint32_t usedFactorDim1 = dim1OnceSizeAlgin / oneBlockUb_;
    copyOutParamScale.blockCount = dim0OnceSize;
    copyOutParamScale.blockLen = usedFactorDim1;
    copyOutParamScale.srcStride = 0;
    copyOutParamScale.dstStride = outputScaleRowBytes_ - copyOutParamScale.blockLen;
    int64_t offsetScale =
        (blockOffset_ + dim0LoopIdx * factorDim0Size_) * outputScaleRowBytes_ + dim1LoopIdx * factorDim1Size_ * SCALE_ONCE_NUM;
    DataCopyPad<uint8_t, PaddingMode::Compact>(scaleGm_[offsetScale], mxScaleLocal, copyOutParamScale);
    outQuey_.FreeTensor(outLocal);
    outQueScale_.FreeTensor(mxScaleLocal);
}
} // namespace SwigluMxQuant
#endif // SWIGLU_MX_QUANT_LAST_LAST_H
