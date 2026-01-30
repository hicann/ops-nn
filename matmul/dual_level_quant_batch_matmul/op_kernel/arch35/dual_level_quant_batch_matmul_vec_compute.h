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
 * \file dual_level_quant_batch_matmul_vec_compute.h
 * \brief
 */
#ifndef DUAL_LEVEL_QUANT_BATCH_MATMUL_VEC_COMPUTE_H
#define DUAL_LEVEL_QUANT_BATCH_MATMUL_VEC_COMPUTE_H

#include "kernel_basic_intf.h"
#include "dual_level_quant_batch_matmul_vf.h"
#include "op_kernel/math_util.h"
#include "tool_arch35.h"

using AscendC::GetSubBlockIdx;
using AscendC::HardEvent;
using AscendC::SetFlag;
using AscendC::TPosition;
using AscendC::WaitFlag;

namespace DualLevelQuantBatchMatmul::Arch35 {
#define DLQBMM_VEC_COMPUTE_TEMPLATE_PARAM \
    template <typename x1Level0ScaleType, typename x2Level0ScaleType, typename biasType, typename yType, bool hasBias>

#define DLQBMM_VEC_COMPUTE_CLASS \
    DualLevelQuantBatchMatmulVectorCompute<x1Level0ScaleType, x2Level0ScaleType, biasType, yType, hasBias>

DLQBMM_VEC_COMPUTE_TEMPLATE_PARAM
class DualLevelQuantBatchMatmulVectorCompute {
public:
    __aicore__ inline DualLevelQuantBatchMatmulVectorCompute(){};

    __aicore__ inline void Init(
        __gm__ x1Level0ScaleType* x1Level0Scale, __gm__ x2Level0ScaleType* x2Level0Scale, __gm__ biasType* bias,
        __gm__ yType* y);
    __aicore__ inline void CopyX1Level0ScaleToUb(
        const DualLevelQbmmBasicBlockOffsetParams& basicBlockParam, const L0CopyAndCalcParams& l0Params);
    __aicore__ inline void CopyX2Level0ScaleAndBiasToUb(
        const DualLevelQbmmBasicBlockOffsetParams& basicBlockParam, const L0CopyAndCalcParams& l0Params);
    __aicore__ inline void InitUb(uint64_t cFp32BufId, uint64_t nL1Offset);
    __aicore__ inline void AntiQuantCompute(
        uint64_t cFp32BufId, uint64_t cvLoopIdx, uint64_t mL1Offset, uint64_t nL1Offset, LocalTensor<float>& cTmpFp32Ub,
        const DualLevelQbmmBasicBlockOffsetParams& basicBlockParam, const L0CopyAndCalcParams& l0Params);
    __aicore__ inline void CopyUbToGm(
        uint64_t cFp32BufId, uint64_t mL1Offset, uint64_t nL1Offset,
        const DualLevelQbmmBasicBlockOffsetParams& basicBlockParam, const L0CopyAndCalcParams& l0Params);

    __aicore__ inline void WaitVectorFlag();
    __aicore__ inline void SetMte3ToV(uint64_t cFp32BufId);
    __aicore__ inline void WaitMte3ToV(uint64_t cFp32BufId);
    __aicore__ inline void SetAndWaitMte2ToV();
    __aicore__ inline void WaitVToMte2ForX1();
    __aicore__ inline void SetVToMte2ForX1();
    __aicore__ inline void WaitVToMte2ForX2();
    __aicore__ inline void SetVToMTE2ForX2();

private:
    static constexpr uint64_t CV_LOOP_NUM = 2;
    static constexpr uint64_t C_TMP_FP32_UB_OFFSET = 64 * 128;
    static constexpr uint64_t SINGLE_REG_PROCESS_SIZE = 64;

    static constexpr uint64_t EVENT_V_MTE2_ID_X1 = 3;
    static constexpr uint64_t EVENT_V_MTE2_ID_X2 = 0;
    static constexpr uint64_t EVENT_MTE3_V_ID = 0;
    static constexpr uint64_t EVENT_V_MTE3_ID = 0;
    static constexpr uint64_t EVENT_MTE2_V_ID = 0;

    static constexpr uint64_t BIAS_B32_OFFSET = 256;
    static constexpr uint64_t B_SCALE_B32_OFFSET = 256;
    static constexpr uint64_t X1_SCALE_B32_OFFSET = 128 * 32;
    static constexpr uint64_t C_B32_OFFSET = 128 * 128; // n方向循环将累加存储区域切为两块，每一块的size为128 * 128

    static constexpr uint64_t X1_LEVEL0_SCALE_B32_OFFSET = 64 * 32;
    static constexpr uint64_t X1_LEVEL0_SCALE_B32_STORE_INNER_SIZE = 32; // x1Level0ScaleUB上存放数据的内轴最大长度
    uint64_t mte2BufIdx_ = 0;
    uint64_t mte2X1Level0ScaleBufIdx_ = 0;

    GlobalTensor<float> x1Level0ScaleB32Gm_;
    GlobalTensor<float> x2Level0ScaleB32Gm_;
    GlobalTensor<float> biasB32Gm_;
    GlobalTensor<yType> yGm_;

    LocalTensor<float> x1Level0ScaleB32Ub_{TPosition::VECIN, 84 * 1024, 32 * 1024}; // shape(128,32) * 2 , 16KB * 2;
    LocalTensor<float> x2Level0ScaleB32Ub_{TPosition::VECIN, 116 * 1024, 2 * 1024}; // shape(256,) * 2 , 1k* 2;
    LocalTensor<float> biasB32Ub_{TPosition::VECIN, 118 * 1024, 2 * 1024};          // shape(256,) * 2 , 1k* 2;
    LocalTensor<float> cFp32Ub_{TPosition::VECIN, 120 * 1024, 128 * 1024};          // shape(128, 256) * 2 , 128KB;
};

DLQBMM_VEC_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void DLQBMM_VEC_COMPUTE_CLASS::Init(
    __gm__ x1Level0ScaleType* x1Level0Scale, __gm__ x2Level0ScaleType* x2Level0Scale, __gm__ biasType* bias,
    __gm__ yType* y)
{
    x1Level0ScaleB32Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x1Level0Scale));
    x2Level0ScaleB32Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x2Level0Scale));
    biasB32Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(bias));
    yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ yType*>(y));

    for (uint64_t i = 0; i < DOUBLE_BUFFER_NUM; i++) {
        SetFlag<HardEvent::V_MTE2>(EVENT_V_MTE2_ID_X2 + i);
        SetFlag<HardEvent::MTE3_V>(EVENT_MTE3_V_ID + i);
    }
    SetFlag<HardEvent::V_MTE2>(EVENT_V_MTE2_ID_X1);
}

DLQBMM_VEC_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void DLQBMM_VEC_COMPUTE_CLASS::CopyX1Level0ScaleToUb(
    const DualLevelQbmmBasicBlockOffsetParams& basicBlockParam, const L0CopyAndCalcParams& l0Params)
{
    uint64_t vectorShiftOffset =
        GetSubBlockIdx() * basicBlockParam.mUbSize * basicBlockParam.level0ScaleGmGroupNum; // 不同核的偏移量
    uint64_t x1Level0ScaleGroupNum =
        basicBlockParam.kGmOffset + basicBlockParam.level0ScaleKUbSize > basicBlockParam.kSize ?
            Ops::Base::CeilDiv(basicBlockParam.kSize - basicBlockParam.kGmOffset, basicBlockParam.level0GroupSize) :
            Ops::Base::CeilDiv(basicBlockParam.level0ScaleKUbSize, basicBlockParam.level0GroupSize);

    if (basicBlockParam.level0ScaleKUbSize > basicBlockParam.kSize) {
        uint64_t x1Level0ScaleBlockLen = basicBlockParam.mUbSize * x1Level0ScaleGroupNum;
        DataCopyPad2D<float>(
            x1Level0ScaleB32Ub_[mte2X1Level0ScaleBufIdx_ % DOUBLE_BUFFER_NUM * X1_SCALE_B32_OFFSET],
            x1Level0ScaleB32Gm_[basicBlockParam.mGmOffset * basicBlockParam.level0ScaleGmGroupNum + vectorShiftOffset],
            1, x1Level0ScaleBlockLen, X1_SCALE_B32_OFFSET, x1Level0ScaleBlockLen);
        if (l0Params.mL0Size < l0Params.mL1Size) {
            // x1Level0ScaleBlockLen可能会导致地址32B非对齐
            // m方向存在第二轮循环时，第一轮会搬运64*x1Level0ScaleGroupNum，满足32B对齐；不存在第二轮循环时没有必要搬运
            DataCopyPad2D<float>(
                x1Level0ScaleB32Ub_
                    [mte2X1Level0ScaleBufIdx_ % DOUBLE_BUFFER_NUM * X1_SCALE_B32_OFFSET + x1Level0ScaleBlockLen],
                x1Level0ScaleB32Gm_
                    [basicBlockParam.mGmOffset * basicBlockParam.level0ScaleGmGroupNum + vectorShiftOffset +
                     l0Params.mL0Size * basicBlockParam.level0ScaleGmGroupNum],
                1, x1Level0ScaleBlockLen, X1_SCALE_B32_OFFSET, x1Level0ScaleBlockLen);
        }
    } else {
        DataCopyPad2D<float>(
            x1Level0ScaleB32Ub_[mte2X1Level0ScaleBufIdx_ % DOUBLE_BUFFER_NUM * X1_SCALE_B32_OFFSET],
            x1Level0ScaleB32Gm_
                [basicBlockParam.mGmOffset * basicBlockParam.level0ScaleGmGroupNum + vectorShiftOffset +
                 basicBlockParam.kGmOffset / basicBlockParam.level0GroupSize],
            basicBlockParam.mUbSize, x1Level0ScaleGroupNum, X1_LEVEL0_SCALE_B32_STORE_INNER_SIZE,
            basicBlockParam.level0ScaleGmGroupNum);
        if (l0Params.mL0Size < l0Params.mL1Size) {
            DataCopyPad2D<float>(
                x1Level0ScaleB32Ub_
                    [mte2X1Level0ScaleBufIdx_ % DOUBLE_BUFFER_NUM * X1_SCALE_B32_OFFSET + X1_LEVEL0_SCALE_B32_OFFSET],
                x1Level0ScaleB32Gm_
                    [basicBlockParam.mGmOffset * basicBlockParam.level0ScaleGmGroupNum + vectorShiftOffset +
                     l0Params.mL0Size * basicBlockParam.level0ScaleGmGroupNum +
                     basicBlockParam.kGmOffset / basicBlockParam.level0GroupSize],
                basicBlockParam.mUbSize, x1Level0ScaleGroupNum, X1_LEVEL0_SCALE_B32_STORE_INNER_SIZE,
                basicBlockParam.level0ScaleGmGroupNum);
        }
    }
}

DLQBMM_VEC_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void DLQBMM_VEC_COMPUTE_CLASS::CopyX2Level0ScaleAndBiasToUb(
    const DualLevelQbmmBasicBlockOffsetParams& basicBlockParam, const L0CopyAndCalcParams& l0Params)
{
    uint64_t ubX2ScaleBlockLen = Ops::Base::CeilDiv(l0Params.kL1Size, basicBlockParam.level0GroupSize);
    DataCopyPad2D<float>(
        x2Level0ScaleB32Ub_[mte2BufIdx_ % DOUBLE_BUFFER_NUM * B_SCALE_B32_OFFSET],
        x2Level0ScaleB32Gm_
            [Ops::Base::CeilDiv(basicBlockParam.kGmOffset, basicBlockParam.level0GroupSize) * basicBlockParam.nSize +
             basicBlockParam.nGmOffset],
        ubX2ScaleBlockLen, l0Params.nL1Size, basicBlockParam.nSize, B_SCALE_B32_OFFSET);

    if constexpr (hasBias) {
        if (basicBlockParam.kGmOffset != 0) {
            return;
        }
        // BIAS 的 N 轴的搬运量
        DataCopyPad2D<float>(
            biasB32Ub_[mte2BufIdx_ % DOUBLE_BUFFER_NUM * BIAS_B32_OFFSET], biasB32Gm_[basicBlockParam.nGmOffset], 1,
            l0Params.nL1Size, BIAS_B32_OFFSET, l0Params.nL1Size);
    }
}

DLQBMM_VEC_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void DLQBMM_VEC_COMPUTE_CLASS::InitUb(uint64_t cFp32BufId, uint64_t nL1Offset)
{
    if constexpr (hasBias) {
        // cFp32BufId是一个用于确定是左侧块还是右侧块的索引，以基本块为（256, 256）例，每一块的大小是(128,128)
        //  左侧           右侧
        //  (64,128)    (64,128)
        //  (64,128)    (64,128)
        // 4:bias的每个数据为4 byte; 512:每次处理(128 * 4)byte的数据
        InitUbToBias(
            C_B32_OFFSET * 4 / 512,
            (__ubuf__ float*)biasB32Ub_.GetPhyAddr(mte2BufIdx_ % DOUBLE_BUFFER_NUM * BIAS_B32_OFFSET + nL1Offset),
            (__ubuf__ float*)cFp32Ub_.GetPhyAddr(cFp32BufId * C_B32_OFFSET));
    } else {
        // 没有bias全填充成0
        // 4:bias的每个数据为4 byte; 256:每次处理(64 * 4)byte的数据
        InitUbToZero(
            C_B32_OFFSET * 4 / 256,
            (__ubuf__ int32_t*)cFp32Ub_.template ReinterpretCast<int32_t>().GetPhyAddr(cFp32BufId * C_B32_OFFSET));
    }
}

DLQBMM_VEC_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void DLQBMM_VEC_COMPUTE_CLASS::AntiQuantCompute(
    uint64_t cFp32BufId, uint64_t cvLoopIdx, uint64_t mL1Offset, uint64_t nL1Offset, LocalTensor<float>& cTmpFp32Ub,
    const DualLevelQbmmBasicBlockOffsetParams& basicBlockParam, const L0CopyAndCalcParams& l0Params)
{
    uint64_t groupNumIdx =
        basicBlockParam.kGmOffset / basicBlockParam.level0GroupSize % X1_LEVEL0_SCALE_B32_STORE_INNER_SIZE;
    // 2:表示在L0C到UB之前会将数据沿M轴切为两半，在计算UB上M轴的偏移量时候需要除2，向上取整是为了应对mL1Offset为奇数的情况
    uint64_t mUbOffset = Ops::Base::CeilDiv<uint64_t>(mL1Offset, 2);
    __ubuf__ float* cTmpFp32Addr =
        (__ubuf__ float*)cTmpFp32Ub.GetPhyAddr(cvLoopIdx % CV_LOOP_NUM * C_TMP_FP32_UB_OFFSET);
    __ubuf__ float* cFp32Addr =
        (__ubuf__ float*)cFp32Ub_.GetPhyAddr(cFp32BufId * C_B32_OFFSET + mUbOffset * L0C_BASE_N);
    __ubuf__ float* x2Level0ScaleAddr = (__ubuf__ float*)x2Level0ScaleB32Ub_.GetPhyAddr(
        mte2BufIdx_ % DOUBLE_BUFFER_NUM * B_SCALE_B32_OFFSET + nL1Offset);
    uint64_t x1ScaleGroupSizeNum = basicBlockParam.level0ScaleKUbSize > basicBlockParam.kSize ?
                                       basicBlockParam.level0ScaleGmGroupNum :
                                       X1_LEVEL0_SCALE_B32_STORE_INNER_SIZE;
    __ubuf__ float* x1Level0ScaleAddr = (__ubuf__ float*)x1Level0ScaleB32Ub_.GetPhyAddr(
        mte2X1Level0ScaleBufIdx_ % DOUBLE_BUFFER_NUM * X1_SCALE_B32_OFFSET + mUbOffset * x1ScaleGroupSizeNum +
        groupNumIdx);
    if (basicBlockParam.kGmOffset + basicBlockParam.l0BaseK < basicBlockParam.kSize) {
        if (l0Params.nL0Size <= SINGLE_REG_PROCESS_SIZE) {
            MulAdd<float>(
                cTmpFp32Addr, cFp32Addr, x1Level0ScaleAddr, x2Level0ScaleAddr, basicBlockParam.mUbSize,
                x1ScaleGroupSizeNum);
        } else {
            MulDoubleAdd<float>(
                cTmpFp32Addr, cFp32Addr, x1Level0ScaleAddr, x2Level0ScaleAddr, basicBlockParam.mUbSize,
                x1ScaleGroupSizeNum);
        }
    } else {
        if (l0Params.nL0Size <= SINGLE_REG_PROCESS_SIZE) {
            MulAdd<yType>(
                cTmpFp32Addr, cFp32Addr, x1Level0ScaleAddr, x2Level0ScaleAddr, basicBlockParam.mUbSize,
                x1ScaleGroupSizeNum);
        } else {
            MulDoubleAdd<yType>(
                cTmpFp32Addr, cFp32Addr, x1Level0ScaleAddr, x2Level0ScaleAddr, basicBlockParam.mUbSize,
                x1ScaleGroupSizeNum);
        }
    }
}

DLQBMM_VEC_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void DLQBMM_VEC_COMPUTE_CLASS::CopyUbToGm(
    uint64_t cFp32BufId, uint64_t mL1Offset, uint64_t nL1Offset,
    const DualLevelQbmmBasicBlockOffsetParams& basicBlockParam, const L0CopyAndCalcParams& l0Params)
{
    SetFlag<HardEvent::V_MTE3>(EVENT_V_MTE3_ID);
    WaitFlag<HardEvent::V_MTE3>(EVENT_V_MTE3_ID);

    uint64_t realML0Size =
        mL1Offset + l0Params.mL0Size > l0Params.mL1Size ? l0Params.mL1Size - l0Params.mL0Size : l0Params.mL0Size;
    uint64_t realNL0Size =
        nL1Offset + l0Params.nL0Size > l0Params.nL1Size ? l0Params.nL1Size - l0Params.nL0Size : l0Params.nL0Size;
    uint64_t mUbSize = Ops::Base::CeilDiv<uint64_t>(realML0Size, 2);
    uint64_t mOutSize = GetSubBlockIdx() == 0 ? mUbSize : realML0Size - mUbSize;

    if (realML0Size <= 0 || realNL0Size <= 0) {
        return;
    }
    constexpr uint64_t vec0MaxMSize = 64; // VEC0核最大值可取64，小于64则取其本身
    // 128:以下情况尾块的情况，需要单独进行参数设置
    if (realML0Size < 128 && l0Params.mL1Size > 128) {
        if (GetSubBlockIdx() == 0) {
            mOutSize = DualLevelQuantBatchMatmul::Arch35::Min<uint64_t>(vec0MaxMSize, realML0Size);
        } else {
            if (realML0Size <= vec0MaxMSize) {
                return;
            }
            mUbSize = vec0MaxMSize;
            mOutSize = realML0Size - mUbSize;
        }
    }

    // yGm_ M轴方向的偏移：GM上的偏移 + 基本块内的偏移量mL1Offset + 不同VEC所产生的偏移
    // yGm_ N轴方向的偏移：GM上的偏移 + 基本块内的偏移量nL1Offset
    // cFp32Ub_ 的偏移,以(256,256)基本快为例：cFp32BufId索引确定左侧还是右侧块，mL1Offset确定M方向的偏移量
    //  左侧           右侧
    //  (64,128)    (64,128)
    //  (64,128)    (64,128)
    if (mOutSize > 0) {
        // 2* ：此时cFp32Ub_上的数据被cast为bf16或fp16，所占数据亮缩小了1倍，因此在设置srcFullDim0参数时需要乘2
        DataCopyPad2D(
            yGm_
                [(basicBlockParam.mGmOffset + mL1Offset + GetSubBlockIdx() * mUbSize) * basicBlockParam.nSize +
                 nL1Offset + basicBlockParam.nGmOffset],
            cFp32Ub_[cFp32BufId * C_B32_OFFSET + L0C_BASE_N * mL1Offset / 2].template ReinterpretCast<yType>(),
            mOutSize, realNL0Size, 2 * L0C_BASE_N, basicBlockParam.nSize);
    }
}

DLQBMM_VEC_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void DLQBMM_VEC_COMPUTE_CLASS::SetMte3ToV(uint64_t cFp32BufId)
{
    SetFlag<HardEvent::MTE3_V>(EVENT_MTE3_V_ID + cFp32BufId);
}

DLQBMM_VEC_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void DLQBMM_VEC_COMPUTE_CLASS::WaitMte3ToV(uint64_t cFp32BufId)
{
    WaitFlag<HardEvent::MTE3_V>(EVENT_MTE3_V_ID + cFp32BufId);
}

DLQBMM_VEC_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void DLQBMM_VEC_COMPUTE_CLASS::WaitVToMte2ForX1()
{
    WaitFlag<HardEvent::V_MTE2>(EVENT_V_MTE2_ID_X1);
}

DLQBMM_VEC_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void DLQBMM_VEC_COMPUTE_CLASS::SetVToMte2ForX1()
{
    SetFlag<HardEvent::V_MTE2>(EVENT_V_MTE2_ID_X1);
    mte2X1Level0ScaleBufIdx_++;
}

DLQBMM_VEC_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void DLQBMM_VEC_COMPUTE_CLASS::WaitVToMte2ForX2()
{
    WaitFlag<HardEvent::V_MTE2>(EVENT_V_MTE2_ID_X2 + mte2BufIdx_ % DOUBLE_BUFFER_NUM);
}

DLQBMM_VEC_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void DLQBMM_VEC_COMPUTE_CLASS::SetVToMTE2ForX2()
{
    SetFlag<HardEvent::V_MTE2>(EVENT_V_MTE2_ID_X2 + mte2BufIdx_ % DOUBLE_BUFFER_NUM);
    mte2BufIdx_++;
}


DLQBMM_VEC_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void DLQBMM_VEC_COMPUTE_CLASS::WaitVectorFlag()
{
    for (uint64_t i = 0; i < DOUBLE_BUFFER_NUM; i++) {
        WaitFlag<HardEvent::V_MTE2>(EVENT_V_MTE2_ID_X2 + i);
        WaitFlag<HardEvent::MTE3_V>(EVENT_MTE3_V_ID + i);
    }
    WaitFlag<HardEvent::V_MTE2>(EVENT_V_MTE2_ID_X1);
}
DLQBMM_VEC_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void DLQBMM_VEC_COMPUTE_CLASS::SetAndWaitMte2ToV()
{
    SetFlag<HardEvent::MTE2_V>(EVENT_MTE2_V_ID);
    WaitFlag<HardEvent::MTE2_V>(EVENT_MTE2_V_ID);
}

} // namespace DualLevelQuantBatchMatmul::Arch35

#endif // DUAL_LEVEL_QUANT_BATCH_MATMUL_VEC_COMPUTE_H
