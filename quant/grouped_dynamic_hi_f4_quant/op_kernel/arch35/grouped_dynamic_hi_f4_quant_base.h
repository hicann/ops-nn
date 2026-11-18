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
 * \file grouped_dynamic_hi_f4_quant_base.h
 * \brief GroupedDynamicHiF4Quant 计算基类：ComputeMaxPer4 / ComputeMaxPer8 / ComputeE6M2 /
 *        ComputeTmpScale / ComputeOut 五段 VF 计算。
 */

#ifndef OPS_NN_GROUPED_DYNAMIC_HIF4_QUANT_BASE_H
#define OPS_NN_GROUPED_DYNAMIC_HIF4_QUANT_BASE_H

#define FLOAT_OVERFLOW_MODE_CTRL 60

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../inc/platform.h"
#include "../inc/kernel_utils.h"
#include "grouped_dynamic_hi_f4_quant_tilingdata.h"

namespace GroupedDynamicHiF4Quant {
using namespace AscendC;
constexpr int64_t BLOCK_SIZE = 64;
constexpr int64_t BYTE_OF_INPUT_X = 2;
constexpr int64_t BYTE_OF_INPUT_GROUP_INDEX = 4;
constexpr int64_t BYTE_OF_OUTPUT_SCALE = 4;
constexpr int64_t Y_NUM_OF_ONE_BYTE = 2;
constexpr int64_t DB_BUFFER = 2;
constexpr int64_t NUM_TWO = 2;
constexpr int64_t NUM_FOUR = 4;
constexpr int64_t NUM_EIGHT = 8;
constexpr int64_t NUM_SIXTENN = 16;
constexpr int64_t UB_BLOCK_SIZE = 32;
constexpr int16_t SHIFT_INDEX = 1;
// CastTrait 统一定义在 namespace 作用域：块内 static constexpr 定义在 host 侧仿真编译
// （openEuler gcc 12.3.1）实例化 Reg::Cast 时会触发编译器 ICE（tsubst segfault）
static constexpr Reg::CastTrait castTraitHalf2Bf16 = {Reg::RegLayout::UNKNOWN, Reg::SatMode::UNKNOWN,
                                                      Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
static constexpr Reg::CastTrait castTraitE6M2 = {Reg::RegLayout::ZERO, Reg::SatMode::SAT, Reg::MaskMergeMode::ZEROING,
                                                 RoundMode::CAST_RINT};
static constexpr Reg::CastTrait castTraitReciprocalCast = {Reg::RegLayout::ZERO, Reg::SatMode::SAT,
                                                           Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
static constexpr Reg::CastTrait castTraitBfloatToE2M1 = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                         Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

template <typename xDtype, typename yDtype>
class GroupedDynamicHiF4QuantBase {
public:
    __aicore__ inline GroupedDynamicHiF4QuantBase(){};

protected:
    __aicore__ inline void BaseInitData(const GroupedDynamicHiF4QuantTilingData* tilingData);
    // 计算每4个数最大值
    __aicore__ inline void ComputeMaxPer4(uint16_t maxPer4LoopNum, uint16_t tailMaxper4, __ubuf__ xDtype* xAddr,
                                          __ubuf__ uint16_t* maxPer4Addr);
    // 计算每8个数最大值
    __aicore__ inline void ComputeMaxPer8(uint16_t maxPer8LoopNum, uint16_t tailMaxper8, __ubuf__ uint16_t* maxPer4Addr,
                                          __ubuf__ uint16_t* maxPer8Addr);
    // 计算每64个数最大值，并求出E6M2
    __aicore__ inline void ComputeE6M2(uint16_t maxPer64LoopNum, uint16_t tail8Maxper64, __ubuf__ uint16_t* maxPer8Addr,
                                       __ubuf__ uint16_t* eaAddr);
    // 计算 tmpScale（量化系数）并拼装输出 scale（Eb/Ec 掩码 masked Add 进位平面累加）
    __aicore__ inline void ComputeTmpScale(uint16_t maxPer64LoopNum, uint16_t tailFullLoop, uint16_t tailPartM4,
                                           __ubuf__ uint16_t* maxPer4Addr, __ubuf__ uint16_t* maxPer8Addr,
                                           __ubuf__ uint16_t* eaAddr, __ubuf__ uint16_t* tmpScaleAddr,
                                           __ubuf__ uint16_t* scaleFrontAddr, __ubuf__ uint16_t* scaleBackAddr);
    // 计算y
    __aicore__ inline void ComputeOut(uint16_t maxPer4LoopNum, uint16_t tailPairLoop, uint16_t tailOdd,
                                      __ubuf__ xDtype* xAddr, __ubuf__ uint16_t* tmpScaleAddr,
                                      __ubuf__ uint32_t* yAddr);

protected:
    int64_t rowSize_ = 0;
    int64_t colSize_ = 0;
    int64_t blockRowSize_ = 0;
    int64_t tailBlockRowSize_ = 0;
    int64_t blockColSize_ = 0;
    int64_t blockRowNum_ = 0;
    int64_t groupNum_ = 0;
    // 由 scale_alg 决定的 Eb/Ec 比较阈值
    // scale_alg=0(默认): Eb 阈值4, Ec 阈值2; scale_alg=1: Eb 阈值3.75, Ec 阈值1.875
    bfloat16_t e8Thresh_ = static_cast<bfloat16_t>(4.0F);
    bfloat16_t e16Thresh_ = static_cast<bfloat16_t>(2.0F);
};

template <typename xDtype, typename yDtype>
__aicore__ inline void GroupedDynamicHiF4QuantBase<xDtype, yDtype>::BaseInitData(
    const GroupedDynamicHiF4QuantTilingData* tilingData)
{
    colSize_ = tilingData->colSize;
    rowSize_ = tilingData->rowSize;
    blockRowNum_ = tilingData->blockRowNum;
    blockRowSize_ = tilingData->blockRowSize;
    tailBlockRowSize_ = tilingData->tailBlockRowSize;
    blockColSize_ = tilingData->blockColSize;
    groupNum_ = tilingData->groupNum;
    e8Thresh_ = (tilingData->scaleAlg == 1) ? static_cast<bfloat16_t>(3.75F) : static_cast<bfloat16_t>(4.0F);
    e16Thresh_ = (tilingData->scaleAlg == 1) ? static_cast<bfloat16_t>(1.875F) : static_cast<bfloat16_t>(2.0F);
}

template <typename xDtype, typename yDtype>
__aicore__ inline void GroupedDynamicHiF4QuantBase<xDtype, yDtype>::ComputeMaxPer4(uint16_t maxPer4LoopNum,
                                                                                   uint16_t tailMaxper4,
                                                                                   __ubuf__ xDtype* xAddr,
                                                                                   __ubuf__ uint16_t* maxPer4Addr)
{
    // 4 条独立行流：相差 1 行(128 元素)、步长 4 行(512 元素)；尾块复用流 0、步长 1 行
    __ubuf__ xDtype* xStream0 = xAddr;
    __ubuf__ xDtype* xStream1 = xAddr + 1 * 128;
    __ubuf__ xDtype* xStream2 = xAddr + 2 * 128;
    __ubuf__ xDtype* xStream3 = xAddr + 3 * 128;
    __VEC_SCOPE__
    {
        Reg::RegTensor<uint16_t> vregIn, vregMax, vregIn1, vregIn2, vregIn3, vregIn4, absMask;
        Reg::RegTensor<bfloat16_t> vregOutbf16;
        Reg::MaskReg maskAll = Reg::CreateMask<uint16_t>();
        Reg::Duplicate(absMask, 0x7fff);
        Reg::Duplicate(vregMax, 0);
        for (uint16_t loopIdx = 0; loopIdx < maxPer4LoopNum; loopIdx++) {
            Reg::DataCopy<xDtype, Reg::PostLiteral::POST_MODE_UPDATE>((Reg::RegTensor<xDtype>&)vregIn1, xStream0, 512);
            Reg::DataCopy<xDtype, Reg::PostLiteral::POST_MODE_UPDATE>((Reg::RegTensor<xDtype>&)vregIn2, xStream1, 512);
            Reg::DataCopy<xDtype, Reg::PostLiteral::POST_MODE_UPDATE>((Reg::RegTensor<xDtype>&)vregIn3, xStream2, 512);
            Reg::DataCopy<xDtype, Reg::PostLiteral::POST_MODE_UPDATE>((Reg::RegTensor<xDtype>&)vregIn4, xStream3, 512);
            Reg::And(vregIn1, vregIn1, absMask, maskAll);
            Reg::And(vregIn2, vregIn2, absMask, maskAll);
            Reg::And(vregIn3, vregIn3, absMask, maskAll);
            Reg::And(vregIn4, vregIn4, absMask, maskAll);
            // 两两归约后合并
            Reg::Max(vregIn1, vregIn1, vregIn2, maskAll);
            Reg::Max(vregIn3, vregIn3, vregIn4, maskAll);
            Reg::Max(vregIn1, vregIn1, vregIn3, maskAll);

            if constexpr (IsSameType<xDtype, half>::value) {
                Reg::Cast<bfloat16_t, xDtype, castTraitHalf2Bf16>(vregOutbf16, (Reg::RegTensor<xDtype>&)vregIn1,
                                                                  maskAll);
                Reg::DataCopy<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(
                    maxPer4Addr, (Reg::RegTensor<uint16_t>&)vregOutbf16, 128, maskAll);
            } else {
                Reg::DataCopy<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(maxPer4Addr, vregIn1, 128, maskAll);
            }
        }
        // 尾块处理
        for (uint16_t loopIdx = 0; loopIdx < tailMaxper4; loopIdx++) {
            Reg::DataCopy<xDtype, Reg::PostLiteral::POST_MODE_UPDATE>((Reg::RegTensor<xDtype>&)vregIn, xStream0, 128);
            Reg::And(vregIn, vregIn, absMask, maskAll);
            Reg::Max(vregMax, vregMax, vregIn, maskAll);
        }
        if constexpr (IsSameType<xDtype, half>::value) {
            Reg::Cast<bfloat16_t, xDtype, castTraitHalf2Bf16>(vregOutbf16, (Reg::RegTensor<xDtype>&)vregMax, maskAll);
            Reg::DataCopy<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(
                maxPer4Addr, (Reg::RegTensor<uint16_t>&)vregOutbf16, 128, maskAll);
        } else {
            Reg::DataCopy<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(maxPer4Addr, vregMax, 128, maskAll);
        }
    }
    return;
}

template <typename xDtype, typename yDtype>
__aicore__ inline void GroupedDynamicHiF4QuantBase<xDtype, yDtype>::ComputeMaxPer8(uint16_t maxPer8LoopNum,
                                                                                   uint16_t tailMaxper8,
                                                                                   __ubuf__ uint16_t* maxPer4Addr,
                                                                                   __ubuf__ uint16_t* maxPer8Addr)
{
    __VEC_SCOPE__
    {
        Reg::RegTensor<uint16_t> vregIn, vregMax, vregIn1, vregIn2;
        Reg::MaskReg maskAll = Reg::CreateMask<uint16_t>();
        Reg::Duplicate(vregMax, 0);
        for (uint16_t loopIdx = 0; loopIdx < maxPer8LoopNum; loopIdx++) {
            Reg::DataCopy<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(vregIn1, maxPer4Addr, 128);
            Reg::DataCopy<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(vregIn2, maxPer4Addr, 128);
            Reg::Max(vregIn1, vregIn1, vregIn2, maskAll);
            Reg::DataCopy<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(maxPer8Addr, vregIn1, 128, maskAll);
        }
        // 尾块处理
        for (uint16_t loopIdx = 0; loopIdx < tailMaxper8; loopIdx++) {
            Reg::DataCopy<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(vregIn, maxPer4Addr, 128);
            Reg::Max(vregMax, vregMax, vregIn, maskAll);
        }
        Reg::DataCopy<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(maxPer8Addr, vregMax, 128, maskAll);
    }
    return;
}

template <typename xDtype, typename yDtype>
__aicore__ inline void GroupedDynamicHiF4QuantBase<xDtype, yDtype>::ComputeE6M2(uint16_t maxPer64LoopNum,
                                                                                uint16_t tail8Maxper64,
                                                                                __ubuf__ uint16_t* maxPer8Addr,
                                                                                __ubuf__ uint16_t* eaAddr)
{
    // 双记录流：相差 1 记录(128 元素)、步长 2 记录(256 元素)；尾块复用流 0、步长 1 记录
    __ubuf__ uint16_t* m8s0 = maxPer8Addr;
    __ubuf__ uint16_t* m8s1 = maxPer8Addr + 1 * 128;
    __VEC_SCOPE__
    {
        Reg::RegTensor<uint16_t> vregMaxper8_0, vregMaxper8_1, vreg0E6m2;
        Reg::RegTensor<bfloat16_t> vregGround, vregGround1, vregNumscale;
        Reg::MaskReg maskAll = Reg::CreateMask<uint8_t>();
        // 1/7 = 0x3e12
        Reg::Duplicate((Reg::RegTensor<uint16_t>&)vregNumscale, 0x3e12);
        for (uint16_t loop1Idx = 0; loop1Idx < maxPer64LoopNum; loop1Idx++) {
            Reg::Duplicate(vregGround, 0);
            Reg::Duplicate(vregGround1, 0);
            // 2 条独立装载并入双链
            for (uint16_t loop2Idx = 0; loop2Idx < 4; loop2Idx++) {
                Reg::DataCopy<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(vregMaxper8_0, m8s0, 256);
                Reg::DataCopy<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(vregMaxper8_1, m8s1, 256);
                Reg::Max(vregGround, vregGround, (Reg::RegTensor<bfloat16_t>&)vregMaxper8_0, maskAll);
                Reg::Max(vregGround1, vregGround1, (Reg::RegTensor<bfloat16_t>&)vregMaxper8_1, maskAll);
            }
            // 双链合并
            Reg::Max(vregGround, vregGround, vregGround1, maskAll);
            Reg::Mul(vregGround, vregGround, vregNumscale, maskAll);
            Reg::Cast<fp8_e6m2_t, bfloat16_t, castTraitE6M2>((Reg::RegTensor<fp8_e6m2_t>&)vreg0E6m2, vregGround,
                                                             maskAll);
            // 搬出 ea（E6M2）
            Reg::DataCopy<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(eaAddr, vreg0E6m2, 128, maskAll);
        }
        // 尾块：链式累加
        Reg::Duplicate(vregGround, 0x0000);
        for (uint16_t loop2Idx = 0; loop2Idx < tail8Maxper64; loop2Idx++) {
            Reg::DataCopy<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(vregMaxper8_0, m8s0, 128);
            Reg::Max(vregGround, vregGround, (Reg::RegTensor<bfloat16_t>&)vregMaxper8_0, maskAll);
        }
        Reg::Mul(vregGround, vregGround, vregNumscale, maskAll);
        Reg::Cast<fp8_e6m2_t, bfloat16_t, castTraitE6M2>((Reg::RegTensor<fp8_e6m2_t>&)vreg0E6m2, vregGround, maskAll);
        Reg::DataCopy<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(eaAddr, vreg0E6m2, 128, maskAll);
    }
    return;
}

template <typename xDtype, typename yDtype>
__aicore__ inline void GroupedDynamicHiF4QuantBase<xDtype, yDtype>::ComputeTmpScale(
    uint16_t maxPer64LoopNum, uint16_t tailFullLoop, uint16_t tailPartM4, __ubuf__ uint16_t* maxPer4Addr,
    __ubuf__ uint16_t* maxPer8Addr, __ubuf__ uint16_t* eaAddr, __ubuf__ uint16_t* tmpScaleAddr,
    __ubuf__ uint16_t* scaleFrontAddr, __ubuf__ uint16_t* scaleBackAddr)
{
    // 尾组专用流起点 = 主循环组数 × 每组记录数（mp8 8 条、mp4/tmpScale 16 条）
    uint32_t tailGroupNum = static_cast<uint32_t>(maxPer64LoopNum);
    __ubuf__ uint16_t* p8Tail = maxPer8Addr + tailGroupNum * 8 * 128;
    __ubuf__ uint16_t* p4Tail = maxPer4Addr + tailGroupNum * 16 * 128;
    __ubuf__ uint16_t* tsTail = tmpScaleAddr + tailGroupNum * 16 * 128;
    // 每组一条记录的流
    __ubuf__ uint16_t* eaStream = eaAddr;
    __ubuf__ uint16_t* scaleFrontStream = scaleFrontAddr;
    __ubuf__ uint16_t* scaleBackStream = scaleBackAddr;
    // 奇数 micro(2b+1) 基址 = 偶数 micro 基址 + 1 条记录
    __ubuf__ uint16_t* maxPer4OddAddr = maxPer4Addr + 128;
    __ubuf__ uint16_t* tmpScaleOddAddr = tmpScaleAddr + 128;
    __VEC_SCOPE__
    {
        // micro 寄存器：下标 0 = 偶数 micro(2b)、1 = 奇数 micro(2b+1)
        Reg::RegTensor<bfloat16_t> vregMaxper4_0, vregMaxper4_1, vregMaxper8;
        Reg::RegTensor<bfloat16_t> vb16_EaRecBf16, vregNum1, vregNumhalf;
        // Eb/Ec 位平面累加器：Ec 偶/奇各一条（偶落偶数位、奇落奇数位），块尾相加合并
        Reg::RegTensor<uint16_t> vregEa, vregEbAcc, vregEcAcc_0, vregEcAcc_1;
        // 权值寄存器：Eb 每轮左移 1 位（置位 8 行块 b），Ec 每轮左移 2 位（置位 micro 2b / 2b+1）
        Reg::RegTensor<uint16_t> vregEbW, vregEcW_0, vregEcW_1, vregMask00FF;
        // 比较掩码：偶/奇各一个，互不覆写
        Reg::MaskReg vb1E8Comp, vb1E16Comp_0, vb1E16Comp_1, maskAll = Reg::CreateMask<uint8_t>();

        Reg::Duplicate(vregNum1, 1.0F);
        Reg::Duplicate(vregNumhalf, static_cast<bfloat16_t>(0.5F));
        Reg::Duplicate(vregMask00FF, 0x00ff);

        // 主循环：每个 64 行组迭代 8 次，第 b 次处理 8 行块 b（Eb 位 b）与 micro 2b/2b+1（Ec 位 2b/2b+1）。
        // 组基址与块内偏移折进 CreateAddrReg（2 对 index/stride）；奇数 micro 用 +1 记录基址。
        for (uint16_t loop1Idx = 0; loop1Idx < maxPer64LoopNum; loop1Idx++) {
            Reg::DataCopy<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(vregEa, eaStream, 128);
            Reg::ReciprocalCast<bfloat16_t, fp8_e6m2_t, castTraitReciprocalCast>(
                vb16_EaRecBf16, (Reg::RegTensor<fp8_e6m2_t>&)vregEa, maskAll);
            Reg::Duplicate(vregEbAcc, 0x0000);
            Reg::Duplicate(vregEcAcc_0, 0x0000);
            Reg::Duplicate(vregEcAcc_1, 0x0000);
            Reg::Duplicate(vregEbW, 0x0001);
            Reg::Duplicate(vregEcW_0, 0x0001);
            Reg::Duplicate(vregEcW_1, 0x0002);
            for (uint16_t roundIdx = 0; roundIdx < static_cast<uint16_t>(NUM_EIGHT); roundIdx++) {
                // p8a: 组基址 + 8 行块 b；p4a: 组基址 + micro 2b（2b+1 复用 p4a 配奇基址）
                Reg::AddrReg p8aReg = Reg::CreateAddrReg<uint16_t>(loop1Idx, static_cast<uint32_t>(NUM_EIGHT) * 128,
                                                                   roundIdx, 128);
                Reg::AddrReg p4aReg = Reg::CreateAddrReg<uint16_t>(loop1Idx, static_cast<uint32_t>(NUM_SIXTENN) * 128,
                                                                   roundIdx, static_cast<uint32_t>(NUM_TWO) * 128);
                // 8 行块 b 的 Eb
                Reg::DataCopy<uint16_t, Reg::LoadDist::DIST_NORM>((Reg::RegTensor<uint16_t>&)vregMaxper8, maxPer8Addr,
                                                                  p8aReg);
                Reg::Mul(vregMaxper8, vregMaxper8, vb16_EaRecBf16, maskAll);
                Reg::CompareScalar<bfloat16_t, CMPMODE::GE>(vb1E8Comp, vregMaxper8, e8Thresh_, maskAll);
                Reg::Add<uint16_t, Reg::MaskMergeMode::MERGING>(vregEbAcc, vregEbAcc, vregEbW, vb1E8Comp);
                Reg::Select(vregMaxper8, vregNumhalf, vregNum1, vb1E8Comp);

                // micro 2b（偶, _0）/ 2b+1（奇, _1）：同类操作逐行配对
                Reg::DataCopy<uint16_t, Reg::LoadDist::DIST_NORM>((Reg::RegTensor<uint16_t>&)vregMaxper4_0, maxPer4Addr,
                                                                  p4aReg);
                Reg::DataCopy<uint16_t, Reg::LoadDist::DIST_NORM>((Reg::RegTensor<uint16_t>&)vregMaxper4_1,
                                                                  maxPer4OddAddr, p4aReg);
                Reg::Mul(vregMaxper4_0, vregMaxper4_0, vb16_EaRecBf16, maskAll);
                Reg::Mul(vregMaxper4_1, vregMaxper4_1, vb16_EaRecBf16, maskAll);
                Reg::Mul(vregMaxper4_0, vregMaxper4_0, vregMaxper8, maskAll);
                Reg::Mul(vregMaxper4_1, vregMaxper4_1, vregMaxper8, maskAll);
                Reg::CompareScalar<bfloat16_t, CMPMODE::GE>(vb1E16Comp_0, vregMaxper4_0, e16Thresh_, maskAll);
                Reg::CompareScalar<bfloat16_t, CMPMODE::GE>(vb1E16Comp_1, vregMaxper4_1, e16Thresh_, maskAll);
                Reg::Add<uint16_t, Reg::MaskMergeMode::MERGING>(vregEcAcc_0, vregEcAcc_0, vregEcW_0, vb1E16Comp_0);
                Reg::Add<uint16_t, Reg::MaskMergeMode::MERGING>(vregEcAcc_1, vregEcAcc_1, vregEcW_1, vb1E16Comp_1);
                Reg::Select(vregMaxper4_0, vregNumhalf, vregNum1, vb1E16Comp_0);
                Reg::Select(vregMaxper4_1, vregNumhalf, vregNum1, vb1E16Comp_1);
                Reg::Mul(vregMaxper4_0, vregMaxper4_0, vregMaxper8, maskAll);
                Reg::Mul(vregMaxper4_1, vregMaxper4_1, vregMaxper8, maskAll);
                Reg::Mul(vregMaxper4_0, vregMaxper4_0, vb16_EaRecBf16, maskAll);
                Reg::Mul(vregMaxper4_1, vregMaxper4_1, vb16_EaRecBf16, maskAll);
                Reg::DataCopy<uint16_t>(tmpScaleAddr, (Reg::RegTensor<uint16_t>&)vregMaxper4_0, p4aReg, maskAll);
                Reg::DataCopy<uint16_t>(tmpScaleOddAddr, (Reg::RegTensor<uint16_t>&)vregMaxper4_1, p4aReg, maskAll);

                // 权值左移推进：Eb +1 位，Ec +2 位
                Reg::ShiftLefts(vregEbW, vregEbW, SHIFT_INDEX, maskAll);
                Reg::ShiftLefts(vregEcW_0, vregEcW_0, static_cast<int16_t>(NUM_TWO), maskAll);
                Reg::ShiftLefts(vregEcW_1, vregEcW_1, static_cast<int16_t>(NUM_TWO), maskAll);
            }
            // 块尾拼装：[Ea,Eb] = E6M2 | EbAcc<<8；Ec = 偶累加器 + 奇累加器（位不重叠）
            Reg::ShiftLefts(vregEbAcc, vregEbAcc, static_cast<int16_t>(NUM_EIGHT), maskAll);
            Reg::And(vregEa, vregEa, vregMask00FF, maskAll);
            Reg::Add(vregEbAcc, vregEbAcc, vregEa, maskAll);
            Reg::DataCopy<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(scaleFrontStream, vregEbAcc, 128, maskAll);
            Reg::Add(vregEcAcc_0, vregEcAcc_0, vregEcAcc_1, maskAll);
            Reg::DataCopy<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(scaleBackStream, vregEcAcc_0, 128, maskAll);
        }
        // 尾组（最后一组，单链）：整 8 行块 + 残 4 行块
        Reg::DataCopy<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(vregEa, eaStream, 128);
        Reg::ReciprocalCast<bfloat16_t, fp8_e6m2_t, castTraitReciprocalCast>(
            vb16_EaRecBf16, (Reg::RegTensor<fp8_e6m2_t>&)vregEa, maskAll);
        Reg::Duplicate(vregEbAcc, 0x0000);
        Reg::Duplicate(vregEcAcc_0, 0x0000);
        Reg::Duplicate(vregEbW, 0x0001);
        Reg::Duplicate(vregEcW_0, 0x0001);
        Reg::Duplicate(vregEcW_1, 0x0002);
        // 整 8 行块：1 mp8 + 2 mp4
        for (uint16_t loop2Idx = 0; loop2Idx < tailFullLoop; loop2Idx++) {
            Reg::DataCopy<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>((Reg::RegTensor<uint16_t>&)vregMaxper8, p8Tail,
                                                                        128);
            Reg::Mul(vregMaxper8, vregMaxper8, vb16_EaRecBf16, maskAll);
            Reg::CompareScalar<bfloat16_t, CMPMODE::GE>(vb1E8Comp, vregMaxper8, e8Thresh_, maskAll);
            Reg::Add<uint16_t, Reg::MaskMergeMode::MERGING>(vregEbAcc, vregEbAcc, vregEbW, vb1E8Comp);
            Reg::Select(vregMaxper8, vregNumhalf, vregNum1, vb1E8Comp);
            Reg::DataCopy<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>((Reg::RegTensor<uint16_t>&)vregMaxper4_0,
                                                                        p4Tail, 128);
            Reg::Mul(vregMaxper4_0, vregMaxper4_0, vb16_EaRecBf16, maskAll);
            Reg::Mul(vregMaxper4_0, vregMaxper4_0, vregMaxper8, maskAll);
            Reg::CompareScalar<bfloat16_t, CMPMODE::GE>(vb1E16Comp_0, vregMaxper4_0, e16Thresh_, maskAll);
            Reg::Add<uint16_t, Reg::MaskMergeMode::MERGING>(vregEcAcc_0, vregEcAcc_0, vregEcW_0, vb1E16Comp_0);
            Reg::Select(vregMaxper4_0, vregNumhalf, vregNum1, vb1E16Comp_0);
            Reg::Mul(vregMaxper4_0, vregMaxper4_0, vregMaxper8, maskAll);
            Reg::Mul(vregMaxper4_0, vregMaxper4_0, vb16_EaRecBf16, maskAll);
            Reg::DataCopy<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(
                tsTail, (Reg::RegTensor<uint16_t>&)vregMaxper4_0, 128, maskAll);
            Reg::DataCopy<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>((Reg::RegTensor<uint16_t>&)vregMaxper4_0,
                                                                        p4Tail, 128);
            Reg::Mul(vregMaxper4_0, vregMaxper4_0, vb16_EaRecBf16, maskAll);
            Reg::Mul(vregMaxper4_0, vregMaxper4_0, vregMaxper8, maskAll);
            Reg::CompareScalar<bfloat16_t, CMPMODE::GE>(vb1E16Comp_0, vregMaxper4_0, e16Thresh_, maskAll);
            Reg::Add<uint16_t, Reg::MaskMergeMode::MERGING>(vregEcAcc_0, vregEcAcc_0, vregEcW_1, vb1E16Comp_0);
            Reg::Select(vregMaxper4_0, vregNumhalf, vregNum1, vb1E16Comp_0);
            Reg::Mul(vregMaxper4_0, vregMaxper4_0, vregMaxper8, maskAll);
            Reg::Mul(vregMaxper4_0, vregMaxper4_0, vb16_EaRecBf16, maskAll);
            Reg::DataCopy<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(
                tsTail, (Reg::RegTensor<uint16_t>&)vregMaxper4_0, 128, maskAll);
            Reg::ShiftLefts(vregEbW, vregEbW, SHIFT_INDEX, maskAll);
            Reg::ShiftLefts(vregEcW_0, vregEcW_0, static_cast<int16_t>(NUM_TWO), maskAll);
            Reg::ShiftLefts(vregEcW_1, vregEcW_1, static_cast<int16_t>(NUM_TWO), maskAll);
        }
        // 残 8 行块：1 mp8 + 1 mp4
        for (uint16_t loop3Idx = 0; loop3Idx < tailPartM4; loop3Idx++) {
            Reg::DataCopy<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>((Reg::RegTensor<uint16_t>&)vregMaxper8, p8Tail,
                                                                        128);
            Reg::Mul(vregMaxper8, vregMaxper8, vb16_EaRecBf16, maskAll);
            Reg::CompareScalar<bfloat16_t, CMPMODE::GE>(vb1E8Comp, vregMaxper8, e8Thresh_, maskAll);
            Reg::Add<uint16_t, Reg::MaskMergeMode::MERGING>(vregEbAcc, vregEbAcc, vregEbW, vb1E8Comp);
            Reg::Select(vregMaxper8, vregNumhalf, vregNum1, vb1E8Comp);
            Reg::DataCopy<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>((Reg::RegTensor<uint16_t>&)vregMaxper4_0,
                                                                        p4Tail, 128);
            Reg::Mul(vregMaxper4_0, vregMaxper4_0, vb16_EaRecBf16, maskAll);
            Reg::Mul(vregMaxper4_0, vregMaxper4_0, vregMaxper8, maskAll);
            Reg::CompareScalar<bfloat16_t, CMPMODE::GE>(vb1E16Comp_0, vregMaxper4_0, e16Thresh_, maskAll);
            Reg::Add<uint16_t, Reg::MaskMergeMode::MERGING>(vregEcAcc_0, vregEcAcc_0, vregEcW_0, vb1E16Comp_0);
            Reg::Select(vregMaxper4_0, vregNumhalf, vregNum1, vb1E16Comp_0);
            Reg::Mul(vregMaxper4_0, vregMaxper4_0, vregMaxper8, maskAll);
            Reg::Mul(vregMaxper4_0, vregMaxper4_0, vb16_EaRecBf16, maskAll);
            Reg::DataCopy<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(
                tsTail, (Reg::RegTensor<uint16_t>&)vregMaxper4_0, 128, maskAll);
        }
        // 尾组拼装：Ec 单累加器已是最终位布局，直存
        Reg::ShiftLefts(vregEbAcc, vregEbAcc, static_cast<int16_t>(NUM_EIGHT), maskAll);
        Reg::And(vregEa, vregEa, vregMask00FF, maskAll);
        Reg::Add(vregEbAcc, vregEbAcc, vregEa, maskAll);
        Reg::DataCopy<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(scaleFrontStream, vregEbAcc, 128, maskAll);
        Reg::DataCopy<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(scaleBackStream, vregEcAcc_0, 128, maskAll);
    }
    return;
}

template <typename xDtype, typename yDtype>
__aicore__ inline void GroupedDynamicHiF4QuantBase<xDtype, yDtype>::ComputeOut(uint16_t maxPer4LoopNum,
                                                                               uint16_t tailPairLoop, uint16_t tailOdd,
                                                                               __ubuf__ xDtype* xAddr,
                                                                               __ubuf__ uint16_t* tmpScaleAddr,
                                                                               __ubuf__ uint32_t* yAddr)
{
    // 4 条独立行流：相差 1 行(128 元素)、步长 4 行(512 元素)；y 每行 16 u32，4 条 y 流相差 16、步长 64；
    // 尾块成对处理（流 0/1、步长 2 行），残行复用流 0
    __ubuf__ xDtype* xStream0 = xAddr;
    __ubuf__ xDtype* xStream1 = xAddr + 1 * 128;
    __ubuf__ xDtype* xStream2 = xAddr + 2 * 128;
    __ubuf__ xDtype* xStream3 = xAddr + 3 * 128;
    __ubuf__ uint32_t* yStream0 = yAddr;
    __ubuf__ uint32_t* yStream1 = yAddr + 1 * 16;
    __ubuf__ uint32_t* yStream2 = yAddr + 2 * 16;
    __ubuf__ uint32_t* yStream3 = yAddr + 3 * 16;
    __VEC_SCOPE__
    {
        Reg::RegTensor<xDtype> vregOut0, vregOut1, vregOut2, vregOut3;
        Reg::RegTensor<bfloat16_t> vregOutbf16_0, vregOutbf16_1, vregOutbf16_2, vregOutbf16_3;
        Reg::RegTensor<fp4x2_e1m2_t> vbOut0, vbOut1, vbOut2, vbOut3;
        Reg::RegTensor<uint16_t> vregScaleTmp;
        Reg::MaskReg maskAll = Reg::CreateMask<uint8_t>();
        for (uint16_t loop1Idx = 0; loop1Idx < maxPer4LoopNum; loop1Idx++) {
            Reg::DataCopy<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(vregScaleTmp, tmpScaleAddr, 128);
            if constexpr (IsSameType<xDtype, half>::value) {
                Reg::DataCopy<xDtype, Reg::PostLiteral::POST_MODE_UPDATE>((Reg::RegTensor<xDtype>&)vregOut0, xStream0,
                                                                          512);
                Reg::DataCopy<xDtype, Reg::PostLiteral::POST_MODE_UPDATE>((Reg::RegTensor<xDtype>&)vregOut1, xStream1,
                                                                          512);
                Reg::DataCopy<xDtype, Reg::PostLiteral::POST_MODE_UPDATE>((Reg::RegTensor<xDtype>&)vregOut2, xStream2,
                                                                          512);
                Reg::DataCopy<xDtype, Reg::PostLiteral::POST_MODE_UPDATE>((Reg::RegTensor<xDtype>&)vregOut3, xStream3,
                                                                          512);
                Reg::Cast<bfloat16_t, xDtype, castTraitHalf2Bf16>(vregOutbf16_0, (Reg::RegTensor<xDtype>&)vregOut0,
                                                                  maskAll);
                Reg::Cast<bfloat16_t, xDtype, castTraitHalf2Bf16>(vregOutbf16_1, (Reg::RegTensor<xDtype>&)vregOut1,
                                                                  maskAll);
                Reg::Cast<bfloat16_t, xDtype, castTraitHalf2Bf16>(vregOutbf16_2, (Reg::RegTensor<xDtype>&)vregOut2,
                                                                  maskAll);
                Reg::Cast<bfloat16_t, xDtype, castTraitHalf2Bf16>(vregOutbf16_3, (Reg::RegTensor<xDtype>&)vregOut3,
                                                                  maskAll);
            } else {
                Reg::DataCopy<xDtype, Reg::PostLiteral::POST_MODE_UPDATE>(vregOutbf16_0, xStream0, 512);
                Reg::DataCopy<xDtype, Reg::PostLiteral::POST_MODE_UPDATE>(vregOutbf16_1, xStream1, 512);
                Reg::DataCopy<xDtype, Reg::PostLiteral::POST_MODE_UPDATE>(vregOutbf16_2, xStream2, 512);
                Reg::DataCopy<xDtype, Reg::PostLiteral::POST_MODE_UPDATE>(vregOutbf16_3, xStream3, 512);
            }
            Reg::Mul(vregOutbf16_0, vregOutbf16_0, (Reg::RegTensor<bfloat16_t>&)vregScaleTmp, maskAll);
            Reg::Mul(vregOutbf16_1, vregOutbf16_1, (Reg::RegTensor<bfloat16_t>&)vregScaleTmp, maskAll);
            Reg::Mul(vregOutbf16_2, vregOutbf16_2, (Reg::RegTensor<bfloat16_t>&)vregScaleTmp, maskAll);
            Reg::Mul(vregOutbf16_3, vregOutbf16_3, (Reg::RegTensor<bfloat16_t>&)vregScaleTmp, maskAll);
            Reg::Cast<fp4x2_e1m2_t, bfloat16_t, castTraitBfloatToE2M1>(
                vbOut0, (Reg::RegTensor<bfloat16_t>&)vregOutbf16_0, maskAll);
            Reg::Cast<fp4x2_e1m2_t, bfloat16_t, castTraitBfloatToE2M1>(
                vbOut1, (Reg::RegTensor<bfloat16_t>&)vregOutbf16_1, maskAll);
            Reg::Cast<fp4x2_e1m2_t, bfloat16_t, castTraitBfloatToE2M1>(
                vbOut2, (Reg::RegTensor<bfloat16_t>&)vregOutbf16_2, maskAll);
            Reg::Cast<fp4x2_e1m2_t, bfloat16_t, castTraitBfloatToE2M1>(
                vbOut3, (Reg::RegTensor<bfloat16_t>&)vregOutbf16_3, maskAll);
            Reg::DataCopy<uint32_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_PACK4_B32>(
                yStream0, (Reg::RegTensor<uint32_t>&)vbOut0, 64, maskAll);
            Reg::DataCopy<uint32_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_PACK4_B32>(
                yStream1, (Reg::RegTensor<uint32_t>&)vbOut1, 64, maskAll);
            Reg::DataCopy<uint32_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_PACK4_B32>(
                yStream2, (Reg::RegTensor<uint32_t>&)vbOut2, 64, maskAll);
            Reg::DataCopy<uint32_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_PACK4_B32>(
                yStream3, (Reg::RegTensor<uint32_t>&)vbOut3, 64, maskAll);
        }
        // 尾块：1 条 scale + 成对流 0/1（步长 2 行）
        Reg::DataCopy<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(vregScaleTmp, tmpScaleAddr, 128);
        for (uint16_t loop2Idx = 0; loop2Idx < tailPairLoop; loop2Idx++) {
            if constexpr (IsSameType<xDtype, half>::value) {
                Reg::DataCopy<xDtype, Reg::PostLiteral::POST_MODE_UPDATE>((Reg::RegTensor<xDtype>&)vregOut0, xStream0,
                                                                          256);
                Reg::DataCopy<xDtype, Reg::PostLiteral::POST_MODE_UPDATE>((Reg::RegTensor<xDtype>&)vregOut1, xStream1,
                                                                          256);
                Reg::Cast<bfloat16_t, xDtype, castTraitHalf2Bf16>(vregOutbf16_0, (Reg::RegTensor<xDtype>&)vregOut0,
                                                                  maskAll);
                Reg::Cast<bfloat16_t, xDtype, castTraitHalf2Bf16>(vregOutbf16_1, (Reg::RegTensor<xDtype>&)vregOut1,
                                                                  maskAll);
            } else {
                Reg::DataCopy<xDtype, Reg::PostLiteral::POST_MODE_UPDATE>(vregOutbf16_0, xStream0, 256);
                Reg::DataCopy<xDtype, Reg::PostLiteral::POST_MODE_UPDATE>(vregOutbf16_1, xStream1, 256);
            }
            Reg::Mul(vregOutbf16_0, vregOutbf16_0, (Reg::RegTensor<bfloat16_t>&)vregScaleTmp, maskAll);
            Reg::Mul(vregOutbf16_1, vregOutbf16_1, (Reg::RegTensor<bfloat16_t>&)vregScaleTmp, maskAll);
            Reg::Cast<fp4x2_e1m2_t, bfloat16_t, castTraitBfloatToE2M1>(
                vbOut0, (Reg::RegTensor<bfloat16_t>&)vregOutbf16_0, maskAll);
            Reg::Cast<fp4x2_e1m2_t, bfloat16_t, castTraitBfloatToE2M1>(
                vbOut1, (Reg::RegTensor<bfloat16_t>&)vregOutbf16_1, maskAll);
            Reg::DataCopy<uint32_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_PACK4_B32>(
                yStream0, (Reg::RegTensor<uint32_t>&)vbOut0, 32, maskAll);
            Reg::DataCopy<uint32_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_PACK4_B32>(
                yStream1, (Reg::RegTensor<uint32_t>&)vbOut1, 32, maskAll);
        }
        // 残行：复用流 0
        for (uint16_t loop3Idx = 0; loop3Idx < tailOdd; loop3Idx++) {
            if constexpr (IsSameType<xDtype, half>::value) {
                Reg::DataCopy<xDtype, Reg::PostLiteral::POST_MODE_UPDATE>((Reg::RegTensor<xDtype>&)vregOut0, xStream0,
                                                                          256);
                Reg::Cast<bfloat16_t, xDtype, castTraitHalf2Bf16>(vregOutbf16_0, (Reg::RegTensor<xDtype>&)vregOut0,
                                                                  maskAll);
            } else {
                Reg::DataCopy<xDtype, Reg::PostLiteral::POST_MODE_UPDATE>(vregOutbf16_0, xStream0, 256);
            }
            Reg::Mul(vregOutbf16_0, vregOutbf16_0, (Reg::RegTensor<bfloat16_t>&)vregScaleTmp, maskAll);
            Reg::Cast<fp4x2_e1m2_t, bfloat16_t, castTraitBfloatToE2M1>(
                vbOut0, (Reg::RegTensor<bfloat16_t>&)vregOutbf16_0, maskAll);
            Reg::DataCopy<uint32_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_PACK4_B32>(
                yStream0, (Reg::RegTensor<uint32_t>&)vbOut0, 32, maskAll);
        }
    }
    return;
}

} // namespace GroupedDynamicHiF4Quant
#endif // OPS_NN_GROUPED_DYNAMIC_HIF4_QUANT_BASE_H
