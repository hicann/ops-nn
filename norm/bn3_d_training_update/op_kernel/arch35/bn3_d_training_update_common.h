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
 * \file bn3_d_training_update_common.h
 * \brief BN3DTrainingUpdate arch35(regbase)公共件:向量宽度、随路 cast
 * 的寄存器载入/存储、非对齐访存惯用法、以及**唯一一份通道数学**(FoldRegs / VFFoldChunk)。
 */

#ifndef BN3_D_TRAINING_UPDATE_COMMON_H
#define BN3_D_TRAINING_UPDATE_COMMON_H

#include <cstdint>
#include "kernel_operator.h"
#include "op_kernel/platform_util.h"
#include "bn3_d_training_update_tiling_data.h"

namespace Bn3dTrainingUpdateOps {
using namespace AscendC;

using AscendC::MicroAPI::LoadDist;
using AscendC::MicroAPI::MaskMergeMode;
using AscendC::MicroAPI::MaskPattern;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::RegTensor;
using AscendC::MicroAPI::StoreDist;
using AscendC::MicroAPI::UnalignRegForLoad;
using AscendC::MicroAPI::UnalignRegForStore;

// UB 对齐粒度与一个 fp32 向量寄存器的车道数:均由平台信息推导。
constexpr int64_t UB_BLOCK_BYTES = Ops::Base::GetUbBlockSize();
constexpr uint32_t VL_FP32 = Ops::Base::GetVRegSize() / sizeof(float);

// b16 → fp32 加宽(升精度):RoundMode 无关。
constexpr AscendC::MicroAPI::CastTrait kCastB16ToFp32 = {AscendC::MicroAPI::RegLayout::ZERO,
                                                         AscendC::MicroAPI::SatMode::UNKNOWN, MaskMergeMode::ZEROING,
                                                         AscendC::RoundMode::UNKNOWN};
// fp32 → b16 收窄(降精度):单次就近舍入 CAST_RINT。
constexpr AscendC::MicroAPI::CastTrait kCastFp32ToB16 = {AscendC::MicroAPI::RegLayout::ZERO,
                                                         AscendC::MicroAPI::SatMode::NO_SAT, MaskMergeMode::ZEROING,
                                                         AscendC::RoundMode::CAST_RINT};
// fp32 除法高精度模式:默认 Div 与真值可差 1 ULP,在近零处可观测。
static constexpr AscendC::Reg::DivSpecificMode kHighPrecisionDiv = {AscendC::Reg::MaskMergeMode::ZEROING, true};

// ------------------------------------------------------------------ 载入 / 存储

// ALIGNED:按整 VL 载入 T_X 并随路升 fp32(x/y 的偏移恒为 VL 的整数倍时使用)。
template <typename T_X>
__simd_callee__ inline void LoadAlignedXToFp32(RegTensor<float>& dst, __ubuf__ T_X* src, uint32_t offset, MaskReg& mask)
{
    if constexpr (IsSameType<T_X, float>::value) {
        AscendC::Reg::LoadAlign<float, LoadDist::DIST_NORM>(dst, src + offset);
    } else {
        RegTensor<T_X> raw;
        AscendC::Reg::LoadAlign<T_X, LoadDist::DIST_UNPACK_B16>(raw, src + offset);
        AscendC::Reg::Cast<float, T_X, kCastB16ToFp32>(dst, raw, mask);
    }
}

// ALIGNED:把 fp32 寄存器降回 T_X 写回(单次 CAST_RINT 舍入)。
template <typename T_X>
__simd_callee__ inline void StoreAlignedYFromFp32(__ubuf__ T_X* dst, RegTensor<float>& src, uint32_t offset,
                                                  MaskReg& mask)
{
    if constexpr (IsSameType<T_X, float>::value) {
        AscendC::Reg::StoreAlign<float, StoreDist::DIST_NORM>(dst + offset, src, mask);
    } else {
        RegTensor<T_X> narrowed;
        AscendC::Reg::Cast<T_X, float, kCastFp32ToB16>(narrowed, src, mask);
        AscendC::Reg::StoreAlign<T_X, StoreDist::DIST_PACK_B32>(dst + offset, narrowed, mask);
    }
}

// UNALIGNED:流式载入 T_X 并随路升 fp32,src 按 stride 前进(channel-last 的瓦片起点不保证 VL 对齐)。
template <typename T_X>
__simd_callee__ inline void LoadUnalignedXToFp32(__ubuf__ T_X*& src, RegTensor<float>& dst, UnalignRegForLoad& state,
                                                 MaskReg& mask, uint32_t stride)
{
    if constexpr (IsSameType<T_X, float>::value) {
        AscendC::MicroAPI::LoadUnAlign<float, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(dst, state, src,
                                                                                                stride);
    } else {
        RegTensor<T_X> raw;
        RegTensor<T_X> unpacked;
        AscendC::MicroAPI::LoadUnAlign<T_X, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(raw, state, src, stride);
        AscendC::MicroAPI::UnPack((RegTensor<uint32_t>&)unpacked, (RegTensor<uint16_t>&)raw);
        AscendC::Reg::Cast<float, T_X, kCastB16ToFp32>(dst, unpacked, mask);
    }
}

// UNALIGNED:流式写回 T_X,dst 按 stride 前进。
template <typename T_X>
__simd_callee__ inline void StoreUnalignedYFromFp32(__ubuf__ T_X*& dst, RegTensor<float>& src,
                                                    UnalignRegForStore& state, MaskReg& mask, uint32_t stride)
{
    if constexpr (IsSameType<T_X, float>::value) {
        AscendC::MicroAPI::StoreUnAlign<float, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(dst, src, state,
                                                                                                 stride);
    } else {
        RegTensor<T_X> narrowed;
        RegTensor<T_X> packed;
        AscendC::Reg::Cast<T_X, float, kCastFp32ToB16>(narrowed, src, mask);
        AscendC::MicroAPI::Pack((RegTensor<uint16_t>&)packed, (RegTensor<uint32_t>&)narrowed);
        AscendC::MicroAPI::StoreUnAlign<T_X, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(dst, packed, state,
                                                                                               stride);
    }
}

// raw fp32(系数 tile:数据本来就是 fp32,不需要 cast)。
__simd_callee__ inline void LoadFp32Aligned(RegTensor<float>& dst, __ubuf__ float* src, uint32_t offset)
{
    AscendC::Reg::LoadAlign<float, LoadDist::DIST_NORM>(dst, src + offset);
}

__simd_callee__ inline void StoreFp32Aligned(__ubuf__ float* dst, RegTensor<float>& src, uint32_t offset, MaskReg& mask)
{
    AscendC::Reg::StoreAlign<float, StoreDist::DIST_NORM>(dst + offset, src, mask);
}

// ------------------------------------------------------------------ 通道数学

// 一个 VL 块上的通道折叠(唯一一份通道数学)。**非折叠式**:只算 3 个量,不再算
// `addend = offset − mult·bm` —— offset 由调用方**原样**放进系数向量第 3 路,在热循环里最后相加。
//   bm      = sum * numRec                        // 乘倒数,非除法
//   saveVar = square_sum * numRec − bm²           // 平方差形式 E[x²]−μ²
//   mult    = scale / sqrt(saveVar + eps)         // +eps → sqrt → div(高精度模式)
// 共 **7 个向量算子**(Muls/Muls/Mul/Sub/Adds/Sqrt/Div)。
__simd_callee__ inline void FoldRegs(RegTensor<float>& sumReg, RegTensor<float>& sqReg, RegTensor<float>& scaleReg,
                                     RegTensor<float>& multReg, RegTensor<float>& bmReg, RegTensor<float>& saveVarReg,
                                     MaskReg& mask, float numRec, float epsilon)
{
    RegTensor<float> tmpReg;
    AscendC::Reg::Muls(bmReg, sumReg, numRec, mask);
    AscendC::Reg::Muls(saveVarReg, sqReg, numRec, mask);
    AscendC::Reg::Mul(tmpReg, bmReg, bmReg, mask);
    AscendC::Reg::Sub(saveVarReg, saveVarReg, tmpReg, mask);
    AscendC::Reg::Adds(tmpReg, saveVarReg, epsilon, mask);
    AscendC::Reg::Sqrt(tmpReg, tmpReg, mask);
    AscendC::Reg::Div<float, &kHighPrecisionDiv>(multReg, scaleReg, tmpReg, mask);
}

// VF 承载点 ①:按 VL 分块折叠 len(≤ kParamChunkLen)个通道。
// 4 路输入里**只有 sum/square_sum/scale 参与运算**;offset 只做**直通搬移**到第 3 路系数
// (非折叠式的"offset 不进折叠、由调用方原样放进系数向量"),不出现在任何算术里。
// 4 个出参:mult / bm / offset(直通)/ saveVar —— 相 X 取前三路,相 S 取 bm/saveVar。
__simd_vf__ inline void VFFoldChunk(__ubuf__ float* sumAddr, __ubuf__ float* sqAddr, __ubuf__ float* scaleAddr,
                                    __ubuf__ float* offsetAddr, __ubuf__ float* multAddr, __ubuf__ float* bmAddr,
                                    __ubuf__ float* offsetOutAddr, __ubuf__ float* saveVarAddr, float numRec,
                                    float epsilon, uint32_t len)
{
    __VEC_SCOPE__
    {
        RegTensor<float> sumReg;
        RegTensor<float> sqReg;
        RegTensor<float> scaleReg;
        RegTensor<float> offsetReg;
        RegTensor<float> multReg;
        RegTensor<float> bmReg;
        RegTensor<float> saveVarReg;
        const uint16_t loopCount = static_cast<uint16_t>((len + VL_FP32 - 1) / VL_FP32);
        for (uint16_t i = 0; i < loopCount; ++i) {
            const uint32_t offset = static_cast<uint32_t>(i) * VL_FP32;
            uint32_t maskCount = len - offset;
            MaskReg mask = AscendC::MicroAPI::UpdateMask<float>(maskCount);
            LoadFp32Aligned(sumReg, sumAddr, offset);
            LoadFp32Aligned(sqReg, sqAddr, offset);
            LoadFp32Aligned(scaleReg, scaleAddr, offset);
            LoadFp32Aligned(offsetReg, offsetAddr, offset);
            FoldRegs(sumReg, sqReg, scaleReg, multReg, bmReg, saveVarReg, mask, numRec, epsilon);
            StoreFp32Aligned(multAddr, multReg, offset, mask);
            StoreFp32Aligned(bmAddr, bmReg, offset, mask);
            StoreFp32Aligned(offsetOutAddr, offsetReg, offset, mask); // offset 直通(不参与算术)
            StoreFp32Aligned(saveVarAddr, saveVarReg, offset, mask);
        }
    }
}

// VF 承载点 ②:running 统计量的 EMA 合并 + 4 路 (C,) 结果的原地写出(仅相 S 单元执行)。
//   meanOut = bm * factor + mean * (1 - factor)
//   varOut  = (saveVar * besselScaler) * factor + variance * (1 - factor)
//   bm(有偏)/ saveVar(有偏)原样写出
__simd_vf__ inline void VFStatEma(__ubuf__ float* bmAddr, __ubuf__ float* saveVarAddr, __ubuf__ float* meanAddr,
                                  __ubuf__ float* varianceAddr, __ubuf__ float* meanOutAddr, __ubuf__ float* varOutAddr,
                                  float factor, float oneMinusFactor, float besselScaler, uint32_t len)
{
    __VEC_SCOPE__
    {
        RegTensor<float> bmReg;
        RegTensor<float> saveVarReg;
        RegTensor<float> meanReg;
        RegTensor<float> varianceReg;
        RegTensor<float> meanOutReg;
        RegTensor<float> varOutReg;
        RegTensor<float> tmpReg;
        const uint16_t loopCount = static_cast<uint16_t>((len + VL_FP32 - 1) / VL_FP32);
        for (uint16_t i = 0; i < loopCount; ++i) {
            const uint32_t offset = static_cast<uint32_t>(i) * VL_FP32;
            uint32_t maskCount = len - offset;
            MaskReg mask = AscendC::MicroAPI::UpdateMask<float>(maskCount);
            LoadFp32Aligned(bmReg, bmAddr, offset);
            LoadFp32Aligned(saveVarReg, saveVarAddr, offset);
            LoadFp32Aligned(meanReg, meanAddr, offset);
            LoadFp32Aligned(varianceReg, varianceAddr, offset);

            AscendC::Reg::Muls(meanOutReg, bmReg, factor, mask);
            AscendC::Reg::Muls(tmpReg, meanReg, oneMinusFactor, mask);
            AscendC::Reg::Add(meanOutReg, meanOutReg, tmpReg, mask);

            AscendC::Reg::Muls(tmpReg, saveVarReg, besselScaler, mask);
            AscendC::Reg::Muls(varOutReg, tmpReg, factor, mask);
            AscendC::Reg::Muls(tmpReg, varianceReg, oneMinusFactor, mask);
            AscendC::Reg::Add(varOutReg, varOutReg, tmpReg, mask);

            StoreFp32Aligned(meanOutAddr, meanOutReg, offset, mask);
            StoreFp32Aligned(varOutAddr, varOutReg, offset, mask);
        }
    }
}

// ------------------------------------------------------------------ channel-first 热循环

// VF 承载点 ③:channel-first 的通道窗热循环。逐平面 **3 次单元素广播**(mult / bm / offset),
// 平面内的 x/y 流走与 CL 逐行**同一对**非对齐惯用法(`LoadUnAlignPre`+`LoadUnAlign` /
// `StoreUnAlign`+每平面一次 `StoreUnAlignPost`)。
// **非折叠式**:y = (x − bm)·mult + offset(Sub → Mul → Add,三次舍入)。
// 系数取用的"非零元素偏移广播"仓内已上库(bn_training_update_v3 的同款用法)。
template <typename T_X>
__simd_vf__ inline void VFChannelFirst(__ubuf__ T_X* xAddr, __ubuf__ T_X* yAddr, __ubuf__ float* coeffMult,
                                       __ubuf__ float* coeffBm, __ubuf__ float* coeffOffset, uint32_t planes,
                                       uint32_t count)
{
    __VEC_SCOPE__
    {
        RegTensor<float> multReg;
        RegTensor<float> bmReg;
        RegTensor<float> addReg;
        RegTensor<float> xReg;
        RegTensor<float> yReg;
        const uint16_t planeCount = static_cast<uint16_t>(planes);
        const uint16_t loopCount = static_cast<uint16_t>((count + VL_FP32 - 1) / VL_FP32);
        for (uint16_t row = 0; row < planeCount; ++row) {
            const uint32_t rowBase = static_cast<uint32_t>(row) * count;
            __ubuf__ T_X* xRow = xAddr + rowBase;
            __ubuf__ T_X* yRow = yAddr + rowBase;
            AscendC::Reg::LoadAlign<float, LoadDist::DIST_BRC_B32>(multReg, coeffMult + row);
            AscendC::Reg::LoadAlign<float, LoadDist::DIST_BRC_B32>(bmReg, coeffBm + row);
            AscendC::Reg::LoadAlign<float, LoadDist::DIST_BRC_B32>(addReg, coeffOffset + row);
            UnalignRegForLoad xState;
            UnalignRegForStore yState;
            AscendC::MicroAPI::LoadUnAlignPre(xState, xRow);
            for (uint16_t i = 0; i < loopCount; ++i) {
                const uint32_t offset = static_cast<uint32_t>(i) * VL_FP32;
                const uint32_t remain = count - offset;
                const uint32_t active = remain < VL_FP32 ? remain : VL_FP32;
                uint32_t maskCount = active;
                MaskReg mask = AscendC::MicroAPI::UpdateMask<float>(maskCount);
                LoadUnalignedXToFp32<T_X>(xRow, xReg, xState, mask, active);
                AscendC::Reg::Sub(yReg, xReg, bmReg, mask);   // (x − bm)
                AscendC::Reg::Mul(yReg, yReg, multReg, mask); // × multiplier
                AscendC::Reg::Add(yReg, yReg, addReg, mask);  // + offset
                StoreUnalignedYFromFp32<T_X>(yRow, yReg, yState, mask, active);
            }
            AscendC::MicroAPI::StoreUnAlignPost(yRow, yState, 0);
        }
    }
}

// ------------------------------------------------------------------ channel-last 热循环

// 系数寄存器驻留的块数上界:块外置后常驻寄存器 O(1) = 3 个,与 R 无关 ⇒ 该上界不再是
// 寄存器约束(设 3 覆盖 cLen ≤ 3·VL = 192)。R 更大时回落"行优先 + 行内逐块读系数"。
constexpr int64_t kCoeffResidentBlocks = 3;

// __simd_callee__:一个 (行, VL 块) 片段的非折叠式逐元素计算。
//   y = (x − bm)·mult + offset   (Sub → Mul → Add,三次舍入)
// 系数由调用方**以寄存器传入**(驻留路径传块外预读的三路寄存器,回落路径传行内逐块读的寄存器)。
// 非对齐访存成对(Pre 由调用方在片段首做、Post 在片段末做)。
template <typename T_X>
__simd_callee__ inline void RowBlockMulSubAdd(__ubuf__ T_X*& xRow, __ubuf__ T_X*& yRow, UnalignRegForLoad& xState,
                                              UnalignRegForStore& yState, RegTensor<float>& multReg,
                                              RegTensor<float>& bmReg, RegTensor<float>& offReg, uint32_t active)
{
    uint32_t maskCount = active;
    MaskReg mask = AscendC::MicroAPI::UpdateMask<float>(maskCount);
    RegTensor<float> xReg;
    RegTensor<float> yReg;
    LoadUnalignedXToFp32<T_X>(xRow, xReg, xState, mask, active);
    AscendC::Reg::Sub(yReg, xReg, bmReg, mask);
    AscendC::Reg::Mul(yReg, yReg, multReg, mask);
    AscendC::Reg::Add(yReg, yReg, offReg, mask);
    StoreUnalignedYFromFp32<T_X>(yRow, yReg, yState, mask, active);
}

// VF 承载点 ④:channel-last 热循环(**块外置、行内置**)。
//   `R = ceil(cLen/VL) <= kCoeffResidentBlocks` 时:外层遍历 VL 块(**系数只依赖块** ⇒ 三路系数
//   常驻 3 个寄存器,与 R 无关 = O(1)),内层遍历行 —— 每 (行,块) 片段重新 Pre/Post,因为相邻行在 UB
//   内相隔 `cLen`,不是连续流,无法共用一个非对齐状态。
//   R 更大时回落"行优先 + 行内逐块读系数"(结构与回落分支一致:同顺序、同掩码、同一对惯用法;
//   差异仅为本轮强制的两处——系数 2 路→3 路、算子 2 个→3 个)。
// 行内系数恒按 VL 整数倍偏移**对齐读**,行内下标 == 通道下标 ⇒ 无相位回绕、无周期展开。
// **VF 内不得 return**(后端 `Outputs are not allowed in vector function`)⇒ 完整 if / else 分支体。
template <typename T_X>
__simd_vf__ inline void VFChannelLast(__ubuf__ T_X* xAddr, __ubuf__ T_X* yAddr, __ubuf__ float* coeffMult,
                                      __ubuf__ float* coeffBm, __ubuf__ float* coeffOffset, uint32_t rows,
                                      uint32_t cLen)
{
    __VEC_SCOPE__
    {
        const uint16_t rowCount = static_cast<uint16_t>(rows);
        const uint16_t blockCount = static_cast<uint16_t>((cLen + VL_FP32 - 1) / VL_FP32);
        if (blockCount <= static_cast<uint16_t>(kCoeffResidentBlocks)) {
            RegTensor<float> multReg;
            RegTensor<float> bmReg;
            RegTensor<float> offReg;
            for (uint16_t blk = 0; blk < blockCount; ++blk) {
                const uint32_t blkOffset = static_cast<uint32_t>(blk) * VL_FP32;
                const uint32_t blkRemain = cLen - blkOffset;
                const uint32_t blkActive = blkRemain < VL_FP32 ? blkRemain : VL_FP32;
                LoadFp32Aligned(multReg, coeffMult, blkOffset); // 每瓦片每块 3 次读(rows 倍摊薄)
                LoadFp32Aligned(bmReg, coeffBm, blkOffset);
                LoadFp32Aligned(offReg, coeffOffset, blkOffset);
                for (uint16_t row = 0; row < rowCount; ++row) {
                    const uint32_t rowBase = static_cast<uint32_t>(row) * cLen + blkOffset;
                    __ubuf__ T_X* xRow = xAddr + rowBase;
                    __ubuf__ T_X* yRow = yAddr + rowBase;
                    UnalignRegForLoad xState;
                    UnalignRegForStore yState;
                    AscendC::MicroAPI::LoadUnAlignPre(xState, xRow);
                    RowBlockMulSubAdd<T_X>(xRow, yRow, xState, yState, multReg, bmReg, offReg, blkActive);
                    AscendC::MicroAPI::StoreUnAlignPost(yRow, yState, 0);
                }
            }
        } else {
            RegTensor<float> multReg;
            RegTensor<float> bmReg;
            RegTensor<float> offReg;
            RegTensor<float> xReg;
            RegTensor<float> yReg;
            for (uint16_t row = 0; row < rowCount; ++row) {
                const uint32_t rowBase = static_cast<uint32_t>(row) * cLen;
                __ubuf__ T_X* xRow = xAddr + rowBase;
                __ubuf__ T_X* yRow = yAddr + rowBase;
                UnalignRegForLoad xState;
                UnalignRegForStore yState;
                AscendC::MicroAPI::LoadUnAlignPre(xState, xRow);
                for (uint16_t i = 0; i < blockCount; ++i) {
                    const uint32_t offset = static_cast<uint32_t>(i) * VL_FP32;
                    const uint32_t remain = cLen - offset;
                    const uint32_t active = remain < VL_FP32 ? remain : VL_FP32;
                    uint32_t maskCount = active;
                    MaskReg mask = AscendC::MicroAPI::UpdateMask<float>(maskCount);
                    LoadUnalignedXToFp32<T_X>(xRow, xReg, xState, mask, active);
                    LoadFp32Aligned(multReg, coeffMult, offset);
                    LoadFp32Aligned(bmReg, coeffBm, offset);
                    LoadFp32Aligned(offReg, coeffOffset, offset);
                    AscendC::Reg::Sub(yReg, xReg, bmReg, mask);
                    AscendC::Reg::Mul(yReg, yReg, multReg, mask);
                    AscendC::Reg::Add(yReg, yReg, offReg, mask);
                    StoreUnalignedYFromFp32<T_X>(yRow, yReg, yState, mask, active);
                }
                AscendC::MicroAPI::StoreUnAlignPost(yRow, yState, 0);
            }
        }
    }
}

} // namespace Bn3dTrainingUpdateOps

#endif // BN3_D_TRAINING_UPDATE_COMMON_H
