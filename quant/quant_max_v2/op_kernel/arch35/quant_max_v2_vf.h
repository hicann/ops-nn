/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// QuantMaxV2 向量函数，通过 asc_vf_call 发射；仅接收 UB 地址和标量参数，不在 VF 内执行标量 UB
// 访问，组最大值通过广播加载续接。

#ifndef QUANT_MAX_V2_VF_H
#define QUANT_MAX_V2_VF_H

#include "kernel_operator.h"
#include "quant_max_v2_common.h"

namespace QuantMaxV2 {
using namespace AscendC;

// 清除 BF16/FP16 符号位，非负绝对值可以按 uint16 比较。
constexpr uint16_t BF16_ABS_MASK = 0x7fff;

// 半字宽度：位级求最大值时右移/左移取高低半字。
constexpr int16_t B16_SHIFT_BITS = 16;

// 四路展开因子（FP32 首遍累加链与跨核归并）。
constexpr uint16_t VF_UNROLL_FACTOR = 4;

template <typename T, typename U, uint64_t RoundMode, bool WriteY, bool ComputeMax, bool DualChain = false>
__simd_vf__ inline void GroupedQuantComputeVf(const __ubuf__ T* xAddr, __ubuf__ U* outAddr, __ubuf__ float* maxAddr,
                                              float scaleValue, uint32_t dataCount, int64_t groupIndex,
                                              uint32_t vectorLength);

__simd_callee__ inline void GroupedQuantFp32MaxVf(const __ubuf__ float* xAddr, __ubuf__ float* maxAddr,
                                                  uint32_t dataCount, int64_t groupIndex, uint32_t vectorLength);

template <typename T, typename U, uint64_t RoundMode>
__simd_callee__ inline void GroupedQuantWidenMaxVf(const __ubuf__ T* xAddr, __ubuf__ float* maxAddr, uint32_t dataCount,
                                                   int64_t groupIndex, uint32_t vectorLength);

template <typename T, typename U, uint64_t RoundMode>
__simd_callee__ inline void GroupedQuantQuantMaxVf(const __ubuf__ T* xAddr, __ubuf__ U* outAddr,
                                                   __ubuf__ float* maxAddr, float scaleValue, uint32_t dataCount,
                                                   int64_t groupIndex, uint32_t vectorLength);

template <typename T, typename U, uint64_t RoundMode, bool DualChain>
__simd_callee__ inline void GroupedQuantQuantOnlyVf(const __ubuf__ T* xAddr, __ubuf__ U* outAddr, float scaleValue,
                                                    uint32_t dataCount, uint32_t vectorLength);

__simd_vf__ inline void GroupedQuantBf16TileMaxVf(const __ubuf__ uint16_t* xAddr, __ubuf__ uint16_t* tileMaxAddr,
                                                  uint32_t dataCount, uint32_t tileMaxCount, uint32_t vectorLengthB16);

template <typename T, typename U, uint64_t RoundMode>
__simd_vf__ inline void GroupedQuantBf16GroupMaxVf(const __ubuf__ uint16_t* tileMaxAddr, __ubuf__ float* groupMaxAddr,
                                                   uint32_t count, uint32_t vectorLengthB16);

__simd_vf__ inline void GroupedQuantMergeSingleGroupVf(const __ubuf__ float* groupMaxAddr, __ubuf__ float* mergedAddr,
                                                       uint32_t count, uint32_t vectorLength);

__simd_vf__ inline void GroupedQuantMergeScaleVf(const __ubuf__ float* groupMaxAddr, __ubuf__ float* mergedAddr,
                                                 uint32_t numGroups, uint16_t actualCoreNum, int64_t groupStride,
                                                 uint32_t vectorLength);

__simd_vf__ inline void GroupedQuantMergeAmaxVf(const __ubuf__ float* groupMaxAddr, __ubuf__ float* mergedAddr,
                                                uint32_t numGroups, uint16_t actualCoreNum, int64_t groupStride,
                                                uint32_t vectorLength);

template <typename T, typename U, uint64_t RoundMode>
__simd_vf__ inline void GroupedQuantCastAmaxVf(const __ubuf__ float* mergedAddr, __ubuf__ T* amaxAddr,
                                               uint32_t numGroups, uint32_t vectorLength);

template <typename T, typename U, uint64_t RoundMode>
__simd_vf__ inline void GroupedQuantEmitYScaleVf(const __ubuf__ float* amaxAddr, __ubuf__ float* yScaleAddr,
                                                 uint32_t numGroups, float invDstTypeMax, uint32_t vectorLength);

template <typename T, typename U, uint64_t RoundMode, bool WriteY, bool ComputeMax, bool DualChain>
// 量化主计算 VF：静态单趟与动态第二遍共用的 tile 级计算，本算子向量侧热路径核心。
// 计算：
//   WriteY      y[i] = Cast(x[i] * scaleValue, dstType, roundMode)，逐向量量化写 outAddr；
//   ComputeMax  组内绝对值最大值续接归约 max(groupMax, max(|x[0..dataCount)|))，结果写 maxAddr[groupIndex]（首元素单
//   lane 存储）。
// 参数：
//   xAddr         输入 tile 首地址（UB，只读）；bf16/fp16 经 UNPACK_B16 装载（每个 32 位槽承载一个 16 位元素）。
//   outAddr       y 输出 tile 首地址（UB）；WriteY=false 时调用方传 nullptr，本函数不访问。
//   maxAddr       组最大值数组（UB，float，按 groupIndex 寻址）；ComputeMax=true 时先读旧值广播续接、
//                 归约后写回同槽；仅量化时调用方仍传值但不访问。
//   scaleValue    标量量化系数（乘性），由调用方标量传入：静态=外部给定 scale[g]，动态二遍=dstTypeMax/amax[g]。
//   dataCount     本 tile 元素数；尾块为实际长度（<vectorLength），尾部向量使用精确掩码。
//   groupIndex    组号，仅用于 maxAddr 寻址；tensor 路恒为 0。
//   vectorLength  fp32 域向量长度（lane 数 = VF_BYTES/sizeof(float)）；16 位输入按 UNPACK 布局以同长度消费。
// 模板组合（三种实例化场景）：
//   <true, true>  静态单趟：量化与求最大值同路，配对环双链 + bf16 位级 abs/max 支路；
//   <false, true> 动态第一遍（仅求最大值）：fp32 走四路独立累加链；fp16 走通用单链（else 分支）；
//                 bf16 不经过此实例，改走专用 GroupedQuantBf16TileMaxVf（位级 16 位暂存）；
//   <true, false> 动态第二遍（仅量化）：scaleValue = dstTypeMax / amax[g]，amax 为 0 时 scaleValue = 0，y 输出全零；
//   DualChain     仅 <true,false> + bf16 生效：配对环（双
//   Load→Cast→Mul→Cast→Store）编译期分离实例化，单链变体不承担配对环寄存器压力。
__simd_vf__ inline void GroupedQuantComputeVf(const __ubuf__ T* xAddr, __ubuf__ U* outAddr, __ubuf__ float* maxAddr,
                                              float scaleValue, uint32_t dataCount, int64_t groupIndex,
                                              uint32_t vectorLength)
{
    // 按编译期场景分派到四个职责函数，本函数只做转发（各场景的寄存器/循环形态见各自文档）。
    if constexpr (WriteY && ComputeMax) {
        GroupedQuantQuantMaxVf<T, U, RoundMode>(xAddr, outAddr, maxAddr, scaleValue, dataCount, groupIndex,
                                                vectorLength);
    } else if constexpr (WriteY) {
        GroupedQuantQuantOnlyVf<T, U, RoundMode, DualChain>(xAddr, outAddr, scaleValue, dataCount, vectorLength);
    } else if constexpr (IsSameType<T, float>::value) {
        GroupedQuantFp32MaxVf((const __ubuf__ float*)xAddr, maxAddr, dataCount, groupIndex, vectorLength);
    } else {
        GroupedQuantWidenMaxVf<T, U, RoundMode>(xAddr, maxAddr, dataCount, groupIndex, vectorLength);
    }
}

// 动态第一遍（fp32 输入，仅求最大值）：四路独立累加链打破 Max 串行依赖。
// 计算：maxAddr[groupIndex] = max(旧值, max(|x[0..dataCount)|))
// fp32 输入天然在 fp32 域，四路 Max 累加零转换成本；其余 dtype 不走本函数
// （fp16 走 GroupedQuantWidenMaxVf，bf16 走 GroupedQuantBf16TileMaxVf 位级链）。
__simd_callee__ inline void GroupedQuantFp32MaxVf(const __ubuf__ float* xAddr, __ubuf__ float* maxAddr,
                                                  uint32_t dataCount, int64_t groupIndex, uint32_t vectorLength)
{
    // AscendC 的 LoadAlign 形参为非 const 指针：以只读别名转回；本函数不得经该别名写入。
    __ubuf__ float* xRd = (__ubuf__ float*)xAddr;
    AscendC::Reg::RegTensor<float> vregFloatX;
    AscendC::Reg::RegTensor<float> vregFloatAbsT;
    AscendC::Reg::RegTensor<float> vregTileMaxFp32;
    AscendC::Reg::MaskReg mask = AscendC::Reg::CreateMask<float>();
    AscendC::Reg::MaskReg maskAll = AscendC::Reg::CreateMask<float>();
    uint32_t countOne = 1;
    AscendC::Reg::MaskReg maskOne = AscendC::Reg::UpdateMask<float>(countOne);

    // 广播加载前一 tile 写入的最大值以续接归约；前一 VF 存储先于本次向量加载，VF 内不执行标量 UB 访问。
    // 本组函数均求最大值，装载无条件执行（原 ComputeMax 门控在本层恒真）。
    AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(vregTileMaxFp32, maxAddr + groupIndex);
    // FP32 首遍用四路累积打破 Max 串行依赖；整向量掩码移出循环，仅尾部使用精确掩码。
    uint32_t fullCount = vectorLength;
    AscendC::Reg::MaskReg maskFull = AscendC::Reg::UpdateMask<float>(fullCount);
    const uint16_t fullLoops = static_cast<uint16_t>(dataCount / vectorLength);
    uint32_t tailCount = dataCount % vectorLength;
    AscendC::Reg::RegTensor<float> vregAcc1;
    AscendC::Reg::RegTensor<float> vregAcc2;
    AscendC::Reg::RegTensor<float> vregAcc3;
    // 四路累积均从广播的组最大值开始。
    AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(vregAcc1, maxAddr + groupIndex);
    AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(vregAcc2, maxAddr + groupIndex);
    AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(vregAcc3, maxAddr + groupIndex);

    const uint16_t unroll4 = (fullLoops / VF_UNROLL_FACTOR) * VF_UNROLL_FACTOR;
    for (uint16_t block = 0; block < unroll4; block += VF_UNROLL_FACTOR) {
        AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(vregFloatX, xRd + block * vectorLength);
        AscendC::Reg::Abs<float>(vregFloatAbsT, vregFloatX, maskFull);
        AscendC::Reg::Max<float, AscendC::Reg::MaskMergeMode::MERGING>(vregTileMaxFp32, vregTileMaxFp32, vregFloatAbsT,
                                                                       maskFull);
        AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(vregFloatX, xRd + (block + 1) * vectorLength);
        AscendC::Reg::Abs<float>(vregFloatAbsT, vregFloatX, maskFull);
        AscendC::Reg::Max<float, AscendC::Reg::MaskMergeMode::MERGING>(vregAcc1, vregAcc1, vregFloatAbsT, maskFull);
        AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(vregFloatX, xRd + (block + 2) * vectorLength);
        AscendC::Reg::Abs<float>(vregFloatAbsT, vregFloatX, maskFull);
        AscendC::Reg::Max<float, AscendC::Reg::MaskMergeMode::MERGING>(vregAcc2, vregAcc2, vregFloatAbsT, maskFull);
        AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(vregFloatX, xRd + (block + 3) * vectorLength);
        AscendC::Reg::Abs<float>(vregFloatAbsT, vregFloatX, maskFull);
        AscendC::Reg::Max<float, AscendC::Reg::MaskMergeMode::MERGING>(vregAcc3, vregAcc3, vregFloatAbsT, maskFull);
    }
    // 剩余整向量使用单路累积。
    for (uint16_t i = unroll4; i < fullLoops; ++i) {
        AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(vregFloatX, xRd + i * vectorLength);
        AscendC::Reg::Abs<float>(vregFloatAbsT, vregFloatX, maskFull);
        AscendC::Reg::Max<float, AscendC::Reg::MaskMergeMode::MERGING>(vregTileMaxFp32, vregTileMaxFp32, vregFloatAbsT,
                                                                       maskFull);
    }
    // 尾向量统一以掩码处理，不在 VF 体内引入运行时分支（KERNEL-13）：tailCount==0 时掩码为空，
    // Abs/Max 不改变累加器。空掩码仍会发起一次整向量装载，故地址在无尾量时回退到最后一个
    // 整向量，保证不越过 tile 的 UB 缓冲区；无整向量（dataCount < vectorLength）时从首元素装载。
    const uint32_t tailOffset = tailCount != 0 ? fullLoops * vectorLength :
                                                 (fullLoops != 0 ? (fullLoops - 1) * vectorLength : 0);
    // UpdateMask 为 POST_UPDATE 语义（入参被改写成剩余计数），必须传副本，否则其后的 tailCount
    // 判断/复算会读到被消耗后的值。
    uint32_t tailMaskCount = tailCount;
    mask = AscendC::Reg::UpdateMask<float>(tailMaskCount);
    AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(vregFloatX, xRd + tailOffset);
    AscendC::Reg::Abs<float>(vregFloatAbsT, vregFloatX, mask);
    AscendC::Reg::Max<float, AscendC::Reg::MaskMergeMode::MERGING>(vregTileMaxFp32, vregTileMaxFp32, vregFloatAbsT,
                                                                   mask);
    // 合并四路结果，再统一归约和存储。
    AscendC::Reg::Max<float, AscendC::Reg::MaskMergeMode::MERGING>(vregTileMaxFp32, vregTileMaxFp32, vregAcc1, maskAll);
    AscendC::Reg::Max<float, AscendC::Reg::MaskMergeMode::MERGING>(vregAcc2, vregAcc2, vregAcc3, maskAll);
    AscendC::Reg::Max<float, AscendC::Reg::MaskMergeMode::MERGING>(vregTileMaxFp32, vregTileMaxFp32, vregAcc2, maskAll);
    // 求最大值收尾：跨 lane 归约（Reduce MAX）成单值后写回 maxAddr[groupIndex]（单 lane 存储）。
    // 本组函数无位级暂存（bf16 位级支路在 GroupedQuantQuantMaxVf / Bf16TileMaxVf 内），直接归约 fp32。
    AscendC::Reg::Reduce<Reg::ReduceType::MAX>(vregTileMaxFp32, vregTileMaxFp32, maskAll);
    AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(maxAddr + groupIndex,
                                                                                     vregTileMaxFp32, maskOne);
}

// 动态第一遍（fp16 输入，仅求最大值）：UNPACK_B16 装载 + 一条加宽 Cast 后单路 Abs/Max。
// 计算：maxAddr[groupIndex] = max(旧值, max(|x[0..dataCount)|))
// 与 fp32 分支的区别只在装载：fp16 需先加宽到 fp32 域；未做四路累加（单条 Cast 后
// Max 依赖链已由装载延迟摊薄，实测无额外收益）。
template <typename T, typename U, uint64_t RoundMode>
__simd_callee__ inline void GroupedQuantWidenMaxVf(const __ubuf__ T* xAddr, __ubuf__ float* maxAddr, uint32_t dataCount,
                                                   int64_t groupIndex, uint32_t vectorLength)
{
    // AscendC 的 LoadAlign 形参为非 const 指针：以只读别名转回；本函数不得经该别名写入。
    __ubuf__ T* xRd = (__ubuf__ T*)xAddr;
    AscendC::Reg::RegTensor<T> vregX;
    AscendC::Reg::RegTensor<float> vregFloatAbsT;
    AscendC::Reg::RegTensor<float> vregTileMaxFp32;
    AscendC::Reg::RegTensor<float> vregFloatX;
    AscendC::Reg::MaskReg mask = AscendC::Reg::CreateMask<float>();
    AscendC::Reg::MaskReg maskAll = AscendC::Reg::CreateMask<float>();
    uint32_t countOne = 1;
    AscendC::Reg::MaskReg maskOne = AscendC::Reg::UpdateMask<float>(countOne);

    // 广播加载前一 tile 写入的最大值以续接归约；前一 VF 存储先于本次向量加载，VF 内不执行标量 UB 访问。
    // 本组函数均求最大值，装载无条件执行（原 ComputeMax 门控在本层恒真）。
    AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(vregTileMaxFp32, maxAddr + groupIndex);

    const uint16_t fullLoops = static_cast<uint16_t>(dataCount / vectorLength);
    uint32_t tailCount = dataCount % vectorLength;
    uint32_t fullCount = vectorLength;
    AscendC::Reg::MaskReg maskFull = AscendC::Reg::UpdateMask<float>(fullCount);
    for (uint16_t i = 0; i < fullLoops; ++i) {
        AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(vregX, xRd + i * vectorLength);
        // fp16 用半精度加宽 trait；bf16 实例化在调用方被改道、不会执行，但保持与原三路装载逐位一致。
        if constexpr (IsSameType<T, half>::value) {
            AscendC::Reg::Cast<float, half, QuantMaxV2Base<T, U, RoundMode>::CAST_TRAIT_HALF_TO_FP32>(vregFloatX, vregX,
                                                                                                      maskFull);
        } else {
            AscendC::Reg::Cast<float, bfloat16_t, QuantMaxV2Base<T, U, RoundMode>::CAST_TRAIT_BF16_TO_FP32>(
                vregFloatX, (AscendC::Reg::RegTensor<bfloat16_t>&)vregX, maskFull);
        }
        AscendC::Reg::Abs<float>(vregFloatAbsT, vregFloatX, maskFull);
        AscendC::Reg::Max<float, AscendC::Reg::MaskMergeMode::MERGING>(vregTileMaxFp32, vregTileMaxFp32, vregFloatAbsT,
                                                                       maskFull);
    }
    // 尾向量统一以掩码处理，不在 VF 体内引入运行时分支（KERNEL-13）：tailCount==0 时掩码为空，
    // Abs/Max 不改变累加器。空掩码仍会发起一次整向量装载，故地址在无尾量时回退到最后一个
    // 整向量，保证不越过 tile 的 UB 缓冲区；无整向量（dataCount < vectorLength）时从首元素装载。
    // 注意：UpdateMask 为 POST_UPDATE 语义（入参被改写成剩余计数），地址须先由原始 tailCount
    // 算出，并把副本交给 UpdateMask。
    const uint32_t tailOffset = tailCount != 0 ? fullLoops * vectorLength :
                                                 (fullLoops != 0 ? (fullLoops - 1) * vectorLength : 0);
    uint32_t tailMaskCount = tailCount;
    mask = AscendC::Reg::UpdateMask<float>(tailMaskCount);
    AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(vregX, xRd + tailOffset);
    if constexpr (IsSameType<T, half>::value) {
        AscendC::Reg::Cast<float, half, QuantMaxV2Base<T, U, RoundMode>::CAST_TRAIT_HALF_TO_FP32>(vregFloatX, vregX,
                                                                                                  mask);
    } else {
        AscendC::Reg::Cast<float, bfloat16_t, QuantMaxV2Base<T, U, RoundMode>::CAST_TRAIT_BF16_TO_FP32>(
            vregFloatX, (AscendC::Reg::RegTensor<bfloat16_t>&)vregX, mask);
    }
    AscendC::Reg::Abs<float>(vregFloatAbsT, vregFloatX, mask);
    AscendC::Reg::Max<float, AscendC::Reg::MaskMergeMode::MERGING>(vregTileMaxFp32, vregTileMaxFp32, vregFloatAbsT,
                                                                   mask);
    // 求最大值收尾：跨 lane 归约（Reduce MAX）成单值后写回 maxAddr[groupIndex]（单 lane 存储）。
    // 本组函数无位级暂存（bf16 位级支路在 GroupedQuantQuantMaxVf / Bf16TileMaxVf 内），直接归约 fp32。
    AscendC::Reg::Reduce<Reg::ReduceType::MAX>(vregTileMaxFp32, vregTileMaxFp32, maskAll);
    AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(maxAddr + groupIndex,
                                                                                     vregTileMaxFp32, maskOne);
}

// 静态单趟（量化 + 求最大值同路）：配对环双链使加载与计算重叠，bf16 走位级 abs/max 支路。
// 计算：y[i] = Cast(x[i] * scaleValue, dstType, roundMode)；maxAddr[groupIndex] =
//       max(旧值, max(|x[0..dataCount)|))
// 配对环对全 dtype 生效（两路最大值链交替推进）；bf16 的位级支路省去 fp32 域加宽比较，
// 结果以 16 位位模式暂存、收尾一次性还原。
template <typename T, typename U, uint64_t RoundMode>
__simd_callee__ inline void GroupedQuantQuantMaxVf(const __ubuf__ T* xAddr, __ubuf__ U* outAddr,
                                                   __ubuf__ float* maxAddr, float scaleValue, uint32_t dataCount,
                                                   int64_t groupIndex, uint32_t vectorLength)
{
    // 本特化恒为量化 + 求最大值（原 WriteY/ComputeMax 模板参在本函数内为常量）。
    constexpr bool WriteY = true;
    constexpr bool ComputeMax = true;
    // AscendC 的 LoadAlign 形参为非 const 指针：以只读别名转回；本函数不得经该别名写入。
    __ubuf__ T* xRd = (__ubuf__ T*)xAddr;
    AscendC::Reg::RegTensor<T> vregX;
    AscendC::Reg::RegTensor<float> vregFloatAbsT;
    AscendC::Reg::RegTensor<float> vregTileMaxFp32;
    constexpr bool staticBf16Bits = IsSameType<T, bfloat16_t>::value;
    AscendC::Reg::RegTensor<uint32_t> vregBitsMax;
    AscendC::Reg::RegTensor<uint32_t> vregBitsMax1;
    AscendC::Reg::RegTensor<uint32_t> vregBitsAbs;
    AscendC::Reg::RegTensor<uint32_t> vregBitsMask;
    AscendC::Reg::RegTensor<float> vregFloatX;
    AscendC::Reg::RegTensor<float> vregS;
    AscendC::Reg::RegTensor<float> vregFloatY;
    AscendC::Reg::RegTensor<U> vregY;
    AscendC::Reg::MaskReg mask = AscendC::Reg::CreateMask<float>();
    AscendC::Reg::MaskReg maskAll = AscendC::Reg::CreateMask<float>();
    uint32_t countOne = 1;
    AscendC::Reg::MaskReg maskOne = AscendC::Reg::UpdateMask<float>(countOne);

    // 广播加载前一 tile 写入的最大值以续接归约；前一 VF 存储先于本次向量加载，VF 内不执行标量 UB 访问。
    // 本组函数均求最大值，装载无条件执行（原 ComputeMax 门控在本层恒真）。
    AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(vregTileMaxFp32, maxAddr + groupIndex);
    if constexpr (staticBf16Bits) {
        // DIST_UNPACK_B16 将 BF16 数据放在 B32 通道低半部；前一非负最大值可精确表示为 BF16。
        AscendC::Reg::ShiftRights(vregBitsMax, (AscendC::Reg::RegTensor<uint32_t>&)vregTileMaxFp32,
                                  static_cast<int16_t>(B16_SHIFT_BITS), maskAll);
        AscendC::Reg::Duplicate<uint32_t>(vregBitsMask, BF16_ABS_MASK, maskAll);
    }
    // 量化路径：把量化系数 scaleValue 广播进向量寄存器。
    if constexpr (WriteY) {
        AscendC::Reg::Duplicate<float>(vregS, scaleValue, mask);
    }

    const uint16_t fullLoops = static_cast<uint16_t>(dataCount / vectorLength);
    uint32_t tailCount = dataCount % vectorLength;
    uint32_t fullCount = vectorLength;
    AscendC::Reg::MaskReg maskFull = AscendC::Reg::UpdateMask<float>(fullCount);
    uint16_t firstSingleLoop = 0;
    // 静态量化使用两条独立加载/计算链，使加载与计算重叠；每条最大值链交替推进，保留原类型转换和存储规则。
    AscendC::Reg::RegTensor<T> vregX1;
    AscendC::Reg::RegTensor<float> vregFloatX1;
    AscendC::Reg::RegTensor<float> vregFloatAbs1;
    AscendC::Reg::RegTensor<float> vregTileMaxFp32_1;
    AscendC::Reg::RegTensor<float> vregFloatY1;
    AscendC::Reg::RegTensor<U> vregY1;
    AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(vregTileMaxFp32_1, maxAddr + groupIndex);
    // 第二路链的位级初值：与主路同法右移取高半字。
    if constexpr (staticBf16Bits) {
        AscendC::Reg::ShiftRights(vregBitsMax1, (AscendC::Reg::RegTensor<uint32_t>&)vregTileMaxFp32_1,
                                  static_cast<int16_t>(B16_SHIFT_BITS), maskAll);
    }
    const uint16_t pairedLoops = (fullLoops / 2) * 2;
    for (uint16_t i = 0; i < pairedLoops; i += 2) {
        // fp32 输入：原生装载即用，无需类型转换。
        if constexpr (IsSameType<T, float>::value) {
            AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(vregFloatX, xRd + i * vectorLength);
            AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(vregFloatX1,
                                                                              xRd + (i + 1) * vectorLength);
            // fp16 输入：UNPACK_B16 装载（32 位槽承载一个 half，保证 lane 对齐）后 Cast 到 fp32。
        } else if constexpr (IsSameType<T, half>::value) {
            AscendC::Reg::LoadAlign<half, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(vregX, xRd + i * vectorLength);
            AscendC::Reg::LoadAlign<half, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(vregX1,
                                                                                   xRd + (i + 1) * vectorLength);
            AscendC::Reg::Cast<float, half, QuantMaxV2Base<T, U, RoundMode>::CAST_TRAIT_HALF_TO_FP32>(vregFloatX, vregX,
                                                                                                      maskFull);
            AscendC::Reg::Cast<float, half, QuantMaxV2Base<T, U, RoundMode>::CAST_TRAIT_HALF_TO_FP32>(vregFloatX1,
                                                                                                      vregX1, maskFull);
            // bf16 输入：UNPACK_B16 装载后处理（配对环内直接移位构造 fp32 位模式）。
        } else {
            AscendC::Reg::LoadAlign<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(vregX, xRd + i * vectorLength);
            AscendC::Reg::LoadAlign<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(vregX1,
                                                                                         xRd + (i + 1) * vectorLength);
            // UNPACK_B16 每个32位槽承载一个BF16；左移16位直接构造FP32位模式。
            // 仅改变静态双路输入展开，乘法和输出Cast仍使用原FP32路径。
            AscendC::Reg::ShiftLefts((AscendC::Reg::RegTensor<uint32_t>&)vregFloatX,
                                     (AscendC::Reg::RegTensor<uint32_t>&)vregX, static_cast<int16_t>(B16_SHIFT_BITS),
                                     maskFull);
            AscendC::Reg::ShiftLefts((AscendC::Reg::RegTensor<uint32_t>&)vregFloatX1,
                                     (AscendC::Reg::RegTensor<uint32_t>&)vregX1, static_cast<int16_t>(B16_SHIFT_BITS),
                                     maskFull);
        }
        // 位级最大值更新（配对环双路）：清符号位后按位模式比较，免除 fp32 域转换。
        if constexpr (staticBf16Bits) {
            AscendC::Reg::And(vregBitsAbs, (AscendC::Reg::RegTensor<uint32_t>&)vregX, vregBitsMask, maskFull);
            AscendC::Reg::Max<uint32_t, AscendC::Reg::MaskMergeMode::MERGING>(vregBitsMax, vregBitsMax, vregBitsAbs,
                                                                              maskFull);
            AscendC::Reg::And(vregBitsAbs, (AscendC::Reg::RegTensor<uint32_t>&)vregX1, vregBitsMask, maskFull);
            AscendC::Reg::Max<uint32_t, AscendC::Reg::MaskMergeMode::MERGING>(vregBitsMax1, vregBitsMax1, vregBitsAbs,
                                                                              maskFull);
            // fp32 域最大值更新（配对环双路）：Abs 后 Max，两路各自推进。
        } else {
            AscendC::Reg::Abs<float>(vregFloatAbsT, vregFloatX, maskFull);
            AscendC::Reg::Abs<float>(vregFloatAbs1, vregFloatX1, maskFull);
            AscendC::Reg::Max<float, AscendC::Reg::MaskMergeMode::MERGING>(vregTileMaxFp32, vregTileMaxFp32,
                                                                           vregFloatAbsT, maskFull);
            AscendC::Reg::Max<float, AscendC::Reg::MaskMergeMode::MERGING>(vregTileMaxFp32_1, vregTileMaxFp32_1,
                                                                           vregFloatAbs1, maskFull);
        }
        AscendC::Reg::Mul(vregFloatY, vregFloatX, vregS, maskFull);
        AscendC::Reg::Mul(vregFloatY1, vregFloatX1, vregS, maskFull);
        // 目标类型 HiF8：fp32 乘积按 roundMode（round/hybrid）Cast 到目标类型。
        if constexpr (IsSameType<U, hifloat8_t>::value) {
            AscendC::Reg::Cast<U, float, QuantMaxV2Base<T, U, RoundMode>::CAST_TRAIT_FP32_TO_HIFP8>(vregY, vregFloatY,
                                                                                                    maskFull);
            AscendC::Reg::Cast<U, float, QuantMaxV2Base<T, U, RoundMode>::CAST_TRAIT_FP32_TO_HIFP8>(vregY1, vregFloatY1,
                                                                                                    maskFull);
            // 目标类型 E5M2：fp32 乘积按 RINT Cast 到目标类型。
        } else if constexpr (IsSameType<U, fp8_e5m2_t>::value) {
            AscendC::Reg::Cast<U, float, QuantMaxV2Base<T, U, RoundMode>::CAST_TRAIT_FP32_TO_FP8E5M2>(vregY, vregFloatY,
                                                                                                      maskFull);
            AscendC::Reg::Cast<U, float, QuantMaxV2Base<T, U, RoundMode>::CAST_TRAIT_FP32_TO_FP8E5M2>(
                vregY1, vregFloatY1, maskFull);
            // 目标类型 E4M3：fp32 乘积按 RINT Cast 到目标类型（其余 FP8 目标）。
        } else {
            AscendC::Reg::Cast<U, float, QuantMaxV2Base<T, U, RoundMode>::CAST_TRAIT_FP32_TO_FP8E4M3>(vregY, vregFloatY,
                                                                                                      maskFull);
            AscendC::Reg::Cast<U, float, QuantMaxV2Base<T, U, RoundMode>::CAST_TRAIT_FP32_TO_FP8E4M3>(
                vregY1, vregFloatY1, maskFull);
        }
        AscendC::Reg::StoreAlign<U, Reg::StoreDist::DIST_PACK4_B32>(outAddr + i * vectorLength, vregY, maskFull);
        AscendC::Reg::StoreAlign<U, Reg::StoreDist::DIST_PACK4_B32>(outAddr + (i + 1) * vectorLength, vregY1, maskFull);
    }
    // 位级双路最大值合并为单路。
    if constexpr (staticBf16Bits) {
        AscendC::Reg::Max<uint32_t, AscendC::Reg::MaskMergeMode::MERGING>(vregBitsMax, vregBitsMax, vregBitsMax1,
                                                                          maskAll);
        // fp32 域双路最大值合并为单路。
    } else {
        AscendC::Reg::Max<float, AscendC::Reg::MaskMergeMode::MERGING>(vregTileMaxFp32, vregTileMaxFp32,
                                                                       vregTileMaxFp32_1, maskAll);
    }
    firstSingleLoop = pairedLoops;
    // 静态最多剩一个整向量；动态继续原循环。
    for (uint16_t i = firstSingleLoop; i < fullLoops; ++i) {
        // fp32 输入：原生装载即用，无需类型转换。
        if constexpr (IsSameType<T, float>::value) {
            AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(vregFloatX, xRd + i * vectorLength);
            // fp16 输入：UNPACK_B16 装载（32 位槽承载一个 half，保证 lane 对齐）后 Cast 到 fp32。
        } else if constexpr (IsSameType<T, half>::value) {
            AscendC::Reg::LoadAlign<half, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(vregX, xRd + i * vectorLength);
            AscendC::Reg::Cast<float, half, QuantMaxV2Base<T, U, RoundMode>::CAST_TRAIT_HALF_TO_FP32>(vregFloatX, vregX,
                                                                                                      maskFull);
            // bf16 输入：UNPACK_B16 装载后处理（配对环内直接移位构造 fp32 位模式）。
        } else {
            AscendC::Reg::LoadAlign<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(vregX, xRd + i * vectorLength);
            AscendC::Reg::Cast<float, bfloat16_t, QuantMaxV2Base<T, U, RoundMode>::CAST_TRAIT_BF16_TO_FP32>(
                vregFloatX, vregX, maskFull);
        }
        // 位级最大值更新（单路整向量）。
        if constexpr (staticBf16Bits) {
            AscendC::Reg::And(vregBitsAbs, (AscendC::Reg::RegTensor<uint32_t>&)vregX, vregBitsMask, maskFull);
            AscendC::Reg::Max<uint32_t, AscendC::Reg::MaskMergeMode::MERGING>(vregBitsMax, vregBitsMax, vregBitsAbs,
                                                                              maskFull);
            // fp32 域最大值更新：Abs 后 Max 续接累加。
        } else if constexpr (ComputeMax) {
            AscendC::Reg::Abs<float>(vregFloatAbsT, vregFloatX, maskFull);
            AscendC::Reg::Max<float, AscendC::Reg::MaskMergeMode::MERGING>(vregTileMaxFp32, vregTileMaxFp32,
                                                                           vregFloatAbsT, maskFull);
        }
        // 量化写出：乘系数后按目标类型 Cast 并打包存储。
        if constexpr (WriteY) {
            AscendC::Reg::Mul(vregFloatY, vregFloatX, vregS, maskFull);
            // 目标类型 HiF8：fp32 乘积按 roundMode（round/hybrid）Cast 到目标类型。
            if constexpr (IsSameType<U, hifloat8_t>::value) {
                AscendC::Reg::Cast<U, float, QuantMaxV2Base<T, U, RoundMode>::CAST_TRAIT_FP32_TO_HIFP8>(
                    vregY, vregFloatY, maskFull);
                AscendC::Reg::StoreAlign<U, Reg::StoreDist::DIST_PACK4_B32>(outAddr + i * vectorLength, vregY,
                                                                            maskFull);
                // 目标类型 E5M2：fp32 乘积按 RINT Cast 到目标类型。
            } else if constexpr (IsSameType<U, fp8_e5m2_t>::value) {
                AscendC::Reg::Cast<U, float, QuantMaxV2Base<T, U, RoundMode>::CAST_TRAIT_FP32_TO_FP8E5M2>(
                    vregY, vregFloatY, maskFull);
                AscendC::Reg::StoreAlign<U, Reg::StoreDist::DIST_PACK4_B32>(outAddr + i * vectorLength, vregY,
                                                                            maskFull);
                // 目标类型 E4M3：fp32 乘积按 RINT Cast 到目标类型（其余 FP8 目标）。
            } else {
                AscendC::Reg::Cast<U, float, QuantMaxV2Base<T, U, RoundMode>::CAST_TRAIT_FP32_TO_FP8E4M3>(
                    vregY, vregFloatY, maskFull);
                AscendC::Reg::StoreAlign<U, Reg::StoreDist::DIST_PACK4_B32>(outAddr + i * vectorLength, vregY,
                                                                            maskFull);
            }
        }
    }
    // 尾向量统一以掩码处理（KERNEL-13：VF 体内不允许运行时分支）：tailCount==0 时掩码为空，
    // 装载仍会发起但所有运算不改变状态；地址在无尾量时回退到最后一个整向量，保证不越界。
    // 注意：UpdateMask 为 POST_UPDATE 语义（入参被改写成剩余计数），地址必须先由原始 tailCount
    // 算出，并把副本交给 UpdateMask，否则尾块会落到前一个向量的位置（整块量化结果错位）。
    const uint32_t tailOffset = tailCount != 0 ? fullLoops * vectorLength :
                                                 (fullLoops != 0 ? (fullLoops - 1) * vectorLength : 0);
    uint32_t tailMaskCount = tailCount;
    mask = AscendC::Reg::UpdateMask<float>(tailMaskCount);
    {
        // fp32 输入：原生装载即用，无需类型转换。
        if constexpr (IsSameType<T, float>::value) {
            AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(vregFloatX, xRd + tailOffset);
            // fp16 输入：UNPACK_B16 装载（32 位槽承载一个 half，保证 lane 对齐）后 Cast 到 fp32。
        } else if constexpr (IsSameType<T, half>::value) {
            AscendC::Reg::LoadAlign<half, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(vregX, xRd + tailOffset);
            AscendC::Reg::Cast<float, half, QuantMaxV2Base<T, U, RoundMode>::CAST_TRAIT_HALF_TO_FP32>(vregFloatX, vregX,
                                                                                                      mask);
            // bf16 输入：UNPACK_B16 装载后处理（配对环内直接移位构造 fp32 位模式）。
        } else {
            AscendC::Reg::LoadAlign<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(vregX, xRd + tailOffset);
            AscendC::Reg::Cast<float, bfloat16_t, QuantMaxV2Base<T, U, RoundMode>::CAST_TRAIT_BF16_TO_FP32>(
                vregFloatX, vregX, mask);
        }
        // 位级最大值更新（单路，尾向量掩码；空掩码时不改变累加器）。
        if constexpr (staticBf16Bits) {
            AscendC::Reg::And(vregBitsAbs, (AscendC::Reg::RegTensor<uint32_t>&)vregX, vregBitsMask, mask);
            AscendC::Reg::Max<uint32_t, AscendC::Reg::MaskMergeMode::MERGING>(vregBitsMax, vregBitsMax, vregBitsAbs,
                                                                              mask);
            // fp32 域最大值更新：Abs 后 Max 续接累加。
        } else if constexpr (ComputeMax) {
            AscendC::Reg::Abs<float>(vregFloatAbsT, vregFloatX, mask);
            AscendC::Reg::Max<float, AscendC::Reg::MaskMergeMode::MERGING>(vregTileMaxFp32, vregTileMaxFp32,
                                                                           vregFloatAbsT, mask);
        }
        // 量化写出（尾向量）：以剩余元素数的掩码写出，空掩码不落任何字节。
        if constexpr (WriteY) {
            AscendC::Reg::Mul(vregFloatY, vregFloatX, vregS, mask);
            // 目标类型 HiF8：fp32 乘积按 roundMode（round/hybrid）Cast 到目标类型。
            if constexpr (IsSameType<U, hifloat8_t>::value) {
                AscendC::Reg::Cast<U, float, QuantMaxV2Base<T, U, RoundMode>::CAST_TRAIT_FP32_TO_HIFP8>(
                    vregY, vregFloatY, mask);
                AscendC::Reg::StoreAlign<U, Reg::StoreDist::DIST_PACK4_B32>(outAddr + tailOffset, vregY, mask);
                // 目标类型 E5M2：fp32 乘积按 RINT Cast 到目标类型。
            } else if constexpr (IsSameType<U, fp8_e5m2_t>::value) {
                AscendC::Reg::Cast<U, float, QuantMaxV2Base<T, U, RoundMode>::CAST_TRAIT_FP32_TO_FP8E5M2>(
                    vregY, vregFloatY, mask);
                AscendC::Reg::StoreAlign<U, Reg::StoreDist::DIST_PACK4_B32>(outAddr + tailOffset, vregY, mask);
                // 目标类型 E4M3：fp32 乘积按 RINT Cast 到目标类型（其余 FP8 目标）。
            } else {
                AscendC::Reg::Cast<U, float, QuantMaxV2Base<T, U, RoundMode>::CAST_TRAIT_FP32_TO_FP8E4M3>(
                    vregY, vregFloatY, mask);
                AscendC::Reg::StoreAlign<U, Reg::StoreDist::DIST_PACK4_B32>(outAddr + tailOffset, vregY, mask);
            }
        }
    }
    // 位级最大值结果左移 16 位还原 BF16 位模式（非负最大值可被 BF16 精确表示）。
    if constexpr (staticBf16Bits) {
        AscendC::Reg::Reduce<Reg::ReduceType::MAX>(vregBitsMax, vregBitsMax, maskAll);
        AscendC::Reg::ShiftLefts((AscendC::Reg::RegTensor<uint32_t>&)vregTileMaxFp32, vregBitsMax,
                                 static_cast<int16_t>(B16_SHIFT_BITS), maskAll);
    }
    // 求最大值收尾：把组内最大值写回 maxAddr[groupIndex]（单 lane 首元素存储）。
    if constexpr (ComputeMax) {
        // fp32 域支路：先跨 lane 归约（Reduce MAX）成单值再存储。
        if constexpr (!staticBf16Bits) {
            AscendC::Reg::Reduce<Reg::ReduceType::MAX>(vregTileMaxFp32, vregTileMaxFp32, maskAll);
        }
        AscendC::Reg::StoreAlign<float, Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(maxAddr + groupIndex, vregTileMaxFp32,
                                                                                maskOne);
    }
}

// 动态第二遍（仅量化）：scaleValue = dstTypeMax / amax[g]（amax 为 0 时系数 0，y 输出全零）。
// 计算：y[i] = Cast(x[i] * scaleValue, dstType, roundMode)
// DualChain 仅 bf16 生效（双 Load→Cast→Mul→Cast→Store，配对环编译期分离实例化）；
// fp16/fp32 保持单链（fp16 的 vec 是 U 的分子，配对会降 2~4pt；fp32 不在关键路径上）。
template <typename T, typename U, uint64_t RoundMode, bool DualChain>
__simd_callee__ inline void GroupedQuantQuantOnlyVf(const __ubuf__ T* xAddr, __ubuf__ U* outAddr, float scaleValue,
                                                    uint32_t dataCount, uint32_t vectorLength)
{
    // 本特化恒为仅量化（原 WriteY/ComputeMax 模板参在本函数内为常量）。
    constexpr bool WriteY = true;
    constexpr bool ComputeMax = false;
    // AscendC 的 LoadAlign 形参为非 const 指针：以只读别名转回；本函数不得经该别名写入。
    __ubuf__ T* xRd = (__ubuf__ T*)xAddr;
    AscendC::Reg::RegTensor<T> vregX;
    // 复用单链/尾块循环的原文：位级支路在本场景恒为编译期假，变量仅为满足其声明需要。
    constexpr bool staticBf16Bits = false;
    AscendC::Reg::RegTensor<uint32_t> vregBitsMax;
    AscendC::Reg::RegTensor<uint32_t> vregBitsAbs;
    AscendC::Reg::RegTensor<uint32_t> vregBitsMask;
    AscendC::Reg::RegTensor<float> vregFloatAbsT;
    AscendC::Reg::RegTensor<float> vregTileMaxFp32;
    AscendC::Reg::RegTensor<float> vregFloatX;
    AscendC::Reg::RegTensor<float> vregS;
    AscendC::Reg::RegTensor<float> vregFloatY;
    AscendC::Reg::RegTensor<U> vregY;
    AscendC::Reg::MaskReg mask = AscendC::Reg::CreateMask<float>();
    AscendC::Reg::MaskReg maskAll = AscendC::Reg::CreateMask<float>();
    // 量化路径：把量化系数 scaleValue 广播进向量寄存器。
    if constexpr (WriteY) {
        AscendC::Reg::Duplicate<float>(vregS, scaleValue, mask);
    }

    const uint16_t fullLoops = static_cast<uint16_t>(dataCount / vectorLength);
    uint32_t tailCount = dataCount % vectorLength;
    uint32_t fullCount = vectorLength;
    AscendC::Reg::MaskReg maskFull = AscendC::Reg::UpdateMask<float>(fullCount);
    uint16_t firstSingleLoop = 0;
    // 双链仅 bf16 + DualChain 实例化存在；其余 dtype 走下方单链（配对会降 fp16 的 vec 利用率，fp32 不在关键路径）。
    if constexpr (DualChain && IsSameType<T, bfloat16_t>::value) {
        // 动态第二遍的 BF16 双链软件流水：两条独立的加载→转换→乘→转换→存储链，
        // 使第 i+1 个向量的加载与第 i 个向量的运算重叠；形态与静态配对环一致，只是去掉最大值链。
        // 按模板参数分离实例化，单链变体不承担配对环的寄存器压力。
        // FP16 保持单链（vec 是 U 的分子，配对会降 2~4pt）；FP32 保持单链（vec 不在关键路径上）。
        AscendC::Reg::RegTensor<bfloat16_t> vregX1;
        AscendC::Reg::RegTensor<float> vregFloatX1;
        AscendC::Reg::RegTensor<float> vregFloatY1;
        AscendC::Reg::RegTensor<U> vregY1;
        const uint16_t pairedLoops = (fullLoops / 2) * 2;
        for (uint16_t i = 0; i < pairedLoops; i += 2) {
            AscendC::Reg::LoadAlign<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(vregX, xRd + i * vectorLength);
            AscendC::Reg::LoadAlign<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(vregX1,
                                                                                         xRd + (i + 1) * vectorLength);
            AscendC::Reg::Cast<float, bfloat16_t, QuantMaxV2Base<T, U, RoundMode>::CAST_TRAIT_BF16_TO_FP32>(
                vregFloatX, vregX, maskFull);
            AscendC::Reg::Cast<float, bfloat16_t, QuantMaxV2Base<T, U, RoundMode>::CAST_TRAIT_BF16_TO_FP32>(
                vregFloatX1, vregX1, maskFull);
            AscendC::Reg::Mul(vregFloatY, vregFloatX, vregS, maskFull);
            AscendC::Reg::Mul(vregFloatY1, vregFloatX1, vregS, maskFull);
            // 目标类型 HiF8：fp32 乘积按 roundMode（round/hybrid）Cast 到目标类型。
            if constexpr (IsSameType<U, hifloat8_t>::value) {
                AscendC::Reg::Cast<U, float, QuantMaxV2Base<T, U, RoundMode>::CAST_TRAIT_FP32_TO_HIFP8>(
                    vregY, vregFloatY, maskFull);
                AscendC::Reg::Cast<U, float, QuantMaxV2Base<T, U, RoundMode>::CAST_TRAIT_FP32_TO_HIFP8>(
                    vregY1, vregFloatY1, maskFull);
                // 目标类型 E5M2：fp32 乘积按 RINT Cast 到目标类型。
            } else if constexpr (IsSameType<U, fp8_e5m2_t>::value) {
                AscendC::Reg::Cast<U, float, QuantMaxV2Base<T, U, RoundMode>::CAST_TRAIT_FP32_TO_FP8E5M2>(
                    vregY, vregFloatY, maskFull);
                AscendC::Reg::Cast<U, float, QuantMaxV2Base<T, U, RoundMode>::CAST_TRAIT_FP32_TO_FP8E5M2>(
                    vregY1, vregFloatY1, maskFull);
                // 目标类型 E4M3：fp32 乘积按 RINT Cast 到目标类型（其余 FP8 目标）。
            } else {
                AscendC::Reg::Cast<U, float, QuantMaxV2Base<T, U, RoundMode>::CAST_TRAIT_FP32_TO_FP8E4M3>(
                    vregY, vregFloatY, maskFull);
                AscendC::Reg::Cast<U, float, QuantMaxV2Base<T, U, RoundMode>::CAST_TRAIT_FP32_TO_FP8E4M3>(
                    vregY1, vregFloatY1, maskFull);
            }
            AscendC::Reg::StoreAlign<U, Reg::StoreDist::DIST_PACK4_B32>(outAddr + i * vectorLength, vregY, maskFull);
            AscendC::Reg::StoreAlign<U, Reg::StoreDist::DIST_PACK4_B32>(outAddr + (i + 1) * vectorLength, vregY1,
                                                                        maskFull);
        }
        firstSingleLoop = pairedLoops;
    }
    // 静态最多剩一个整向量；动态继续原循环。
    for (uint16_t i = firstSingleLoop; i < fullLoops; ++i) {
        // fp32 输入：原生装载即用，无需类型转换。
        if constexpr (IsSameType<T, float>::value) {
            AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(vregFloatX, xRd + i * vectorLength);
            // fp16 输入：UNPACK_B16 装载（32 位槽承载一个 half，保证 lane 对齐）后 Cast 到 fp32。
        } else if constexpr (IsSameType<T, half>::value) {
            AscendC::Reg::LoadAlign<half, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(vregX, xRd + i * vectorLength);
            AscendC::Reg::Cast<float, half, QuantMaxV2Base<T, U, RoundMode>::CAST_TRAIT_HALF_TO_FP32>(vregFloatX, vregX,
                                                                                                      maskFull);
            // bf16 输入：UNPACK_B16 装载后处理（配对环内直接移位构造 fp32 位模式）。
        } else {
            AscendC::Reg::LoadAlign<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(vregX, xRd + i * vectorLength);
            AscendC::Reg::Cast<float, bfloat16_t, QuantMaxV2Base<T, U, RoundMode>::CAST_TRAIT_BF16_TO_FP32>(
                vregFloatX, vregX, maskFull);
        }
        // 位级最大值更新（单路整向量）。
        if constexpr (staticBf16Bits) {
            AscendC::Reg::And(vregBitsAbs, (AscendC::Reg::RegTensor<uint32_t>&)vregX, vregBitsMask, maskFull);
            AscendC::Reg::Max<uint32_t, AscendC::Reg::MaskMergeMode::MERGING>(vregBitsMax, vregBitsMax, vregBitsAbs,
                                                                              maskFull);
            // fp32 域最大值更新：Abs 后 Max 续接累加。
        } else if constexpr (ComputeMax) {
            AscendC::Reg::Abs<float>(vregFloatAbsT, vregFloatX, maskFull);
            AscendC::Reg::Max<float, AscendC::Reg::MaskMergeMode::MERGING>(vregTileMaxFp32, vregTileMaxFp32,
                                                                           vregFloatAbsT, maskFull);
        }
        // 量化写出：乘系数后按目标类型 Cast 并打包存储。
        if constexpr (WriteY) {
            AscendC::Reg::Mul(vregFloatY, vregFloatX, vregS, maskFull);
            // 目标类型 HiF8：fp32 乘积按 roundMode（round/hybrid）Cast 到目标类型。
            if constexpr (IsSameType<U, hifloat8_t>::value) {
                AscendC::Reg::Cast<U, float, QuantMaxV2Base<T, U, RoundMode>::CAST_TRAIT_FP32_TO_HIFP8>(
                    vregY, vregFloatY, maskFull);
                AscendC::Reg::StoreAlign<U, Reg::StoreDist::DIST_PACK4_B32>(outAddr + i * vectorLength, vregY,
                                                                            maskFull);
                // 目标类型 E5M2：fp32 乘积按 RINT Cast 到目标类型。
            } else if constexpr (IsSameType<U, fp8_e5m2_t>::value) {
                AscendC::Reg::Cast<U, float, QuantMaxV2Base<T, U, RoundMode>::CAST_TRAIT_FP32_TO_FP8E5M2>(
                    vregY, vregFloatY, maskFull);
                AscendC::Reg::StoreAlign<U, Reg::StoreDist::DIST_PACK4_B32>(outAddr + i * vectorLength, vregY,
                                                                            maskFull);
                // 目标类型 E4M3：fp32 乘积按 RINT Cast 到目标类型（其余 FP8 目标）。
            } else {
                AscendC::Reg::Cast<U, float, QuantMaxV2Base<T, U, RoundMode>::CAST_TRAIT_FP32_TO_FP8E4M3>(
                    vregY, vregFloatY, maskFull);
                AscendC::Reg::StoreAlign<U, Reg::StoreDist::DIST_PACK4_B32>(outAddr + i * vectorLength, vregY,
                                                                            maskFull);
            }
        }
    }
    // 尾向量统一以掩码处理（KERNEL-13：VF 体内不允许运行时分支）：tailCount==0 时掩码为空，
    // 装载仍会发起但所有运算不改变状态；地址在无尾量时回退到最后一个整向量，保证不越界。
    // 注意：UpdateMask 为 POST_UPDATE 语义（入参被改写成剩余计数），地址必须先由原始 tailCount
    // 算出，并把副本交给 UpdateMask，否则尾块会落到前一个向量的位置（整块量化结果错位）。
    const uint32_t tailOffset = tailCount != 0 ? fullLoops * vectorLength :
                                                 (fullLoops != 0 ? (fullLoops - 1) * vectorLength : 0);
    uint32_t tailMaskCount = tailCount;
    mask = AscendC::Reg::UpdateMask<float>(tailMaskCount);
    {
        // fp32 输入：原生装载即用，无需类型转换。
        if constexpr (IsSameType<T, float>::value) {
            AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(vregFloatX, xRd + tailOffset);
            // fp16 输入：UNPACK_B16 装载（32 位槽承载一个 half，保证 lane 对齐）后 Cast 到 fp32。
        } else if constexpr (IsSameType<T, half>::value) {
            AscendC::Reg::LoadAlign<half, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(vregX, xRd + tailOffset);
            AscendC::Reg::Cast<float, half, QuantMaxV2Base<T, U, RoundMode>::CAST_TRAIT_HALF_TO_FP32>(vregFloatX, vregX,
                                                                                                      mask);
            // bf16 输入：UNPACK_B16 装载后处理（配对环内直接移位构造 fp32 位模式）。
        } else {
            AscendC::Reg::LoadAlign<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(vregX, xRd + tailOffset);
            AscendC::Reg::Cast<float, bfloat16_t, QuantMaxV2Base<T, U, RoundMode>::CAST_TRAIT_BF16_TO_FP32>(
                vregFloatX, vregX, mask);
        }
        // 位级最大值更新（单路，尾向量掩码；空掩码时不改变累加器）。
        if constexpr (staticBf16Bits) {
            AscendC::Reg::And(vregBitsAbs, (AscendC::Reg::RegTensor<uint32_t>&)vregX, vregBitsMask, mask);
            AscendC::Reg::Max<uint32_t, AscendC::Reg::MaskMergeMode::MERGING>(vregBitsMax, vregBitsMax, vregBitsAbs,
                                                                              mask);
            // fp32 域最大值更新：Abs 后 Max 续接累加。
        } else if constexpr (ComputeMax) {
            AscendC::Reg::Abs<float>(vregFloatAbsT, vregFloatX, mask);
            AscendC::Reg::Max<float, AscendC::Reg::MaskMergeMode::MERGING>(vregTileMaxFp32, vregTileMaxFp32,
                                                                           vregFloatAbsT, mask);
        }
        // 量化写出（尾向量）：以剩余元素数的掩码写出，空掩码不落任何字节。
        if constexpr (WriteY) {
            AscendC::Reg::Mul(vregFloatY, vregFloatX, vregS, mask);
            // 目标类型 HiF8：fp32 乘积按 roundMode（round/hybrid）Cast 到目标类型。
            if constexpr (IsSameType<U, hifloat8_t>::value) {
                AscendC::Reg::Cast<U, float, QuantMaxV2Base<T, U, RoundMode>::CAST_TRAIT_FP32_TO_HIFP8>(
                    vregY, vregFloatY, mask);
                AscendC::Reg::StoreAlign<U, Reg::StoreDist::DIST_PACK4_B32>(outAddr + tailOffset, vregY, mask);
                // 目标类型 E5M2：fp32 乘积按 RINT Cast 到目标类型。
            } else if constexpr (IsSameType<U, fp8_e5m2_t>::value) {
                AscendC::Reg::Cast<U, float, QuantMaxV2Base<T, U, RoundMode>::CAST_TRAIT_FP32_TO_FP8E5M2>(
                    vregY, vregFloatY, mask);
                AscendC::Reg::StoreAlign<U, Reg::StoreDist::DIST_PACK4_B32>(outAddr + tailOffset, vregY, mask);
                // 目标类型 E4M3：fp32 乘积按 RINT Cast 到目标类型（其余 FP8 目标）。
            } else {
                AscendC::Reg::Cast<U, float, QuantMaxV2Base<T, U, RoundMode>::CAST_TRAIT_FP32_TO_FP8E4M3>(
                    vregY, vregFloatY, mask);
                AscendC::Reg::StoreAlign<U, Reg::StoreDist::DIST_PACK4_B32>(outAddr + tailOffset, vregY, mask);
            }
        }
    }
}

// 动态 bf16 第一遍 tile 级绝对值最大值（位级实现）。
// 计算：tileMaxAddr[tileMaxCount] = max(|x[0..dataCount)|)
// 实现要点：清符号位后按 uint16 位模式比较（非负 bf16 位模式与数值同序，免 fp32 转换），结果以 16
// 位暂存；四路独立累加链打破 Max 串行依赖；尾向量用精确掩码，防 UB 旧填充值参与最大值计算。
__simd_vf__ inline void GroupedQuantBf16TileMaxVf(const __ubuf__ uint16_t* xAddr, __ubuf__ uint16_t* tileMaxAddr,
                                                  uint32_t dataCount, uint32_t tileMaxCount, uint32_t vectorLengthB16)
{
    // AscendC 的 LoadAlign 形参为非 const 指针：以只读别名转回；本函数不得经该别名写入。
    __ubuf__ uint16_t* xRd = (__ubuf__ uint16_t*)xAddr;

    const uint16_t loopNum = (dataCount + vectorLengthB16 - 1) / vectorLengthB16;

    AscendC::Reg::RegTensor<uint16_t> vregInput;
    AscendC::Reg::RegTensor<uint16_t> vregAbs;
    AscendC::Reg::RegTensor<uint16_t> vregAbsMask;
    AscendC::Reg::RegTensor<uint16_t> vregTileMax0;
    AscendC::Reg::RegTensor<uint16_t> vregTileMax1;
    AscendC::Reg::RegTensor<uint16_t> vregTileMax2;
    AscendC::Reg::RegTensor<uint16_t> vregTileMax3;
    AscendC::Reg::MaskReg mask = AscendC::Reg::CreateMask<uint16_t>();
    AscendC::Reg::MaskReg maskAll = AscendC::Reg::CreateMask<uint16_t>();
    uint32_t fullCount = vectorLengthB16;
    AscendC::Reg::MaskReg maskFull = AscendC::Reg::UpdateMask<uint16_t>(fullCount);
    uint32_t one = 1;
    AscendC::Reg::MaskReg maskOne = AscendC::Reg::UpdateMask<uint16_t>(one);

    AscendC::Reg::Duplicate<uint16_t>(vregAbsMask, BF16_ABS_MASK, maskAll);
    AscendC::Reg::Duplicate<uint16_t>(vregTileMax0, 0, maskAll);
    AscendC::Reg::Duplicate<uint16_t>(vregTileMax1, 0, maskAll);
    AscendC::Reg::Duplicate<uint16_t>(vregTileMax2, 0, maskAll);
    AscendC::Reg::Duplicate<uint16_t>(vregTileMax3, 0, maskAll);
    uint16_t fullLoopNum = ((dataCount / vectorLengthB16) / VF_UNROLL_FACTOR) * VF_UNROLL_FACTOR;
    for (uint16_t block = 0; block < fullLoopNum; block += VF_UNROLL_FACTOR) {
        AscendC::Reg::LoadAlign<uint16_t, AscendC::Reg::LoadDist::DIST_NORM>(vregInput, xRd + block * vectorLengthB16);
        AscendC::Reg::And(vregAbs, vregInput, vregAbsMask, maskFull);
        AscendC::Reg::Max<uint16_t, AscendC::Reg::MaskMergeMode::MERGING>(vregTileMax0, vregTileMax0, vregAbs,
                                                                          maskFull);
        AscendC::Reg::LoadAlign<uint16_t, AscendC::Reg::LoadDist::DIST_NORM>(vregInput,
                                                                             xRd + (block + 1) * vectorLengthB16);
        AscendC::Reg::And(vregAbs, vregInput, vregAbsMask, maskFull);
        AscendC::Reg::Max<uint16_t, AscendC::Reg::MaskMergeMode::MERGING>(vregTileMax1, vregTileMax1, vregAbs,
                                                                          maskFull);
        AscendC::Reg::LoadAlign<uint16_t, AscendC::Reg::LoadDist::DIST_NORM>(vregInput,
                                                                             xRd + (block + 2) * vectorLengthB16);
        AscendC::Reg::And(vregAbs, vregInput, vregAbsMask, maskFull);
        AscendC::Reg::Max<uint16_t, AscendC::Reg::MaskMergeMode::MERGING>(vregTileMax2, vregTileMax2, vregAbs,
                                                                          maskFull);
        AscendC::Reg::LoadAlign<uint16_t, AscendC::Reg::LoadDist::DIST_NORM>(vregInput,
                                                                             xRd + (block + 3) * vectorLengthB16);
        AscendC::Reg::And(vregAbs, vregInput, vregAbsMask, maskFull);
        AscendC::Reg::Max<uint16_t, AscendC::Reg::MaskMergeMode::MERGING>(vregTileMax3, vregTileMax3, vregAbs,
                                                                          maskFull);
    }
    uint16_t tailStart = fullLoopNum;
    for (uint16_t i = tailStart; i < loopNum; ++i) {
        // 最多剩三个整向量和一个尾向量；尾部必须使用精确掩码，防止 UB 旧填充值参与最大值计算。
        uint32_t remaining = static_cast<uint32_t>(dataCount - i * vectorLengthB16);
        mask = AscendC::Reg::UpdateMask<uint16_t>(remaining);
        AscendC::Reg::LoadAlign<uint16_t, AscendC::Reg::LoadDist::DIST_NORM>(vregInput, xRd + i * vectorLengthB16);
        AscendC::Reg::And(vregAbs, vregInput, vregAbsMask, mask);
        AscendC::Reg::Max<uint16_t, AscendC::Reg::MaskMergeMode::MERGING>(vregTileMax0, vregTileMax0, vregAbs, mask);
    }

    AscendC::Reg::Max<uint16_t, AscendC::Reg::MaskMergeMode::MERGING>(vregTileMax0, vregTileMax0, vregTileMax1,
                                                                      maskAll);
    AscendC::Reg::Max<uint16_t, AscendC::Reg::MaskMergeMode::MERGING>(vregTileMax2, vregTileMax2, vregTileMax3,
                                                                      maskAll);
    AscendC::Reg::Max<uint16_t, AscendC::Reg::MaskMergeMode::MERGING>(vregTileMax0, vregTileMax0, vregTileMax2,
                                                                      maskAll);
    AscendC::Reg::Reduce<Reg::ReduceType::MAX>(vregTileMax0, vregTileMax0, maskAll);
    AscendC::Reg::StoreAlign<uint16_t, Reg::StoreDist::DIST_FIRST_ELEMENT_B16>(tileMaxAddr + tileMaxCount, vregTileMax0,
                                                                               maskOne);
}

template <typename T, typename U, uint64_t RoundMode>
// 动态 bf16 第一遍收尾：本组内全部 tile 的 16 位最大值归约为组最大值。
// 计算：groupMaxAddr[0] = (float)max(tileMax[0..count))
// 16 位最大值经 Cast 到 fp32 后单 lane 存储，供 MergeScale 跨核归并消费。
__simd_vf__ inline void GroupedQuantBf16GroupMaxVf(const __ubuf__ uint16_t* tileMaxAddr, __ubuf__ float* groupMaxAddr,
                                                   uint32_t count, uint32_t vectorLengthB16)
{
    // AscendC 的 LoadAlign 形参为非 const 指针：以只读别名转回；本函数不得经该别名写入。
    __ubuf__ uint16_t* tileMaxRd = (__ubuf__ uint16_t*)tileMaxAddr;

    AscendC::Reg::RegTensor<uint16_t> vregInput;
    AscendC::Reg::RegTensor<uint16_t> vregTileMax;
    AscendC::Reg::RegTensor<bfloat16_t> vregTileMaxBf16;
    AscendC::Reg::RegTensor<float> vregTileMaxFp32;
    AscendC::Reg::MaskReg maskAll = AscendC::Reg::CreateMask<uint16_t>();
    AscendC::Reg::MaskReg tileMask = AscendC::Reg::CreateMask<uint16_t>();
    uint32_t one = 1;
    AscendC::Reg::MaskReg maskOne = AscendC::Reg::UpdateMask<float>(one);
    const uint16_t loopNum = static_cast<uint16_t>((count + vectorLengthB16 - 1) / vectorLengthB16);
    AscendC::Reg::Duplicate<uint16_t>(vregTileMax, 0, maskAll);
    for (uint16_t i = 0; i < loopNum; ++i) {
        uint32_t remaining = count - i * vectorLengthB16;
        tileMask = AscendC::Reg::UpdateMask<uint16_t>(remaining);
        AscendC::Reg::LoadAlign<uint16_t, AscendC::Reg::LoadDist::DIST_NORM>(vregInput,
                                                                             tileMaxRd + i * vectorLengthB16);
        AscendC::Reg::Max<uint16_t, AscendC::Reg::MaskMergeMode::MERGING>(vregTileMax, vregTileMax, vregInput,
                                                                          tileMask);
    }
    AscendC::Reg::Reduce<Reg::ReduceType::MAX>(vregTileMax, vregTileMax, maskAll);
    vregTileMaxBf16 = (AscendC::Reg::RegTensor<bfloat16_t>&)vregTileMax;
    AscendC::Reg::Cast<float, bfloat16_t, QuantMaxV2Base<T, U, RoundMode>::CAST_TRAIT_BF16_TO_FP32>(
        vregTileMaxFp32, vregTileMaxBf16, maskOne);
    AscendC::Reg::StoreAlign<float, Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(groupMaxAddr, vregTileMaxFp32, maskOne);
}

// 单组（G=1）跨核归并：各核的部分最大值在核间矩阵中连续存放。
// 计算：mergedAddr[0] = max(groupMax[0..count))
// 动态 per-tensor 场景：各核各写一个元素，归并核合成全局 amax。
__simd_vf__ inline void GroupedQuantMergeSingleGroupVf(const __ubuf__ float* groupMaxAddr, __ubuf__ float* mergedAddr,
                                                       uint32_t count, uint32_t vectorLength)
{
    // AscendC 的 LoadAlign 形参为非 const 指针：以只读别名转回；本函数不得经该别名写入。
    __ubuf__ float* groupMaxRd = (__ubuf__ float*)groupMaxAddr;

    AscendC::Reg::RegTensor<float> vregMax;
    AscendC::Reg::RegTensor<float> vregPart;
    AscendC::Reg::MaskReg maskAll = AscendC::Reg::CreateMask<float>();
    AscendC::Reg::MaskReg mask = AscendC::Reg::CreateMask<float>();
    AscendC::Reg::Duplicate<float>(vregMax, 0.0f, maskAll);
    uint16_t loops = (count + vectorLength - 1) / vectorLength;
    for (uint16_t i = 0; i < loops; ++i) {
        mask = AscendC::Reg::UpdateMask<float>(count);
        AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(vregPart, groupMaxRd + i * vectorLength);
        AscendC::Reg::Max<float, AscendC::Reg::MaskMergeMode::MERGING>(vregMax, vregMax, vregPart, mask);
    }
    AscendC::Reg::Reduce<AscendC::Reg::ReduceType::MAX>(vregMax, vregMax, maskAll);
    uint32_t one = 1;
    AscendC::Reg::MaskReg maskOne = AscendC::Reg::UpdateMask<float>(one);
    AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(mergedAddr, vregMax, maskOne);
}

// 动态 G>=8 跨核归并各组的逐组最大值（MergeScale 按需归并路径）。
// 计算：merged[g] = max over cores(groupMax[core * groupStride + g])，g = 0..numGroups-1
// 按需归并：core0/core1 归并全量组段，其余核只归并本核实际覆盖的组段；四路展开（4 核一组）打破跨核 Max 串行链，剩余不足
// 4 核用单链续接（Max 结合律保证等价）。归并结果供 EmitYScale 与 amax 输出消费。
__simd_vf__ inline void GroupedQuantMergeScaleVf(const __ubuf__ float* groupMaxAddr, __ubuf__ float* mergedAddr,
                                                 uint32_t numGroups, uint16_t actualCoreNum, int64_t groupStride,
                                                 uint32_t vectorLength)
{
    // AscendC 的 LoadAlign 形参为非 const 指针：以只读别名转回；本函数不得经该别名写入。
    __ubuf__ float* groupMaxRd = (__ubuf__ float*)groupMaxAddr;

    AscendC::Reg::RegTensor<float> vregMax;
    AscendC::Reg::RegTensor<float> vregMax1;
    AscendC::Reg::RegTensor<float> vregMax2;
    AscendC::Reg::RegTensor<float> vregMax3;
    AscendC::Reg::RegTensor<float> vregPart;
    AscendC::Reg::MaskReg mask = AscendC::Reg::CreateMask<float>();
    uint32_t count = numGroups;
    uint16_t loops = (numGroups + vectorLength - 1) / vectorLength;
    for (uint16_t i = 0; i < loops; ++i) {
        mask = AscendC::Reg::UpdateMask<float>(count);
        AscendC::Reg::Duplicate<float>(vregMax, 0.0f, mask);
        AscendC::Reg::Duplicate<float>(vregMax1, 0.0f, mask);
        AscendC::Reg::Duplicate<float>(vregMax2, 0.0f, mask);
        AscendC::Reg::Duplicate<float>(vregMax3, 0.0f, mask);
        uint16_t fullCoreNum = (actualCoreNum / VF_UNROLL_FACTOR) * VF_UNROLL_FACTOR;
        for (uint16_t core = 0; core < fullCoreNum; core += VF_UNROLL_FACTOR) {
            AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(
                vregPart, groupMaxRd + core * groupStride + i * vectorLength);
            AscendC::Reg::Max<float, AscendC::Reg::MaskMergeMode::ZEROING>(vregMax, vregMax, vregPart, mask);
            AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(
                vregPart, groupMaxRd + (core + 1) * groupStride + i * vectorLength);
            AscendC::Reg::Max<float, AscendC::Reg::MaskMergeMode::ZEROING>(vregMax1, vregMax1, vregPart, mask);
            AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(
                vregPart, groupMaxRd + (core + 2) * groupStride + i * vectorLength);
            AscendC::Reg::Max<float, AscendC::Reg::MaskMergeMode::ZEROING>(vregMax2, vregMax2, vregPart, mask);
            AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(
                vregPart, groupMaxRd + (core + 3) * groupStride + i * vectorLength);
            AscendC::Reg::Max<float, AscendC::Reg::MaskMergeMode::ZEROING>(vregMax3, vregMax3, vregPart, mask);
        }
        for (uint16_t core = fullCoreNum; core < actualCoreNum; ++core) {
            // VF 内不增加运行时分支；最多三个剩余核使用单链归约，Max 满足结合律。
            AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(
                vregPart, groupMaxRd + core * groupStride + i * vectorLength);
            AscendC::Reg::Max<float, AscendC::Reg::MaskMergeMode::ZEROING>(vregMax, vregMax, vregPart, mask);
        }
        AscendC::Reg::Max<float, AscendC::Reg::MaskMergeMode::ZEROING>(vregMax, vregMax, vregMax1, mask);
        AscendC::Reg::Max<float, AscendC::Reg::MaskMergeMode::ZEROING>(vregMax2, vregMax2, vregMax3, mask);
        AscendC::Reg::Max<float, AscendC::Reg::MaskMergeMode::ZEROING>(vregMax, vregMax, vregMax2, mask);
        AscendC::Reg::StoreAlign<float>(mergedAddr + i * vectorLength, vregMax, mask);
    }
}

// 静态/动态 G>=8 跨步归并各核的逐组最大值（单链版本，amax 输出路径）。
// 计算：merged[g] = max over cores(groupMax[core * groupStride + g])，g = 0..numGroups-1
// 与 GroupedQuantMergeScaleVf 相同的核 x 组跨步布局，单链实现；静态 G>=64 分层归并时，core0 对 8 个 slab
// 行的最终归并也复用本接口。
__simd_vf__ inline void GroupedQuantMergeAmaxVf(const __ubuf__ float* groupMaxAddr, __ubuf__ float* mergedAddr,
                                                uint32_t numGroups, uint16_t actualCoreNum, int64_t groupStride,
                                                uint32_t vectorLength)
{
    // AscendC 的 LoadAlign 形参为非 const 指针：以只读别名转回；本函数不得经该别名写入。
    __ubuf__ float* groupMaxRd = (__ubuf__ float*)groupMaxAddr;

    AscendC::Reg::RegTensor<float> vregMax;
    AscendC::Reg::RegTensor<float> vregPart;
    AscendC::Reg::MaskReg mask = AscendC::Reg::CreateMask<float>();
    uint32_t count = numGroups;
    uint16_t loops = (numGroups + vectorLength - 1) / vectorLength;
    for (uint16_t i = 0; i < loops; ++i) {
        mask = AscendC::Reg::UpdateMask<float>(count);
        AscendC::Reg::Duplicate<float>(vregMax, 0.0f, mask);
        for (uint16_t core = 0; core < actualCoreNum; ++core) {
            AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(
                vregPart, groupMaxRd + core * groupStride + i * vectorLength);
            AscendC::Reg::Max<float, AscendC::Reg::MaskMergeMode::ZEROING>(vregMax, vregMax, vregPart, mask);
        }
        AscendC::Reg::StoreAlign<float>(mergedAddr + i * vectorLength, vregMax, mask);
    }
}

template <typename T, typename U, uint64_t RoundMode>
// amax 输出转换：将归并后的 fp32 逐组最大值落盘为输出 dtype。
// 计算：amax[g] = (T)merged[g]，g = 0..numGroups-1
// fp32 直存；half/bf16 先 Cast（RINT 语义）再 PACK_B32 打包存储。
__simd_vf__ inline void GroupedQuantCastAmaxVf(const __ubuf__ float* mergedAddr, __ubuf__ T* amaxAddr,
                                               uint32_t numGroups, uint32_t vectorLength)
{
    // AscendC 的 LoadAlign 形参为非 const 指针：以只读别名转回；本函数不得经该别名写入。
    __ubuf__ float* mergedRd = (__ubuf__ float*)mergedAddr;

    AscendC::Reg::RegTensor<float> vregFloat;
    AscendC::Reg::RegTensor<T> vregOut;
    AscendC::Reg::MaskReg mask = AscendC::Reg::CreateMask<float>();
    AscendC::Reg::MaskReg maskU8 = AscendC::Reg::CreateMask<uint8_t>();
    uint32_t count = numGroups;
    uint16_t vfLoopNum = (numGroups + vectorLength - 1) / vectorLength;
    for (uint16_t i = 0; i < vfLoopNum; ++i) {
        mask = AscendC::Reg::UpdateMask<float>(count);
        AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(vregFloat, mergedRd + i * vectorLength);
        // 半精度输出：fp32 最大值先 Cast 到 T（RINT 语义）再 PACK_B32 打包存储。
        if constexpr (IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value) {
            AscendC::Reg::Cast<T, float, QuantMaxV2Base<T, U, RoundMode>::CAST_TRAIT_FP32_TO_X_DETYPE>(
                vregOut, vregFloat, maskU8);
            AscendC::Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(amaxAddr + i * vectorLength, vregOut, mask);
            // fp32 输出：类型一致，直接存储无需转换。
        } else {
            AscendC::Reg::StoreAlign<T>(amaxAddr + i * vectorLength, vregFloat, mask);
        }
    }
}

template <typename T, typename U, uint64_t RoundMode>
// 动态收尾 y_scale 输出（core1 执行）：把每组 amax 换算成量化系数。
// 计算：yScale[g] = max(amax[g], 0) * invDstTypeMax（即 amax[g] / dstTypeMax）
// amax 恒非负，钳零为防御性语义；invDstTypeMax 经模板标量传入，乘倒数替代除法，输出满足反量化关系 x ≈ y * y_scale。
__simd_vf__ inline void GroupedQuantEmitYScaleVf(const __ubuf__ float* amaxAddr, __ubuf__ float* yScaleAddr,
                                                 uint32_t numGroups, float invDstTypeMax, uint32_t vectorLength)
{
    // AscendC 的 LoadAlign 形参为非 const 指针：以只读别名转回；本函数不得经该别名写入。
    __ubuf__ float* amaxRd = (__ubuf__ float*)amaxAddr;

    AscendC::Reg::RegTensor<float> vregAmax;
    AscendC::Reg::RegTensor<float> vregScale;
    AscendC::Reg::RegTensor<float> vregZero;
    AscendC::Reg::RegTensor<float> vregInv;
    AscendC::Reg::MaskReg mask = AscendC::Reg::CreateMask<float>();
    AscendC::Reg::MaskReg maskAll = AscendC::Reg::CreateMask<float>();
    AscendC::Reg::Duplicate<float>(vregZero, 0.0f, maskAll);
    AscendC::Reg::Duplicate<float>(vregInv, invDstTypeMax, maskAll);
    const uint16_t loops = static_cast<uint16_t>((numGroups + vectorLength - 1) / vectorLength);
    for (uint16_t i = 0; i < loops; ++i) {
        // 逐轮更新剩余数量，保持尾掩码精确；固定数量会使最后一块读取越过 groupStride 填充区。
        uint32_t remaining = numGroups - i * vectorLength;
        mask = AscendC::Reg::UpdateMask<float>(remaining);
        AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(vregAmax, amaxRd + i * vectorLength);
        AscendC::Reg::Max<float, AscendC::Reg::MaskMergeMode::ZEROING>(vregAmax, vregAmax, vregZero, mask);
        AscendC::Reg::Mul(vregScale, vregAmax, vregInv, mask);
        AscendC::Reg::StoreAlign<float>(yScaleAddr + i * vectorLength, vregScale, mask);
    }
}

} // namespace QuantMaxV2
#endif // QUANT_MAX_V2_VF_H
