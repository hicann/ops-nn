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
 * \file reduce_common_regbase.h
 * \brief reduce common regbase file
 */
#ifndef REDUCE_COMMON_REGBASE_H_RMS_NORM
#define REDUCE_COMMON_REGBASE_H_RMS_NORM
#include "kernel_operator.h"

namespace NormCommon {
using namespace AscendC;
using AscendC::Reg::CreateMask;
using AscendC::Reg::LoadDist;
using AscendC::Reg::LocalMemBar;
using AscendC::Reg::MaskPattern;
using AscendC::Reg::MaskReg;
using AscendC::Reg::MemType;
using AscendC::Reg::RegTensor;
using AscendC::Reg::StoreDist;
using AscendC::Reg::UpdateMask;

namespace NormCommonRegbase {
__aicore__ inline constexpr uint32_t GetVRegSize()
{
#if __CCE_AICORE__ == 310
    return AscendC::VECTOR_REG_WIDTH;
#else
    return 256U;
#endif
}

template <typename T>
__aicore__ inline T CeilDiv(T a, T b)
{
    using type = typename std::conditional<sizeof(T) == sizeof(uint8_t) || sizeof(T) == sizeof(uint16_t), uint32_t,
                                           uint64_t>::type;
    type res = (static_cast<type>(a) + static_cast<type>(b) - 1) / static_cast<type>(b);
    return static_cast<T>(res);
}

template <typename T>
__aicore__ inline T CeilAlign(T a, T b)
{
    using type = typename std::conditional<sizeof(T) == sizeof(uint8_t) || sizeof(T) == sizeof(uint16_t), uint32_t,
                                           uint64_t>::type;
    type res = (static_cast<type>(a) + static_cast<type>(b) - 1) / static_cast<type>(b) * static_cast<type>(b);
    return static_cast<T>(res);
}

template <typename T>
__aicore__ inline T Aligned(T value, T alignment)
{
    if (alignment == 0) {
        return value;
    }
    return (value + alignment - 1) / alignment * alignment;
}

} // namespace NormCommonRegbase

constexpr int32_t VL_SIZE = NormCommonRegbase::GetVRegSize();
constexpr int32_t V_LENGTH = (VL_SIZE / static_cast<int32_t>(sizeof(float)));
constexpr uint32_t ONCE_VECTOR_SIZE = 256;
constexpr uint16_t DICHOTOMY_ADD_COEFF = 2;

constexpr AscendC::Reg::CastTrait castTraitB162B32 = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::UNKNOWN,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};

constexpr AscendC::Reg::CastTrait castTraitB322B16 = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::NO_SAT,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

__aicore__ inline void DichotomyAdd(RegTensor<float>& dstReg, __local_mem__ float* src, uint16_t outerLoop,
                                    uint16_t innerLoop, uint32_t lastNum)
{
    RegTensor<float> tmpReg1;
    RegTensor<float> tmpReg2;
    RegTensor<float> tmpReg3;
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    MaskReg pregMain = CreateMask<float, MaskPattern::ALL>();
    for (uint16_t k = 0; k < outerLoop; k++) {
        innerLoop = innerLoop / DICHOTOMY_ADD_COEFF;
        for (uint16_t i = 0; i < innerLoop; i++) {
            DataCopy(tmpReg1, src + i * V_LENGTH);
            DataCopy(tmpReg2, src + (i + innerLoop) * V_LENGTH);
            Add(tmpReg3, tmpReg1, tmpReg2, pregMain);
            DataCopy(src + i * V_LENGTH, tmpReg3, pregMain);
        }
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    }
    uint32_t sreg0 = lastNum;
    MaskReg pregLoop = UpdateMask<float>(sreg0);
    DataCopy(tmpReg3, src);
    ReduceSum(dstReg, tmpReg3, pregLoop);
}

template <typename U>
__aicore__ inline void LoadTwoCloseRegVF(RegTensor<U>& dstA, RegTensor<U>& dstB, __local_mem__ U* srcAddr,
                                         uint16_t offset)
{
    if constexpr (IsSameType<U, float>::value) {
        DataCopy(dstA, srcAddr + offset);
        DataCopy(dstB, srcAddr + offset + V_LENGTH);
    } else {
        DataCopy<U, LoadDist::DIST_UNPACK_B16>(dstA, srcAddr + offset);
        DataCopy<U, LoadDist::DIST_UNPACK_B16>(dstB, srcAddr + offset + V_LENGTH);
    }
}

template <typename U>
__aicore__ inline void CastAddVF(RegTensor<float>& dstReg, RegTensor<U>& src1Reg, RegTensor<U>& src2Reg,
                                 MaskReg& pregLoop)
{
    if constexpr (IsSameType<U, float>::value) {
        Add(dstReg, src1Reg, src2Reg, pregLoop);
    } else {
        RegTensor<float> src1RegFp32, src2RegFp32;
        Cast<float, U, castTraitB162B32>(src1RegFp32, src1Reg, pregLoop);
        Cast<float, U, castTraitB162B32>(src2RegFp32, src2Reg, pregLoop);
        Add(dstReg, src1RegFp32, src2RegFp32, pregLoop);
    }
}

/**
 * @brief Load and cast to fp32 reg.
 * @param offset idx of VF loop.
 */
template <typename T>
__aicore__ inline void LoadCastRegVF(RegTensor<float>& dstTensor, __local_mem__ T* srcAddr, uint16_t offset,
                                     MaskReg& pregLoop)
{
    if constexpr (IsSameType<T, float>::value) {
        DataCopy(dstTensor, srcAddr + offset * V_LENGTH);
    } else {
        RegTensor<T> loadTmp;
        DataCopy<T, LoadDist::DIST_UNPACK_B16>(loadTmp, srcAddr + offset * V_LENGTH);
        Cast<float, T, castTraitB162B32>(dstTensor, loadTmp, pregLoop);
    }
}

template <typename T>
__aicore__ inline void CastStoreTwoCloseRegVF(__local_mem__ T* dstAddr, RegTensor<float>& srcA, RegTensor<float>& srcB,
                                              uint16_t offset, MaskReg& pregLoop)
{
    if constexpr (IsSameType<T, float>::value) {
        DataCopy(dstAddr + offset, srcA, pregLoop);
        DataCopy(dstAddr + offset + V_LENGTH, srcB, pregLoop);
    } else {
        RegTensor<T> srcATmp, srcBTmp;
        Cast<T, float, castTraitB322B16>(srcATmp, srcA, pregLoop);
        Cast<T, float, castTraitB322B16>(srcBTmp, srcB, pregLoop);
        DataCopy<T, StoreDist::DIST_PACK_B32>(dstAddr + offset, srcATmp, pregLoop);
        DataCopy<T, StoreDist::DIST_PACK_B32>(dstAddr + offset + V_LENGTH, srcBTmp, pregLoop);
    }
}

/**
 * @brief Use VF to Compute reduceSum.
 *        dstLocal = reduceSum((x1+x2)^2)
 *        If HAS_XOUT is true, return xOut = (x1.to(float) + x2.to(float)).to(dtype).
 *        If HAS_XOUT_FP32 is true, return xOutFp32 = x1.to(float) + x2.to(float).
 *        If IS_RSTD is true, dstLocal = 1.0 / sqrt(avgFactor * reduceSum((x1+x2)^2) + epsilon)
 *        Use float32 VL_LENGTH
 */
template <typename U, bool HAS_XOUT = false, bool HAS_XOUT_FP32 = false, bool IS_RSTD = false>
__aicore__ inline void ReduceSumRstd(LocalTensor<float>& dstLocal, LocalTensor<U>& xOutLocal,
                                     LocalTensor<float>& xOutFp32Local, LocalTensor<U>& x1Local,
                                     LocalTensor<U>& x2Local, LocalTensor<float>& workLocal, uint32_t dstOffset,
                                     uint32_t count, uint32_t powerSplit, float avgFactor = 1.0f, float epsilon = 0.0f)
{
    uint32_t remainTile = count - powerSplit;
    uint32_t remainSreg = remainTile;
    uint16_t remainRepeats = remainTile / (2 * V_LENGTH);

    uint32_t masterTile = powerSplit - remainTile;
    uint32_t masterSreg = masterTile;
    uint16_t masterRepeats = masterTile / (2 * V_LENGTH);

    uint32_t mergeTile = powerSplit / (2 * V_LENGTH);
    uint32_t mergeSreg = mergeTile;
    uint16_t mergeRepeats = mergeTile / (2 * V_LENGTH);

    uint32_t meanTile = mergeRepeats == 0 ? mergeTile : mergeRepeats;
    uint32_t meanSreg = meanTile;

    __local_mem__ U* x1MainAddr = (__ubuf__ U*)x1Local.GetPhyAddr();
    __local_mem__ U* x1TailAddr = (__ubuf__ U*)x1Local.GetPhyAddr() + int64_t(powerSplit);
    __local_mem__ U* x1MasterAddr = (__ubuf__ U*)x1Local.GetPhyAddr() + int64_t(remainTile);
    __local_mem__ U* x2MainAddr = (__ubuf__ U*)x2Local.GetPhyAddr();
    __local_mem__ U* x2TailAddr = (__ubuf__ U*)x2Local.GetPhyAddr() + int64_t(powerSplit);
    __local_mem__ U* x2MasterAddr = (__ubuf__ U*)x2Local.GetPhyAddr() + int64_t(remainTile);
    __local_mem__ U* xOutMainAddr;
    __local_mem__ U* xOutTailAddr;
    __local_mem__ U* xOutMasterAddr;
    if constexpr (HAS_XOUT) {
        xOutMainAddr = (__ubuf__ U*)xOutLocal.GetPhyAddr();
        xOutTailAddr = (__ubuf__ U*)xOutLocal.GetPhyAddr() + int64_t(powerSplit);
        xOutMasterAddr = (__ubuf__ U*)xOutLocal.GetPhyAddr() + int64_t(remainTile);
    }
    __local_mem__ float* xOutFp32MainAddr;
    __local_mem__ float* xOutFp32TailAddr;
    __local_mem__ float* xOutFp32MasterAddr;
    if constexpr (HAS_XOUT_FP32) {
        xOutFp32MainAddr = (__ubuf__ float*)xOutFp32Local.GetPhyAddr();
        xOutFp32TailAddr = (__ubuf__ float*)xOutFp32Local.GetPhyAddr() + int64_t(powerSplit);
        xOutFp32MasterAddr = (__ubuf__ float*)xOutFp32Local.GetPhyAddr() + int64_t(remainTile);
    }
    __local_mem__ float* workAddr = (__ubuf__ float*)workLocal.GetPhyAddr();
    __local_mem__ float* dstAddr = (__ubuf__ float*)dstLocal.GetPhyAddr();

    __VEC_SCOPE__
    {
        RegTensor<float> mainA, mainB, tailA, tailB, vSum, vDupReg;
        RegTensor<U> x1MainA, x1MainB, x1TailA, x1TailB;
        RegTensor<U> x2MainA, x2MainB, x2TailA, x2TailB;
        MaskReg pregMerge = CreateMask<float, MaskPattern::VL1>();
        MaskReg pregLoop;

        for (uint16_t i = 0; i < (uint16_t)remainRepeats; ++i) {
            pregLoop = UpdateMask<float>(remainSreg);
            uint16_t offset = i * 2 * V_LENGTH;
            // 1. Copy in reg
            LoadTwoCloseRegVF(x1MainA, x1MainB, x1MainAddr, offset);
            LoadTwoCloseRegVF(x1TailA, x1TailB, x1TailAddr, offset);
            LoadTwoCloseRegVF(x2MainA, x2MainB, x2MainAddr, offset);
            LoadTwoCloseRegVF(x2TailA, x2TailB, x2TailAddr, offset);
            // 2. Cast add
            CastAddVF(mainA, x1MainA, x2MainA, pregLoop);
            CastAddVF(tailA, x1TailA, x2TailA, pregLoop);
            CastAddVF(mainB, x1MainB, x2MainB, pregLoop);
            CastAddVF(tailB, x1TailB, x2TailB, pregLoop);
            if constexpr (HAS_XOUT) {
                CastStoreTwoCloseRegVF(xOutMainAddr, mainA, mainB, offset, pregLoop);
                CastStoreTwoCloseRegVF(xOutTailAddr, tailA, tailB, offset, pregLoop);
            }
            if constexpr (HAS_XOUT_FP32) {
                CastStoreTwoCloseRegVF(xOutFp32MainAddr, mainA, mainB, offset, pregLoop);
                CastStoreTwoCloseRegVF(xOutFp32TailAddr, tailA, tailB, offset, pregLoop);
            }
            // 3. Cal x^2
            Mul(mainA, mainA, mainA, pregLoop);
            Mul(tailA, tailA, tailA, pregLoop);
            Mul(mainB, mainB, mainB, pregLoop);
            Mul(tailB, tailB, tailB, pregLoop);
            Add(mainA, mainA, tailA, pregLoop);
            Add(mainB, mainB, tailB, pregLoop);
            Add(mainA, mainA, mainB, pregLoop);
            ReduceSum(vSum, mainA, pregLoop);
            DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(workAddr + i, vSum, pregMerge);
        }
        for (uint16_t i = 0; i < (uint16_t)masterRepeats; ++i) {
            uint16_t offset = i * 2 * V_LENGTH;
            pregLoop = UpdateMask<float>(masterSreg);
            // 1. Copy in reg
            LoadTwoCloseRegVF(x1MainA, x1MainB, x1MasterAddr, offset);
            LoadTwoCloseRegVF(x2MainA, x2MainB, x2MasterAddr, offset);
            // 2. Cast add
            CastAddVF(mainA, x1MainA, x2MainA, pregLoop);
            CastAddVF(mainB, x1MainB, x2MainB, pregLoop);
            if constexpr (HAS_XOUT) {
                CastStoreTwoCloseRegVF(xOutMasterAddr, mainA, mainB, offset, pregLoop);
            }
            if constexpr (HAS_XOUT_FP32) {
                CastStoreTwoCloseRegVF(xOutFp32MasterAddr, mainA, mainB, offset, pregLoop);
            }
            // 3. Cal x^2
            Mul(mainA, mainA, mainA, pregLoop);
            Mul(mainB, mainB, mainB, pregLoop);
            Add(mainA, mainA, mainB, pregLoop);
            ReduceSum(vSum, mainA, pregLoop);
            DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(workAddr + remainRepeats + i, vSum, pregMerge);
        }
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        for (uint16_t i = 0; i < (uint16_t)mergeRepeats; ++i) {
            pregLoop = UpdateMask<float>(mergeSreg);
            uint16_t offset = i * 2 * V_LENGTH;
            LoadTwoCloseRegVF(mainA, mainB, workAddr, offset);
            Add(mainA, mainA, mainB, pregLoop);
            ReduceSum(vSum, mainA, pregLoop);
            DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(workAddr + i, vSum, pregMerge);
        }
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        pregLoop = UpdateMask<float>(meanSreg);
        DataCopy(mainA, workAddr);
        ReduceSum(vSum, mainA, pregLoop);
        if constexpr (IS_RSTD) {
            Muls(vSum, vSum, avgFactor, pregMerge);
            Adds(vSum, vSum, epsilon, pregMerge);
            Sqrt(vSum, vSum, pregMerge);
            Duplicate(vDupReg, float(1.0), pregMerge);
            Div(vSum, vDupReg, vSum, pregMerge);
        }
        DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dstAddr + dstOffset, vSum, pregMerge);
    }
}

/**
 * @brief Use VF to Compute reduceSum(multi line).
 *        dstLocal = reduceSum((x1+x2)^2)
 *        If HAS_XOUT_FP32 is true, return xOutFp32 = x1.to(float) + x2.to(float).
 *        If IS_RSTD is true, dstLocal = 1.0 / sqrt(avgFactor * reduceSum((x1+x2)^2) + epsilon)
 *        Use float32 VL_LENGTH
 */
#include "reduce_common_regbase_part1.h"
#include "reduce_common_regbase_part2.h"

} // namespace NormCommon

#endif // REDUCE_COMMON_REGBASE_H_RMS_NORM
