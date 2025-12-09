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
using AscendC::MicroAPI::CreateMask;
using AscendC::MicroAPI::LoadDist;
using AscendC::MicroAPI::LocalMemBar;
using AscendC::MicroAPI::MaskPattern;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::MemType;
using AscendC::MicroAPI::RegTensor;
using AscendC::MicroAPI::StoreDist;
using AscendC::MicroAPI::UpdateMask;

namespace NormCommonRegbase {
__aicore__ inline constexpr uint32_t GetVRegSize()
{
    return 256U;
}

template <typename T>
__aicore__ inline T CeilDiv(T a, T b)
{
    using type = typename std::conditional<
        sizeof(T) == sizeof(uint8_t) || sizeof(T) == sizeof(uint16_t), uint32_t, uint64_t>::type;
    type res = (static_cast<type>(a) + static_cast<type>(b) - 1) / static_cast<type>(b);
    return static_cast<T>(res);
}

template <typename T>
__aicore__ inline T CeilAlign(T a, T b)
{
    using type = typename std::conditional<
        sizeof(T) == sizeof(uint8_t) || sizeof(T) == sizeof(uint16_t), uint32_t, uint64_t>::type;
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

} // namespace

constexpr int32_t VL_SIZE = NormCommonRegbase::GetVRegSize();
constexpr int32_t V_LENGTH = (VL_SIZE / static_cast<int32_t>(sizeof(float)));
constexpr uint32_t ONCE_VECTOR_SIZE = 256;

constexpr AscendC::MicroAPI::CastTrait castTraitB162B32 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};

constexpr AscendC::MicroAPI::CastTrait castTraitB322B16 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

template <typename U>
__aicore__ inline void LoadTwoCloseRegVF(
    RegTensor<U>& dstA, RegTensor<U>& dstB, __local_mem__ U* srcAddr, uint16_t offset)
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
__aicore__ inline void CastAddVF(
    RegTensor<float>& dstReg, RegTensor<U>& src1Reg, RegTensor<U>& src2Reg, MaskReg& pregLoop)
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
__aicore__ inline void LoadCastRegVF(
    RegTensor<float>& dstTensor, __local_mem__ T* srcAddr, uint16_t offset, MaskReg& pregLoop)
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
__aicore__ inline void CastStoreTwoCloseRegVF(
    __local_mem__ T* dstAddr, RegTensor<float>& srcA, RegTensor<float>& srcB, uint16_t offset, MaskReg& pregLoop)
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
    LocalTensor<float>& xOutFp32Local, LocalTensor<U>& x1Local, LocalTensor<U>& x2Local, LocalTensor<float>& workLocal,
    uint32_t dstOffset, uint32_t count, uint32_t powerSplit, float avgFactor = 1.0f, float epsilon = 0.0f)
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
        MaskReg pregMain = CreateMask<float, MaskPattern::ALL>();
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
template <typename U, bool HAS_XOUT = false, bool HAS_XOUT_FP32 = false, bool IS_RSTD = false>
__aicore__ inline void ReduceSumRstdMulti(
    LocalTensor<float>& rstdLocal, LocalTensor<U>& xOutLocal, LocalTensor<float>& xOutFp32Local,
    LocalTensor<U>& x1Local, LocalTensor<U>& x2Local, LocalTensor<float>& workLocal, uint32_t rstdOffsetStart,
    uint32_t count, uint32_t powerSplit, uint32_t repeatTimes, float avgFactor = 1.0f, float epsilon = 0.0f)
{
    uint32_t rstdOffset = rstdOffsetStart;
    uint32_t remainTile = count - powerSplit;
    uint16_t remainRepeats = remainTile / (2 * V_LENGTH);

    uint32_t masterTile = powerSplit - remainTile;
    uint16_t masterRepeats = masterTile / (2 * V_LENGTH);

    uint32_t mergeTile = powerSplit / (2 * V_LENGTH);
    uint16_t mergeRepeats = mergeTile / (2 * V_LENGTH);

    uint32_t meanTile = mergeRepeats == 0 ? mergeTile : mergeRepeats;

    __local_mem__ U* x1MainAddr = (__ubuf__ U*)x1Local.GetPhyAddr();
    __local_mem__ U* x1TailAddr = (__ubuf__ U*)x1Local.GetPhyAddr() + int64_t(powerSplit);
    __local_mem__ U* x1MasterAddr = (__ubuf__ U*)x1Local.GetPhyAddr() + int64_t(remainTile);
    __local_mem__ U* x2MainAddr = (__ubuf__ U*)x2Local.GetPhyAddr();
    __local_mem__ U* x2TailAddr = (__ubuf__ U*)x2Local.GetPhyAddr() + int64_t(powerSplit);
    __local_mem__ U* x2MasterAddr = (__ubuf__ U*)x2Local.GetPhyAddr() + int64_t(remainTile);
    __local_mem__ U *xOutMainAddr, *xOutTailAddr, *xOutMasterAddr;
    if constexpr (HAS_XOUT) {
        xOutMainAddr = (__ubuf__ U*)xOutLocal.GetPhyAddr();
        xOutTailAddr = (__ubuf__ U*)xOutLocal.GetPhyAddr() + int64_t(powerSplit);
        xOutMasterAddr = (__ubuf__ U*)xOutLocal.GetPhyAddr() + int64_t(remainTile);
    }
    __local_mem__ float *xOutFp32MainAddr, *xOutFp32TailAddr, *xOutFp32MasterAddr;
    if constexpr (HAS_XOUT_FP32) {
        xOutFp32MainAddr = (__ubuf__ float*)xOutFp32Local.GetPhyAddr();
        xOutFp32TailAddr = (__ubuf__ float*)xOutFp32Local.GetPhyAddr() + int64_t(powerSplit);
        xOutFp32MasterAddr = (__ubuf__ float*)xOutFp32Local.GetPhyAddr() + int64_t(remainTile);
    }
    __local_mem__ float* workAddr = (__ubuf__ float*)workLocal.GetPhyAddr();
    __local_mem__ float* rstdAddr = (__ubuf__ float*)rstdLocal.GetPhyAddr();

    __VEC_SCOPE__
    {
        MaskReg pregMain = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregMerge = CreateMask<float, MaskPattern::VL1>();

        for (uint16_t row = 0; row < (uint16_t)repeatTimes; row++) {
            uint32_t remainSreg = remainTile;
            uint32_t masterSreg = masterTile;
            uint32_t mergeSreg = mergeTile;
            uint32_t meanSreg = meanTile;
            RegTensor<U> x1MainA, x1MainB, x1TailA, x1TailB;
            RegTensor<U> x2MainA, x2MainB, x2TailA, x2TailB;
            RegTensor<float> mainA, mainB, tailA, tailB, vSum, vDupReg, rstdReg;
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
                pregLoop = UpdateMask<float>(masterSreg);
                uint16_t offset = i * 2 * V_LENGTH;
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
                Div(rstdReg, vDupReg, vSum, pregMerge);
            }
            DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(rstdAddr + rstdOffset, rstdReg, pregMerge);

            rstdOffset++;
            x1MainAddr += int64_t(count);
            x1TailAddr += int64_t(count);
            x1MasterAddr += int64_t(count);
            x2MainAddr += int64_t(count);
            x2TailAddr += int64_t(count);
            x2MasterAddr += int64_t(count);
            if constexpr (HAS_XOUT) {
                xOutMainAddr += int64_t(count);
                xOutTailAddr += int64_t(count);
                xOutMasterAddr += int64_t(count);
            }
            if constexpr (HAS_XOUT_FP32) {
                xOutFp32MainAddr += int64_t(count);
                xOutFp32TailAddr += int64_t(count);
                xOutFp32MasterAddr += int64_t(count);
            }
        }
    }
}

/*!
 * @brief Compute ReduceSum mean
 *        IS_RSTD: if True, will cal rstd otherwise sum.
 * @param dstLocal dst levelTensor
 * @param srcLocal src LevelTensor
 * @param offset dst offset
 * @param count src level size, must be ONCE_VECTOR_SIZE
 * @param avgFactor avgFactor for cal rstd
 * @param epsilon epsilon for cal rstd
 * @return
 */
template <bool IS_RSTD>
__aicore__ inline void LevelMergeRstd(
    LocalTensor<float>& dstLocal, LocalTensor<float> srcLocal, uint64_t offset, uint32_t count, float avgFactor = 1.0f,
    float epsilon = 0.0f)
{
    uint64_t calCount = count / 4; // Div 4 for VF parallel execution.
    uint32_t sreg = (uint32_t)(calCount);
    uint16_t repeatTimes = CeilDivision(calCount, V_LENGTH);
    uint32_t meanTile = repeatTimes;

    __local_mem__ float* src1Addr = (__ubuf__ float*)srcLocal.GetPhyAddr() + 0 * calCount;
    __local_mem__ float* src2Addr = (__ubuf__ float*)srcLocal.GetPhyAddr() + 1 * calCount;
    __local_mem__ float* src3Addr = (__ubuf__ float*)srcLocal.GetPhyAddr() + 2 * calCount;
    __local_mem__ float* src4Addr = (__ubuf__ float*)srcLocal.GetPhyAddr() + 3 * calCount;
    __local_mem__ float* dstAddr = (__ubuf__ float*)dstLocal.GetPhyAddr();

    __VEC_SCOPE__
    {
        RegTensor<float> vRegA, vRegB, vRegC, vRegD, dstReg, vSum, vDupReg;
        MaskReg pregMerge = CreateMask<float, MaskPattern::VL1>();
        MaskReg pregLoop;
        for (uint16_t i = 0; i < repeatTimes; ++i) {
            pregLoop = UpdateMask<float>(sreg);
            DataCopy(vRegA, src1Addr + i * V_LENGTH);
            DataCopy(vRegB, src2Addr + i * V_LENGTH);
            DataCopy(vRegC, src3Addr + i * V_LENGTH);
            DataCopy(vRegD, src4Addr + i * V_LENGTH);
            Add(vRegA, vRegA, vRegB, pregLoop);
            Add(vRegC, vRegC, vRegD, pregLoop);
            Add(dstReg, vRegA, vRegC, pregLoop);
            ReduceSum(vSum, dstReg, pregLoop);
            if constexpr (IS_RSTD) {
                Muls(vSum, vSum, avgFactor, pregMerge);
                Adds(vSum, vSum, epsilon, pregMerge);
                Sqrt(vSum, vSum, pregMerge);
                Duplicate(vDupReg, float(1.0), pregMerge);
                Div(vSum, vDupReg, vSum, pregMerge);
            }
            DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dstAddr + offset, vSum, pregMerge);
        }
    }
}

/*!
 * @brief compute final ReduceSum result
 *        IS_RSTD: if True, will cal rstd otherwise sum.
 * @param dstLocal dst Tensor
 * @param offset dst offset
 * @param level1Local level1 Tensor
 * @param level2Local level2 Tensor
 * @param level3Local level3 Tensor
 * @param level1 level1 elements
 * @param level2 level2 elements
 * @param level3 level3 elements
 * @param avgFactor avgFactor for cal rstd
 * @param epsilon epsilon for cal rstd
 * @return
 */
template <bool IS_RSTD>
__aicore__ inline void ComputeMultiLevelRstd(
    LocalTensor<float>& dstLocal, uint32_t offset, LocalTensor<float>& level1Local, LocalTensor<float>& level2Local,
    LocalTensor<float>& level3Local, uint32_t& level1, uint32_t& level2, float avgFactor = 1.0f, float epsilon = 0.0f)
{
    if (level1 > 0 && level1 < ONCE_VECTOR_SIZE) {
        LevelMergeRstd<IS_RSTD>(dstLocal, level1Local, offset, ONCE_VECTOR_SIZE, avgFactor, epsilon);
    } else if (level2 > 0 && level2 < ONCE_VECTOR_SIZE) {
        LevelMergeRstd<IS_RSTD>(dstLocal, level2Local, offset, ONCE_VECTOR_SIZE, avgFactor, epsilon);
    } else {
        LevelMergeRstd<IS_RSTD>(dstLocal, level3Local, offset, ONCE_VECTOR_SIZE, avgFactor, epsilon);
    }
}
} // namespace NormCommon

#endif // _REDUCE_COMMON_REBASE_H_
