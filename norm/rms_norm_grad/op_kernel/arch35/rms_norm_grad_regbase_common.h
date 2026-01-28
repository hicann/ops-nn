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
 * \file rms_norm_grad_regbase_common.h
 * \brief RmsNormGrad regbase common
 */
#ifndef OPS_BUILT_IN_TBE_IMPL_ASCENDC_RMS_NORM_GRAD_REGBASE_COMMON_H
#define OPS_BUILT_IN_TBE_IMPL_ASCENDC_RMS_NORM_GRAD_REGBASE_COMMON_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../inc/platform.h"

namespace RmsNormGrad {
using namespace AscendC;
using namespace AscendC::MicroAPI;
using AscendC::MicroAPI::CreateMask;
using AscendC::MicroAPI::LoadDist;
using AscendC::MicroAPI::LocalMemBar;
using AscendC::MicroAPI::MaskPattern;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::MemType;
using AscendC::MicroAPI::RegTensor;
using AscendC::MicroAPI::StoreDist;
using AscendC::MicroAPI::UpdateMask;

namespace RmsNormGradRegbase {
__aicore__ inline constexpr uint32_t GetVRegSize()
{
#if __CCE_AICORE__ == 310
    return AscendC::VECTOR_REG_WIDTH;
#else
    return 256U;
#endif
}
} // namespace

constexpr uint32_t ONCE_VECTOR_SIZE = 256;
constexpr uint32_t V_LENGTH = RmsNormGradRegbase::GetVRegSize() / sizeof(float);
constexpr uint32_t FLOAT_NUM_BLOCK = 8;
constexpr uint32_t HALF_NUM_BLOCK = 16;
constexpr uint32_t FLOAT_NUM_2VL = 128;
constexpr uint32_t DB_NUM = 2;
constexpr uint32_t DEPTH_TWO = 2;
constexpr uint32_t NUM_TWO = 2;
constexpr uint32_t UB_FACTOR_DX_FULL_LOAD = 6144;
constexpr uint32_t UB_FACTOR_DX_SPLIT_D = 4096;

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

template <typename T>
__aicore__ inline T Min(T left, T right)
{
    return (left < right ? left : right);
}

__aicore__ inline int32_t findPowerTwo(int32_t n)
{
    // find max power of 2 no more than n (32 bit)
    n |= n >> 1;  // 将最高位的1向右扩展1位， 11XX..X
    n |= n >> 2;  // 将最高位的1向右扩展2位， 1111X..X
    n |= n >> 4;  // 将最高位的1向右扩展4位
    n |= n >> 8;  // 将最高位的1向右扩展8位
    n |= n >> 16; // 将最高位的1向右扩展16位
    return (n + 1) >> 1;
}

template <typename T>
__aicore__ inline void LoadAndCast(
    RegTensor<float>& dstReg, __local_mem__ T* srcAddr, MaskReg& maskReg, uint32_t srcOffset)
{
    if constexpr (IsSameType<T, float>::value) {
        DataCopy(dstReg, srcAddr + srcOffset);
    } else {
        RegTensor<T> dstRegB16;
        // DataCopy<T, LoadDist::DIST_UNPACK_B16>(dstRegB16, srcAddr + ubFactorD_ + srcOffset);  // 后续不做偏移
        DataCopy<T, LoadDist::DIST_UNPACK_B16>(dstRegB16, srcAddr + srcOffset); // 后续不做偏移
        Cast<float, T, castTraitB162B32>(dstReg, dstRegB16, maskReg);
    }
}

/*!
 * Compute ReduceSum mean
 *
 * @param dstLocal dst levelTensor
 * @param srcLocal src LevelTenso
 * @param offset dst offset
 * @param count src level size
 * @return
 */
__aicore__ inline void LevelMerge(
    LocalTensor<float>& dstLocal, LocalTensor<float> srcLocal, uint64_t offset, uint32_t count)
{
    uint64_t calCount = count / 4;
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
        RegTensor<float> vRegA, vRegB, vRegC, vRegD, dstReg, vMean;
        MaskReg pregMerge = CreateMask<float, MaskPattern::VL1>();
        MaskReg pregLoop;
        for (uint16_t i = 0; i < repeatTimes; ++i) {
            pregLoop = UpdateMask<float>(sreg);
            DataCopy(vRegA, src1Addr + static_cast<uint32_t>(i * V_LENGTH));
            DataCopy(vRegB, src2Addr + static_cast<uint32_t>(i * V_LENGTH));
            DataCopy(vRegC, src3Addr + static_cast<uint32_t>(i * V_LENGTH));
            DataCopy(vRegD, src4Addr + static_cast<uint32_t>(i * V_LENGTH));
            Add(vRegA, vRegA, vRegB, pregLoop);
            Add(vRegC, vRegC, vRegD, pregLoop);
            Add(dstReg, vRegA, vRegC, pregLoop);
            ReduceSum(vMean, dstReg, pregLoop);
            DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dstAddr + static_cast<uint32_t>(offset), vMean, pregMerge);
        }
    }
}

/*!
 * The num of each level elements is 256, ReduceSum these elements and store to the next level.
 * @param level1Local level1Tensor
 * @param level2Local level2Tensor
 * @param level3Local level3Tensor
 * @param level1 level1 elements
 * @param level2 level2 elements
 * @param level3 level3 elements
 * @return void
 */
__aicore__ inline void ComputeMultiLevelReduce(
    LocalTensor<float>& level1Local, LocalTensor<float>& level2Local, LocalTensor<float>& level3Local, uint32_t& level1,
    uint32_t& level2, uint32_t& level3)
{
    if (level1 == ONCE_VECTOR_SIZE) {
        LevelMerge(level2Local, level1Local, level2, ONCE_VECTOR_SIZE);
        level1 = 0;
        level2 += 1;
    }
    if (level2 == ONCE_VECTOR_SIZE) {
        LevelMerge(level3Local, level2Local, level3, ONCE_VECTOR_SIZE);
        level2 = 0;
        level3 += 1;
    }
}

/*!
 * compute final ReduceSum result
 * @param dstLocal dst Tensor
 * @param offset dst offset
 * @param level1Local level1 Tensor
 * @param level2Local level2 Tensor
 * @param level3Local level3 Tensor
 * @param level1 level1 elements
 * @param level2 level2 elements
 * @param level3 level3 elements
 * @return
 */
__aicore__ inline void ComputeMultiLevelMean(
    LocalTensor<float>& dstLocal, uint32_t offset, LocalTensor<float>& level1Local, LocalTensor<float>& level2Local,
    LocalTensor<float>& level3Local, uint32_t& level1, uint32_t& level2)
{
    if (level1 > 0 && level1 < ONCE_VECTOR_SIZE) {
        LevelMerge(dstLocal, level1Local, offset, ONCE_VECTOR_SIZE);
    } else if (level2 > 0 && level2 < ONCE_VECTOR_SIZE) {
        LevelMerge(dstLocal, level2Local, offset, ONCE_VECTOR_SIZE);
    } else {
        LevelMerge(dstLocal, level3Local, offset, ONCE_VECTOR_SIZE);
    }
}

/*!
 * ReduceSum impl by half add.
 * @param dstLocal dst Tensor
 * @param srcLocal src Tensor
 * @param workLocal  temp Tensor
 * @param offset dst offset
 * @param count count aligned compute elements.
 * @param powerSplit 2 ** k = powerSplit
 * @return void
 */
__aicore__ inline void ReduceSumImpl(
    LocalTensor<float>& dstLocal, LocalTensor<float>& srcLocal, LocalTensor<float>& workLocal, uint32_t offset,
    uint32_t count, uint32_t powerSplit)
{
    uint32_t remainTile = count - powerSplit;
    uint32_t remainSreg = remainTile;
    uint32_t remainRepeats = remainTile / (2 * V_LENGTH);

    uint32_t masterTile = powerSplit - remainTile;
    uint32_t masterSreg = masterTile;
    uint16_t masterRepeats = masterTile / (2 * V_LENGTH);

    uint32_t mergeTile = powerSplit / (2 * V_LENGTH);
    uint32_t mergeSreg = mergeTile;
    uint32_t mergeRepeats = mergeTile / (2 * V_LENGTH);

    uint32_t meanTile = mergeRepeats == 0 ? mergeTile : mergeRepeats;
    uint32_t meanSreg = meanTile;

    __local_mem__ float* mainAddr = (__ubuf__ float*)srcLocal.GetPhyAddr();
    __local_mem__ float* tailAddr = (__ubuf__ float*)srcLocal.GetPhyAddr() + int64_t(powerSplit);
    __local_mem__ float* masterAddr = (__ubuf__ float*)srcLocal.GetPhyAddr() + int64_t(remainTile);
    __local_mem__ float* workAddr = (__ubuf__ float*)workLocal.GetPhyAddr();
    __local_mem__ float* dstAddr = (__ubuf__ float*)dstLocal.GetPhyAddr();

    __VEC_SCOPE__
    {
        RegTensor<float> mainA, mainB, tailA, tailB, vMean;
        MaskReg pregMain = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregMerge = CreateMask<float, MaskPattern::VL1>();
        MaskReg pregLoop;

        for (uint16_t i = 0; i < (uint16_t)remainRepeats; ++i) {
            pregLoop = UpdateMask<float>(remainSreg);
            DataCopy(mainA, mainAddr + static_cast<uint32_t>((i * 2 + 0) * V_LENGTH));
            DataCopy(mainB, mainAddr + static_cast<uint32_t>((i * 2 + 1) * V_LENGTH));
            DataCopy(tailA, tailAddr + static_cast<uint32_t>((i * 2 + 0) * V_LENGTH));
            DataCopy(tailB, tailAddr + static_cast<uint32_t>((i * 2 + 1) * V_LENGTH));

            Add(mainA, mainA, tailA, pregLoop);
            Add(mainB, mainB, tailB, pregLoop);
            Add(mainA, mainA, mainB, pregLoop);
            ReduceSum(vMean, mainA, pregLoop);
            DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(workAddr + static_cast<uint32_t>(i), vMean, pregMerge);
        }
        for (uint16_t i = 0; i < (uint16_t)masterRepeats; ++i) {
            pregLoop = UpdateMask<float>(masterSreg);
            DataCopy(mainA, masterAddr + static_cast<uint32_t>((i * 2 + 0) * V_LENGTH));
            DataCopy(mainB, masterAddr + static_cast<uint32_t>((i * 2 + 1) * V_LENGTH));
            Add(mainA, mainA, mainB, pregLoop);
            ReduceSum(vMean, mainA, pregLoop);
            DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(workAddr + static_cast<uint32_t>(remainRepeats + i), vMean, pregMerge);
        }
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        for (uint16_t i = 0; i < (uint16_t)mergeRepeats; ++i) {
            pregLoop = UpdateMask<float>(mergeSreg);
            DataCopy(mainA, workAddr + static_cast<uint32_t>((i * 2 + 0) * V_LENGTH));
            DataCopy(mainB, workAddr + static_cast<uint32_t>((i * 2 + 1) * V_LENGTH));
            Add(mainA, mainA, mainB, pregLoop);
            ReduceSum(vMean, mainA, pregLoop);
            DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(workAddr + static_cast<uint32_t>(i), vMean, pregMerge);
        }
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        {
            pregLoop = UpdateMask<float>(meanSreg);
            DataCopy(mainA, workAddr + 0);
            ReduceSum(vMean, mainA, pregLoop);
            DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dstAddr + offset, vMean, pregMerge);
        }
    }
}

/*!
 * MultiReduceSum(AR pattern) impl by half add.
 * @param dstLocal dst Tensor
 * @param srcLocal src Tensor
 * @param workLocal  temp Tensor
 * @param rows calc rows once
 * @param colsAlign2VL cols aligned 512B.
 * @param powerSplit 2 ** k = powerSplit
 * @return void
 */
__aicore__ inline void MultiReduceSumImpl(
    LocalTensor<float>& dstLocal, LocalTensor<float>& srcLocal, LocalTensor<float>& workLocal, uint32_t rows,
    uint32_t colsAlign2VL, uint32_t powerSplit)
{
    uint32_t remainTile = colsAlign2VL - powerSplit;
    uint32_t remainRepeats = remainTile / (2 * V_LENGTH);

    uint32_t masterTile = powerSplit - remainTile;
    uint16_t masterRepeats = masterTile / (2 * V_LENGTH);

    uint32_t mergeTile = powerSplit / (2 * V_LENGTH);
    uint32_t mergeRepeats = mergeTile / (2 * V_LENGTH);

    uint32_t meanTile = mergeRepeats == 0 ? mergeTile : mergeRepeats;

    __local_mem__ float* workAddr = (__ubuf__ float*)workLocal.GetPhyAddr();
    __local_mem__ float* dstAddr = (__ubuf__ float*)dstLocal.GetPhyAddr();

    __VEC_SCOPE__
    {
        RegTensor<float> mainA, mainB, tailA, tailB, vMean;
        for (uint16_t r = 0; r < (uint16_t)rows; ++r) {
            uint32_t remainSreg = remainTile;
            uint32_t masterSreg = masterTile;
            uint32_t mergeSreg = mergeTile;
            uint32_t meanSreg = meanTile;

            __local_mem__ float* mainAddr = (__ubuf__ float*)srcLocal.GetPhyAddr() + r * colsAlign2VL;
            __local_mem__ float* tailAddr =
                (__ubuf__ float*)srcLocal.GetPhyAddr() + r * colsAlign2VL + int64_t(powerSplit);
            __local_mem__ float* masterAddr =
                (__ubuf__ float*)srcLocal.GetPhyAddr() + r * colsAlign2VL + int64_t(remainTile);

            MaskReg pregMain = CreateMask<float, MaskPattern::ALL>();
            MaskReg pregMerge = CreateMask<float, MaskPattern::VL1>();
            MaskReg pregLoop;

            for (uint16_t i = 0; i < (uint16_t)remainRepeats; ++i) {
                pregLoop = UpdateMask<float>(remainSreg);
                DataCopy(mainA, mainAddr + static_cast<uint32_t>((i * 2 + 0) * V_LENGTH));
                DataCopy(mainB, mainAddr + static_cast<uint32_t>((i * 2 + 1) * V_LENGTH));
                DataCopy(tailA, tailAddr + static_cast<uint32_t>((i * 2 + 0) * V_LENGTH));
                DataCopy(tailB, tailAddr + static_cast<uint32_t>((i * 2 + 1) * V_LENGTH));

                Add(mainA, mainA, tailA, pregLoop);
                Add(mainB, mainB, tailB, pregLoop);
                Add(mainA, mainA, mainB, pregLoop);
                ReduceSum(vMean, mainA, pregLoop);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(workAddr + static_cast<uint32_t>(i), vMean, pregMerge);
            }
            for (uint16_t i = 0; i < (uint16_t)masterRepeats; ++i) {
                pregLoop = UpdateMask<float>(masterSreg);
                DataCopy(mainA, masterAddr + static_cast<uint32_t>((i * 2 + 0) * V_LENGTH));
                DataCopy(mainB, masterAddr + static_cast<uint32_t>((i * 2 + 1) * V_LENGTH));
                Add(mainA, mainA, mainB, pregLoop);
                ReduceSum(vMean, mainA, pregLoop);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(workAddr + static_cast<uint32_t>(remainRepeats + i), vMean, pregMerge);
            }
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
            for (uint16_t i = 0; i < (uint16_t)mergeRepeats; ++i) {
                pregLoop = UpdateMask<float>(mergeSreg);
                DataCopy(mainA, workAddr + static_cast<uint32_t>((i * 2 + 0) * V_LENGTH));
                DataCopy(mainB, workAddr + static_cast<uint32_t>((i * 2 + 1) * V_LENGTH));
                Add(mainA, mainA, mainB, pregLoop);
                ReduceSum(vMean, mainA, pregLoop);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(workAddr + static_cast<uint32_t>(i), vMean, pregMerge);
            }
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
            {
                pregLoop = UpdateMask<float>(meanSreg);
                DataCopy(mainA, workAddr + 0);
                ReduceSum(vMean, mainA, pregLoop);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dstAddr + static_cast<uint32_t>(r), vMean, pregMerge);
            }
        }
    }
}

} // namespace RmsNormGrad
#endif // OPS_BUILT_IN_TBE_IMPL_ASCENDC_RMS_NORM_GRAD_REGBASE_COMMON_H