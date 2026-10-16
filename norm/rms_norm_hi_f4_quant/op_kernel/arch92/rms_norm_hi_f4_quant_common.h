/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RMS_NORM_HI_F4_QUANT_COMMON_H
#define RMS_NORM_HI_F4_QUANT_COMMON_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_utils.h"
#include "../inc/platform.h"
#include "op_kernel/platform_util.h"
#include "rms_norm_hi_f4_quant_tiling_data.h"

#define FLOAT_OVERFLOW_MODE_CTRL 60

namespace RmsNormHiF4Quant {
using namespace AscendC;
using namespace AscendC::MicroAPI;
using AscendC::MicroAPI::Add;
using AscendC::MicroAPI::CreateMask;
using AscendC::MicroAPI::LoadDist;
using AscendC::MicroAPI::LocalMemBar;
using AscendC::MicroAPI::MaskPattern;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::MemType;
using AscendC::MicroAPI::RegTensor;
using AscendC::MicroAPI::StoreDist;
using AscendC::MicroAPI::UpdateMask;

template <typename Tp, Tp v>
struct IntegralConstant {
    static constexpr Tp value = v;
};
using trueType = IntegralConstant<bool, true>;
using falseType = IntegralConstant<bool, false>;
template <typename, typename>
struct IsSame : public falseType {};
template <typename Tp>
struct IsSame<Tp, Tp> : public trueType {};

constexpr int64_t DIGIT_ONE = 1;
constexpr int64_t DIGIT_TWO = 2;
constexpr int64_t DIGIT_FOUR = 4;
constexpr int64_t DIGIT_EIGHT = 8;
constexpr int64_t DOUBLE_BUFFER_NUM = 2;
constexpr int64_t HIF4_BLOCK_SIZE = 64;
constexpr int64_t MX_STEP_PROCESS_NUM = 256;
constexpr int64_t DQ_HIFP4_GROUP_ELE_NUM = 64;
constexpr int64_t DQ_HIFP4_GROUP_BYTE_NUM = 32;
constexpr int64_t DQ_HIFP4_SCALE_ALIGN = 256;
constexpr int64_t DQ_HIFP4_MAXPER4_ALIGN = 128;
constexpr int64_t DQ_HIFP4_MAXPER8_ALIGN = 128;
constexpr int64_t DQ_HIFP4_SCALETMP_ALIGN = 256;
constexpr int64_t DQ_HIFP4_BLOCK_BYTE = 32;
constexpr int64_t DQ_HIFP4_CAST_COPYOUT_PACK4_NUM = 4;

constexpr static uint32_t VL_F32 = platform::GetVRegSize() / sizeof(float);
constexpr static uint32_t BLOCK_F32_ALIGN_NUM = Ops::Base::GetUbBlockSize() / sizeof(float);
constexpr static uint32_t UB_BLOCK_SIZE = Ops::Base::GetUbBlockSize();

constexpr int64_t HIF4_GROUP_PER_INNER_LOOP = 8;
constexpr int64_t HIF4_GROUP_PER_MAX8_LOOP = 16;
constexpr int64_t HIF4_GROUP_PER_SCALE_LOOP = 256;
constexpr int64_t HIF4_GROUP_PER_OUT_LOOP = 2;
constexpr int64_t AR_RECOMPUTE_SUM_BUFFER_BYTES = 32;
constexpr int64_t AR_RECOMPUTE_SUM_LEN = AR_RECOMPUTE_SUM_BUFFER_BYTES / sizeof(float);

constexpr AscendC::MicroAPI::CastTrait castTraitFp82B16 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};

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

__aicore__ inline uint64_t CeilDiv(uint64_t x, uint64_t y) { return y == 0 ? x : (x + y - 1) / y; }

__aicore__ inline uint64_t CeilAlign(uint64_t x, uint64_t y) { return y == 0 ? x : (x + y - 1) / y * y; }

template <typename T>
__aicore__ inline T Min(T a, T b)
{
    return a > b ? b : a;
}

template <typename T_IN>
__aicore__ inline void LoadTensorForDtypeTIn(__local_mem__ T_IN* src, RegTensor<float>& dst, MaskReg& preg,
                                             uint32_t offset)
{
    if constexpr (IsSameType<T_IN, float>::value) {
        DataCopy<float, LoadDist::DIST_NORM>(dst, src + offset);
    } else {
        RegTensor<T_IN> xIn;
        DataCopy<T_IN, LoadDist::DIST_UNPACK_B16>(xIn, src + offset);
        Cast<float, T_IN, castTraitB162B32>(dst, xIn, preg);
    }
}

template <typename T_OUT>
__aicore__ inline void StoreTensorForDtypeTOut(__local_mem__ T_OUT* dst, RegTensor<float>& src, MaskReg& preg,
                                               uint32_t offset)
{
    if constexpr (IsSameType<T_OUT, float>::value) {
        DataCopy<T_OUT, StoreDist::DIST_NORM>(dst + offset, src, preg);
    } else {
        RegTensor<T_OUT> xOut;
        Cast<T_OUT, float, castTraitB322B16>(xOut, src, preg);
        DataCopy<T_OUT, StoreDist::DIST_PACK_B32>(dst + offset, xOut, preg);
    }
}

__aicore__ inline int64_t GetCacheId(const int64_t idx)
{
    return AscendC::ScalarGetCountOfValue<1>(idx ^ (idx + 1)) - 1;
}

__aicore__ inline void UpdateCache(const LocalTensor<float>& dstTensor, const LocalTensor<float>& srcTensor,
                                   const int64_t cacheId, const int64_t stride)
{
    uint16_t innerLoopTimes = cacheId;
    uint32_t innerLoopStride = stride;
    __local_mem__ float* dst = (__local_mem__ float*)dstTensor.GetPhyAddr();
    __local_mem__ float* cache = (__local_mem__ float*)dstTensor.GetPhyAddr() + cacheId * stride;
    __local_mem__ float* src = (__local_mem__ float*)srcTensor.GetPhyAddr();

    __VEC_SCOPE__
    {
        RegTensor<float> aReg, bReg;
        MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();

        DataCopy(aReg, (__local_mem__ float*)src);
        for (uint16_t j = 0; j < innerLoopTimes; ++j) {
            DataCopy(bReg, dst + j * innerLoopStride);
            AscendC::MicroAPI::Add(aReg, aReg, bReg, pregOne);
        }
        DataCopy((__local_mem__ float*)cache, aReg, pregOne);
    }
}

template <typename T_X>
__aicore__ inline void CalculateCopyInToFP32(LocalTensor<T_X>& xLocal, LocalTensor<float>& xFp32Tmp, uint32_t count)
{
    __local_mem__ T_X* xInUb = (__local_mem__ T_X*)xLocal.GetPhyAddr();
    __local_mem__ float* xFp32TmpBuf = (__local_mem__ float*)xFp32Tmp.GetPhyAddr();

    uint16_t loops = (count + VL_F32 - 1) / VL_F32;
    uint32_t sreg = count;
    __VEC_SCOPE__
    {
        RegTensor<float> xReg;
        MaskReg pregLoop;
        for (uint16_t vi = 0; vi < loops; ++vi) {
            uint32_t offset = vi * VL_F32;
            pregLoop = UpdateMask<float>(sreg);
            LoadTensorForDtypeTIn<T_X>(xInUb, xReg, pregLoop, offset);
            AscendC::MicroAPI::DataCopy<float, StoreDist::DIST_NORM_B32>(xFp32TmpBuf + offset, xReg, pregLoop);
        }
    }
}

__aicore__ inline void CalculateSquareReduceSumLessThanVL(__local_mem__ float* xFp32Tmp, __local_mem__ float* xReduceUb,
                                                          uint32_t reduceNum)
{
    __VEC_SCOPE__
    {
        RegTensor<float> x, vMean;
        uint32_t sreg0 = reduceNum;
        MaskReg pregLoop = UpdateMask<float>(sreg0);
        MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();

        LoadTensorForDtypeTIn<float>(xFp32Tmp, x, pregLoop, 0);
        AscendC::MicroAPI::ReduceSum(vMean, x, pregLoop);
        AscendC::MicroAPI::DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(xReduceUb, vMean, pregOne);
    }
}

__aicore__ inline void CalculateSquareReduceSumLessThanTwoVL(__local_mem__ float* xFp32Tmp,
                                                             __local_mem__ float* xReduceUb, uint32_t reduceNum)
{
    uint32_t tailLen = reduceNum - VL_F32;
    __VEC_SCOPE__
    {
        RegTensor<float> x, xFold, sumReg, vMean;
        MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();
        MaskReg pregTail = UpdateMask<float>(tailLen);

        LoadTensorForDtypeTIn<float>(xFp32Tmp, x, pregFull, 0);
        LoadTensorForDtypeTIn<float>(xFp32Tmp + VL_F32, xFold, pregTail, 0);
        AscendC::MicroAPI::ShiftLefts((RegTensor<uint32_t>&)xFold, (RegTensor<uint32_t>&)xFold, static_cast<int16_t>(0),
                                      pregTail);
        AscendC::MicroAPI::Add(sumReg, x, xFold, pregFull);
        AscendC::MicroAPI::ReduceSum(vMean, sumReg, pregFull);
        AscendC::MicroAPI::DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(xReduceUb, vMean, pregOne);
    }
}

template <int32_t LAST_LOOP_NUMS>
__aicore__ inline void CalculateSquareReduceSumCommon(__local_mem__ float* xFp32Tmp, __local_mem__ float* xReduceUb,
                                                      __local_mem__ float* tmpUb, uint32_t reduceNum,
                                                      uint32_t binAddQuotient)
{
    uint16_t binaryAddQuotientLoop = (binAddQuotient + VL_F32 - 1) / VL_F32;
    uint32_t lastBinaryAddNum = binAddQuotient / VL_F32;

    uint32_t binaryAddRemainder = reduceNum - binAddQuotient;
    uint16_t binaryAddRemainderCeilLoop = (binaryAddRemainder + VL_F32 - 1) / VL_F32;
    uint16_t binaryAddRemainderFloorLoop = binaryAddRemainder / VL_F32;
    __VEC_SCOPE__
    {
        RegTensor<float> x, xFold, sumReg, vMean;
        MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();
        MaskReg pregLoop;

        for (uint16_t r = 0; r < binaryAddRemainderFloorLoop; ++r) {
            uint32_t offset = r * VL_F32;
            LoadTensorForDtypeTIn<float>(xFp32Tmp, x, pregFull, offset);
            LoadTensorForDtypeTIn<float>(xFp32Tmp + binAddQuotient, xFold, pregFull, offset);
            AscendC::MicroAPI::Add(sumReg, x, xFold, pregFull);
            AscendC::MicroAPI::ReduceSum(vMean, sumReg, pregFull);
            AscendC::MicroAPI::DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(tmpUb + static_cast<uint32_t>(r),
                                                                                  vMean, pregOne);
        }
        uint32_t sregRemainder = binaryAddRemainder - binaryAddRemainderFloorLoop * VL_F32;
        for (uint16_t r = 0; r < static_cast<uint16_t>(binaryAddRemainderCeilLoop - binaryAddRemainderFloorLoop); ++r) {
            pregLoop = UpdateMask<float>(sregRemainder);
            LoadTensorForDtypeTIn<float>(xFp32Tmp + binaryAddRemainderFloorLoop * VL_F32, x, pregFull, 0);
            LoadTensorForDtypeTIn<float>(xFp32Tmp + binaryAddRemainderFloorLoop * VL_F32 + binAddQuotient, xFold,
                                         pregLoop, 0);
            AscendC::MicroAPI::ShiftLefts((RegTensor<uint32_t>&)xFold, (RegTensor<uint32_t>&)xFold,
                                          static_cast<int16_t>(0), pregLoop);
            AscendC::MicroAPI::Add(sumReg, x, xFold, pregFull);
            AscendC::MicroAPI::ReduceSum(vMean, sumReg, pregFull);
            AscendC::MicroAPI::DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                tmpUb + static_cast<uint32_t>(binaryAddRemainderFloorLoop), vMean, pregOne);
        }
        for (uint16_t r = 0; r < static_cast<uint16_t>(binaryAddQuotientLoop - binaryAddRemainderCeilLoop); ++r) {
            LoadTensorForDtypeTIn<float>(xFp32Tmp + binaryAddRemainderCeilLoop * VL_F32, x, pregFull, 0);
            AscendC::MicroAPI::ReduceSum(vMean, x, pregFull);
            AscendC::MicroAPI::DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                tmpUb + static_cast<uint32_t>(binaryAddRemainderCeilLoop + r), vMean, pregOne);
        }
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        if constexpr (LAST_LOOP_NUMS == 1) {
            MaskReg pregLast = UpdateMask<float>(lastBinaryAddNum);
            DataCopy(x, tmpUb);
            AscendC::MicroAPI::ReduceSum(vMean, x, pregLast);
            AscendC::MicroAPI::DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(xReduceUb, vMean, pregOne);
        } else if constexpr (LAST_LOOP_NUMS == 2) {
            lastBinaryAddNum -= VL_F32;
            MaskReg pregLast = UpdateMask<float>(lastBinaryAddNum);
            DataCopy(x, tmpUb);
            DataCopy(xFold, tmpUb + VL_F32);
            AscendC::MicroAPI::ShiftLefts((RegTensor<uint32_t>&)xFold, (RegTensor<uint32_t>&)xFold,
                                          static_cast<int16_t>(0), pregLast);
            AscendC::MicroAPI::Add(sumReg, x, xFold, pregFull);
            AscendC::MicroAPI::ReduceSum(vMean, sumReg, pregFull);
            AscendC::MicroAPI::DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(xReduceUb, vMean, pregOne);
        }
    }
}

__aicore__ inline void CalculateSquareReduceSum(LocalTensor<float>& xFp32Local, LocalTensor<float>& xReduceLocal,
                                                LocalTensor<float>& binaryAddBuffTmp, uint32_t reduceNum,
                                                uint32_t binAddQuotient)
{
    __local_mem__ float* xReduceUb = (__local_mem__ float*)xReduceLocal.GetPhyAddr();
    __local_mem__ float* tmpUb = (__local_mem__ float*)binaryAddBuffTmp.GetPhyAddr();
    __local_mem__ float* xFp32Tmp = (__local_mem__ float*)xFp32Local.GetPhyAddr();

    if (reduceNum <= VL_F32) {
        CalculateSquareReduceSumLessThanVL(xFp32Tmp, xReduceUb, reduceNum);
    } else if (reduceNum <= VL_F32 + VL_F32) {
        CalculateSquareReduceSumLessThanTwoVL(xFp32Tmp, xReduceUb, reduceNum);
    } else if (reduceNum <= VL_F32 * VL_F32 * DIGIT_TWO) {
        CalculateSquareReduceSumCommon<DIGIT_ONE>(xFp32Tmp, xReduceUb, tmpUb, reduceNum, binAddQuotient);
    } else {
        CalculateSquareReduceSumCommon<DIGIT_TWO>(xFp32Tmp, xReduceUb, tmpUb, reduceNum, binAddQuotient);
    }
}

__aicore__ inline void CalculateRstd(LocalTensor<float>& xReduceLocal, LocalTensor<float>& rstdLocal, uint32_t curRows,
                                     float avgFactor, float epsilon)
{
    static constexpr float POS_INF = 3.40282366920938E+38;
    static constexpr float SCALAR1 = -0.5;
    static constexpr float SCALAR2 = 1.5;
    static constexpr float SCALAR3 = 0.5;
    static constexpr float SCALAR0 = -99.99;

    __local_mem__ float* rstdInUb = (__local_mem__ float*)rstdLocal.GetPhyAddr();
    __local_mem__ float* xReduceUb = (__local_mem__ float*)xReduceLocal.GetPhyAddr();
    uint16_t loopRows = static_cast<uint16_t>((curRows + VL_F32 - 1) / VL_F32);
    __VEC_SCOPE__
    {
        RegTensor<float> var, rstd, r, y, s, t, one, scalar1, t1, t2, t3, t4, scalarInf, scalarZero;
        MaskReg cmpRegZero, cmpRegInf;
        MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregLoop;

        uint32_t sreg = static_cast<uint32_t>(curRows);
        for (uint16_t i = 0; i < loopRows; ++i) {
            pregLoop = UpdateMask<float>(sreg);
            AscendC::MicroAPI::Duplicate(scalarInf, POS_INF, pregLoop);
            AscendC::MicroAPI::Duplicate(scalarZero, float(0.0), pregLoop);
            AscendC::MicroAPI::Duplicate(one, float(1.0), pregLoop);
            AscendC::MicroAPI::Duplicate(scalar1, SCALAR3, pregLoop);
            AscendC::MicroAPI::Duplicate(t1, SCALAR2, pregLoop);
            AscendC::MicroAPI::Duplicate(s, float(1.0), pregLoop);
            AscendC::MicroAPI::DataCopy(var, xReduceUb + i * VL_F32);
            AscendC::MicroAPI::Muls(var, var, avgFactor, pregLoop);
            AscendC::MicroAPI::Adds(var, var, epsilon, pregLoop);
            AscendC::MicroAPI::Maxs(var, var, SCALAR0, pregLoop);
            AscendC::MicroAPI::Div(r, one, var, pregLoop);
            AscendC::MicroAPI::Sqrt(y, r, pregLoop);
            AscendC::MicroAPI::Muls(t, var, SCALAR1, pregLoop);
            AscendC::MicroAPI::Mul(t, t, y, pregLoop);
            AscendC::MicroAPI::Mula(t1, t, y, pregLoop);
            AscendC::MicroAPI::Mul(rstd, y, t1, pregLoop);
            AscendC::MicroAPI::Muls(t3, var, float(-1.0), pregLoop);
            AscendC::MicroAPI::Mula(s, t3, r, pregLoop);
            AscendC::MicroAPI::Muls(t4, rstd, float(-1.0), pregLoop);
            AscendC::MicroAPI::Mula(r, t4, rstd, pregLoop);
            AscendC::MicroAPI::Mula(s, var, r, pregLoop);
            AscendC::MicroAPI::Mul(s, s, rstd, pregLoop);
            AscendC::MicroAPI::Mula(rstd, s, scalar1, pregLoop);
            AscendC::MicroAPI::CompareScalar(cmpRegZero, var, POS_INF, pregLoop);
            AscendC::MicroAPI::Select(rstd, scalarZero, rstd, cmpRegZero);
            AscendC::MicroAPI::CompareScalar(cmpRegInf, var, float(0.0), pregLoop);
            AscendC::MicroAPI::Select(rstd, scalarInf, rstd, cmpRegInf);
            AscendC::MicroAPI::DataCopy(rstdInUb + i * VL_F32, rstd, pregLoop);
        }
    }
}

} // namespace RmsNormHiF4Quant
#endif
