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
 * \file foreach_add_list_regbase.h
 * \brief Register-only compute policy for ForeachAddList on Ascend950.
 *        Computes y = x1 + alpha * x2 for each tensor in the list.
 *        FP16/BF16 promote to FP32 registers (mirrors the SIMT cast paths);
 *        FP32 uses native Axpy; INT32 uses Duplicate+Mul+Add.
 */

#ifndef FOREACH_ADD_LIST_REGBASE_H
#define FOREACH_ADD_LIST_REGBASE_H

#include "../../foreach_utils/arch35/foreach_flat_flow_regbase.h"

namespace ForeachAddList {
using namespace AscendC;

constexpr static Reg::CastTrait ADDL_CAST_STORAGE_TO_FLOAT = {
    Reg::RegLayout::ZERO,
    Reg::SatMode::UNKNOWN,
    Reg::MaskMergeMode::ZEROING,
    RoundMode::UNKNOWN,
};

constexpr static Reg::CastTrait ADDL_CAST_FLOAT_TO_STORAGE = {
    Reg::RegLayout::ZERO,
    Reg::SatMode::NO_SAT,
    Reg::MaskMergeMode::ZEROING,
    RoundMode::CAST_RINT,
};

constexpr static Reg::CastTrait ADDL_CAST_STORAGE_TO_INT = {
    Reg::RegLayout::ZERO,
    Reg::SatMode::UNKNOWN,
    Reg::MaskMergeMode::ZEROING,
    RoundMode::UNKNOWN,
};

constexpr static Reg::CastTrait ADDL_CAST_INT_TO_STORAGE = {
    Reg::RegLayout::ZERO,
    Reg::SatMode::NO_SAT,
    Reg::MaskMergeMode::ZEROING,
    RoundMode::CAST_NONE,
};

template <typename ScalarT>
__simd_callee__ inline float AddListScalarToFloat(ScalarT value)
{
    if constexpr (IsSameType<ScalarT, bfloat16_t>::value) {
        uint16_t raw = *reinterpret_cast<uint16_t*>(&value);
        uint32_t bits = static_cast<uint32_t>(raw) << 16;
        return *reinterpret_cast<float*>(&bits);
    } else {
        return static_cast<float>(value);
    }
}

// x1 = x1 + alpha * x2; the scalar must already be converted to CalcT by the caller
template <typename CalcT>
__simd_callee__ inline void AddListApply(Reg::RegTensor<CalcT, Reg::RegTraitNumOne>& x1Reg,
                                         Reg::RegTensor<CalcT, Reg::RegTraitNumOne>& x2Reg, CalcT scalar,
                                         Reg::MaskReg& mask)
{
    if constexpr (IsSameType<CalcT, int32_t>::value) {
        Reg::RegTensor<int32_t, Reg::RegTraitNumOne> scalarReg;
        Reg::Duplicate(scalarReg, scalar, mask);
        Reg::Mul(x2Reg, x2Reg, scalarReg, mask);
        Reg::Add(x1Reg, x1Reg, x2Reg, mask);
    } else {
        Reg::Axpy(x1Reg, x2Reg, scalar, mask);
    }
}

template <typename T>
__simd_vf__ inline void AddListPromoteVF(__ubuf__ T* x1Addr, __ubuf__ T* x2Addr, __ubuf__ T* outputAddr,
                                         float scalarValue, uint32_t dataCount, uint16_t repeatTimes)
{
    constexpr uint32_t VL = VECTOR_REG_WIDTH / sizeof(float);
    uint32_t remain = static_cast<uint32_t>(dataCount);
    Reg::MaskReg mask;
    Reg::RegTensor<T, Reg::RegTraitNumOne> x1Storage;
    Reg::RegTensor<T, Reg::RegTraitNumOne> x2Storage;
    Reg::RegTensor<T, Reg::RegTraitNumOne> outputStorage;
    Reg::RegTensor<float, Reg::RegTraitNumOne> x1Float;
    Reg::RegTensor<float, Reg::RegTraitNumOne> x2Float;
    for (uint16_t repeatIdx = 0; repeatIdx < repeatTimes; ++repeatIdx) {
        mask = Reg::UpdateMask<float, Reg::RegTraitNumOne>(remain);
        Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x1Storage, x1Addr + repeatIdx * VL);
        Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x2Storage, x2Addr + repeatIdx * VL);
        Reg::Cast<float, T, ADDL_CAST_STORAGE_TO_FLOAT>(x1Float, x1Storage, mask);
        Reg::Cast<float, T, ADDL_CAST_STORAGE_TO_FLOAT>(x2Float, x2Storage, mask);
        AddListApply<float>(x1Float, x2Float, scalarValue, mask);
        Reg::Cast<T, float, ADDL_CAST_FLOAT_TO_STORAGE>(outputStorage, x1Float, mask);
        Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(outputAddr + repeatIdx * VL, outputStorage, mask);
    }
}

template <typename T>
__simd_vf__ inline void AddListNativeVF(__ubuf__ T* x1Addr, __ubuf__ T* x2Addr, __ubuf__ T* outputAddr, T scalarValue,
                                        uint32_t dataCount, uint16_t repeatTimes)
{
    constexpr uint32_t VL = VECTOR_REG_WIDTH / sizeof(T);
    uint32_t remain = static_cast<uint32_t>(dataCount);
    Reg::MaskReg mask;
    Reg::RegTensor<T, Reg::RegTraitNumOne> x1Reg;
    Reg::RegTensor<T, Reg::RegTraitNumOne> x2Reg;
    for (uint16_t repeatIdx = 0; repeatIdx < repeatTimes; ++repeatIdx) {
        mask = Reg::UpdateMask<T, Reg::RegTraitNumOne>(remain);
        Reg::LoadAlign<T, Reg::LoadDist::DIST_NORM>(x1Reg, x1Addr + repeatIdx * VL);
        Reg::LoadAlign<T, Reg::LoadDist::DIST_NORM>(x2Reg, x2Addr + repeatIdx * VL);
        AddListApply<T>(x1Reg, x2Reg, scalarValue, mask);
        Reg::StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(outputAddr + repeatIdx * VL, x1Reg, mask);
    }
}

// int16: compute in int32, truncate to low 16 bits (two's-complement wrap). dav_3510's int->int
// vcvt saturates on overflow, so mask with 0xFFFF before the narrowing cast.
template <typename T>
__simd_vf__ inline void AddListIntPromoteVF(__ubuf__ T* x1Addr, __ubuf__ T* x2Addr, __ubuf__ T* outputAddr,
                                            int32_t scalarValue, uint32_t dataCount, uint16_t repeatTimes)
{
    constexpr uint32_t VL = VECTOR_REG_WIDTH / sizeof(int32_t);
    uint32_t remain = static_cast<uint32_t>(dataCount);
    Reg::MaskReg mask;
    Reg::RegTensor<int32_t, Reg::RegTraitNumOne> x1Int;
    Reg::RegTensor<int32_t, Reg::RegTraitNumOne> x2Int;
    Reg::RegTensor<int32_t, Reg::RegTraitNumOne> maskLowReg;
    for (uint16_t repeatIdx = 0; repeatIdx < repeatTimes; ++repeatIdx) {
        mask = Reg::UpdateMask<int32_t, Reg::RegTraitNumOne>(remain);
        Reg::RegTensor<T, Reg::RegTraitNumOne> x1Storage;
        Reg::RegTensor<T, Reg::RegTraitNumOne> x2Storage;
        Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x1Storage, x1Addr + repeatIdx * VL);
        Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x2Storage, x2Addr + repeatIdx * VL);
        Reg::Cast<int32_t, T, ADDL_CAST_STORAGE_TO_INT>(x1Int, x1Storage, mask);
        Reg::Cast<int32_t, T, ADDL_CAST_STORAGE_TO_INT>(x2Int, x2Storage, mask);
        AddListApply<int32_t>(x1Int, x2Int, scalarValue, mask);
        Reg::Duplicate(maskLowReg, 0xFFFF, mask);
        Reg::And(x1Int, x1Int, maskLowReg, mask);
        Reg::RegTensor<uint16_t, Reg::RegTraitNumOne> outputStorage;
        Reg::Cast<uint16_t, int32_t, ADDL_CAST_INT_TO_STORAGE>(outputStorage, x1Int, mask);
        Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(outputAddr + repeatIdx * VL,
                                                          (Reg::RegTensor<T, Reg::RegTraitNumOne>&)outputStorage, mask);
    }
}

// int8/uint8 promote to int32 lanes and compute x1 + alpha * x2 in int32 (per requirement).
// Loads use DIST_UNPACK4_B8 (b8 -> low byte of each b32 lane, per selectwithbytesmask_3510_impl.h);
// sign/zero extension is done with shifts/AND instead of casts (dav_3510 has no proven b8->b32 vcvt
// for this layout): int8: (v << 24) >> 24 arithmetic; uint8: v & 0xFF.
// Result is truncated with & 0xFF (low byte = two's-complement wrap for both int8 and uint8)
// and stored via DIST_PACK4_B32 (b32 lanes -> b8 memory, per quantize_impl.h).
template <typename T>
__simd_vf__ inline void AddListInt8PromoteVF(__ubuf__ T* x1Addr, __ubuf__ T* x2Addr, __ubuf__ T* outputAddr,
                                             int32_t scalarValue, uint32_t dataCount, uint16_t repeatTimes)
{
    constexpr uint32_t VL = VECTOR_REG_WIDTH / sizeof(int32_t);
    uint32_t remain = static_cast<uint32_t>(dataCount);
    Reg::MaskReg mask;
    Reg::RegTensor<T, Reg::RegTraitNumOne> x1Storage;
    Reg::RegTensor<T, Reg::RegTraitNumOne> x2Storage;
    Reg::RegTensor<int32_t, Reg::RegTraitNumOne> x1Int;
    Reg::RegTensor<int32_t, Reg::RegTraitNumOne> x2Int;
    Reg::RegTensor<int32_t, Reg::RegTraitNumOne> scalarReg;
    Reg::RegTensor<int32_t, Reg::RegTraitNumOne> maskLowReg;
    Reg::RegTensor<uint8_t, Reg::RegTraitNumOne> outputStorage;
    for (uint16_t repeatIdx = 0; repeatIdx < repeatTimes; ++repeatIdx) {
        mask = Reg::UpdateMask<int32_t, Reg::RegTraitNumOne>(remain);
        Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK4_B8>(x1Storage, x1Addr + repeatIdx * VL);
        Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK4_B8>(x2Storage, x2Addr + repeatIdx * VL);
        if constexpr (IsSameType<T, int8_t>::value) {
            Reg::ShiftLefts(x1Int, (Reg::RegTensor<int32_t, Reg::RegTraitNumOne>&)x1Storage, static_cast<int16_t>(24),
                            mask);
            Reg::ShiftRights(x1Int, x1Int, static_cast<int16_t>(24), mask);
            Reg::ShiftLefts(x2Int, (Reg::RegTensor<int32_t, Reg::RegTraitNumOne>&)x2Storage, static_cast<int16_t>(24),
                            mask);
            Reg::ShiftRights(x2Int, x2Int, static_cast<int16_t>(24), mask);
        } else {
            Reg::Duplicate(maskLowReg, 0xFF, mask);
            Reg::And(x1Int, (Reg::RegTensor<int32_t, Reg::RegTraitNumOne>&)x1Storage, maskLowReg, mask);
            Reg::And(x2Int, (Reg::RegTensor<int32_t, Reg::RegTraitNumOne>&)x2Storage, maskLowReg, mask);
        }
        Reg::Duplicate(scalarReg, scalarValue, mask);
        Reg::Mul(x2Int, x2Int, scalarReg, mask);
        Reg::Add(x1Int, x1Int, x2Int, mask);
        Reg::Duplicate(maskLowReg, 0xFF, mask);
        Reg::And(x1Int, x1Int, maskLowReg, mask);
        Reg::Cast<uint8_t, int32_t, ADDL_CAST_INT_TO_STORAGE>(outputStorage, x1Int, mask);
        Reg::StoreAlign<uint8_t, Reg::StoreDist::DIST_PACK4_B32>(
            reinterpret_cast<__ubuf__ uint8_t*>(outputAddr + repeatIdx * VL), outputStorage, mask);
    }
}

template <typename T, typename ScalarT>
struct AddListPolicy {
    static constexpr bool kUsesExtraUb = false;

    __aicore__ static inline void Run(LocalTensor<T>* inputs, LocalTensor<T> output, ScalarT scalar, uint32_t dataCount)
    {
        __ubuf__ T* x1Addr = reinterpret_cast<__ubuf__ T*>(inputs[0].GetPhyAddr());
        __ubuf__ T* x2Addr = reinterpret_cast<__ubuf__ T*>(inputs[1].GetPhyAddr());
        __ubuf__ T* outputAddr = reinterpret_cast<__ubuf__ T*>(output.GetPhyAddr());
        if constexpr (IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value) {
            constexpr uint32_t VL = VECTOR_REG_WIDTH / sizeof(float);
            uint16_t repeatTimes = static_cast<uint16_t>(CeilDivision(dataCount, VL));
            float scalarFloat = AddListScalarToFloat(scalar);
            asc_vf_call<AddListPromoteVF<T>>(x1Addr, x2Addr, outputAddr, scalarFloat, dataCount, repeatTimes);
        } else if constexpr (IsSameType<T, int16_t>::value) {
            constexpr uint32_t VL = VECTOR_REG_WIDTH / sizeof(int32_t);
            uint16_t repeatTimes = static_cast<uint16_t>(CeilDivision(dataCount, VL));
            asc_vf_call<AddListIntPromoteVF<T>>(x1Addr, x2Addr, outputAddr, scalar, dataCount, repeatTimes);
        } else if constexpr (IsSameType<T, int8_t>::value || IsSameType<T, uint8_t>::value) {
            constexpr uint32_t VL = VECTOR_REG_WIDTH / sizeof(int32_t);
            uint16_t repeatTimes = static_cast<uint16_t>(CeilDivision(dataCount, VL));
            asc_vf_call<AddListInt8PromoteVF<T>>(x1Addr, x2Addr, outputAddr, scalar, dataCount, repeatTimes);
        } else {
            constexpr uint32_t VL = VECTOR_REG_WIDTH / sizeof(T);
            uint16_t repeatTimes = static_cast<uint16_t>(CeilDivision(dataCount, VL));
            asc_vf_call<AddListNativeVF<T>>(x1Addr, x2Addr, outputAddr, scalar, dataCount, repeatTimes);
        }
    }
};

template <typename T, typename ScalarT, typename Tiling>
using ForeachAddListRegbase = ForeachFlatRegbase::FlatFlowKernel<T, ScalarT, Tiling, 2, AddListPolicy<T, ScalarT>>;
} // namespace ForeachAddList

#endif // FOREACH_ADD_LIST_REGBASE_H
