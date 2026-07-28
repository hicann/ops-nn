/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * NOTE: Portions of this code were AI-generated and have been technically reviewed for functional accuracy.
 */

/*!
 * \file l2_normalize_grad_regbase_common.h
 * \brief L2NormalizeGrad arch35 (Ascend950 / regbase) common idioms.
 *
 * Shared constants + MicroAPI helpers for the three DX templates.
 * Compute is done in fp32 (x/y/dy cast to fp32 on load, dx cast back on store),
 * matching the A2 reference (l2_normalize_grad.py) fp32 accumulation.
 */
#ifndef L2_NORMALIZE_GRAD_REGBASE_COMMON_H
#define L2_NORMALIZE_GRAD_REGBASE_COMMON_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "l2_normalize_grad_tiling_data.h"

namespace L2NormalizeGrad {
using namespace AscendC;
using namespace AscendC::MicroAPI;
using AscendC::MicroAPI::CreateMask;
using AscendC::MicroAPI::LoadDist;
using AscendC::MicroAPI::MaskPattern;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::RegTensor;
using AscendC::MicroAPI::StoreDist;
using AscendC::MicroAPI::UpdateMask;

namespace L2NormalizeGradRegbase {
__aicore__ inline constexpr uint32_t GetVRegSize()
{
#if __CCE_AICORE__ == 310
    return AscendC::VECTOR_REG_WIDTH;
#else
    return 256U;
#endif
}
} // namespace L2NormalizeGradRegbase

constexpr uint32_t V_LENGTH = L2NormalizeGradRegbase::GetVRegSize() / sizeof(float); // fp32 lanes per VL (=64)
constexpr uint32_t FLOAT_NUM_BLOCK = 8;                                              // fp32 elements per 32B block
constexpr uint32_t HALF_NUM_BLOCK = 16;                                              // fp16 elements per 32B block
constexpr uint32_t FLOAT_NUM_2VL = 128; // 2 * V_LENGTH; ReduceSum AR row alignment
constexpr uint32_t DB_NUM = 2;
constexpr uint32_t DEPTH_TWO = 2;
// Full-row load is chosen when D (reduced-axis length) fits this many fp32 elements.
constexpr uint32_t UB_FACTOR_DX_FULL_LOAD = 6144;
// Split-D processes the reduced axis in chunks of this many fp32 elements (2VL aligned).
constexpr uint32_t UB_FACTOR_DX_SPLIT_D = 4096;

// b16 (fp16) -> fp32 widening cast, zeroing masked-off lanes.
constexpr AscendC::MicroAPI::CastTrait castTraitB162B32 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};

// fp32 -> b16 (fp16) narrowing cast, round-to-nearest, zeroing masked-off lanes.
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

// Load a VL of T from UB into a fp32 RegTensor, casting fp16 -> fp32 (fp32 passes through).
template <typename T>
__aicore__ inline void LoadAndCast(RegTensor<float>& dstReg, __local_mem__ T* srcAddr, MaskReg& maskReg,
                                   uint32_t srcOffset)
{
    if constexpr (IsSameType<T, float>::value) {
        DataCopy(dstReg, srcAddr + srcOffset);
    } else {
        RegTensor<T> dstRegB16;
        DataCopy<T, LoadDist::DIST_UNPACK_B16>(dstRegB16, srcAddr + srcOffset);
        Cast<float, T, castTraitB162B32>(dstReg, dstRegB16, maskReg);
    }
}

// Store a fp32 dx RegTensor to UB as T (fp32 passes through, fp16 casts + packs).
template <typename T>
__aicore__ inline void StoreDx(__local_mem__ T* dstAddr, uint32_t dstOffset, RegTensor<float>& dxReg, MaskReg& maskReg)
{
    if constexpr (IsSameType<T, float>::value) {
        DataCopy(dstAddr + dstOffset, dxReg, maskReg);
    } else {
        RegTensor<T> dxRegB16;
        Cast<T, float, castTraitB322B16>(dxRegB16, dxReg, maskReg);
        DataCopy<T, StoreDist::DIST_PACK_B32>(dstAddr + dstOffset, dxRegB16, maskReg);
    }
}

} // namespace L2NormalizeGrad
#endif // L2_NORMALIZE_GRAD_REGBASE_COMMON_H
