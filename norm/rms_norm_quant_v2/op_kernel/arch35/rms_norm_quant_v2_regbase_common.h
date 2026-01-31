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
 * \file rms_norm_quant_v2_regbase_common.h
 * \brief
 */
#ifndef RMS_NORM_QUANT_V2_REGBASE_COMMON_H_
#define RMS_NORM_QUANT_V2_REGBASE_COMMON_H_

#include "kernel_operator.h"
#include "../inc/platform.h"
#include "../../norm_common/reduce_common_regbase.h"
#include "../../rms_norm/arch35/rms_norm_regbase_common.h"

namespace RmsNormQuantV2 {
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
using namespace NormCommon;
using namespace NormCommon::NormCommonRegbase;

constexpr int64_t ALIGN_512_FACTOR = 512;
constexpr int64_t ALIGN_32_FACTOR = 32;
constexpr int64_t BLOCK_SIZE = platform::GetUbBlockSize();
constexpr static uint32_t VL_FP32 = platform::GetVRegSize() / sizeof(float);
constexpr static uint32_t BLK_B32 = BLOCK_SIZE / sizeof(float);

constexpr int64_t B32_BLOCK_NUM = 8;
constexpr int64_t B8_BLOCK_NUM = 32;
constexpr int32_t CONST_FACTOR_2 = 2;
constexpr uint32_t SUM_COUNT = 2;

constexpr int32_t NUM_ZERO = 0;
constexpr int32_t NUM_ONE = 1;
constexpr int32_t NUM_TWO = 2;
constexpr int32_t NUM_THREE = 3;
constexpr int32_t NUM_FOUR = 4;

constexpr int32_t VL_SIZE = GetVRegSize();
constexpr int32_t V_LENGTH = (VL_SIZE / sizeof(float));


static constexpr uint32_t DOUBLE_BUFFER_NUM = 2;
constexpr int64_t AR_RECOMPUTE_SUM_BUFFER_BTYES = 32;
constexpr int64_t AR_RECOMPUTE_SUM_LEN = AR_RECOMPUTE_SUM_BUFFER_BTYES / sizeof(float);

constexpr AscendC::MicroAPI::CastTrait castTraitF162F32 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};

constexpr AscendC::MicroAPI::CastTrait castTraitF322F16 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

constexpr AscendC::MicroAPI::CastTrait castTraitI82F16 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

constexpr AscendC::MicroAPI::CastTrait castTraitI322F16 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

constexpr AscendC::MicroAPI::CastTrait castTraitFp322Int32 = {
    AscendC::MicroAPI::RegLayout::UNKNOWN,
    AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};
constexpr AscendC::MicroAPI::CastTrait castTraitInt322Fp32 = {
    AscendC::MicroAPI::RegLayout::UNKNOWN,
    AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};
constexpr AscendC::MicroAPI::CastTrait castTraitFp322Fp16 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_ODD,
};
constexpr AscendC::MicroAPI::CastTrait castTraitFp162Int8 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};
constexpr AscendC::MicroAPI::CastTrait castTraitFp322Fp8 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    RoundMode::CAST_RINT,
};
constexpr AscendC::MicroAPI::CastTrait castTraitFp322Hifp8 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    RoundMode::CAST_ROUND,
};

__aicore__ inline int64_t GetCacheId(const int64_t idx)
{
    return AscendC::ScalarGetCountOfValue<1>(idx ^ (idx + NUM_ONE)) - NUM_ONE;
}

template <typename T_IN>
__aicore__ inline void LoadScalarForDtypeTIn(
    __local_mem__ T_IN* src, RegTensor<float>& dst, MaskReg& preg, uint32_t offset)
{
    if constexpr (IsSameType<T_IN, float>::value) {
        DataCopy<float, LoadDist::DIST_BRC_B32>(dst, src + offset);
    } else if constexpr (IsSameType<T_IN, int8_t>::value) {
        RegTensor<T_IN> xIn;
        RegTensor<half> xInTmp;
        DataCopy<T_IN, LoadDist::DIST_BRC_B8>(xIn, src + offset);
        Cast<half, T_IN, castTraitI82F16>(xInTmp, xIn, preg);
        Cast<float, half, castTraitF162F32>(dst, xInTmp, preg);
    } else if constexpr (IsSameType<T_IN, int32_t>::value) {
        RegTensor<T_IN> xIn;
        DataCopy<int32_t, LoadDist::DIST_BRC_B32>(xIn, src + offset);
        Cast<float, T_IN, castTraitInt322Fp32>(dst, xIn, preg);
    } else {
        RegTensor<T_IN> xIn;
        DataCopy<T_IN, LoadDist::DIST_BRC_B16>(xIn, src + offset);
        Cast<float, T_IN, castTraitF162F32>(dst, xIn, preg);
    }
}

template <typename T_IN>
__aicore__ inline void LoadTensorForDtypeTIn(
    __local_mem__ T_IN* src, RegTensor<float>& dst, MaskReg& preg, uint32_t offset)
{
    if constexpr (IsSameType<T_IN, float>::value) {
        DataCopy<float, LoadDist::DIST_NORM>(dst, src + offset);
    } else if constexpr (IsSameType<T_IN, int8_t>::value) {
        RegTensor<T_IN> xIn;
        RegTensor<half> xInTmp;
        DataCopy<T_IN, LoadDist::DIST_UNPACK4_B8>(xIn, src + offset);
        Cast<half, T_IN, castTraitI82F16>(xInTmp, xIn, preg);
        Cast<float, half, castTraitF162F32>(dst, xInTmp, preg);
    } else if constexpr (IsSameType<T_IN, int32_t>::value) {
        RegTensor<T_IN> xIn;
        DataCopy<int32_t, LoadDist::DIST_NORM>(xIn, src + offset);
        Cast<float, T_IN, castTraitInt322Fp32>(dst, xIn, preg);
    } else {
        RegTensor<T_IN> xIn;
        DataCopy<T_IN, LoadDist::DIST_UNPACK_B16>(xIn, src + offset);
        Cast<float, T_IN, castTraitF162F32>(dst, xIn, preg);
    }
}

template <typename T_OUT>
__aicore__ inline void StoreTensorForDtypeTOut(
    __local_mem__ T_OUT* dst, RegTensor<float>& xRegFp32, MaskReg& preg, MaskReg& mask4Int4, uint32_t offset)
{
    if constexpr (IsSameType<T_OUT, int8_t>::value) {
        RegTensor<T_OUT> xOut;
        RegTensor<half> xRegFp16;
        Cast<half, float, castTraitFp322Fp16>(xRegFp16, xRegFp32, preg);
        Cast<int8_t, half, castTraitFp162Int8>(xOut, xRegFp16, preg);
        DataCopy<T_OUT, StoreDist::DIST_PACK4_B32>(dst + offset, xOut, preg);
    } else if constexpr (IsSameType<T_OUT, uint8_t>::value) {
        RegTensor<half> xRegFp16;
        RegTensor<uint16_t> xUint16;
        RegTensor<uint8_t> xUint8;
        Cast<half, float, castTraitFp322Fp16>(xRegFp16, xRegFp32, preg);
        Pack(xUint16, (RegTensor<uint32_t>&)xRegFp16);
        Cast<int4x2_t, half, castTraitFp162Int8>((RegTensor<int4x2_t>&)xUint8, (RegTensor<half>&)xUint16, preg);
        DataCopy<uint8_t, StoreDist::DIST_PACK4_B32>(dst + offset / 2, xUint8, mask4Int4);
    } else if constexpr (IsSameType<T_OUT, fp8_e4m3fn_t>::value || IsSameType<T_OUT, fp8_e5m2_t>::value) {
        RegTensor<T_OUT> xOut;
        Cast<T_OUT, float, castTraitFp322Fp8>(xOut, xRegFp32, preg);
        DataCopy<T_OUT, StoreDist::DIST_PACK4_B32>(dst + offset, xOut, preg);
    } else if constexpr (IsSameType<T_OUT, hifloat8_t>::value) {
        RegTensor<T_OUT> xOut;
        Cast<T_OUT, float, castTraitFp322Hifp8>(xOut, xRegFp32, preg);
        DataCopy<T_OUT, StoreDist::DIST_PACK4_B32>(dst + offset, xOut, preg);
    }
}
} // namespace RmsNormQuantV2
#endif // RMS_NORM_QUANT_V2_REGBASE_COMMON_H_
