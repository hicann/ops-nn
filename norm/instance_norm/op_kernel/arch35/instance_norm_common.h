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
 * \file instance_norm_common.h
 * \brief
 */
#ifndef INSTANCE_NORM_AR_COMMON_H_
#define INSTANCE_NORM_AR_COMMON_H_

#include "instance_norm_tiling_data.h"
#include "kernel_operator.h"
#include "../inc/platform.h"

namespace InstanceNormOps {
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

constexpr uint32_t VL_FP32 = platform::GetVRegSize() / sizeof(float);
constexpr uint32_t VL_F32 = VECTOR_REG_WIDTH / sizeof(float);
constexpr uint32_t BLOCK_SIZE = platform::GetUbBlockSize();
constexpr uint32_t BLK_B32 = BLOCK_SIZE / sizeof(float);
constexpr uint32_t DOUBLE_BUFFER_NUM = 2;

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

template <typename T_IN>
__aicore__ inline void LoadTensorForDtypeTIn(
    __local_mem__ T_IN* src, RegTensor<float>& dst, MaskReg& preg, uint32_t offset)
{
    if constexpr (IsSameType<T_IN, float>::value) {
        DataCopy<float, LoadDist::DIST_NORM>(dst, src + offset);
    } else {
        RegTensor<T_IN> xIn;
        DataCopy<T_IN, LoadDist::DIST_UNPACK_B16>(xIn, src + offset);
        Cast<float, T_IN, castTraitB162B32>(dst, xIn, preg);
    }
}

template <typename T_IN>
__aicore__ inline void LoadScalarForDtypeTIn(
    __local_mem__ T_IN* src, RegTensor<float>& dst, MaskReg& preg, uint32_t offset)
{
    if constexpr (IsSameType<T_IN, float>::value) {
        DataCopy<float, LoadDist::DIST_BRC_B32>(dst, src + offset);
    } else {
        RegTensor<T_IN> xIn;
        DataCopy<T_IN, LoadDist::DIST_BRC_B16>(xIn, src + offset);
        Cast<float, T_IN, castTraitB162B32>(dst, xIn, preg);
    }
}

template <typename T_OUT>
__aicore__ inline void StoreTensorForDtypeTOut(
    __local_mem__ T_OUT* dst, RegTensor<float>& src, MaskReg& preg, uint32_t offset)
{
    if constexpr (IsSameType<T_OUT, float>::value) {
        DataCopy<T_OUT, StoreDist::DIST_NORM>(dst + offset, src, preg);
    } else {
        RegTensor<T_OUT> xOut;
        Cast<T_OUT, float, castTraitB322B16>(xOut, src, preg);
        DataCopy<T_OUT, StoreDist::DIST_PACK_B32>(dst + offset, xOut, preg);
    }
}

template <typename T_OUT>
__aicore__ inline void StoreOneElementForDtypeTOut(
    __local_mem__ T_OUT* dst, RegTensor<float>& src, MaskReg& preg, uint32_t offset)
{
    if constexpr (IsSameType<T_OUT, float>::value) {
        DataCopy<T_OUT, StoreDist::DIST_FIRST_ELEMENT_B32>(dst + offset, src, preg);
    } else {
        RegTensor<T_OUT> xOut;
        Cast<T_OUT, float, castTraitB322B16>(xOut, src, preg);
        DataCopy<T_OUT, StoreDist::DIST_FIRST_ELEMENT_B16>(dst + offset, xOut, preg);
    }
}
} // namespace InstanceNormOps
#endif // INSTANCE_NORM_AR_COMMON_H_
