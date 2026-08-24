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
 * \file batch_norm_ext2_base.h
 * \brief
 */
#ifndef NORM_BATCH_NORM_EXT2_BASE_H
#define NORM_BATCH_NORM_EXT2_BASE_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "../inc/platform.h"
#include "../inc/kernel_utils.h"
#include "../../norm_common/reduce_common_regbase.h"

namespace BatchNormExt2Ops {
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
using AscendC::Reg::LoadAlign;
using AscendC::Reg::Reduce;
using AscendC::Reg::StoreAlign;

constexpr static int64_t DOUBLE_BUFFER = 2;
constexpr static int32_t BUFFER_DEPTH = 1;
static constexpr uint16_t VECTOR_LENGTH = platform::GetVRegSize();
static constexpr uint16_t VL_FP32 = VECTOR_LENGTH / sizeof(float);
static constexpr int64_t BLOCK_SIZE = platform::GetUbBlockSize();
constexpr static uint32_t FLOAT_BYTES = 4;
constexpr static float POS_INF = 3.40282366920938E+38;
constexpr static float zero = 0.0f;
constexpr static int64_t NDDMA_THRESHOLD = 32;
constexpr static int64_t NDDMA_SECOND_DIM = 1;
constexpr static int64_t NDDMA_THIRD_DIM = 2;
constexpr static int64_t NDDMA_DIM_NUM = 3;
constexpr static AscendC::MicroAPI::CastTrait castTraitB162B32 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};

constexpr static AscendC::MicroAPI::CastTrait castTraitB322B16 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

template <typename T>
__aicore__ inline void LoadTensorForDtypeT(AscendC::MicroAPI::RegTensor<float>& dst, __ubuf__ T* src,
                                           AscendC::MicroAPI::MaskReg& preg, uint32_t offset)
{
    if constexpr (IsSameType<T, float>::value) {
        LoadAlign<float, LoadDist::DIST_NORM>(dst, (__ubuf__ float*)src + offset);
    } else { // fp16、bf16
        RegTensor<T> xFp16;
        LoadAlign<T, LoadDist::DIST_UNPACK_B16>(xFp16, ((__ubuf__ T*)src + offset));
        Cast<float, T, castTraitB162B32>(dst, xFp16, preg);
    }
}

template <typename T>
__aicore__ inline void LoadTensorForDtypeTBrc(RegTensor<float>& dst, __ubuf__ T* src, MaskReg& preg, uint32_t offset)
{
    if constexpr (IsSameType<T, float>::value) {
        LoadAlign<float, LoadDist::DIST_BRC_B32>(dst, (__ubuf__ float*)src + offset);
    } else { // fp16、bf16
        RegTensor<T> xFp16;
        LoadAlign<T, LoadDist::DIST_BRC_B16>(xFp16, ((__ubuf__ T*)src + offset));
        Cast<float, T, castTraitB162B32>(dst, xFp16, preg);
    }
}

template <typename T>
__aicore__ inline void LoadTwoTensorForDtypeT(RegTensor<float>& dst1, RegTensor<float>& dst2, __ubuf__ T* src1,
                                              __ubuf__ T* src2, MaskReg& dst1Preg, MaskReg& dst2Preg,
                                              uint32_t src1Offset, uint32_t src2Offset)
{
    if constexpr (IsSameType<T, half>::value) {
        RegTensor<half> xFp16R;
        RegTensor<half> xFp16Q;
        LoadAlign<half, LoadDist::DIST_UNPACK_B16>(xFp16Q, ((__ubuf__ half*)(src1) + (src1Offset)));
        LoadAlign<half, LoadDist::DIST_UNPACK_B16>(xFp16R, ((__ubuf__ half*)(src2) + (src2Offset)));
        Cast<float, half, castTraitB162B32>(dst1, xFp16Q, dst1Preg);
        Cast<float, half, castTraitB162B32>(dst2, xFp16R, dst2Preg);
    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
        RegTensor<bfloat16_t> xFp16R;
        RegTensor<bfloat16_t> xFp16Q;
        LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(xFp16Q, ((__ubuf__ bfloat16_t*)(src1) + (src1Offset)));
        LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(xFp16R, ((__ubuf__ bfloat16_t*)(src2) + (src2Offset)));
        Cast<float, bfloat16_t, castTraitB162B32>(dst1, xFp16Q, dst1Preg);
        Cast<float, bfloat16_t, castTraitB162B32>(dst2, xFp16R, dst2Preg);
    } else {
        LoadAlign(dst1, ((__ubuf__ float*)(src1) + (src1Offset)));
        LoadAlign(dst2, ((__ubuf__ float*)(src2) + (src2Offset)));
    }
}

template <typename T>
__aicore__ inline void LoadTwoTensorForDtypeTBrc(RegTensor<float>& dst1, RegTensor<float>& dst2, __ubuf__ T* src1,
                                                 __ubuf__ T* src2, MaskReg& dst1Preg, MaskReg& dst2Preg,
                                                 uint32_t src1Offset, uint32_t src2Offset)
{
    if constexpr (IsSameType<T, half>::value) {
        RegTensor<half> xFp16Q;
        RegTensor<half> xFp16R;
        LoadAlign<half, LoadDist::DIST_BRC_B16>(xFp16Q, ((__ubuf__ half*)(src1) + (src1Offset)));
        LoadAlign<half, LoadDist::DIST_BRC_B16>(xFp16R, ((__ubuf__ half*)(src2) + (src2Offset)));
        Cast<float, half, castTraitB162B32>(dst1, xFp16Q, dst1Preg);
        Cast<float, half, castTraitB162B32>(dst2, xFp16R, dst2Preg);
    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
        RegTensor<bfloat16_t> xFp16Q;
        RegTensor<bfloat16_t> xFp16R;
        LoadAlign<bfloat16_t, LoadDist::DIST_BRC_B16>(xFp16Q, ((__ubuf__ bfloat16_t*)(src1) + (src1Offset)));
        LoadAlign<bfloat16_t, LoadDist::DIST_BRC_B16>(xFp16R, ((__ubuf__ bfloat16_t*)(src2) + (src2Offset)));
        Cast<float, bfloat16_t, castTraitB162B32>(dst1, xFp16Q, dst1Preg);
        Cast<float, bfloat16_t, castTraitB162B32>(dst2, xFp16R, dst2Preg);
    } else {
        LoadAlign<float, LoadDist::DIST_BRC_B32>(dst1, ((__ubuf__ float*)(src1) + (src1Offset)));
        LoadAlign<float, LoadDist::DIST_BRC_B32>(dst2, ((__ubuf__ float*)(src2) + (src2Offset)));
    }
}

template <typename T>
__aicore__ inline void StoreTensorForDtypeT(__ubuf__ T* dst, AscendC::MicroAPI::RegTensor<float>& src,
                                            AscendC::MicroAPI::MaskReg& preg, uint32_t offset)
{
    if constexpr (IsSameType<T, float>::value) {
        StoreAlign<T, AscendC::MicroAPI::StoreDist::DIST_NORM>(dst + offset, src, preg);
    } else {
        AscendC::MicroAPI::RegTensor<T> xFp16;
        Cast<T, float, castTraitB322B16>(xFp16, src, preg);
        StoreAlign<T, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(dst + offset, xFp16, preg);
    }
}

} // namespace BatchNormExt2Ops

#endif // NORM_BATCH_NORM_EXT2_BASE_H
