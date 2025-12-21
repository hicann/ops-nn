/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file group_norm_grad_common.h
 * \brief
 */
#ifndef GROUP_NORM_GRAD_COMMON_H
#define GROUP_NORM_GRAD_COMMON_H
#pragma once

#include "kernel_operator.h"
#include "../../norm_common/reduce_common_regbase.h"

namespace GroupNormGrad {
using namespace AscendC;
using AscendC::MicroAPI::CreateMask;
using AscendC::MicroAPI::DataCopyUnAlignPost;
using AscendC::MicroAPI::DataCopyUnAlignPre;
using AscendC::MicroAPI::LoadDist;
using AscendC::MicroAPI::LocalMemBar;
using AscendC::MicroAPI::MaskPattern;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::MemType;
using AscendC::MicroAPI::RegTensor;
using AscendC::MicroAPI::StoreDist;
using AscendC::MicroAPI::UnalignReg;
using AscendC::MicroAPI::UpdateMask;
using namespace NormCommon;
using namespace NormCommon::NormCommonRegbase;

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
__aicore__ inline void LoadTwoTensorForDtypeT(
    __local_mem__ T* src1, __local_mem__ T* src2, RegTensor<float>& dst1, RegTensor<float>& dst2, MaskReg& dst1Preg,
    MaskReg& dst2Preg, uint32_t src1Offset, uint32_t src2Offset)
{
    if constexpr (IsSameType<T, half>::value) {
        RegTensor<half> xFp16Q;
        RegTensor<half> xFp16R;
        DataCopy<half, LoadDist::DIST_UNPACK_B16>(xFp16Q, ((__local_mem__ half*)(src1) + (src1Offset)));
        DataCopy<half, LoadDist::DIST_UNPACK_B16>(xFp16R, ((__local_mem__ half*)(src2) + (src2Offset)));
        Cast<float, half, castTraitB162B32>(dst1, xFp16Q, dst1Preg);
        Cast<float, half, castTraitB162B32>(dst2, xFp16R, dst2Preg);
    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
        RegTensor<bfloat16_t> xFp16Q;
        RegTensor<bfloat16_t> xFp16R;
        DataCopy<bfloat16_t, LoadDist::DIST_UNPACK_B16>(xFp16Q, ((__local_mem__ bfloat16_t*)(src1) + (src1Offset)));
        DataCopy<bfloat16_t, LoadDist::DIST_UNPACK_B16>(xFp16R, ((__local_mem__ bfloat16_t*)(src2) + (src2Offset)));
        Cast<float, bfloat16_t, castTraitB162B32>(dst1, xFp16Q, dst1Preg);
        Cast<float, bfloat16_t, castTraitB162B32>(dst2, xFp16R, dst2Preg);
    } else {
        DataCopy(dst1, ((__local_mem__ float*)(src1) + (src1Offset)));
        DataCopy(dst2, ((__local_mem__ float*)(src2) + (src2Offset)));
    }
}

template <typename T>
__aicore__ inline void LoadOneTensorForDtypeT(
    __local_mem__ T* input, RegTensor<float>& dst, MaskReg& preg, uint32_t offset)
{
    if constexpr (IsSameType<T, half>::value) {
        RegTensor<half> xFp16;
        DataCopy<half, LoadDist::DIST_UNPACK_B16>(xFp16, ((__local_mem__ half*)(input) + (offset)));
        Cast<float, half, castTraitB162B32>(dst, xFp16, preg);
    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
        RegTensor<bfloat16_t> xBf16;
        DataCopy<bfloat16_t, LoadDist::DIST_UNPACK_B16>(xBf16, ((__local_mem__ bfloat16_t*)(input) + (offset)));
        Cast<float, bfloat16_t, castTraitB162B32>(dst, xBf16, preg);
    } else {
        DataCopy(dst, ((__local_mem__ float*)(input) + (offset)));
    }
}

template <typename T>
__aicore__ inline void LoadUnAlignOneTensor(
    __local_mem__ T*& input, RegTensor<float>& dst, UnalignReg& uSrc, MaskReg& preg, uint32_t postUpdateStride)
{
    if constexpr (IsSameType<T, half>::value) {
        RegTensor<half> xFp16;
        RegTensor<half> xFp16UnPack;
        DataCopyUnAlign(xFp16, uSrc, input, postUpdateStride);
        UnPack((RegTensor<uint32_t>&)xFp16UnPack, (RegTensor<uint16_t>&)xFp16);
        Cast<float, half, castTraitB162B32>(dst, xFp16UnPack, preg);
    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
        RegTensor<bfloat16_t> xBf16;
        RegTensor<bfloat16_t> xBf16UnPack;
        DataCopyUnAlign(xBf16, uSrc, input, postUpdateStride);
        UnPack((RegTensor<uint32_t>&)xBf16UnPack, (RegTensor<uint16_t>&)xBf16);
        Cast<float, bfloat16_t, castTraitB162B32>(dst, xBf16UnPack, preg);
    } else {
        DataCopyUnAlign(dst, uSrc, input, postUpdateStride);
    }
}

template <typename T>
__aicore__ inline void StoreOneTensorForDtypeT(
    __local_mem__ T* output, RegTensor<float>& src, MaskReg& preg, uint32_t offset)
{
    if constexpr (IsSameType<T, half>::value) {
        RegTensor<half> xFp16;
        Cast<half, float, castTraitB322B16>(xFp16, src, preg);
        DataCopy<half, StoreDist::DIST_PACK_B32>(((__local_mem__ half*)(output) + offset), xFp16, preg);
    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
        RegTensor<bfloat16_t> xBf16;
        Cast<bfloat16_t, float, castTraitB322B16>(xBf16, src, preg);
        DataCopy<bfloat16_t, StoreDist::DIST_PACK_B32>(output + offset, xBf16, preg);
    } else {
        DataCopy(output + offset, src, preg);
    }
}

template <typename T>
__aicore__ inline void StoreUnAlignOneTensor(
    __local_mem__ T*& output, RegTensor<float>& src, UnalignReg& uValue, MaskReg& preg, uint32_t postUpdateStride)
{
    if constexpr (IsSameType<T, half>::value) {
        RegTensor<half> xFp16;
        RegTensor<half> xFp16Pack;
        Cast<half, float, castTraitB322B16>(xFp16, src, preg);
        Pack((RegTensor<uint16_t>&)xFp16Pack, (RegTensor<uint32_t>&)xFp16);
        DataCopyUnAlign(output, xFp16Pack, uValue, postUpdateStride);
    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
        RegTensor<bfloat16_t> xBf16;
        RegTensor<bfloat16_t> xBf16Pack;
        Cast<bfloat16_t, float, castTraitB322B16>(xBf16, src, preg);
        Pack((RegTensor<uint16_t>&)xBf16Pack, (RegTensor<uint32_t>&)xBf16);
        DataCopyUnAlign(output, xBf16Pack, uValue, postUpdateStride);
    } else {
        DataCopyUnAlign(output, src, uValue, postUpdateStride);
    }
}

template <typename T>
__aicore__ inline void VFCastFloat2T(
    const __ubuf__ T* ubAddrOut, const __ubuf__ float* ubAddrIn, const uint32_t length, const uint32_t vecLen)
{
    uint16_t loopCnt = CeilDiv(length, vecLen);
    __VEC_SCOPE__
    {
        __ubuf__ float* srcAddr = (__ubuf__ float*)ubAddrIn;
        __ubuf__ T* dstAddr = (__ubuf__ T*)ubAddrOut;
        uint32_t sregMask = (uint32_t)length;
        MaskReg preg;
        uint32_t sregvl = (uint32_t)vecLen;

        for (uint16_t i = 0; i < loopCnt; ++i) {
            preg = UpdateMask<float>(sregMask);
            if constexpr (IsSameType<T, half>::value) {
                RegTensor<half> vregB16;
                RegTensor<float> vregF32;
                DataCopy(vregF32, srcAddr + i * sregvl);
                Cast<half, float, castTraitB322B16>(vregB16, vregF32, preg);
                DataCopy<half, StoreDist::DIST_PACK_B32>(dstAddr + i * sregvl, vregB16, preg);
            } else if constexpr (IsSameType<T, bfloat16_t>::value) {
                RegTensor<bfloat16_t> vregBF16;
                RegTensor<float> vregF32;
                DataCopy(vregF32, srcAddr + i * sregvl);
                Cast<bfloat16_t, float, castTraitB322B16>(vregBF16, vregF32, preg);
                DataCopy<bfloat16_t, StoreDist::DIST_PACK_B32>(dstAddr + i * sregvl, vregBF16, preg);
            }
        }
    }
}

template <typename T>
__aicore__ inline void VFCastT2Float(
    const __ubuf__ float* ubAddrOut, const __ubuf__ T* ubAddrIn, const uint32_t length, const uint32_t vecLen)
{
    __ubuf__ T* srcAddr = (__ubuf__ T*)ubAddrIn;
    __ubuf__ float* dstAddr = (__ubuf__ float*)ubAddrOut;
    uint16_t loopCnt = CeilDiv(length, vecLen);
    __VEC_SCOPE__
    {
        uint32_t sregMask = (uint32_t)length;
        MaskReg preg;
        uint32_t sregvl = (uint32_t)vecLen;

        for (uint16_t i = 0; i < loopCnt; ++i) {
            preg = UpdateMask<float>(sregMask);
            if constexpr (IsSameType<T, half>::value) {
                RegTensor<half> vregB16;
                RegTensor<float> vregF32;
                DataCopy<half, LoadDist::DIST_UNPACK_B16>(vregB16, srcAddr + i * sregvl);
                Cast<float, half, castTraitB162B32>(vregF32, vregB16, preg);
                DataCopy(dstAddr + i * sregvl, vregF32, preg);
            } else if constexpr (IsSameType<T, bfloat16_t>::value) {
                RegTensor<bfloat16_t> vregBF16;
                RegTensor<float> vregF32;
                DataCopy<bfloat16_t, LoadDist::DIST_UNPACK_B16>(vregBF16, srcAddr + i * sregvl);
                Cast<float, bfloat16_t, castTraitB162B32>(vregF32, vregBF16, preg);
                DataCopy(dstAddr + i * sregvl, vregF32, preg);
            }
        }
    }
}
} // namespace GroupNormGrad
#endif