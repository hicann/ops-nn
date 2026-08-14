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
using AscendC::MicroAPI::LoadDist;
using AscendC::MicroAPI::LoadUnAlignPre;
using AscendC::MicroAPI::LocalMemBar;
using AscendC::MicroAPI::MaskPattern;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::MemType;
using AscendC::MicroAPI::RegTensor;
using AscendC::MicroAPI::StoreDist;
using AscendC::MicroAPI::StoreUnAlignPost;
using AscendC::MicroAPI::UnalignRegForLoad;
using AscendC::MicroAPI::UnalignRegForStore;
using AscendC::MicroAPI::UpdateMask;
using namespace NormCommon;
using namespace NormCommon::NormCommonRegbase;

constexpr int DOUBLE_BUFFER = 2;
constexpr int TRIPLE_BUFFER = 3;

__aicore__ inline constexpr uint32_t GetVRegSize()
{
#if __CCE_AICORE__ == 310
    return AscendC::VECTOR_REG_WIDTH;
#else
    return 256U;
#endif
}

__aicore__ inline constexpr uint32_t GetUbBlockSize() { return 32U; }

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
__aicore__ inline void LoadTwoTensorForDtypeT(__ubuf__ T* src1, __ubuf__ T* src2, RegTensor<float>& dst1,
                                              RegTensor<float>& dst2, MaskReg& dst1Preg, MaskReg& dst2Preg,
                                              uint32_t src1Offset, uint32_t src2Offset)
{
    if constexpr (IsSameType<T, half>::value) {
        RegTensor<half> xFp16Q;
        RegTensor<half> xFp16R;
        LoadAlign<half, LoadDist::DIST_UNPACK_B16>(xFp16Q, ((__ubuf__ half*)(src1) + (src1Offset)));
        LoadAlign<half, LoadDist::DIST_UNPACK_B16>(xFp16R, ((__ubuf__ half*)(src2) + (src2Offset)));
        Cast<float, half, castTraitB162B32>(dst1, xFp16Q, dst1Preg);
        Cast<float, half, castTraitB162B32>(dst2, xFp16R, dst2Preg);
    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
        RegTensor<bfloat16_t> xFp16Q;
        RegTensor<bfloat16_t> xFp16R;
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
__aicore__ inline void LoadOneTensorForDtypeT(__ubuf__ T* input, RegTensor<float>& dst, MaskReg& preg, uint32_t offset)
{
    if constexpr (IsSameType<T, half>::value) {
        RegTensor<half> xFp16;
        LoadAlign<half, LoadDist::DIST_UNPACK_B16>(xFp16, ((__ubuf__ half*)(input) + (offset)));
        Cast<float, half, castTraitB162B32>(dst, xFp16, preg);
    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
        RegTensor<bfloat16_t> xBf16;
        LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(xBf16, ((__ubuf__ bfloat16_t*)(input) + (offset)));
        Cast<float, bfloat16_t, castTraitB162B32>(dst, xBf16, preg);
    } else {
        LoadAlign(dst, ((__ubuf__ float*)(input) + (offset)));
    }
}

template <typename T>
__aicore__ inline void LoadUnAlignOneTensor(__ubuf__ T*& input, RegTensor<float>& dst, UnalignRegForLoad& uSrc,
                                            MaskReg& preg, uint32_t postUpdateStride)
{
    if constexpr (IsSameType<T, half>::value) {
        RegTensor<half> xFp16;
        RegTensor<half> xFp16UnPack;
        LoadUnAlign(xFp16, uSrc, input, postUpdateStride);
        UnPack((RegTensor<uint32_t>&)xFp16UnPack, (RegTensor<uint16_t>&)xFp16);
        Cast<float, half, castTraitB162B32>(dst, xFp16UnPack, preg);
    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
        RegTensor<bfloat16_t> xBf16;
        RegTensor<bfloat16_t> xBf16UnPack;
        LoadUnAlign(xBf16, uSrc, input, postUpdateStride);
        UnPack((RegTensor<uint32_t>&)xBf16UnPack, (RegTensor<uint16_t>&)xBf16);
        Cast<float, bfloat16_t, castTraitB162B32>(dst, xBf16UnPack, preg);
    } else {
        LoadUnAlign(dst, uSrc, input, postUpdateStride);
    }
}

template <typename T>
__aicore__ inline void StoreOneTensorForDtypeT(__ubuf__ T* output, RegTensor<float>& src, MaskReg& preg,
                                               uint32_t offset)
{
    if constexpr (IsSameType<T, half>::value) {
        RegTensor<half> xFp16;
        Cast<half, float, castTraitB322B16>(xFp16, src, preg);
        StoreAlign<half, StoreDist::DIST_PACK_B32>(((__ubuf__ half*)(output) + offset), xFp16, preg);
    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
        RegTensor<bfloat16_t> xBf16;
        Cast<bfloat16_t, float, castTraitB322B16>(xBf16, src, preg);
        StoreAlign<bfloat16_t, StoreDist::DIST_PACK_B32>(output + offset, xBf16, preg);
    } else {
        StoreAlign(output + offset, src, preg);
    }
}

template <typename T>
__aicore__ inline void StoreUnAlignOneTensor(__ubuf__ T*& output, RegTensor<float>& src, UnalignRegForStore& uValue,
                                             MaskReg& preg, uint32_t postUpdateStride)
{
    if constexpr (IsSameType<T, half>::value) {
        RegTensor<half> xFp16;
        RegTensor<half> xFp16Pack;
        Cast<half, float, castTraitB322B16>(xFp16, src, preg);
        Pack((RegTensor<uint16_t>&)xFp16Pack, (RegTensor<uint32_t>&)xFp16);
        StoreUnAlign(output, xFp16Pack, uValue, postUpdateStride);
    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
        RegTensor<bfloat16_t> xBf16;
        RegTensor<bfloat16_t> xBf16Pack;
        Cast<bfloat16_t, float, castTraitB322B16>(xBf16, src, preg);
        Pack((RegTensor<uint16_t>&)xBf16Pack, (RegTensor<uint32_t>&)xBf16);
        StoreUnAlign(output, xBf16Pack, uValue, postUpdateStride);
    } else {
        StoreUnAlign(output, src, uValue, postUpdateStride);
    }
}

template <typename T>
__aicore__ inline void VFCastFloat2T(const __ubuf__ T* ubAddrOut, const __ubuf__ float* ubAddrIn, const uint32_t length,
                                     const uint32_t vecLen)
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
                LoadAlign(vregF32, srcAddr + i * sregvl);
                Cast<half, float, castTraitB322B16>(vregB16, vregF32, preg);
                StoreAlign<half, StoreDist::DIST_PACK_B32>(dstAddr + i * sregvl, vregB16, preg);
            } else if constexpr (IsSameType<T, bfloat16_t>::value) {
                RegTensor<bfloat16_t> vregBF16;
                RegTensor<float> vregF32;
                LoadAlign(vregF32, srcAddr + i * sregvl);
                Cast<bfloat16_t, float, castTraitB322B16>(vregBF16, vregF32, preg);
                StoreAlign<bfloat16_t, StoreDist::DIST_PACK_B32>(dstAddr + i * sregvl, vregBF16, preg);
            }
        }
    }
}

template <typename T>
__aicore__ inline void VFCastT2Float(const __ubuf__ float* ubAddrOut, const __ubuf__ T* ubAddrIn, const uint32_t length,
                                     const uint32_t vecLen)
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
                LoadAlign<half, LoadDist::DIST_UNPACK_B16>(vregB16, srcAddr + i * sregvl);
                Cast<float, half, castTraitB162B32>(vregF32, vregB16, preg);
                StoreAlign(dstAddr + i * sregvl, vregF32, preg);
            } else if constexpr (IsSameType<T, bfloat16_t>::value) {
                RegTensor<bfloat16_t> vregBF16;
                RegTensor<float> vregF32;
                LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(vregBF16, srcAddr + i * sregvl);
                Cast<float, bfloat16_t, castTraitB162B32>(vregF32, vregBF16, preg);
                StoreAlign(dstAddr + i * sregvl, vregF32, preg);
            }
        }
    }
}

/*
  dbeta = reduceSum(dy)
  dgamma = reduceSum(dy * x)
*/
template <typename T>
__aicore__ inline void VFComputeDbetaDs(const LocalTensor<T>& x, const LocalTensor<T>& dy,
                                        const LocalTensor<float>& dbeta, const LocalTensor<float>& dgamma,
                                        uint32_t eleNumPerC, uint32_t vecLen, uint32_t storeBaseOffset,
                                        uint16_t loopCount)
{
    __ubuf__ T* ubX = (__ubuf__ T*)x.GetPhyAddr();
    __ubuf__ T* ubDy = (__ubuf__ T*)dy.GetPhyAddr();
    __ubuf__ float* ubDbeta = (__ubuf__ float*)dbeta.GetPhyAddr();
    __ubuf__ float* ubDgamma = (__ubuf__ float*)dgamma.GetPhyAddr();
    uint16_t repeatTimes = CeilDiv(eleNumPerC, vecLen);
    __ubuf__ T* curUbX;
    __ubuf__ T* curUbDy;

    __VEC_SCOPE__
    {
        UnalignRegForLoad uSrcX;
        UnalignRegForLoad uSrcDy;
        RegTensor<float> vregDbeta;
        RegTensor<float> vregDgamma;
        RegTensor<float> tempDbeta;
        RegTensor<float> vregX;
        RegTensor<float> vregDy;
        for (uint16_t idx = 0; idx < loopCount; idx++) {
            MaskReg preg;
            uint32_t sreg = (uint32_t)eleNumPerC;
            uint32_t sregvl = (uint32_t)vecLen;
            uint32_t ubOffSet = idx * eleNumPerC;
            MaskReg pregAll = CreateMask<float, MaskPattern::ALL>();
            curUbX = ubX + ubOffSet;
            curUbDy = ubDy + ubOffSet;
            Duplicate(vregDbeta, 0, pregAll);
            Duplicate(vregDgamma, 0, pregAll);
            LoadUnAlignPre(uSrcX, curUbX);
            LoadUnAlignPre(uSrcDy, curUbDy);
            for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
                preg = UpdateMask<float>(sreg);
                LoadUnAlignOneTensor<T>(curUbX, vregX, uSrcX, preg, sregvl);
                LoadUnAlignOneTensor<T>(curUbDy, vregDy, uSrcDy, preg, sregvl);
                MulDstAdd(vregX, vregDy, vregDgamma, preg);
                Add(tempDbeta, vregDbeta, vregDy, preg);
                Move<float, AscendC::MicroAPI::MaskMergeMode::MERGING>(vregDbeta, tempDbeta, preg);
                Move<float, AscendC::MicroAPI::MaskMergeMode::MERGING>(vregDgamma, vregX, preg);
            }
            MaskReg pregMerge = CreateMask<float, MaskPattern::VL1>();
            Reduce<ReduceType::SUM>(vregDbeta, vregDbeta, pregAll);
            StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(ubDbeta + storeBaseOffset + idx, vregDbeta, pregMerge);
            Reduce<ReduceType::SUM>(vregDgamma, vregDgamma, pregAll);
            StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(ubDgamma + storeBaseOffset + idx, vregDgamma,
                                                                 pregMerge);
        }
    }
}

template <typename U>
__aicore__ inline void UpdateCacheStage2Mode2(const LocalTensor<U>& dstTensor, const LocalTensor<U>& srcTensor,
                                              const int64_t cacheId, const int64_t stride, const int64_t count)
{
    uint16_t outerLoopTimes = CeilDiv(static_cast<int64_t>(count * sizeof(U)), static_cast<int64_t>(GetVRegSize()));
    uint16_t innerLoopTimes = cacheId;
    uint32_t outerLoopStride = GetVRegSize() / sizeof(U);
    uint32_t innerLoopStride = stride;
    __ubuf__ U* dst = (__ubuf__ U*)dstTensor.GetPhyAddr();
    __ubuf__ U* cache = (__ubuf__ U*)dstTensor.GetPhyAddr() + cacheId * stride;
    __ubuf__ U* src = (__ubuf__ U*)srcTensor.GetPhyAddr();
    __VEC_SCOPE__
    {
        uint32_t sreg = static_cast<uint32_t>(count);
        RegTensor<U> aReg, bReg;
        MaskReg pMask;
        for (uint16_t i = 0; i < outerLoopTimes; ++i) {
            pMask = UpdateMask<U>(sreg);
            LoadAlign(aReg, (__ubuf__ U*)src + i * outerLoopStride);
            for (uint16_t j = 0; j < innerLoopTimes; ++j) {
                LoadAlign(bReg, (__ubuf__ U*)dst + i * outerLoopStride + j * innerLoopStride);
                Add<U, AscendC::MicroAPI::MaskMergeMode::ZEROING>(aReg, aReg, bReg, pMask);
            }
            StoreAlign((__ubuf__ U*)cache + i * outerLoopStride, aReg, pMask);
        }
    }
}

__aicore__ inline int64_t GetCacheId(const int64_t idx) { return ScalarGetCountOfValue<1>(idx ^ (idx + 1)) - 1; }
} // namespace GroupNormGrad
#endif
