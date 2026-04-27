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
 * \file bucketize_v2_common.h
 * \brief
 */
#ifndef BUCKETIZE_V2_COMMON_H
#define BUCKETIZE_V2_COMMON_H

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#if ASC_DEVKIT_MAJOR >=9
#include "basic_api/kernel_vec_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "op_kernel/platform_util.h"

namespace BucketizeV2 {
using namespace AscendC;

constexpr int32_t ONCE_STORE_REG_NUM = 2;

constexpr static MicroAPI::CastTrait CAST_TRAIT_INT8_TO_INT16 = {
  MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
  MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
  
template <typename T>
__aicore__ inline void CopyGmToUb(LocalTensor<T> xLocal,
  GlobalTensor<T> xGm, int64_t offset,
  uint32_t nBurst, uint32_t copyLen)
{
    DataCopyPadExtParams<T> dataCopyPadExtParams;
    dataCopyPadExtParams.isPad = false;
    dataCopyPadExtParams.leftPadding = 0;
    dataCopyPadExtParams.rightPadding = 0;
    dataCopyPadExtParams.paddingValue = 0;

    DataCopyExtParams dataCoptExtParams;
    dataCoptExtParams.blockCount = nBurst;
    dataCoptExtParams.blockLen = copyLen * sizeof(T);
    dataCoptExtParams.srcStride = 0;
    dataCoptExtParams.dstStride = 0;
    DataCopyPad(xLocal, xGm[offset], dataCoptExtParams, dataCopyPadExtParams);
}

template <typename T>
__aicore__ inline void CopyUbToGm(GlobalTensor<T> xGm, LocalTensor<T> xLocal, 
    int64_t offset, uint32_t nBurst, uint32_t copyLen)
{
    DataCopyExtParams dataCoptExtParams;
    dataCoptExtParams.blockCount = nBurst;
    dataCoptExtParams.blockLen = copyLen * sizeof(T);
    dataCoptExtParams.srcStride = 0;
    dataCoptExtParams.dstStride = 0;
    DataCopyPad(xGm[offset], xLocal, dataCoptExtParams);
}

template <typename TARGET_T, typename ORG_T>
__simd_callee__ inline void StoreTensor(__local_mem__ ORG_T *dstAddr, MicroAPI::RegTensor<ORG_T> &orgReg, MicroAPI::MaskReg preg) 
{
    if constexpr (sizeof(ORG_T) == sizeof(int32_t)) {
        if constexpr (sizeof(TARGET_T) == sizeof(int32_t)) {
            MicroAPI::StoreAlign(dstAddr, orgReg, preg);
        } else {
            MicroAPI::RegTensor<ORG_T> zeroReg;
            MicroAPI::Duplicate(zeroReg, static_cast<ORG_T>(0));
            MicroAPI::StoreAlign<ORG_T, MicroAPI::StoreDist::DIST_INTLV_B32>(dstAddr, orgReg, zeroReg, preg);
        }   
    } else {
        if constexpr (sizeof(TARGET_T) == sizeof(int32_t)) {
            MicroAPI::RegTensor<ORG_T> zeroReg;
            MicroAPI::Duplicate(zeroReg, static_cast<ORG_T>(0));
            MicroAPI::StoreAlign<ORG_T, MicroAPI::StoreDist::DIST_INTLV_B16>(dstAddr, orgReg, zeroReg, preg);
        } else {
            constexpr int16_t stride = ONCE_STORE_REG_NUM * AscendC::VECTOR_REG_WIDTH / sizeof(ORG_T);
            MicroAPI::RegTensor<ORG_T> zeroReg;
            MicroAPI::RegTensor<ORG_T> tmpReg;
            MicroAPI::Duplicate(zeroReg, static_cast<ORG_T>(0));
            MicroAPI::Interleave(orgReg, tmpReg, orgReg, zeroReg);
            MicroAPI::StoreAlign<ORG_T, MicroAPI::StoreDist::DIST_INTLV_B16>(dstAddr, orgReg, zeroReg, preg);
            MicroAPI::StoreAlign<ORG_T, MicroAPI::StoreDist::DIST_INTLV_B16>(dstAddr + stride, tmpReg, zeroReg, preg);
        }        
    }
}

template <typename ORG_T, typename REG_T>
__simd_callee__ inline void LoadTensor(REG_T &dstReg, __local_mem__ ORG_T *srcAddr, MicroAPI::MaskReg preg) 
{
  if constexpr (std::is_same_v<ORG_T, uint8_t>) {
      MicroAPI::LoadAlign<ORG_T, MicroAPI::LoadDist::DIST_UNPACK_B8>((MicroAPI::RegTensor<uint8_t>&)dstReg, srcAddr);
  } else if constexpr (std::is_same_v<ORG_T, int8_t>) {
      MicroAPI::LoadAlign<ORG_T, MicroAPI::LoadDist::DIST_UNPACK_B8>((MicroAPI::RegTensor<int8_t>&)dstReg, srcAddr);
      MicroAPI::Cast<int16_t, int8_t, CAST_TRAIT_INT8_TO_INT16>(dstReg, (MicroAPI::RegTensor<int8_t>&)dstReg, preg);
  } else {
      MicroAPI::LoadAlign(dstReg, srcAddr);
  }
}

template <typename ORG_T, typename REG_T, typename INDICES_REG_T>
__simd_callee__ inline void LoadMidValue(REG_T &midValueReg, __local_mem__ ORG_T *srcPtr, INDICES_REG_T midIndicesReg, MicroAPI::MaskReg preg) 
{
    if constexpr (std::is_same_v<ORG_T, uint8_t>)  {
        MicroAPI::DataCopyGather((MicroAPI::RegTensor<int16_t>&)midValueReg, srcPtr, midIndicesReg, preg);
    } else if constexpr (std::is_same_v<ORG_T, int8_t>) {
        MicroAPI::DataCopyGather((MicroAPI::RegTensor<int16_t>&)midValueReg, srcPtr, midIndicesReg, preg);
        MicroAPI::Cast<int16_t, int8_t, CAST_TRAIT_INT8_TO_INT16>(midValueReg, (MicroAPI::RegTensor<int8_t>&)midValueReg, preg);
    } else {
        MicroAPI::DataCopyGather(midValueReg, srcPtr, midIndicesReg, preg);
    }
}

template <typename X_T, typename B_T, typename Y_T, typename INDICES_T, typename SHIFT_T, typename COMPUTE_T, typename RegT, bool RIGHT>
__simd_vf__ inline void BinaryQueryVF(__local_mem__ X_T* xPtr, __local_mem__ B_T* boundPtr, __local_mem__ INDICES_T* yPtr,
  uint32_t dataLen, uint16_t repeatNum, uint16_t onRepeatSize, uint32_t dstRepeatStride, uint16_t maxIter, uint32_t boundLength) {
    RegT xReg;
    MicroAPI::RegTensor<INDICES_T> leftReg;
    MicroAPI::RegTensor<INDICES_T> rightReg;
    MicroAPI::RegTensor<INDICES_T> subReg;
    MicroAPI::RegTensor<INDICES_T> tmpReg;
    MicroAPI::RegTensor<INDICES_T> midIndicesReg;
    MicroAPI::RegTensor<INDICES_T> midOneReg;
    MicroAPI::RegTensor<SHIFT_T> shiftReg;
    
    RegT midValueReg;
    uint32_t count = dataLen;
    MicroAPI::Duplicate(shiftReg, static_cast<SHIFT_T>(1));
    for(uint16_t i = 0; i < repeatNum; i++) {
        MicroAPI::MaskReg maskReg = MicroAPI::UpdateMask<INDICES_T>(count);
        LoadTensor<X_T, RegT>(xReg, xPtr + i * onRepeatSize, maskReg);
        MicroAPI::Duplicate(leftReg, static_cast<INDICES_T>(0));
        MicroAPI::Duplicate(rightReg, static_cast<INDICES_T>(boundLength));
        MicroAPI::MaskReg pReg = maskReg;
        MicroAPI::MaskReg cmpMaskReg;
        for(uint16_t j = 0; j < maxIter; j++)  {
            MicroAPI::Compare<INDICES_T, CMPMODE::NE>(pReg, leftReg, rightReg, pReg);
            MicroAPI::Sub(subReg, rightReg, leftReg, pReg);
            MicroAPI::ShiftRight(tmpReg, subReg, shiftReg, pReg);
            MicroAPI::Add(midIndicesReg, leftReg, tmpReg, pReg);
            LoadMidValue<B_T, RegT, MicroAPI::RegTensor<INDICES_T>>(midValueReg, boundPtr, midIndicesReg, pReg);
            if constexpr (RIGHT) {
                MicroAPI::Compare<COMPUTE_T, CMPMODE::GT>(cmpMaskReg, midValueReg, xReg, pReg);
            } else {
                MicroAPI::Compare<COMPUTE_T, CMPMODE::GE>(cmpMaskReg, midValueReg, xReg, pReg);
            }
            MicroAPI::MaskNot(cmpMaskReg, cmpMaskReg, pReg);
            MicroAPI::Adds(midOneReg, midIndicesReg, 1, pReg);
            MicroAPI::Select(leftReg, midOneReg, leftReg, cmpMaskReg);
            MicroAPI::Select(rightReg, rightReg, midIndicesReg, cmpMaskReg);
        }
        StoreTensor<Y_T, INDICES_T>(yPtr + i * dstRepeatStride, leftReg, maskReg);
    }  
}

template <typename X_T, typename B_T, typename Y_T, bool RIGHT>
__aicore__ inline void BinaryQuery(LocalTensor<X_T> &xLocal, LocalTensor<B_T> &boundLocal, LocalTensor<Y_T> &yLocal, uint32_t dataLen, uint16_t maxIter, uint32_t boundLength)
{
    using INDICES_T = typename std::conditional<sizeof(B_T) == 1 || sizeof(B_T) == sizeof(int16_t), uint16_t, uint32_t>::type;
    using SHIFT_T = typename std::conditional<sizeof(INDICES_T) == sizeof(int16_t), int16_t, int32_t>::type;
    using COMPUTE_T = typename std::conditional<std::is_same<B_T, uint8_t>::value, uint16_t,
                                                typename std::conditional<std::is_same<B_T, int8_t>::value, int16_t, B_T>::type>::type;
    using RegT = typename std::conditional<
        sizeof(COMPUTE_T) == sizeof(int64_t), MicroAPI::RegTensor<COMPUTE_T, MicroAPI::RegTraitNumTwo>, MicroAPI::RegTensor<COMPUTE_T>>::type;
    __local_mem__ X_T* xPtr = (__local_mem__ X_T*)xLocal.GetPhyAddr();
    __local_mem__ B_T* boundPtr = (__local_mem__ B_T*)boundLocal.GetPhyAddr();
    __local_mem__ INDICES_T* yPtr = (__local_mem__ INDICES_T*)yLocal.GetPhyAddr();
    constexpr uint16_t onRepeatSize = GetVecLen() / sizeof(INDICES_T);
    uint16_t repeatNum = CeilDivision(dataLen, onRepeatSize);
    constexpr uint16_t dstRepeatStride = sizeof(Y_T) / sizeof(INDICES_T) * onRepeatSize;
    asc_vf_call<BinaryQueryVF<X_T, B_T, Y_T, INDICES_T, SHIFT_T, COMPUTE_T, RegT, RIGHT>>(xPtr, boundPtr, yPtr, 
        dataLen, repeatNum, onRepeatSize, dstRepeatStride, maxIter, boundLength);
}

}  // namespace BucketizeV2
#endif  // BUCKETIZE_V2_COMMON_H
