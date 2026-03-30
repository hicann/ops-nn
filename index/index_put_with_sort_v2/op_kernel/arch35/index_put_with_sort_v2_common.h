/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file index_put_with_sort_v2_common.h
 * \brief index_put_with_sort_v2
 */
#ifndef INDEX_PUT_WITH_SORT_V2_COMMON_H
#define INDEX_PUT_WITH_SORT_V2_COMMON_H

#include "kernel_operator.h"
#include "../inc/platform.h"
#include "kernel_tiling/kernel_tiling.h"

using namespace AscendC;

constexpr static AscendC::MicroAPI::CastTrait castTrait16ToFloat = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};

constexpr static AscendC::MicroAPI::CastTrait castTraitFloatTo16 = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};

template <typename PARAM_T>
__aicore__ inline void CopyIn(const LocalTensor<PARAM_T>& dstTensor, const GlobalTensor<PARAM_T>& srcTensor, int64_t dataLen)
{
    DataCopyExtParams copyParams = {
        static_cast<uint16_t>(1), static_cast<uint32_t>(dataLen * sizeof(PARAM_T)), static_cast<uint32_t>(0),
        static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    DataCopyPadExtParams<PARAM_T> padParams = {
        false, static_cast<uint8_t>(0), static_cast<uint8_t>(0), static_cast<PARAM_T>(0)};
    DataCopyPad(dstTensor, srcTensor, copyParams, padParams);
}

template <typename PARAM_T>
__aicore__ inline void CopyOut(const GlobalTensor<PARAM_T>& dstTensor, const LocalTensor<PARAM_T>& srcTensor, int64_t dataLen)
{
    DataCopyExtParams copyParams = {
        static_cast<uint16_t>(1), static_cast<uint32_t>(dataLen * sizeof(PARAM_T)), static_cast<uint32_t>(0),
        static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    DataCopyPad(dstTensor, srcTensor, copyParams);
}

template<typename SELF_TYPE, typename CAST_TYPE>
__aicore__ inline void CastSumValue (LocalTensor<SELF_TYPE>& valueSumLocal, LocalTensor<CAST_TYPE>& castLocal, 
    int64_t colLen, uint32_t vfLen, uint16_t loopCnt) {
    __local_mem__ SELF_TYPE* valueSumAddr = (__ubuf__ SELF_TYPE*)valueSumLocal.GetPhyAddr();
    __local_mem__ CAST_TYPE* castSumAddr = (__ubuf__ CAST_TYPE*)castLocal.GetPhyAddr();
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<SELF_TYPE> valueSumReg;
        AscendC::MicroAPI::RegTensor<CAST_TYPE> valueCastReg;
        AscendC::MicroAPI::MaskReg valueMaskReg;
        uint32_t maskLen = static_cast<uint32_t>(colLen);
        for (uint16_t i = 0; i < loopCnt; i++) {
            valueMaskReg = AscendC::MicroAPI::UpdateMask<CAST_TYPE>(maskLen);
            AscendC::MicroAPI::AddrReg castSumAddrOfst = AscendC::MicroAPI::CreateAddrReg<CAST_TYPE>(i, vfLen);
            AscendC::MicroAPI::AddrReg valueSumAddrOfst = AscendC::MicroAPI::CreateAddrReg<SELF_TYPE>(i, vfLen);
            AscendC::MicroAPI::DataCopy(valueCastReg, castSumAddr, castSumAddrOfst);
            AscendC::MicroAPI::Cast<SELF_TYPE, CAST_TYPE, castTraitFloatTo16>(valueSumReg, valueCastReg, valueMaskReg);
            AscendC::MicroAPI::DataCopy<SELF_TYPE, MicroAPI::StoreDist::DIST_PACK_B32>(valueSumAddr, valueSumReg, valueSumAddrOfst, valueMaskReg);
        }
    }
} 

template <AscendC::HardEvent VAR_T>
__aicore__ inline void EventMsg() {
    event_t event = static_cast<event_t>(GetTPipePtr()->FetchEventID(VAR_T));
    SetFlag<VAR_T>(event);
    WaitFlag<VAR_T>(event);
}

#endif