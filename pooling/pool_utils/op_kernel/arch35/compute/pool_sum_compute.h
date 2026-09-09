/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file pool_sum_compute.h
 * \brief AvgPool NHWC 与 Pool3D NDHWC big kernel 共用的非 fp32 输入按 fp32 部分和累加接口。
 */

#ifndef POOL_UTILS_ARCH35_COMPUTE_POOL_SUM_COMPUTE_H_
#define POOL_UTILS_ARCH35_COMPUTE_POOL_SUM_COMPUTE_H_

#include <cstdint>

#include "kernel_operator.h"
#include "op_kernel/platform_util.h"
#include "op_kernel/math_util.h"

namespace PoolUtils {
namespace Compute {

constexpr AscendC::Reg::CastTrait POOL_SUM_CAST_T2FP32 = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN,
                                                          AscendC::Reg::MaskMergeMode::ZEROING,
                                                          AscendC::RoundMode::UNKNOWN};

/*
 * 功能：将本轮搬入的非 fp32 输入提升为 fp32 后逐元素累加到 fp32 部分和 buffer。
 */
template <typename T>
__aicore__ inline void AccumulateSumFp32(const AscendC::LocalTensor<T>& xLocal,
                                         const AscendC::LocalTensor<float>& sumLocal, int64_t dataCount)
{
    __ubuf__ T* xLocalAddr = (__ubuf__ T*)xLocal.GetPhyAddr();
    __ubuf__ float* sumLocalAddr = (__ubuf__ float*)sumLocal.GetPhyAddr();
    constexpr uint32_t repeatElm = Ops::Base::GetVRegSize() / sizeof(float);
    uint16_t repeatTimes = static_cast<uint16_t>(Ops::Base::CeilDiv(dataCount, static_cast<int64_t>(repeatElm)));
    uint32_t len = dataCount;
    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<T> in;
        AscendC::Reg::RegTensor<float> inFp32;
        AscendC::Reg::RegTensor<float> sum;
        AscendC::Reg::MaskReg mask;
        uint32_t num = len;
        for (uint16_t i = 0; i < repeatTimes; i++) {
            mask = AscendC::Reg::UpdateMask<float>(num);
            auto sumReg = AscendC::Reg::CreateAddrReg<float>(i, static_cast<uint16_t>(repeatElm));
            auto srcReg = AscendC::Reg::CreateAddrReg<T>(i, static_cast<uint16_t>(repeatElm));
            AscendC::Reg::LoadAlign(in, xLocalAddr, srcReg);
            AscendC::Reg::LoadAlign(sum, sumLocalAddr, sumReg);
            AscendC::Reg::UnPack((AscendC::Reg::RegTensor<uint32_t>&)in, (AscendC::Reg::RegTensor<uint16_t>&)in);
            AscendC::Reg::Cast<float, T, POOL_SUM_CAST_T2FP32>(inFp32, in, mask);
            AscendC::Reg::Add(sum, inFp32, sum, mask);
            AscendC::Reg::StoreAlign(sumLocalAddr, sum, sumReg, mask);
        }
    }
}

} // namespace Compute
} // namespace PoolUtils

#endif // POOL_UTILS_ARCH35_COMPUTE_POOL_SUM_COMPUTE_H_
