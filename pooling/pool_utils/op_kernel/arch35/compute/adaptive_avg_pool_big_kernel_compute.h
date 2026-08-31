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
 * \file adaptive_avg_pool_big_kernel_compute.h
 * \brief AdaptiveAvgPool2D/AdaptiveAvgPool3D big kernel 共用的单点存取与窗口求和接口。
 */

#ifndef POOL_UTILS_ARCH35_COMPUTE_ADAPTIVE_AVG_POOL_BIG_KERNEL_COMPUTE_H_
#define POOL_UTILS_ARCH35_COMPUTE_ADAPTIVE_AVG_POOL_BIG_KERNEL_COMPUTE_H_

#include <cstdint>

#include "op_kernel/platform_util.h"
#include "op_kernel/math_util.h"
#include "kernel_operator.h"

namespace PoolUtils {
namespace Compute {

constexpr int32_t ADAPTIVE_POOL_B32_SIZE = 4;

constexpr AscendC::Reg::CastTrait ADAPTIVE_POOL_CAST_B4TOB2 = {
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT, AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};
constexpr AscendC::Reg::CastTrait ADAPTIVE_POOL_CAST_B2TOB4 = {
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN};

/*
 * 功能：将寄存器中的单个 float 结果按输出类型 T 写回 UB 的指定偏移。
 */
template <typename T, typename U>
__aicore__ inline void StoreOneValue(const __ubuf__ void* dstAddr, AscendC::Reg::RegTensor<U>& srcReg,
                                     AscendC::Reg::MaskReg& maskReg, uint32_t offset)
{
    auto addr = (__ubuf__ T*)dstAddr + offset;
    if constexpr (AscendC::IsSameType<T, half>::value) {
        AscendC::Reg::RegTensor<half> regfp16;
        AscendC::Reg::Cast<half, float, ADAPTIVE_POOL_CAST_B4TOB2>(regfp16, srcReg, maskReg);
        AscendC::Reg::StoreAlign<half, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B16>(addr, regfp16, maskReg);
    } else if constexpr (AscendC::IsSameType<T, bfloat16_t>::value) {
        AscendC::Reg::RegTensor<bfloat16_t> regBf16;
        AscendC::Reg::Cast<bfloat16_t, float, ADAPTIVE_POOL_CAST_B4TOB2>(regBf16, srcReg, maskReg);
        AscendC::Reg::StoreAlign<bfloat16_t, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B16>(addr, regBf16, maskReg);
    } else if constexpr (sizeof(T) == ADAPTIVE_POOL_B32_SIZE) {
        AscendC::Reg::StoreAlign<T, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(
            addr, (AscendC::Reg::RegTensor<T>&)srcReg, maskReg);
    } else {
        AscendC::Reg::UnalignRegForStore uReg;
        AscendC::Reg::StoreUnAlign(addr, srcReg, uReg, 1);
        AscendC::Reg::StoreUnAlignPost(addr, uReg, 0);
    }
}

/*
 * 功能：从 UB 的指定偏移读取单个元素并广播到寄存器。
 */
template <typename U>
__aicore__ inline void LoadOneValue(const __ubuf__ void* srcAddr, AscendC::Reg::RegTensor<U>& dstReg,
                                    AscendC::Reg::MaskReg& preg, uint32_t offset)
{
    auto addr = (__ubuf__ U*)srcAddr + offset;
    if constexpr (sizeof(U) == ADAPTIVE_POOL_B32_SIZE) {
        AscendC::Reg::LoadAlign<U, AscendC::Reg::LoadDist::DIST_BRC_B32>(dstReg, addr);
    } else {
        AscendC::Reg::UnalignRegForLoad ureg;
        AscendC::Reg::LoadUnAlignPre(ureg, addr);
        AscendC::Reg::LoadUnAlign(dstReg, ureg, addr, 1);
    }
}

/*
 * 功能：按输入类型 T 加载一段输入并统一提升到 float 寄存器。
 */
template <typename T, typename U>
__aicore__ inline void LoadXLocalToReg(const __ubuf__ void* srcAddr, AscendC::Reg::RegTensor<U>& dstReg,
                                       AscendC::Reg::MaskReg& preg, AscendC::Reg::AddrReg& offset)
{
    if constexpr (AscendC::IsSameType<T, half>::value) {
        AscendC::Reg::RegTensor<half> regfp16;
        AscendC::Reg::LoadAlign<half, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(regfp16, (__ubuf__ half*)srcAddr,
                                                                               offset);
        AscendC::Reg::Cast<float, half, ADAPTIVE_POOL_CAST_B2TOB4>(dstReg, regfp16, preg);
    } else if constexpr (AscendC::IsSameType<T, bfloat16_t>::value) {
        AscendC::Reg::RegTensor<bfloat16_t> regBf16;
        AscendC::Reg::LoadAlign<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(
            regBf16, (__ubuf__ bfloat16_t*)srcAddr, offset);
        AscendC::Reg::Cast<float, bfloat16_t, ADAPTIVE_POOL_CAST_B2TOB4>(dstReg, regBf16, preg);
    } else {
        AscendC::Reg::LoadAlign(dstReg, (__ubuf__ float*)srcAddr, offset);
    }
}

/*
 * 功能：切分场景下把上一轮暂存在 UB 中的部分和累加到当前寄存器结果上。
 */
template <typename U>
__aicore__ inline void UpdateSum(AscendC::Reg::RegTensor<U>& res, const __ubuf__ U* storeLocalAddr, int32_t offset)
{
    // get data from local mem
    AscendC::Reg::MaskReg pregOne = AscendC::Reg::CreateMask<U, AscendC::Reg::MaskPattern::VL1>();
    AscendC::Reg::RegTensor<U> lastRes;

    // get last res from local mem
    LoadOneValue<U>(storeLocalAddr, lastRes, pregOne, offset);

    // calc sum
    AscendC::Reg::Add(res, res, lastRes, pregOne);
    AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_LOAD, AscendC::Reg::MemType::VEC_STORE>();
}

/*
 * 功能：对本轮搬入的输入求和并写入暂存 UB。
 * 说明：NEED_ACCUMULATE 为 true 表示当前处于窗口切分场景，需要先累加上一轮的部分和。
 */
template <typename T, typename U, bool NEED_ACCUMULATE>
__aicore__ inline void ComputeSum(const AscendC::LocalTensor<T>& xLocal, const AscendC::LocalTensor<U>& storeAddLocal,
                                  int64_t dataCount)
{
    __ubuf__ T* xLocalAddr = (__ubuf__ T*)xLocal.GetPhyAddr();
    __ubuf__ U* storeLocalAddr = (__ubuf__ U*)storeAddLocal.GetPhyAddr();

    uint32_t repeatCount = Ops::Base::GetVRegSize() / sizeof(U); // 一个vf需要的次数
    uint16_t repeatTimes = Ops::Base::CeilDiv(static_cast<uint32_t>(dataCount),
                                              repeatCount); // 上取整，获取repeatCount的整数倍
    uint32_t dataCount_ = dataCount;

    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<U> vd0;
        AscendC::Reg::RegTensor<U> vd1;
        AscendC::Reg::RegTensor<U> res;
        AscendC::Reg::MaskReg sumMask = AscendC::Reg::CreateMask<U, AscendC::Reg::MaskPattern::VL1>();
        AscendC::Reg::Duplicate(res, static_cast<U>(0));
        for (uint16_t i = 0; i < repeatTimes; i++) {
            AscendC::Reg::MaskReg p0 = AscendC::Reg::UpdateMask<U>(dataCount_);            // 一次处理数量
            AscendC::Reg::AddrReg offset = AscendC::Reg::CreateAddrReg<T>(i, repeatCount); // 搬运偏移
            LoadXLocalToReg<T, U>(xLocalAddr, vd0, p0, offset);
            AscendC::Reg::Reduce<AscendC::Reg::ReduceType::SUM>(vd1, vd0, p0);
            AscendC::Reg::Add(res, res, vd1, sumMask);
        }
        if constexpr (NEED_ACCUMULATE) {
            UpdateSum<U>(res, storeLocalAddr, 0);
        }
        StoreOneValue<U, U>(storeLocalAddr, res, sumMask, 0);
    }
}

} // namespace Compute
} // namespace PoolUtils

#endif // POOL_UTILS_ARCH35_COMPUTE_ADAPTIVE_AVG_POOL_BIG_KERNEL_COMPUTE_H_
