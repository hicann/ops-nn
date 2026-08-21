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
 * \file compute.h
 * \brief A5 (ascend950) Vector Functions (__simd_vf__)
 */
#ifndef CROSS_ENTROPY_SUM_EXP_AND_INDEX_LOGIT_ARCH35_VF_COMPUTE_H_
#define CROSS_ENTROPY_SUM_EXP_AND_INDEX_LOGIT_ARCH35_VF_COMPUTE_H_

#include "kernel_operator.h"
#include "../cross_entropy_sum_exp_and_index_logit_common.h"

namespace CrossEntropySumExpAndIndexLogit {
using namespace AscendC;

// ===== VF1：mask/offset 计算（独立 VF，循环次数与 VF3 不等价，不融合）=====
// 0/1 常量由 VF 内 Duplicate 广播成 RegTensor（不占 UB，无需 zeroBuf/oneBuf）；
// LoadAlign 无 mask 按 256B 读 target（越界读相邻 UB 被 mask 掉，无害）；
// 比较极性取"在范围内"（t>=start && t<end，闭开区间 [vocabStart, vocabEnd)）：两条件非互斥，
// 用 MaskReg 版 And 求交集得 maskInRange；offset/mask 各一次 Select 即可
// （Select 语义：mask 位为 1 取 src0，为 0 取 src1，故交换 src0/src1 等价于 mask 取反，无需 Not）。
template <typename T>
__simd_vf__ inline void MaskOffsetVF(__ubuf__ int32_t* targetAddr, __ubuf__ int32_t* offsetAddr,
                                     __ubuf__ int32_t* maskAddr, int64_t vocabStart, int64_t vocabEnd, uint32_t curN,
                                     uint16_t repeatTimes)
{
    AscendC::Reg::RegTensor<int32_t> tgtReg;
    AscendC::Reg::RegTensor<int32_t> offReg;
    AscendC::Reg::RegTensor<int32_t> mskReg;
    AscendC::Reg::RegTensor<int32_t> zeroReg;
    AscendC::Reg::RegTensor<int32_t> oneReg;
    AscendC::Reg::MaskReg mask;
    AscendC::Reg::MaskReg maskGeStart;
    AscendC::Reg::MaskReg maskLtEnd;
    AscendC::Reg::MaskReg maskInRange;

    AscendC::Reg::Duplicate(zeroReg, static_cast<int32_t>(0));
    AscendC::Reg::Duplicate(oneReg, static_cast<int32_t>(1));

    for (uint16_t i = 0; i < repeatTimes; ++i) {
        mask = AscendC::Reg::UpdateMask<int32_t>(curN);
        AscendC::Reg::LoadAlign(tgtReg, targetAddr + i * (REPEAT_SIZE / sizeof(int32_t)));
        // t >= vocabStart（闭端）
        AscendC::Reg::Compares<int32_t, CMPMODE::GE>(maskGeStart, tgtReg, static_cast<int32_t>(vocabStart), mask);
        // t < vocabEnd（开端）
        AscendC::Reg::Compares<int32_t, CMPMODE::LT>(maskLtEnd, tgtReg, static_cast<int32_t>(vocabEnd), mask);
        // maskInRange = (t>=start) && (t<end)，等价于 !(t<start || t>=end)
        AscendC::Reg::And(maskInRange, maskGeStart, maskLtEnd, mask);
        AscendC::Reg::Adds(offReg, tgtReg, -static_cast<int32_t>(vocabStart), mask);
        // offset = inRange ? t - start : 0
        AscendC::Reg::Select(offReg, offReg, zeroReg, maskInRange);
        // mask_int = inRange ? 0 : 1（越界为 1，与 target_mask 语义一致）
        AscendC::Reg::Select(mskReg, zeroReg, oneReg, maskInRange);
        AscendC::Reg::StoreAlign(offsetAddr + i * (REPEAT_SIZE / sizeof(int32_t)), offReg, mask);
        AscendC::Reg::StoreAlign(maskAddr + i * (REPEAT_SIZE / sizeof(int32_t)), mskReg, mask);
    }
}

// ===== VF3：exp 计算（显式 for 循环逐 repeat），逐行调用（curN 恒传 1）=====
// BF16 输入先 Cast→FP32（显式 3 个模板参 T/U/CastTrait，S/V 由编译器推导），FP32 直通（Adds 直接以 inReg 为 src）；
// LoadAlign 无 mask 按 256B 读（越界读相邻 UB 被 mask 掉，无害），StoreAlign 带 mask 只写 curV 个元素。
template <typename T>
__simd_vf__ inline void ExpSumTileVF(__ubuf__ T* inAddr, __ubuf__ float* expOutAddr, float gmaxScalar, uint32_t curN,
                                     uint32_t curV, uint16_t repeatTimes)
{
    AscendC::Reg::RegTensor<float> calcReg;
    AscendC::Reg::MaskReg mask;
    // 每 repeat 按 FP32 lane 数步进（REPEAT_SIZE=256B / 4B = 64）：
    //   FP32 用 LoadAlign 读 64 元素；BF16 用 DIST_UNPACK_B16 每 repeat 解包 64 个 BF16。
    uint32_t fp32Lane = REPEAT_SIZE / sizeof(float);

    for (uint16_t i = 0; i < repeatTimes; ++i) {
        mask = AscendC::Reg::UpdateMask<float>(curV);
        if constexpr (std::is_same<T, float>::value) {
            AscendC::Reg::RegTensor<float> inReg;
            AscendC::Reg::LoadAlign(inReg, inAddr + i * fp32Lane);
            AscendC::Reg::Adds(calcReg, inReg, -gmaxScalar, mask);
        } else {
            AscendC::Reg::RegTensor<bfloat16_t> inRegB;
            AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(inRegB, inAddr + i * fp32Lane);
            AscendC::Reg::Cast<float, bfloat16_t, castTraitBf16ToFp32>(calcReg, inRegB, mask);
            AscendC::Reg::Adds(calcReg, calcReg, -gmaxScalar, mask);
        }
        AscendC::Reg::Exp(calcReg, calcReg, mask);
        AscendC::Reg::StoreAlign(expOutAddr + i * fp32Lane, calcReg, mask);
    }
}

} // namespace CrossEntropySumExpAndIndexLogit

#endif // CROSS_ENTROPY_SUM_EXP_AND_INDEX_LOGIT_ARCH35_VF_COMPUTE_H_
