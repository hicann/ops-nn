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
 * \file in_training_reduce_v2_common.h
 * \brief
 */
#ifndef IN_TRAINING_REDUCE_V2_COMMON_H_
#define IN_TRAINING_REDUCE_V2_COMMON_H_

#include "in_training_reduce_v2_tiling_data.h"
#include "kernel_operator.h"
#include "../inc/platform.h"

namespace INTrainingReduceV2Ops {
using namespace AscendC;
// 使用 AscendC::Reg（低阶 VF API 的规范命名空间），在 __NPU_ARCH__==3510 恒被填充（reg_compute struct_intf.h），
// 包构建（device-only pass）与 Kernel 直调单文件 .asc（host+device 双遍）均稳定解析。
using AscendC::Reg::CreateMask;
using AscendC::Reg::LoadDist;
using AscendC::Reg::LocalMemBar;
using AscendC::Reg::MaskPattern;
using AscendC::Reg::MaskReg;
using AscendC::Reg::MemType;
using AscendC::Reg::RegTensor;
using AscendC::Reg::StoreDist;
using AscendC::Reg::UpdateMask;

constexpr uint32_t VL_FP32 = platform::GetVRegSize() / sizeof(float);
constexpr uint32_t BLOCK_SIZE = platform::GetUbBlockSize();
constexpr uint32_t BLK_B32 = BLOCK_SIZE / sizeof(float);
constexpr uint32_t DOUBLE_BUFFER_NUM = 2;

// fp16 → fp32 提升（输入侧）。输出恒 fp32，故本算子无 fp32→fp16 回写的 castTraitB322B16。
constexpr AscendC::Reg::CastTrait castTraitB162B32 = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::UNKNOWN,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};

// 加载一整段（VL 宽）输入到 fp32 寄存器：fp32 直载；fp16 解包后 Cast 提升 fp32。
template <typename T_IN>
__aicore__ inline void LoadTensorForDtypeTIn(__local_mem__ T_IN* src, RegTensor<float>& dst, MaskReg& preg,
                                             uint32_t offset)
{
    if constexpr (IsSameType<T_IN, float>::value) {
        DataCopy<float, LoadDist::DIST_NORM>(dst, src + offset);
    } else {
        RegTensor<T_IN> xIn;
        DataCopy<T_IN, LoadDist::DIST_UNPACK_B16>(xIn, src + offset);
        Cast<float, T_IN, castTraitB162B32>(dst, xIn, preg);
    }
}

// 输出恒 fp32：把 fp32 寄存器的首元素（reduce 结果）写回 UB。
template <typename T_OUT>
__aicore__ inline void StoreOneElementForDtypeTOut(__local_mem__ T_OUT* dst, RegTensor<float>& src, MaskReg& preg,
                                                   uint32_t offset)
{
    DataCopy<T_OUT, StoreDist::DIST_FIRST_ELEMENT_B32>(dst + offset, src, preg);
}
} // namespace INTrainingReduceV2Ops
#endif // IN_TRAINING_REDUCE_V2_COMMON_H_
