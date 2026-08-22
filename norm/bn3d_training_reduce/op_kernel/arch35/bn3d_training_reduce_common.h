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
 * \file bn3d_training_reduce_common.h
 * \brief
 */
#ifndef BN3D_TRAINING_REDUCE_COMMON_H_
#define BN3D_TRAINING_REDUCE_COMMON_H_

#include "bn3d_training_reduce_tiling_data.h"
#include "kernel_operator.h"
#include "../inc/platform.h"

namespace BN3DTrainingReduceOps {
using namespace AscendC;
// 直接使用 AscendC::Reg（低阶 VF API 的规范命名空间；AscendC::MicroAPI 只是其别名）。
// Reg 在 __NPU_ARCH__ == 3510 恒被填充，包构建与 Kernel 直调单文件 .asc 两种编译方式下都稳定解析。
using AscendC::Reg::Compare;
using AscendC::Reg::CreateMask;
using AscendC::Reg::LoadDist;
using AscendC::Reg::Select;
// 多累加槽归并要在同一个 __VEC_SCOPE__ 内跨趟复用 UB：本趟 store 必须先于下趟 load 生效。
using AscendC::Reg::LocalMemBar;
using AscendC::Reg::MaskPattern;
using AscendC::Reg::MaskReg;
using AscendC::Reg::MemType;
using AscendC::Reg::RegTensor;
using AscendC::Reg::StoreDist;
using AscendC::Reg::UpdateMask;

constexpr uint32_t VL_FP32 = platform::GetVRegSize() / sizeof(float);
constexpr uint32_t BLOCK_SIZE = platform::GetUbBlockSize();
constexpr uint32_t DOUBLE_BUFFER_NUM = 2;

// fp16 / bf16 → fp32 提升。输出恒 fp32，故无 fp32 → b16 的回写 trait。
// 平方必须发生在提升之后：低精度下 x*x 会提前溢出/丢精度。
constexpr AscendC::Reg::CastTrait castTraitB162B32 = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::UNKNOWN,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};

// 载入一整段（VL 宽）输入并统一提升为 fp32 寄存器。
// fp32 直载；fp16 / bf16 走同一条 B16 解包 + Cast 路径（两者只是位解释不同，指令序列一致）。
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

// 把 fp32 寄存器的首元素（水平归约结果）写回 UB 的第 offset 个 fp32 槽位。
__aicore__ inline void StoreOneFp32(__local_mem__ float* dst, RegTensor<float>& src, MaskReg& preg, uint32_t offset)
{
    DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dst + offset, src, preg);
}
} // namespace BN3DTrainingReduceOps
#endif // BN3D_TRAINING_REDUCE_COMMON_H_
