/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file group_norm_grad_base.h
 * \brief
 */
#ifndef GROUP_NORM_GRAD_KERNEL_BASE_H
#define GROUP_NORM_GRAD_KERNEL_BASE_H

#pragma once

#include "kernel_operator.h"
#include "op_kernel/platform_util.h"

namespace GroupNormGrad {
using namespace AscendC;

constexpr static uint32_t PF32_PER_BLOCK = Ops::Base::GetUbBlockSize() / sizeof(float); // UB块对应的fp32元素数
constexpr static uint32_t BLOCK_BYTES = Ops::Base::GetUbBlockSize();                    // UB搬运对齐块大小
constexpr static uint32_t VecLen_ = Ops::Base::GetVRegSize() / sizeof(float); // 矢量寄存器长度(fp32元素数)

// group_norm_grad kernel 公共轻量基类：无状态，供 mix 系列模板、
// 独立模板（如 empty）及后续新增 kernel 直接继承
class GroupNormGradBase {
public:
    __aicore__ inline GroupNormGradBase() = default;
    __aicore__ inline ~GroupNormGradBase() = default;
};
} // namespace GroupNormGrad
#endif // GROUP_NORM_GRAD_KERNEL_BASE_H
