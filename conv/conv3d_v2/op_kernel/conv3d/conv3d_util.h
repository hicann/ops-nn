/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file conv3d_util.h
 * \brief
 */

#ifndef CONV3D_UTIL_H
#define CONV3D_UTIL_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_utils.h"
#include "../conv_common/conv_util.h"

using namespace AscendC;

namespace conv3d {
constexpr uint64_t LOAD2D_MAX_REPEAT_TIMES = 255;
constexpr uint8_t RIGHT_MOVE_8 = 8;
constexpr uint32_t L0A_SIZE = 65536;
constexpr uint32_t L0B_SIZE = 65536;
constexpr uint32_t INT16_DIV_INT8 = 2;

__aicore__ inline uint64_t GetCurrentKD(uint64_t tilingKL1, uint64_t cin, uint64_t khxKw)
{
    return conv::CeilDIV(tilingKL1, cin * khxKw);
}
}  // namespace conv3d
#endif  // __CONV3D_UTIL_H__
