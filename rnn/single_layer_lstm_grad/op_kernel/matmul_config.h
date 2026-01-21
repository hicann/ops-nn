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
 * \file matmul_config.h
 * \brief
 */

#ifndef MATMUL_CONFIG_H
#define MATMUL_CONFIG_H

#include <cstdint>
#include <type_traits>
#include "kernel_operator.h"
#include "lib/matmul_intf.h"


constexpr MatmulConfig MM_CFG = GetNormalConfig(
    false  // intrinsicsLimit
);
constexpr MatmulConfig MM_HUGE_CFG = GetNormalConfig(
    true  // intrinsicsLimit
);

#ifdef __CCE_KT_TEST__
void EmptyTestFunc() {};
#endif  // __CCE_KT_TEST__

#endif  // MATMUL_CONFIG_H