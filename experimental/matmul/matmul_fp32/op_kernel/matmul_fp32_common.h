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
 * \file matmul_fp32_common.h
 * \brief
 */
#ifndef __MATMUL_FP32_COMMON_H__
#define __MATMUL_FP32_COMMON_H__

#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace AscendC;
using namespace matmul;

constexpr uint64_t BLOCK_BYTE_SIZE = 32;
constexpr uint64_t BLOCK_SIZE = 16;
constexpr uint64_t ALIGNED_H = 16;

// common MDL config
constexpr MatmulConfig MM_CFG_MDL = GetMDLConfig();

// set enUnitFlag
constexpr MatmulConfig MM_CFG_NO_PRELOAD = GetMDLConfig(false, false, 0, false, false, false, true);

/**
 * if b is 0, return a
 */
__aicore__ inline uint64_t MMFp32CeilDiv(uint64_t a, uint64_t b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

/**
 * if b is 0, return 0
 */
__aicore__ inline uint64_t MMFp32CeilAlign(uint64_t a, uint64_t b)
{
    return MMFp32CeilDiv(a, b) * b;
}

__aicore__ inline uint64_t GetCurrentBlockIdx()
{
    if ASCEND_IS_AIV {
        return GetBlockIdx() / GetTaskRation();
    }
    return GetBlockIdx();
}

#endif // __MATMUL_FP32_COMMON_H__