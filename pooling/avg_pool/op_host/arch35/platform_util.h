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
 * \file platform_util.h
 * \brief platform utility functions for op_host tiling code (ported from canndev runtime/inc/platform.h)
 */

#ifndef AVG_POOL_OP_HOST_ARCH35_PLATFORM_UTIL_H_
#define AVG_POOL_OP_HOST_ARCH35_PLATFORM_UTIL_H_

#include <cstdint>

namespace optiling {
namespace platform {

/**
 * Get the block size of unified buffer in bytes
 */
template <typename T>
inline uint32_t GetUbBlockSize(T* context)
{
    return 32U;
}

/**
 * Get the size of vector registers in bytes
 */
template <typename T>
inline uint32_t GetVRegSize(T* context)
{
    return 256U;
}

/**
 * Get the cache line size in bytes
 */
template <typename T>
inline uint32_t GetCacheLineSize(T* context)
{
    return 256U;
}

}  // namespace platform
}  // namespace optiling

#endif  // AVG_POOL_OP_HOST_ARCH35_PLATFORM_UTIL_H_
