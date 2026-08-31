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
 * \file pool_type_traits.h
 * \brief 池化系列算子 kernel 共用的通用 type trait。
 */

#ifndef POOL_UTILS_POOL_TYPE_TRAITS_H_
#define POOL_UTILS_POOL_TYPE_TRAITS_H_

#include <cstdint>
#include <type_traits>

namespace PoolUtils {
namespace TypeTraits {

// 无符号索引类型到 VCI 指令所需有符号类型的映射，其余类型保持不变。
template <typename T>
struct VciTypeGet {
    using type = typename std::conditional<
        std::is_same<T, uint32_t>::value, int32_t,
        typename std::conditional<
            std::is_same<T, uint16_t>::value, int16_t,
            typename std::conditional<std::is_same<T, uint64_t>::value, int64_t, T>::type>::type>::type;
};

} // namespace TypeTraits
} // namespace PoolUtils

#endif // POOL_UTILS_POOL_TYPE_TRAITS_H_
